#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 11 07:32:54 2019

@author: dudley
"""

from itertools import count

from heapq import heappush, heappop
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter
from scipy.signal import argrelextrema
from scipy.spatial.distance import cdist
from sklearn.feature_extraction.image import extract_patches
from sklearn.linear_model import RANSACRegressor
import networkx as nx

import torch

from vae import ConvAEDeep

'''
The AutoWellCorrelation class is intended to automate the well log correlation 
process using an adapted 1D version of panoramic stiching commonly utilitized 
in image processing to merge two or more overlapping images.  This class will load 
a pre generated project, load a PyTorch model, process the log data, identify 
reasonable well pairs, and then build a connectivity graph based on key points
that were identified using a 1D SIFT(ish) workflow.  This connectivity graph can
then be queried to find the most likely corresponding depths between all wells.

This class also provides a series of utility functions to preview data to ensure
quality control.

'''


class AutoWellCorrelation:
    
    def __init__(self, projPath, modelPath, resampleInc=2, minZ=None, maxZ=None,
                 maxOffset=1000, smSigma=5, numNeighbors=6):
        
        self.projPath = projPath
        self.modelPath = modelPath
        self.resampleInce = resampleInc
        self.minZ = minZ
        self.maxZ = maxZ
        self.maxOffset = maxOffset
        self.smSigma = smSigma
        self.numNeighbors = numNeighbors
        self.windowSize = 256
        self.halfWindow = self.windowSize // 2
        
        self.getModel()
        self.getWellData()   
        self.getWellPairs()
        self.scaleLogData()
        self.computePatches()
        
        
    # Create a NetworkX graph to store likely correlations between all wells 
    # across all depths
    def buildConnectivityGraph(self):
        
        # Define the weighted graph
        self.G = nx.Graph()
        
        # Iterate over all well pairs (u & v are well identifiers, w is distance)
        for u,v,w in self.wellPairs.edges(data=True):
            try:
                
                # Get necessary data and assign to variables
                uid1 = u
                uid2 = v
                tvd1 = self.logData[self.logData['uid'] == uid1].index.values
                tvd2 = self.logData[self.logData['uid'] == uid2].index.values
                
                # Create feature vectors
                vector1 = self.getFeatureVector(uid1)
                vector2 = self.getFeatureVector(uid2)
                
                # Compute DoG and Key Points for each well in pair
                dog1 = self.computeDoG(uid1)
                dog2 = self.computeDoG(uid2)
                kP1 = self.getKeyPoints(dog1)                
                kP2 = self.getKeyPoints(dog2)
                
                # Compute vector and depth distance matrices
                vectorDist = self.computeVectorDistance(vector1, vector2, kP1, kP2)
                tvdDist = self.computeTvdDistance(tvd1, tvd2, kP1, kP2)
                
                # Mask the distance matrices based on maximum allowable offset 
                # between well pair
                vectorMask, tvdMask = self.computeVectorMask(vectorDist, tvdDist)
                
                # Identify matching points and retrieve weights at those locations
                mPs = self.getMatchPoints(vectorMask)
                weights = vectorMask[mPs[:, 1], mPs[:, 0]]
                
                # Iterate over matching points, get depth value and add to Graph
                for num, mP in enumerate(mPs):
                    n1 = (uid1, tvd1[kP1[mP[0]]])
                    n2 = (uid2, tvd2[kP2[mP[1]]])                    
                    self.G.add_edge(n1, n2, weight=weights[num])
            
            except Exception as e:
                print(e)
        
        
    # Compute Difference of Gaussians
    def computeDoG(self, uid, k=1.6, sigmaFactor=2):
        
        # Retrieve log name from data frame
        logName = self.logData.columns[0]
        
        # Get log data for specified well identifier
        logData = self.logData[self.logData['uid'] == uid][logName].values
        
        # Create placeholder for values
        dog = np.zeros((logData.size, 4))
        
        # Iterate over sigma values, compute filtered values, generate difference
        for idx, sigma in enumerate(np.arange(1, 9, 2) * sigmaFactor):
            f1 = gaussian_filter(logData, sigma)
            f2 = gaussian_filter(logData, sigma * k)
            dog[:, idx] = f1 - f2
            
        return dog
    
    
    # Generate patches for each well
    def computePatches(self):
        
        # Retrieve log name from data frame
        logName = self.logData.columns[0]
        
        # Iterate over well identifiers, pad data, extract patches, add to dict
        self.patches = {}
        for uid in self.logData['uid'].unique():
            data = self.logData[self.logData['uid'] == uid]
            log = data[logName].values
            pad = np.pad(log, (self.halfWindow, self.halfWindow), mode='reflect')
            patch = extract_patches(pad, self.windowSize)
            patch = patch[np.newaxis]
            patch= np.moveaxis(patch, 1, 0)            
            self.patches[uid] = patch
            
            
    # Compute a distance matrix for the depth values of each well using Key Points
    def computeTvdDistance(self, tvd1, tvd2, keyPoints1, keyPoints2):
        
        return cdist(tvd1[keyPoints1].reshape(-1, 1), tvd2[keyPoints2].reshape(-1, 1))
    
    
    # Compute a distance matrix for the vector values of each well using Key Points
    def computeVectorDistance(self, vector1, vector2, keyPoints1, keyPoints2, exp=1):
        
        return cdist(vector1[keyPoints1] ** exp, vector2[keyPoints2] ** exp, 'cosine')
    
    
    # Set values of vector distance matrix outside of the allowable offset to INF
    def computeVectorMask(self, vectorDist, tvdDist, exp=3):
        
        vectorMask = vectorDist.copy()
        vectorMask[tvdDist > self.offset] = np.inf    
        
        tvdMask = ((tvdDist / self.offset) + 1) ** exp
        
        return vectorMask, tvdMask
    
    
    # Compute the feature vector for specified well identifier
    def getFeatureVector(self, uid):
        
        # Create a PyTorch tensor from the specified well identifier
        tensor = torch.from_numpy(self.patches[uid]).float().cuda()
        
        # Pass the tensor through the models encode function
        vector = self.model.encode(tensor).cpu().data.numpy()
        
        # Reshape vector
        vector = vector.reshape(vector.shape[0], -1)
        
        return(vector)
        
        
    # Find the key points from the provided dog along the specified axis
    def getKeyPoints(self, dog, axis=1):
        
        # Get the peaks, then troughs from the dog along the specified axis
        kp = argrelextrema(dog[:, axis], np.greater)[0].tolist()
        kp += argrelextrema(dog[:, axis], np.less)[0].tolist()
        kp.sort()
        
        return kp
    
    
    # Function to automatically find well marker provided a source location
    # This could be improved significantly
    def getMarker(self, source):
        
        # Utility variables
        push = heappush
        pop = heappop    
        c = count()
        
        # Placeholders 
        explored = []
        queue = []
        pick = []
        
        # Set up heap
        push(queue, (next(c), source, 0)) # _, node, weight
        
        # Iterate over queue while there are still nodes that haven't been evaluated
        while queue:
            
            # Extract the lowest weight node from the queue with its weight
            _, curnode, weight = pop(queue)
            
            # If well has already been explored continue
            if curnode[0] in explored:
                continue
            
            # Add well to explored list and append node to pick
            explored.append(curnode[0])
            pick.append(curnode)
            
            # Find edges of curnode and add to heap
            for k,v in self.G[curnode].items():
                push(queue, (next(c), k, v['weight']))
                
        return(pick)
    
    
    # Find matching points in the vector mask
    def getMatchPoints(self, vectorMask):
        
        # Find the indices of the minimum value along the 0 axis
        iPts = np.dstack((np.arange(vectorMask.shape[1]), 
                          vectorMask.argmin(axis=0)))[0]
        
        # Find the indices of the minimum value along the 1 axis
        jPts = np.dstack((vectorMask.argmin(axis=1), 
                          np.arange(vectorMask.shape[0])))[0]
        
        # Stack points together, identify points that occurr multiple times
        pts = np.vstack((iPts, jPts))
        uni = np.unique(pts, axis=0, return_counts=True)
        uni = uni[0][uni[1] > 1]
        
        # Pass points through RANSAC to remove outlier points
        reg = RANSACRegressor(residual_threshold=2,
                              min_samples=np.ceil(uni.shape[0] * 0.75))
        reg.fit(uni[:, 0].reshape(-1, 1), uni[:, 1].reshape(-1,1))
        
        return uni[reg.inlier_mask_]
    
    
    # Load the model
    def getModel(self):
        
        self.model = ConvAEDeep().cuda()
        self.model.load_state_dict(torch.load(self.modelPath))
    
    
    # Load the well data
    def getWellData(self):
        
        # Open the HDF5 store
        with pd.HDFStore(self.projPath) as store:
            
            # Retrieve the header data
            self.coords = store['header'].loc[:, ['X', 'Y']]
            
            # Iterate through the well identifiers, load log data, filter by depth,
            # check the lenth of the log, resample, smooth, and add to list
            logData = []
            for uid in self.coords.index.tolist():
                data = store['/log/{}'.format(uid)]
                data['uid'] = uid
                
                if self.minZ is not None and self.maxZ is not None:
                    idx = np.logical_and(data.index > self.minZ, data.index < self.maxZ)
                    data = data.loc[idx]
                elif self.minZ is not None:
                    data = data[data.index > self.minZ]
                elif self.maxZ is not None:
                    data = data[data.index < self.maxZ] 
                    
                if len(data) > 50:
                    data = self.resampleLog(data)
                    data = self.smoothLog(data)
                    
                    logData = self.logData.append(data)
                
            # Create a "master" log data data frame
            self.logData = pd.concat(logData, ignore_index=True)
            
            
    # Identify pairs of wells based on proximity to each other
    def getWellPairs(self):
        
        # Create a distance matrix between wells
        dist = cdist(self.coords.values, self.coords.values)
        
        # Iteratve over distance matrix, find wells that aren't neighbors based
        # on the numNeighbor variable, and set to nan
        for i in range(dist.shape[0]):
            row = dist[i]
            dist[i, row.argsort()[self.numNeighbors:]] = np.nan
            
        # Normalize the distance
        dist = (dist - np.nanmin(dist)) / (np.nanmax(dist) - np.nanmin(dist))
        
        # Create a NetworkX graph from distance matrix and assign well identifiers
        # as well node labels
        self.wellPairs = nx.from_numpy_array(dist)
        mapping = dict(zip(self.wellPairs.nodes(), self.coords.index))
        self.wellPairs = nx.relabel_nodes(self.wellPairs, mapping)
        
        # Remove edges that are nan
        remove = [(u,v) for u,v,w in self.wellPairs.edges(data=True) if np.isnan(w['weight'])]
        self.wellPairs.remove_edges_from(remove)
        
        
    # Resample the log values to a common increment
    def resampleLog(self, logData):
        
        # Get log name and create a placeholder data frame
        logName = logData.columns[0]
        newLogData = pd.DataFrame(columns=self.logData.columns)
        
        # Extract depth and log values
        tvd = logData.index.values
        log = logData[logName].values
        
        # Create an interpolation function mapping depth to log values
        func = interp1d(tvd, log)
        
        # Create resampled depth and log values
        tvdNew = np.arange(np.ceil(tvd.min()), np.floor(tvd.max()), self.resampleInc)
        logNew = func(tvdNew)
        
        # Assign placeholder data frame new values
        newLogData[logName] = logNew
        newLogData.index = tvdNew
        
        return newLogData
    
    
    # Scale the log data
    def scaleLogData(self):
        
        # Get log name and clip log data to P1 and P99
        logName = self.logData.columns[0]
        self.logData[logName] = np.clip(self.logData[logName], 
                                        np.percentile(self.logData[logName], 1), 
                                        np.percentile(self.logData[logName], 99))
        
        # Remove the mean and divide by std
        stats = self.logData[logName].describe()
        self.logData[logName] -= stats.loc['mean']
        self.logData[logName] /= stats.loc['std']
        stats = self.logData[logName].describe()
        
        # Scale between 0 and 1
        self.logData[logName] -= stats.loc['min']
        self.logData[logName] /= (stats.loc['max'] - stats.loc['min'])
            
            
    # Smooth the specified log data
    def smoothLog(self, logData):
        
        # Retrieve log name and if the smooth value is not None apply smoothing
        logName = logData.columns[0]
        if self.smooth is not None:
            logData[logName] = gaussian_filter(logData[logName].values, self.smooth)
            
        return logData