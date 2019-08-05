#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 12 14:28:53 2019

@author: dudley
"""

import lasio, os, argparse
import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
from sklearn.feature_extraction.image import extract_patches
from joblib import Parallel, delayed


'''
This script is intended to create training data that can be fed into the 
convolutional autoencoder found in this repository.  

Run from the command line and provide required command line arguements:
    
--data-name : The name that will be prefixed to all training data
--data-dir : Location all LAS files wished to be included in training data
--output-dir : Location where all output will be saved
--cpu-count : Number of cpu's used while processing
--patch-size : The window of data used to extract patches of log values
--clip-log : Boolean - if anomalous values are present this will clip to p1 / p99
--log-name : Name of log to use in the analysis
--skip-inc : This specificies how many samples to skip in each well log
--depth-inc : Specifies the sampling rate the logs will be resampled to

'''


class CreateTrainingData:
    
    def __init__(self, args):
        
        self.dataName = args.data_name
        self.dataDir = args.data_dir
        self.outputDir = args.output_dir
        self.cpuCount = args.cpu_count
        self.patchSize = args.patch_size
        self.clipLog = args.clip_log
        self.logName = args.log_name
        self.skipInc = args.skip_inc
        self.deptInc = args.depth_inc
        
        self.run()
        
        
    def readLasFiles(self, fileName):
    
        try:
            
            # Read LAS Header
            header = lasio.read(fileName, ignore_data=True)
            
            # Check if log in header, read data, remove Nan, resample to common increment
            if self.logName in list(header.curvesdict.keys()):
                las = lasio.read(fileName)
                df = pd.DataFrame(las.df()[self.logName])
                df = df.replace(np.inf, np.nan)
                df = df.dropna()
                
                if len(df) > 50:
                    dept = df.index
                    log = df[self.logName]
                    deptNew = np.arange(np.ceil(dept.min()), 
                                        np.floor(dept.max()), 
                                        self.deptInc)
                    func = interp1d(dept, log)
                    logNew = func(deptNew)
                    df = pd.DataFrame()
                    df[self.logName] = logNew
                    df['Well'] = os.path.basename(fileName).split('.')[0]
                    
                    return(df)
               
        except Exception as e:
            print(e)    
            
            
    def run(self):
        
        try:
        
            # Get path to all LAS files in directory
            files = os.listdir(self.dataDir)
            files = [os.path.join(self.dataDir, i) for i in files if i.endswith('.las')]
            
            # Get LAS data for specified files
            lasData = Parallel(n_jobs=self.cpuCount)(delayed(self.readLasFiles)(f) for f in files)
            lasData = [i for i in lasData if i is not None]
            lasData = pd.concat(lasData, ignore_index=True)
            
            # Scale data
            if self.clipLog:
                p1 = np.percentile(lasData[self.logName], 1)
                p99 = np.percentile(lasData[self.logName], 99)
                lasData[self.logName] = np.clip(lasData[self.logName], p1, p99)
            stats = lasData[self.logName].describe()
            lasData[self.logName] -= stats.loc['mean']
            lasData[self.logName] /= stats.loc['std']
            stats = lasData[self.logName].describe()
            lasData[self.logName] -= stats.loc['min']
            lasData[self.logName] /= (stats.loc['max'] - stats.loc['min'])
            
            # Extract patches and save to disk
            wellGrps = lasData.groupby('Well')
            Parallel(n_jobs=self.cpuCount)(delayed(self.saveLasPatches)(wellGrps.get_group(i), i) for i in lasData['Well'].unique())
        
        except Exception as e:
            print('Something broke', e)
            
            
    def saveLasPatches(self, lasData, wellName):
    
        try:
            
            #  Pad data, extract patches, save to disk
            window = self.patchSize
            hw = window // 2       
            data = lasData[self.logName].values
            pad = np.pad(data, (hw, hw), 'reflect')
            patch = extract_patches(pad, window, self.skipInc)
            patch = patch[np.newaxis]
            patch = np.moveaxis(patch, 1, 0)
            outputFile = '{}_{}'.format(self.dataName, wellName)
            outputPath = os.path.join(self.outputDir, outputFile)
            np.save(outputPath, patch)
                
        except Exception as e:
            print(e)
        

if __name__ == '__main__':

    # Get command line arguements
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg('--data-name', type=str, default='input')
    arg('--data-dir', type=str)
    arg('--output-dir', type=str)
    arg('--cpu-count', type=int, default=os.cpu_count())
    arg('--patch-size', type=int, default=256)
    arg('--clip-log', type=bool, default=True)
    arg('--log-name', type=str, default='GR')
    arg('--skip-inc', type=int, default=10)
    arg('--depth-inc', type=float, default=1)
    args = parser.parse_args()
    
    CreateTrainingData(args)