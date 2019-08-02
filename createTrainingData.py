#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 12 14:28:53 2019

@author: dudley
"""

import lasio, os
import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
from sklearn.feature_extraction.image import extract_patches
from joblib import Parallel, delayed

 
#lasDirs = {'eagleford' : '/media/hdd/datasets/Eagleford/RAW Data/Kingdom',
#           'kansas' : '/media/hdd/datasets/KansasOG/LAS',
#           'northdakota' : '/media/hdd/datasets/ND/LAS',
#           'ohio' : '/media/hdd/datasets/Ohio'}

lasDirs = {'wy' : '/media/hdd/datasets/WY/LASFiles/'}


def readLasFiles(fileName, fileDir, loc):
    
    try:
        file = os.path.join(fileDir, fileName)
        header = lasio.read(file, ignore_data=True)
        if 'GR' in list(header.curvesdict.keys()):
            las = lasio.read(file)
            df = pd.DataFrame(las.df()['GR'])
            df = df.replace(np.inf, np.nan)
            df = df.dropna()
            if len(df) > 50:
                dept = df.index
                log = df['GR']
                deptInc = np.random.choice(np.arange(5, 35, 5)) / 10
                deptNew = np.arange(np.ceil(dept.min()), 
                                    np.floor(dept.max()), 
                                    deptInc)
                func = interp1d(dept, log)
                logNew = func(deptNew)
                df = pd.DataFrame()
                df['GR'] = logNew
                df['Well'] = fileName.split('.')[0]
                df['Loc'] = loc
                
                return(df)
           
    except Exception as e:
        print(e)


def saveLasPatches(lasData):
    
    try:
        window = 256
        hw = window // 2
        dirName = lasData['Loc'].iloc[0]
        wells = lasData['Well'].unique()
        for well in wells:            
            data = lasData[lasData['Well'] == well]['GR'].values
            pad = np.pad(data, (hw, hw), 'reflect')
            patch = extract_patches(pad, window, 10)
            patch = patch[np.newaxis]
            patch = np.moveaxis(patch, 1, 0)
            np.save('/media/hdd/autoprop/trainingData/{}_{}'.format(dirName, well), patch)
            
    except Exception as e:
        print(e)
        
        
datas = {}
for loc, path in lasDirs.items():
    files = os.listdir(path)[::2]
    datas[loc] = Parallel(n_jobs=6)(delayed(readLasFiles)(i, path, loc) for i in files)
    datas[loc] = [i for i in datas[loc] if i is not None]
    datas[loc] = pd.concat(datas[loc], ignore_index=True)
    datas[loc]['GR'] = np.clip(datas[loc]['GR'], 
                               np.percentile(datas[loc]['GR'], 1),
                               np.percentile(datas[loc]['GR'], 99))
    stats = datas[loc]['GR'].describe()
    datas[loc]['GR'] -= stats.loc['mean']
    datas[loc]['GR'] /= stats.loc['std']
    stats = datas[loc]['GR'].describe()
    datas[loc]['GR'] -= stats.loc['min']
    datas[loc]['GR'] /= (stats.loc['max'] - stats.loc['min'])
    
#lasData = Parallel(n_jobs=6)(delayed(getLasData)(k, v) for k, v in lasDirs.items())
Parallel(n_jobs=6)(delayed(saveLasPatches)(data) for data in datas.values())