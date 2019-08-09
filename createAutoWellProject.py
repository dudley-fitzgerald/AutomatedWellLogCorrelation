#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 22 11:16:21 2019

@author: dudley
"""

import os, lasio, argparse
import pandas as pd


'''
This script is intended to create a project file containing all of the data
necessary to run the auto correlation script.  A HDF5 data store will be created
that contains well header information as well as well log data, both stored as
pandas data frames.  In order for this script to run, the LAS file name must 
be the UWI/API number.  

Run from the command line and provide required command line arguements:
    
--proj-name : The name that of the project, this will be the file name
--proj-dir : Location where the project file will be saved
--header-dir : Location of the header file for the wells in the analysis
--las-dir : Location of the LAS files to be used in the analysis
--uwi-col : Name of the column in the header file to be used as the well identifier
--x-col : Name of the column in the header file to be used as the X coordinate
--y-col : Name of the column in the header file to be used as the Y coordinate
--log-name : Name of the well log to be used in the analysis
--uwi-subset : List of UWIs to use in the analysis

'''


class CreateProject:
    
    def __init__(self, args):
        
        self.projName = args.proj_name
        self.projDir = args.proj_dir
        self.headerDir = args.header_dir
        self.lasDir = args.las_dir
        self.uwiCol = args.uwi_col
        self.xCol = args.x_col
        self.yCol = args.y_col
        self.logName = args.log_name
        self.uwiSubset = args.uwi_subset
        
        self.makeProjectDir()
        self.readHeaderFile()
        self.readLasFiles()
    
    
    # Determine if project directory exists, if not create it
    def makeProjectDir(self):
        
        if not os.path.exists(self.projDir):
            os.makedirs(self.projDir)
            
            
    # Read the header file and save it to directory
    def readHeaderFile(self):
        
        # Read header and subset file to appropriate columns
        self.header = pd.read_csv(self.headerDir)
        self.header = self.header[[self.uwiCol, self.xCol, self.yCol]]
        self.header.columns = ['UWI', 'X', 'Y']
        
        # If a subset of UWI's have been specified, limit file to those entries
        if len(self.uwiSubset) > 0:
            self.header = self.header[self.header['UWI'].isin(self.uwiSubset)]
        
        # Create a HDF5 data store and save file
        path = os.path.join(self.projDir, '{}.eg'.format(self.projName))
        with pd.HDFStore(path, 'a') as store:
            store['header'] = self.header
            
        
    # Read LAS files and save to directory
    def readLasFiles(self):
        
        # Open HDF5 data store and save LAS files as pandas DataFrames
        path = os.path.join(self.projDir, '{}.eg'.format(self.projName))
        with pd.HDFStore(path, 'a') as store:
            
            # Get list of files in LAS directory
            if os.path.exists(self.lasDir):
                for f in os.listdir(self.lasDir):
                    try:
                        # Get UWI name from file name
                        uwi = f.split('.')[0]
                        # Get index of well from header
                        uid = self.header[self.header['UWI'] == int(uwi)].index[0]
                        # Read LAS data
                        path = os.path.join(self.lasDir, f)
                        las = lasio.read(path, ignore_data=True)
                        # Determine if LAS file has appropriate log, remove
                        # null values, and save
                        if self.logName in las.df().columns:
                            df = pd.DataFrame(las.df()[self.logName]).dropna()
                            store['/log/{}'.format(uid)] = df
                            
                    except Exception as e:
                        print(e) 
                    
                    
if __name__ == '__main__':

    # Get command line arguements
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg('--proj-name', type=str, default='autoWell')
    arg('--proj-dir', type=str)
    arg('--header-dir', type=str)
    arg('--las-dir', type=str)
    arg('--uwi-col', type=str, default='UWI')
    arg('--x-col', type=str, default='X')
    arg('--y-col', type=str, default='Y')
    arg('--log-name', type=str, default='GR')
    arg('--uwi-subset', type=list, default=[])
    args = parser.parse_args()
    
    CreateProject(args)