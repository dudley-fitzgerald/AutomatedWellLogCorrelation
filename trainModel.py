#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 12 19:01:52 2018

@author: dudley
"""

import matplotlib.pyplot as plt
import numpy as np
import os, argparse

import torch
from torch import nn
from torch.autograd import Variable
from torch.optim import Adam

from vae import ConvAEDeep


'''
This script is intended to train a 1D Convolutional Autoencoder for the purpose
of generating feature vectors from input well logs that can be used in the 
process of automated well correlation.  

Run from the command line and provide required command line arguements:
    
--model-name : Name of model being trained
--data-dir : Location training data
--output-dir : Location where model will be saved
--imgs-dir : Location where the images will be saved
--save-imgs : Save images or not
--epochs : Number of epochs to train model
--batch-size : Not exactly batch size, how to split the training data into batches
--split-ratio : Test, train split ratio
--skip-inc : Decimation factor, skip training data to reduce memory overhead

'''


class TrainModel:
    
    def __init__(self, args):
        
        self.modelName = args.model_name
        self.dataDir = args.data_dir
        self.dataSplitRatio = args.split_ratio
        self.outputDir = args.output_dir
        self.saveImgs = args.save_imgs
        self.imgsDir = args.imgs_dir
        self.epochs = args.epochs
        self.batchSize = args.batch_size
        self.skipInc = args.skip_inc
        
        self.run()
        
     
    # Retrieve the training data from disk and format appropriately
    def getTrainingData(self):
        
        # Get files from directory, subsample, and shuffle
        files = os.listdir(self.dataDir)[::self.skipInc]
        np.random.shuffle(files)
        
        # Determine split and load data
        trainTestSplit = int(len(files) * self.dataSplitRatio)
        trainData = [np.load(os.path.join(self.dataDir, f)) for f in files[:trainTestSplit]]
        trainData = np.vstack(trainData)
        testData  = [np.load(os.path.join(self.dataDir, f)) for f in files[trainTestSplit:]]
        testData = np.vstack(testData)
        self.data = {'train' : trainData, 'test' : testData}
        
        print('Training data loaded')
        
        
    # Create a plot of training & testing loss and Input vs Predicted data
    def plotData(self, trainLoss, testLoss, inImg, outImg, epoch, phase):
        
        # Initialize plot
        f, ax = plt.subplots(1, 2, figsize=(20, 5))
        
        # Create plot of testing and training loss
        ax[0].plot(trainLoss, label='Train loss')
        ax[0].plot(testLoss, label='Test loss')
        ax[0].set_ylabel('Loss (log)')
        ax[0].legend()
        ax[0].set_yscale('log')
        
        # Create plot of Input curve vs Predicted curve
        ax[1].set_title('Curves')
        ax[1].plot(inImg, c='b', label='Target')
        ax[1].plot(outImg, c='r', label='Predicted')
        ax[1].set_yticklabels([])
        ax[1].legend()
        
        plt.subplots_adjust(wspace=0)
        path = os.path.join(self.imgsDir,
                            'epoch_{}_{}.png'.format(self.modelName, epoch, phase))
        f.savefig(path)
        
        
    # Run the script
    def run(self):
        
        self.setPaths()
        self.getTrainingData()
        self.setModel()
        self.trainModel()
        self.saveModel()
        
        
    # Save model to disk
    def saveModel(self):
        
        path = os.path.join(self.outputDir, self.modelName)
        torch.save(self.model.state_dict(), path)
        
        print('Model saved')
        
        
    # Import model, set optimizer and criterion
    def setModel(self):
        
        self.model = ConvAEDeep().cuda()
        self.optimizer = Adam(self.model.parameters(), lr=1e-3)
        self.criterion = nn.MSELoss().cuda()
        
        print('Model set')
        
        
    # Assess whether paths provided exist, if not create
    def setPaths(self):
        
        if not os.path.exists(self.dataDir):
            print('Directory of Training Data does not exist')
            
        if not os.path.exists(self.outputDir):
            os.mkdir(self.outputDir)
            
        if not os.path.exists(self.imgsDir):
            os.mkdir(self.imgsDir)
        
    
    # Train model
    def trainModel(self):
        
        # Record of running losses
        trainLoss = []
        testLoss = []
        
        # Iterate over model 
        for epoch in range(self.epochs) :
            
            # For each epoch differentiate between testing and training pahses
            for phase in ['train', 'test']:
                
                # Set the model mode and determine the appropriate number of batches
                if phase =='train':
                    self.model.train()
                    shape = self.data['train'].shape[0]
                    splits = np.array_split(range(shape), self.batchSize)
                else:
                    self.model.eval()
                    shape = self.data['test'].shape[0]
                    splits = np.array_split(range(shape), self.batchSize)           
                
                # Iterate over batches, get outputs, compute error, update model weights
                running_loss = 0.0
                for num, batch in enumerate(splits):
                    inputs = Variable(torch.from_numpy(self.data[phase][batch])).float().cuda()
                    self.optimizer.zero_grad()
                    
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = self.model(inputs)
                        loss = self.criterion(outputs, inputs)
                        if phase == 'train':
                            loss.backward()
                            self.optimizer.step()
                            
                    running_loss += loss.item() * inputs.size(0)
                    
                # Capture epoch loss
                epoch_loss = running_loss / inputs.size(0)
                if phase == 'train':
                    trainLoss.append(epoch_loss)
                else:
                    testLoss.append(epoch_loss)
                    
                # Save images 
                if self.saveImgs:
                    imgNum = np.random.randint(0, inputs.size(0))
                    inImg = inputs[imgNum].cpu().data.numpy().squeeze()
                    outImg = outputs[imgNum].cpu().data.numpy().squeeze()
                    self.plotData(trainLoss, testLoss, inImg, outImg, epoch, phase)
                    
                print('-' * 50)
                print('Epoch {} {}'.format(epoch, phase))
                print('Loss: {}'.format(epoch_loss))
                
                
                
if __name__ == '__main__':

    # Get command line arguements
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg('--model-name', type=str, default='model')
    arg('--data-dir', type=str)
    arg('--output-dir', type=str)
    arg('--imgs-dir', type=str)
    arg('--save-imgs', type=bool, default=True)
    arg('--epochs', type=int, default=250)
    arg('--batch-size', type=int, default=25)
    arg('--split-ratio', type=float, default=0.75)
    arg('--skip-inc', type=int, default=1)
    args = parser.parse_args()
    
    TrainModel(args)