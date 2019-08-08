#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  9 07:21:09 2019

@author: dudley
"""

import torch
from torch import nn
from torch.autograd import Variable
    
    
class ConvAEDeep(nn.Module):
    
    def __init__(self):
        
        super(ConvAEDeep, self).__init__()
        
        # Create pooling and activation functions
        self.pool = nn.MaxPool1d(2, 2)
        self.relu = nn.ReLU(inplace=True)
        
        # Create 1D convolutional encoding layers
        self.conv1 = nn.Conv1d(1, 16, 3, padding=1)  
        self.conv2 = nn.Conv1d(16, 32, 3, padding=1)
        self.conv3 = nn.Conv1d(32, 64, 3, padding=1)
        self.conv4 = nn.Conv1d(64, 128, 3, padding=1)
        self.conv5 = nn.Conv1d(128, 256, 3, padding=1)        
        
        # Create 1D inverse convolutional decoding layers
        self.dec5 = nn.ConvTranspose1d(256, 128, 2, stride=2)
        self.dec4 = nn.ConvTranspose1d(128, 64, 2, stride=2)
        self.dec3 = nn.ConvTranspose1d(64, 32, 2, stride=2)
        self.dec2 = nn.ConvTranspose1d(32, 16, 2, stride=2)
        self.dec1 = nn.ConvTranspose1d(16, 1, 2, stride=2)
        
        
    # Encoding portion of Auto Encoder
    def encode(self, x):
        
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(self.pool(x)))
        x = self.relu(self.conv3(self.pool(x)))
        x = self.relu(self.conv4(self.pool(x)))
        
        return(x)
        
        
    # Decoding portion of Auto Encoder
    def decode(self, x):
        
        x = self.pool(x)
        x = self.relu(self.dec4(x))
        x = self.relu(self.dec3(x))
        x = self.relu(self.dec2(x))
        x = self.relu(self.dec1(x))
        
        return(x)


    def forward(self, x):
        
        x = self.encode(x)
        x = self.decode(x)
        x = torch.tanh(x)
                
        return x