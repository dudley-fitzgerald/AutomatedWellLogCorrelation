#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  9 07:21:09 2019

@author: dudley
"""

import os

import torch
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms   
    
    
class ConvAEDeep(nn.Module):
    
    def __init__(self):
        
        super(ConvAEDeep, self).__init__()
        
        self.pool = nn.MaxPool1d(2, 2)
        self.relu = nn.ReLU(inplace=True)
        
        self.conv1 = nn.Conv1d(1, 16, 3, padding=1)  
        self.conv2 = nn.Conv1d(16, 32, 3, padding=1)
        self.conv3 = nn.Conv1d(32, 64, 3, padding=1)
        self.conv4 = nn.Conv1d(64, 128, 3, padding=1)
        self.conv5 = nn.Conv1d(128, 256, 3, padding=1)        
        
        self.dec5 = nn.ConvTranspose1d(256, 128, 2, stride=2)
        self.dec4 = nn.ConvTranspose1d(128, 64, 2, stride=2)
        self.dec3 = nn.ConvTranspose1d(64, 32, 2, stride=2)
        self.dec2 = nn.ConvTranspose1d(32, 16, 2, stride=2)
        self.dec1 = nn.ConvTranspose1d(16, 1, 2, stride=2)
        
        
    def encode(self, x):
        
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(self.pool(x)))
        x = self.relu(self.conv3(self.pool(x)))
        x = self.relu(self.conv4(self.pool(x)))
#        x = self.relu(self.conv5(self.pool(x)))
        
        return(x)
        
        
    def decode(self, x):
        
        x = self.pool(x)
#        x = self.relu(self.dec5(x))
        x = self.relu(self.dec4(x))
        x = self.relu(self.dec3(x))
        x = self.relu(self.dec2(x))
        x = self.relu(self.dec1(x))
        
        return(x)
        
        
    def interpolate(self, v1, v2, alpha):
        
        o1 = self.encode(v1)
        o2 = self.encode(v2)
        o = self.decode((o1 * alpha) + ((1 - alpha) * o2))
        return(torch.tanh(o))
        
        
    def reparametrize(self, mu, logvar):
        
        var = logvar.exp()
        std = var.sqrt()
        eps = Variable(torch.cuda.FloatTensor(std.size()).normal_())
        return eps.mul(std).add(mu)


    def forward(self, x):
        
        x = self.encode(x)
        x = self.decode(x)
        x = torch.tanh(x)
                
        return x



class ConvAEDeepBN(nn.Module):
    
    def __init__(self):
        
        super(ConvAEDeepBN, self).__init__()
        
        self.pool = nn.MaxPool1d(2, 2)
        self.relu = nn.ReLU(inplace=True)
        
        self.conv1 = nn.Conv1d(1, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm1d(32)
        self.conv2 = nn.Conv1d(32, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm1d(64)
        self.conv3 = nn.Conv1d(64, 128, 3, padding=1)
        self.bn3 = nn.BatchNorm1d(128)
        self.conv4 = nn.Conv1d(128, 256, 3, padding=1)
        self.bn4 = nn.BatchNorm1d(256)
        
        self.dec4 = nn.ConvTranspose1d(256, 128, 2, stride=2)
        self.dec3 = nn.ConvTranspose1d(128, 64, 2, stride=2)
        self.dec2 = nn.ConvTranspose1d(64, 32, 2, stride=2)
        self.dec1 = nn.ConvTranspose1d(32, 1, 2, stride=2)
        
        
    def encode(self, x):
        
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(self.pool(x))))
        x = self.relu(self.bn3(self.conv3(self.pool(x))))
        x = self.relu(self.bn4(self.conv4(self.pool(x))))
        
        return(x)
        
        
    def decode(self, x):
        
        x = self.pool(x)
        x = self.relu(self.dec4(x))
        x = self.relu(self.dec3(x))
        x = self.relu(self.dec2(x))
        x = self.relu(self.dec1(x))
        
        return(x)
        
        
    def interpolate(self, v1, v2, alpha):
        
        o1 = self.encode(v1)
        o2 = self.encode(v2)
        o = self.decode((o1 * alpha) + ((1 - alpha) * o2))
        return(torch.tanh(o))


    def forward(self, x):
        
        x = self.encode(x)
        x = self.decode(x)
        x = torch.tanh(x)
                
        return x