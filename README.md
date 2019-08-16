# Automated well log correlation using Python

This repository contains a preliminary (and admittely incomplete) set of scripts that can be used to correlate two or more well logs automatically without any user input.

## Background

Well log correlation is the process of identifying equivalent geological units / features between two or more well logs.  This is commonly done by identifying corresponding patterns between well logs from different wells and assigning them to a particular marker.  This process allows us to develop an understanding of the subsurface geology and answer a variety of different questions.

Correlating well logs is a time consuming, tedious task.  Depending on the number of wells,  number of formations that are being interpreted, quality of the data, and the complexity of the geology this process can take days to months to complete.  Additionally, this process is susceptible to bias as we each bring our own individual background and experiences, as well as prone to error as there is ambiguity in the well logs and general complexity of correlation.

## Overview of process

The process captured in this repository leverages a variety of different capabilities from the computer vision and image processing community.  Essentially, the approach outlined here is a 1D adaptation of panoramic stiching using feature vectors generated from a 1D Convolutional Autoencoder.  The basic idea being to identify a series of matching points between a pair of well logs from locally minimal values in a cost matrix that was generated from two sets of feature vectors.

### Generating feature vectors

In order to generate feature vectors for each well log in the analysis we need a couple of things:

1. Input data
	1. Well header with surface coordinates
	2. Well log data in LAS format
