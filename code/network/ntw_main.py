#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 20 21:51:49 2025

Visual Braille Silico - main script to train the networks used in the experiment 

One script to run them all: 
for both AlexNets and CORnet Zs, train on the Latin script alone, 
on Latin+Braille, on Latin+Line (detials below)

@author: Filippo Cerpelloni
"""

# Import personal functions, then all libraries are coming from there
import sys
sys.path.append('../')

# Import custom functions for training + cornet
from ntw_functions import *
from src.vbs_functions import *



### ---------------------------------------------------------------------------
### Load options 

opt = vbs_option()



### ---------------------------------------------------------------------------
### AlexNet models 

## Train five instances on Dutch words in the latin script alone
#
# Train on 20 epochs to parallel the literate but naive subjects of the 
# Visual Braille Expertise study, to be compared with networks trained on 
# Latin+Braille and Latin+Line
# 
# Function arguments:
# - opt: the paths and general settings
# - script: the script to train, a.k.a. which dataset to load
# - sub: the number of instances to train
# - epochs: how long to train
# - learning rate
# - size of the training set
# - size of the batches 

network_train_alexnets(opt, 'latin', 5, 20, 1e-4, 0.7, 100)


## Train five instances on Dutch words introducin g a novel script
#
# Start from epoch 10 of the Latin laone training, and train for 10 additional 
# epochs to parallel the expert subjects of the Visual Braille Expertise and 
# VB Traaining studies, to be compared with "naive" instances 
# 
# Function arguments:
# - opt: the paths and general settings
# - script: the script to train, a.k.a. which dataset to load
# - sub: the number of instances to train
# - epochs: how long to train
# - learning rate
# - size of the training set
# - size of the batches 

# Braille training 
network_train_alexnets(opt, 'braille', 5, 10, 1e-4, 0.7, 100)

# Line training
network_train_alexnets(opt, 'line', 5, 10, 1e-4, 0.7, 100)



### ---------------------------------------------------------------------------
### CORnet (Z) models 

## Train five instances on Dutch words in the latin script alone
#
# Perform the same trainings listed above, with the particularity that the networks
# are all considered literate already (weights from Agrawal and Dehaene, 2024). 
# Train on 15 epochs to reproduce the performances of the AlexNets for the 
# specific alphabets we are using.

# Latin 
network_train_cornets(opt, 'latin', 5, 15, 1e-4, 0.7, 100)

# Latin + Braille 
network_train_cornets(opt, 'braille', 5, 15, 1e-4, 0.7, 100)

# Latin + Line 
network_train_cornets(opt, 'line', 5, 15, 1e-4, 0.7, 100)


