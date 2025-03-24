#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 20 21:51:49 2025

Main script to train the networks
One script to run them all

- latin script training x5
- latin-braille training x5
- latin-line training x5

@author: cerpelloni
"""

# Import personal functions, then all libraries are coming from there
import sys
sys.path.append('../')

# Import custom functions for training + cornet
from network_functions import *
from network_option import *

### ---------------------------------------------------------------------------
### LOAD OPTIONS 

# Only paths for now
opt = network_option()


### ---------------------------------------------------------------------------
### ALEXNETS

## TRAIN FIVE INSTANCES ON LATIN SCRIPT
#
# 1. Load the Latin script (LT) dataset with relative classes 
# 2. Take five instances of alexnet, with classic weights from ImageNet
# 3. Reset the last layer of alexnet to delete ImageNet's categories and replace 
#    them with our 1000 words
# 4. Train this instance with the given hyperparameters
# 5. Save learning curves in outputs/figures/literate/latin
# 6. Save weights for each epoch in outputs/weights/literate/latin
#
# Function specifes: 
# - which dataset (latin, braille, line)
# - how many instances (subjects)
# - how many epochs
# - learning rate
# - size of the training set
# - size of the batches

network_train_alexnets(opt, 'latin', 5, 10, 1e-4, 0.7, 100)


## RE-TRAIN THE SAME FIVE INSTANCES ON NOVEL SCRIPTS
#
# Same training procedure, regarding the script 
# 1. Load the Latin + novel_script (BR/LN_LT) dataset with relative classes 
# 2. Take five instances of alexnet, without weights
# 3. Load weights from one of the "literate" "subjects"
# 4. Train again on new dataset
# 5. Save learning curves in outputs/figures/literate/(braille or line)
# 6. Save weights for each epoch in outputs/weights/literate/(braille or line)

# Braille training 
network_train_alexnets(opt, 'braille', 5, 10, 1e-4, 0.7, 100)

# Line training
network_train_alexnets(opt, 'line', 5, 10, 1e-4, 0.7, 100)


### ---------------------------------------------------------------------------
### CORNETS (Z)

## TRAIN FIVE INSTANCES ON NOVEL SCRIPTS
#
# Use networks from previous studies (Agrawal and Dehaene's 2024 paper) and 
# train them on our scripts
#
# 1. Load the Latin + novel_script (BR/LN_LT) dataset with relative classes 
# 2. Load cornet Z models from github:dicarlolab/CORnet
# 3. Apply weights from Agrawal and Dehaene's French literate network
# 4. Train again on new dataset
# 5. Save learning curves in outputs/figures/literate/(braille or line)
# 6. Save weights for each epoch in outputs/weights/literate/(braille or line)

# Latin training, to probe for weights' validity
network_train_cornets(opt, 'latin', 1, 10, 1e-4, 0.7, 100)

# Braille training
network_train_cornets(opt, 'braille', 5, 10, 1e-4, 0.7, 100)

# Line training
network_train_cornets(opt, 'line', 5, 10, 1e-4, 0.7, 100)


