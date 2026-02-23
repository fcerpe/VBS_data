#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 28 14:42:49 2025

Visual Braille Silico - main script to extract activations from the networks

@author: Filippo Cerpelloni
"""

# Import personal functions, then all libraries are coming from there
import sys
sys.path.append('../')

# Import custom functions to store paths, extraction of activations
from src.vbs_functions import * 
from act_functions import * 


### ---------------------------------------------------------------------------
### Load options 

opt = vbs_option()



### ---------------------------------------------------------------------------
### Network activations for different classes of linguistic content
### (see Visual Braille Expertise)

# Extract the activations of one network's iteration for all the stimuli in the 
# test set. Process all the relevant layers of the chosen network and save 
# activations and distance matrices for later stats and visualization. 
# 
# Function arguments:
# - opt: general paths and settings
# - network: from which network to extract activations
# - sub: script is subject specific, otherwise it might break the kernel
# - training: which expertise does the network possesses
# - test: which experiment to test (i.e. which dataset to present)

## AlexNet models 

# Latin+Braille training: expert networks
extract_activations(opt, 'alexnet', 0, 'LTBR', 'VBE', 'last', 'euclidean')
extract_activations(opt, 'alexnet', 1, 'LTBR', 'VBE', 'last', 'euclidean')
extract_activations(opt, 'alexnet', 2, 'LTBR', 'VBE', 'last', 'euclidean')
extract_activations(opt, 'alexnet', 3, 'LTBR', 'VBE', 'last', 'euclidean')
extract_activations(opt, 'alexnet', 4, 'LTBR', 'VBE', 'last', 'euclidean')

# # Latin script training: naive networks
extract_activations(opt, 'alexnet', 0, 'LT', 'VBE', 'last', 'euclidean')
extract_activations(opt, 'alexnet', 1, 'LT', 'VBE', 'last', 'euclidean')
extract_activations(opt, 'alexnet', 2, 'LT', 'VBE', 'last', 'euclidean')
extract_activations(opt, 'alexnet', 3, 'LT', 'VBE', 'last', 'euclidean')
extract_activations(opt, 'alexnet', 4, 'LT', 'VBE', 'last', 'euclidean')


## CORnet Z models 

# Latin+Braille training: expert networks
cornet_activations(opt, 'cornet', 0, 'LTBR', 'VBE', 'last', 'euclidean')
cornet_activations(opt, 'cornet', 1, 'LTBR', 'VBE', 'last', 'euclidean')
cornet_activations(opt, 'cornet', 2, 'LTBR', 'VBE', 'last', 'euclidean')
cornet_activations(opt, 'cornet', 3, 'LTBR', 'VBE', 'last', 'euclidean')
cornet_activations(opt, 'cornet', 4, 'LTBR', 'VBE', 'last', 'euclidean')

# Latin script training: naive networks
cornet_activations(opt, 'cornet', 0, 'LT', 'VBE', 'last', 'euclidean')
cornet_activations(opt, 'cornet', 1, 'LT', 'VBE', 'last', 'euclidean')
cornet_activations(opt, 'cornet', 2, 'LT', 'VBE', 'last', 'euclidean')
cornet_activations(opt, 'cornet', 3, 'LT', 'VBE', 'last', 'euclidean')
cornet_activations(opt, 'cornet', 4, 'LT', 'VBE', 'last', 'euclidean')



### ---------------------------------------------------------------------------
### Network classification of stimuli during learning
### (see Visual Braille Training)

# Extract the performance of one network's iteration at classifying the stimuli
# in the test set. Process images, save results and distance matrices for 
# later stats and visualization. 
# 
# Function arguments:
# - opt: general paths and settings
# - network: from which network to extract activations
# - sub: script is subject specific, otherwise it might break the kernel
# - training: which expertise does the network possesses
# - test: which experiment to test (i.e. which dataset to present)
# - epoch(s): when to test performance

## AlexNet models
# Consider beginning of the expertise acquisition (epochs 11 and 12), beginning
# of the accuracy's plateau, end of training

# Latin+Braille training: expert networks
extract_timepoint_performances(opt, 'alexnet', 0, 'LTBR', 'VBT', [13, 14, 16, 17, 18, 19])
extract_timepoint_performances(opt, 'alexnet', 1, 'LTBR', 'VBT', [13, 14, 16, 17, 18, 19])
extract_timepoint_performances(opt, 'alexnet', 2, 'LTBR', 'VBT', [13, 14, 16, 17, 18, 19])
extract_timepoint_performances(opt, 'alexnet', 3, 'LTBR', 'VBT', [13, 14, 16, 17, 18, 19])
extract_timepoint_performances(opt, 'alexnet', 4, 'LTBR', 'VBT', [13, 14, 16, 17, 18, 19])

# Latin+Line training: expert networks
extract_timepoint_performances(opt, 'alexnet', 0, 'LTLN', 'VBT', [13, 14, 16, 17, 18, 19])
extract_timepoint_performances(opt, 'alexnet', 1, 'LTLN', 'VBT', [13, 14, 16, 17, 18, 19])
extract_timepoint_performances(opt, 'alexnet', 2, 'LTLN', 'VBT', [13, 14, 16, 17, 18, 19])
extract_timepoint_performances(opt, 'alexnet', 3, 'LTLN', 'VBT', [13, 14, 16, 17, 18, 19])
extract_timepoint_performances(opt, 'alexnet', 4, 'LTLN', 'VBT', [13, 14, 16, 17, 18, 19])


## CORnet Z models 
# Consider beginning of the expertise acquisition (epochs 1 and 2), divergence 
# between scripts (epochs 5 adn 10), end of training

# Latin+Braille training: expert networks
extract_timepoint_performances(opt, 'cornet', 0, 'LTBR', 'VBT', [3, 4, 6, 7, 8, 9, 11, 12, 13, 14])
extract_timepoint_performances(opt, 'cornet', 1, 'LTBR', 'VBT', [3, 4, 6, 7, 8, 9, 11, 12, 13, 14])
extract_timepoint_performances(opt, 'cornet', 2, 'LTBR', 'VBT', [3, 4, 6, 7, 8, 9, 11, 12, 13, 14])
extract_timepoint_performances(opt, 'cornet', 3, 'LTBR', 'VBT', [3, 4, 6, 7, 8, 9, 11, 12, 13, 14])
extract_timepoint_performances(opt, 'cornet', 4, 'LTBR', 'VBT', [3, 4, 6, 7, 8, 9, 11, 12, 13, 14])

# Latin+Line training: expert networks
extract_timepoint_performances(opt, 'cornet', 0, 'LTLN', 'VBT', [3, 4, 6, 7, 8, 9, 11, 12, 13, 14])
extract_timepoint_performances(opt, 'cornet', 1, 'LTLN', 'VBT', [3, 4, 6, 7, 8, 9, 11, 12, 13, 14])
extract_timepoint_performances(opt, 'cornet', 2, 'LTLN', 'VBT', [3, 4, 6, 7, 8, 9, 11, 12, 13, 14])
extract_timepoint_performances(opt, 'cornet', 3, 'LTLN', 'VBT', [3, 4, 6, 7, 8, 9, 11, 12, 13, 14])
extract_timepoint_performances(opt, 'cornet', 4, 'LTLN', 'VBT', [3, 4, 6, 7, 8, 9, 11, 12, 13, 14])

# Latin script training: naive networks
extract_timepoint_performances(opt, 'cornet', 0, 'LT', 'VBT', [3, 4, 6, 7, 8, 9, 11, 12, 13, 14])
extract_timepoint_performances(opt, 'cornet', 1, 'LT', 'VBT', [3, 4, 6, 7, 8, 9, 11, 12, 13, 14])
extract_timepoint_performances(opt, 'cornet', 2, 'LT', 'VBT', [3, 4, 6, 7, 8, 9, 11, 12, 13, 14])
extract_timepoint_performances(opt, 'cornet', 3, 'LT', 'VBT', [3, 4, 6, 7, 8, 9, 11, 12, 13, 14])
extract_timepoint_performances(opt, 'cornet', 4, 'LT', 'VBT', [3, 4, 6, 7, 8, 9, 11, 12, 13, 14])




