#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 28 14:42:49 2025

Main script to extract activations from the networks
One script to run them all

@author: Filippo Cerpelloni
"""

# Import personal functions, then all libraries are coming from there
import sys
sys.path.append('../')

# Import custom functions to store paths, extraction of activations
from act_option import *
from act_functions import * 


### ---------------------------------------------------------------------------
### LOAD OPTIONS 

# Only paths for now
opt = act_option()


### ---------------------------------------------------------------------------
### ALEXNETS

# From a given network and test set, extract the relevant activation or 
# classification performance
extract_activations(model_name, nSub, experiment)

# Take network with training weights
# Present again images but with weights frozen
# get activations at any layer of any network
#   try thingsvision in addition to Andrea's script
# Make RDMs with the results
#   resurface old scripts for letters
# ???
# Results

