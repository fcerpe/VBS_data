#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 24 16:30:23 2025

Visual Braille Silico - main script to present letters to AlexNet

Mostly meant to be a pilot, and an exercise to get used to DNNs. This is why this
script is mostly standalone and it's not integrated in the main pipeline

@author: Filippo Cerpelloni
"""

# Import personal functions, then all libraries are coming from there
import sys
sys.path.append('../')

# Import custom functions to store paths, extraction of activations
from src.vbs_functions import * 
from let_functions import *



## ----------------------------------------------------------------------------
## Load options 

opt = vbs_option()

extract_letters_activations(opt)



## ----------------------------------------------------------------------------
## Create stimuli

# Starting from letter images in the Latin alphabet (Arial), Braille, Line Braille,
# creave variations in thickness and size
# Inputs:
# - options
# - fonts list (F1: Arial, F5: Braille, F6: Line Braille)
# - thickness variations
# - size variations

create_letters_variations(opt, ['F1', 'F5', 'F6'], [3,6], [15,30])


# TODO resurface scripts that ensure that the letters are matched for pixel density


## ----------------------------------------------------------------------------
## Present stimuli to AlexNet

# TODO description 
# TODO check that let_variation_comparison and functions actually match

extract_letters_activations(opt)


## ----------------------------------------------------------------------------
## Plot results

# TODO description 

plot_letters_representations(opt)


## ----------------------------------------------------------------------------
## Compare representations of letters 

# TODO add description

stats_letters_representations(opt)








