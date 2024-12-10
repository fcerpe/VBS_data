#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 12 09:45:17 2024

Visual Braille Silico - main script to create stimuli

@author: cerpelloni
"""
 
import stim_create_letters_variations
import stim_create_words_variations
import stim_functions


### Check for datasets presence

# Scripts to create the stimuli manipulate them at the pixel level, meaning
# that they can be very looong. IF possible, avoid any unnecessary step. 

# Check if datasets are already present in inputs/datasets

# Check if datasets exists as .zip files 

# Check if final stimuli are present in inputs/words/stimuli and inputs/letters/stimuli

# Check if variations of thickness are already present (that's the longest step)

# If none of these steps have been already done (highly unlikely), 
# make the stimuli from scratch.



### Create letter stimuli

# From basic images of letters in each script (Latin Arial, Braille, Line Braille)
# create visual variations in the size and thickness of the single letters
# 
# Saves everthing in: ../../inputs/letters/variations
stim_create_letters_variations 



### Create word stimuli

# From word images created in Matlab (porting of script soon)  
# create visual variations in thickness, size,  and y positions
#
# From the original words, implement the same variations made on the single letters
stim_create_words_variations



### Check pixel densisity of the datasets



### Combine the images into the datasets





# Later
## network_main
### Take istances of alexnet
### Test on letters without training
### Train on words

## analyses_main
### RDMs and correlations between chunks of letters 
### Decoding and whatever on words 

## visualization_main 
### Plot cool stuff


