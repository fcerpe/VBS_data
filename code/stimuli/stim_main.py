#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 12 09:45:17 2024

Visual Braille Silico - main script to create stimuli

@author: cerpelloni
"""

import stim_functions as stim


### Check for datasets presence

# TO-DO: add pre-checks.
# Scripts to create the stimuli manipulate them at the pixel level, meaning
# that they can be very looong. IF possible, avoid any unnecessary step. 
# - Check if datasets are already present in inputs/datasets
# - Check if datasets exists as .zip files 
# - Check if final stimuli are present in inputs/words/stimuli and inputs/letters/stimuli
# - Check if variations of thickness are already present (that's the longest step)
# If none of these steps have been already done (highly unlikely), 
# make the stimuli from scratch.



### Create letter stimuli

# From basic images of letters in each script (Latin Arial, Braille, Line Braille)
# create visual variations in the size and thickness of the single letters
# 
# Saves everthing in: ../../inputs/letters/variations
# stim_create_letters_variations 



### Create word stimuli

# Create images for the training datasets.
# 1. take as input the path of the folder containing word images in different fonts (made in Matlab) 
# 2. make a temporary folder to store the result variations
# 3. vary the size of the words
# 4. vary the position of the word on the x and y axes
# 5. organize the stimuli into a dataset or a zipped test folder
# 6. delete the created images, now stored elsewhere 

# Training set for network's literacy and expertise
stim.create_words_variations('../../inputs/words/stimuli_training')

# Test set to replicate Visual Braille Expertise (VBE) experiment
stim.create_words_variations('../../inputs/words/stimuli_test_vbe')

# Test set to replicate Visual Braille Training (VBT) experiment
stim.create_words_variations('../../inputs/words/stimuli_test_vbt')





