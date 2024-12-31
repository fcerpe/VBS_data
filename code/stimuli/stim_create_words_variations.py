#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 12 10:41:33 2024

@author: cerpelloni
"""
import sys
sys.path.append('../src')

import os
import glob
from PIL import Image
import stim_functions as stim
import resize_images as ri


## Find the raw word images 

word_dir = "../../inputs/words/raw" 
word_paths = glob.glob(os.path.join(word_dir, '*_F[0-9].png'))

## Resize images to be 1000x1000
# ri.resize_word_stimuli(700, '../../inputs/words/raw', False)
# ri.resize_word_stimuli(1000, '../../inputs/words/raw', False)

## ---------------------------------------------------------------------------
## OBSOLETE: temporarily kept here
## Create the thickness variations
# Legend: T1 = -6, T2 = -3, T3 = original, T4 = +3, T5 = +6 steps
# for path in word_paths: 
#     # Extract letter information to keep track of size changes
#     word_info = stim.parce_filename(path)
#     # Open the image and create array to modify
#     img = Image.open(path).convert("RGB")
#     # Define the variations of thickness (Y,X)
#     thicknesses = [3,6]
#     # Enlarge and shrink the images and save them in 'words/variations'. 
#     # Here, we create the remaining 20/25 stimuli needed      
#     stim.thicken_image(path, img, thicknesses, word_info)
## ----------------------------------------------------------------------------
    
    
## Create size variations 

# Loop through all the new words to create size (S) variations
for path in word_paths:
    
    # Extract script and letter to save the new image with the correct name
    word_info = stim.parce_filename(path)
    
    # Open the image 
    img = Image.open(path).convert("RGB")
    
    ## Define size variations 
    # s1 = - 30%
    # s2 = - 15%
    # s3 = original size
    # s4 = + 15%
    # s5 = + 30%
    
    # Define the sizes (in percentage of increase, e.g. 30 -> 130% of original image size)
    # !! sizes must go in increasing order, to avoid naming confusion
    sizes = [15,30]
    
    # Resize the images and save them in 'letters/variations'
    stim.resize_image(path, img, sizes, word_info)
    

## Create position variations

# Refresh list to include all the variations F* and S*
word_dir = "../../inputs/words/variations"
word_paths = glob.glob(os.path.join(word_dir, '*S[0-9].png'))

# Loop through all the words again to create horizontal (X) and vertical (Y) variations
for path in word_paths:
    
    # Extract script and letter to save the new image with the correct name
    word_info = stim.parce_filename(path)
    
    # Open the image 
    img = Image.open(path).convert("RGB")
    
    ## Define horizontal variations 
    #  X1  X2  X3  X4  X5  X6  X7  X8  X9  X10 X11
    # -50 -40 -30 -20 -10  0   10  20  30  40  50  
    x_positions = [-50, -40, -30, -20, -10, 0, 10, 20, 30, 40, 50]
    
    ## Define vertical variations 
    #  Y1  Y2  Y3  Y4  Y5
    # -40 -20   0  20  40
    y_positions = [-40, -20, 0, 20, 40]
    
    # Shift the imges on the x-axis 
    stim.shift_image(path, img, x_positions, y_positions, word_info)
    







