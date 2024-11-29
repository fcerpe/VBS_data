#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 12 10:41:33 2024

@author: cerpelloni
"""

import os
import glob
from PIL import Image
import stim_functions as stim

# Find the words in the correct folders

br_dir = "../../input/words/braille" 
br_paths = glob.glob(os.path.join(br_dir, 'br_*.png'))
ln_dir = "../../input/words/line"
ln_paths = glob.glob(os.path.join(ln_dir, 'ln_*.png'))
lt_dir = "../../input/words/latin"
lt_paths = glob.glob(os.path.join(lt_dir, 'lt_*.png'))

word_paths = br_paths + ln_paths + lt_paths 

# First, loop to create new variations:
# t1 = -X steps
# t2 = -Y steps 
# t3 = original thickness
# t4 = +Y steps
# t5 = +X steps
for path in word_paths: 
    
    # Extract letter information to keep track of size changes
    word_info = stim.parce_filename(path)
    
    # Open the image and create array to modify
    img = Image.open(path).convert("RGB")
    
    # Define the variations of thickness (Y,X)
    thicknesses = [3,6]
    
    # Enlarge and shrink the images and save them in 'words/variations'. 
    # Here, we create the remaining 20/25 stimuli needed      
    stim.thicken_image(path, img, thicknesses, word_info)
    
    
## Then, take t1-t5 and create size variations 
# Find the newly created thickness variations
word_dir = "../../input/words/variations"
word_paths = glob.glob(os.path.join(word_dir, '*T[0-9].png'))

## Then, variate the size of each letter
# Loop will produce 5 size variations:
# s1 = -X%
# s2 = -Y%
# s3 = original size
# s4 = +Y%
# s5 = +X%
for path in word_paths:
    
    # Extract script and letter to save the new image with the correct name
    word_info = stim.parce_filename(path)
    
    # Open the image 
    img = Image.open(path).convert("RGB")
    
    # Define the sizes (in percentage of increase, e.g. 20 -> 120% of original image size)
    # IMPORTANT: put the sizes in increasing order, to avoid confusion in the naming of variations
    sizes = [15,30]
    
    # Resize the images and save them in 'letters/variations'. We just create 5/25 stimuli needed
    stim.resize_image(path, img, sizes, word_info)
    
    # After all the manipulations, cast the images to be 1000x1000 pixels
    
    
    # Move them in ../../input/words/stimuli
    