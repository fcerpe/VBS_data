#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 13 13:13:15 2024

Simple script to create stimuli that will be used in the Visual Braille Silico (VBS) project

From images of Latin (Arial), Braille, and Line Braille, create the variations needed for the experiment

Steps:
    - load images
    - vary the size (-X -Y +Y +X)
    - from the size variations, vary the thickness (-A -B +B +A)
    - save all the stimuli in "letters/state_of_the_art"
    
TODO:
    - change 90deg corner function to accomodate for black lines thinner than 3 pixel, to capute end of shrinking spectrum
    - improve identify pixels algorithm to look for the color with fewer pixels, to speed-up expansion
    - make separate function to adjust pixels
    - most cases are hard-coded, not nice and can probably be generalized

@author: Filippo Cerpelloni
"""

import os
import glob

from PIL import Image
import stim_src as stim


# Pick which stimuli to process

br_dir = "letters/braille" 
br_paths = glob.glob(os.path.join(br_dir, 'br_*.png'))
ln_dir = "letters/line"
ln_paths = glob.glob(os.path.join(ln_dir, 'ln_*.png'))
lt_dir = "letters/latin"
lt_paths = glob.glob(os.path.join(lt_dir, 'lt_*.png'))

letter_paths = br_paths + ln_paths + lt_paths

# First, loop to create new variations:
# t1 = -X steps
# t2 = -Y steps 
# t3 = original thickness
# t4 = +Y steps
# t5 = +X steps
for path in letter_paths: 
    
    # Extract letter information to keep track of size changes
    letter_info = stim.parce_filename(path)
    
    # Open the image and create array to modify
    img = Image.open(path).convert("RGB")
    
    # Define the variations of thickness (Y,X)
    thicknesses = [3,6]
    
    # Enlarge and shrink the images and save them in 'letters/variations'. Here, we create the remaining 20/25 stimuli needed
    stim.thicken_image(path, img, thicknesses, letter_info)
    
    
## Then, take t1-t5 and create size variations 
# Find the newly created thickness variations
letter_dir = "letters/variations"
letter_paths = glob.glob(os.path.join(letter_dir, '*.png'))

## Then, variate the size of each letter
# Loop will produce 5 size variations:
# s1 = -X%
# s2 = -Y%
# s3 = original size
# s4 = +Y%
# s5 = +X%
for path in letter_paths:
    
    # Extract script and letter to save the new image with the correct name
    letter_info = stim.parce_filename(path)
    
    # Open the image 
    img = Image.open(path).convert("RGB")
    
    # Define the sizes (in percentage of increase, e.g. 20 -> 120% of original image size)
    # IMPORTANT: put the sizes in increasing order, to avoid confusion in the naming of variations
    sizes = [15,30]
    
    # Resize the images and save them in 'letters/variations'. We just create 5/25 stimuli needed
    stim.resize_image(img, sizes, letter_info)
        




    















