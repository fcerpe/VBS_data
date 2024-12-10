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


## Find the raw letter images 
# concatenate them in one 

br_dir = "../../inputs/letters/braille" 
br_paths = glob.glob(os.path.join(br_dir, 'br_*.png'))
ln_dir = "../../inputs/letters/line"
ln_paths = glob.glob(os.path.join(ln_dir, 'ln_*.png'))
lt_dir = "../../inputs/letters/latin"
lt_paths = glob.glob(os.path.join(lt_dir, 'lt_*.png'))

letter_paths = br_paths + ln_paths + lt_paths


## Create the thickness variations
# Legend
# t1 = -6 steps
# t2 = -3 steps 
# t3 = original thickness
# t4 = +3 steps
# t5 = +6 steps

for path in letter_paths: 
    
    # Extract letter information to keep track of size changes
    letter_info = stim.parce_filename(path)
    
    # Open the image and create array to modify
    img = Image.open(path).convert("RGB")
    
    # Define the variations of thickness (Y,X)
    thicknesses = [3,6]
    
    # Enlarge and shrink the images and save them in 'letters/variations'. Here, we create the remaining 20/25 stimuli needed
    stim.thicken_image(path, img, thicknesses, letter_info)
    
    
## Add size variations 
 
# Look for the new list of stimuli, the one that includes the newly created
# T variations
letter_dir = "../../inputs/letters/variations"
letter_paths = glob.glob(os.path.join(letter_dir, '*.png'))

# Loop through all the new letters to create size (S) variations
for path in letter_paths:
    
    # Extract script and letter to save the new image with the correct name
    letter_info = stim.parce_filename(path)
    
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
    
    # Resize the images and save them in 'letters/variations'. We just create 5/25 stimuli needed
    stim.resize_image(img, sizes, letter_info)
        


    















