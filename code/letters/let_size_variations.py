#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 11 13:31:33 2024

Create variations in the Braille and Line scripts, in the size of the stimuli

@author: Filippo Cerpelloni
"""

import os
import glob
import shutil

from PIL import Image
import numpy as np

# From path, extract which script and which letter is being processed
# Output examples: 'ln_a', 'br_v', 'lt_r'
def parce_filename(image_path):
    # Extract the file name from the path
    filename = os.path.basename(image_path)
    
    # Remove the file extension
    filename_no_ext = os.path.splitext(filename)[0]
    
    # Split the filename by '_' and paste together the first two parts
    parts = filename_no_ext.split('_')
    
    return parts[0], parts[1]



### Main script

# Both scripts are in the same folder now
letter_dir = "letters/to_variate"
letter_paths = glob.glob(os.path.join(letter_dir, '*.png'))



for path in letter_paths:
    
    # Extract script and letter to save the new image with the correct name
    script, letter = parce_filename(path)
    variation = "20-percent-bigger"
    
    # Open the image and create copies
    letter_image = Image.open(path).convert("RGB")

    # Resizing factor - expresses size as percentage of the previous image
    # e.g. +10% -> 110% of original image -> 1.1
    resize = 1.2
    
    # Resize image
    # make sure that parameters are constant to avoid different aspect ratios
    enlarged_image = letter_image.resize((int(letter_image.width * resize), 
                                          int(letter_image.height * resize)), Image.LANCZOS)

    # Calculate the coordinates to crop the image to 500x500 from the center
    left = (enlarged_image.width - 500) / 2
    top = (enlarged_image.height - 500) / 2
    right = (enlarged_image.width + 500) / 2
    bottom = (enlarged_image.height + 500) / 2
    
    # Crop image
    cropped_image = enlarged_image.crop((left, top, right, bottom))


    # Adjust filename to note the change     
    cropped_savename = f'letters/variations/{script}_{letter}_{variation}.png'
    
    # Save image
    cropped_image.save(cropped_savename)



