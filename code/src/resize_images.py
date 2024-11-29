#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 12 22:57:31 2024

@author: cerpelloni
"""

from PIL import Image
import os
import glob
import shutil

## Resize the word stimuli
#
# Stimuli are made in Matlab to be 1430x1430 pixels
# In different moments, resizing is required:
# - enlarging them to 1859x1859 ensures that the manipulations of thickness 
#   can be performed smoothly
# - shrinking them to 1000 px standardize them
#
# Take as input:
# - target size
# - folder in which to find the stimuli to resize
# - 'move' argument, whether to move the files into other folders
#   this is used for enlarging to 1859x1859, moves them from matlab_import to 
#   braille, line, latin folders
def resize_word_stimuli(target_size, folder, move): 

    # Define folder and dimensions
    source_path = folder
    word_paths = glob.glob(os.path.join(source_path, '*.png'))
    
    # If 'move' is set to true, store everything in 'matlab_enlarged' and move later
    if move:
        dest_path = "../../input/words/matlab_enlarged"
        
    else:
        dest_path = folder
        
    # Define the target dimensions
    background_size = (target_size, target_size)
    
    
    # Create a white background
    background = Image.new("RGB", background_size, (255, 255, 255))
    
    # Iterate over all images in the folder
    for path in word_paths:
    
        # Extract filename
        filename = path.split("/")[-1]
        
        # Open the image
        figure = Image.open(path).convert("RGB")
        
        # Resize figure
        figure.thumbnail(background_size, Image.LANCZOS)
                
        # Calculate position to center the figure
        x_position = (background_size[0] - figure.width) // 2
        y_position = (background_size[1] - figure.height) // 2
        
        # Paste the figure onto the background
        background.paste(figure, (x_position, y_position))
        
        # Overwrite the original image
        background.save(f'{dest_path}/{filename}')
        
    # If chosen to, move new images to corresponding folders    
    if move:
        
        # Define the destinations
        braille_path = os.path.join(dest_path, "../braille")
        line_path = os.path.join(dest_path, "../line")
        latin_path = os.path.join(dest_path, "../latin")
                
        # Create destination folders if they don't exist
        os.makedirs(braille_path, exist_ok=True)
        os.makedirs(line_path, exist_ok=True)
        os.makedirs(latin_path, exist_ok=True)
        
        # Iterate through all the images
        for filename in os.listdir(dest_path):
            
            # Get the image path
            file_path = os.path.join(dest_path, filename)
            
            # Assign the image to the corresponding folder
            if filename.startswith("br_"):
                shutil.move(file_path, braille_path)
                
            elif filename.startswith("ln_"):
                shutil.move(file_path, line_path)
                
            elif filename.startswith("lt_"):
                shutil.move(file_path, latin_path)
