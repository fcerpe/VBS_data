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
import dataset_functions as dataset
import store_files as stf

### ---------------------------------------------------------------------------
### CREATE WORD VARIATIONS

def create_words_variations(stim_path): 
    
    # Get the words to work on based on the defined path 
    # Also, define a variable 'data_type' related to the type of data we are dealing with:
    # - VBE for the expertise experiment replication
    # - VBT for the training experiment replication
    # - VBS for data strictly related to the silico experiment (this one)
    if stim_path[-3:] == 'vbe':
        word_paths = glob.glob(os.path.join(stim_path, '*.png'))
        data_type = 'VBE'
    elif stim_path[-3:] == 'vbt':
        word_paths = glob.glob(os.path.join(stim_path, '*_F[0-9].png'))
        data_type = 'VBT'
    else:
        word_paths = glob.glob(os.path.join(stim_path, '*_F[0-9].png'))
        data_type = 'VBS'
        
    # Resize images to be 1000x1000
    ri.resize_word_stimuli(700, stim_path, False)
    ri.resize_word_stimuli(1000, stim_path, False)
    
    # Make temporary directory "temp_variations" to store the new images 
    temp_path = os.path.join(stim_path, '..', 'temp_variations')
    os.makedirs(temp_path, exist_ok = True)
    
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
        resize_image(path, img, sizes, word_info)

    
    ## Create position variations

    # Refresh list to include all the variations F* and S*
    word_paths = glob.glob(os.path.join(temp_path, '*S[0-9].png'))

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
        shift_image(path, img, x_positions, y_positions, word_info)
        
    
    ## Move stimuli to datasets/test and zip them 
    
    # Prune size-only variations
    for file in glob.glob(os.path.join(temp_path, "*S[0-9].png")):
        os.remove(file)
    
    # Test sets are moved, training sets are organized
    if data_type == 'VBE': 
        
        # Move (rename) the folder to datasets
        os.rename(temp_path, '../../inputs/datasets/test_vbe')
        
        # Zip it to be stored on GIN
        stf.zip_folder('../../inputs/datasets/test_vbe')

    elif data_type == 'VBT':

        # Move and zip
        os.rename(temp_path, '../../inputs/datasets/test_vbt')
        stf.zip_folder('../../inputs/datasets/test_vbt')
        
    else:
        
        # Create all the datasets
        dataset.create_dataset_LT()
        dataset.create_dataset_BRLT()
        dataset.create_dataset_LNLT()
        
        # Zip the datasets
        stf.zip_subfolders('../../inputs/datasets/LT')
        stf.zip_subfolders('../../inputs/datasets/BR_LT')
        stf.zip_subfolders('../../inputs/datasets/LN_LT')
        
        # Then delete the folder
        os.rmdir(temp_path)
        
        
    
### ---------------------------------------------------------------------------
### MAIN VARIATIONS FUNCTIONS

# Position variations
# From an image, will create copies that are shifted on the x and y axes
def shift_image(path, image, x_positions, y_positions, stim_info):

    
    # Loop through all combinations of x and y positions
    for iX, x in enumerate(x_positions):
        
        for iY, y in enumerate(y_positions):
            
            # Create white image on which to paste the shifted word
            canvas = Image.new("RGB", image.size, (255, 255, 255)) 
            
            # Paste the original image at the shifted position
            canvas.paste(image, (x, y))
            
            # Add information to the name and save the image
            shifted_savename = f'../../inputs/words/temp_variations/{stim_info}_X{iX+1}_Y{iY+1}.png'
            canvas.save(shifted_savename)
            


# Resizing function
# From given sizes and image, will create the 5 (actually 4) variations needed
def resize_image(path, image, sizes, stim_info):
    
    # Assuming images passed are squared (they should be), use the original side 
    # to crop enlarged /shrank images
    resize_factor = image.width
    
    ## Make letter the same
    # Just adjust name and save original
    cropped_savename = f'../../inputs/words/temp_variations/{stim_info}_S3.png'
    image.save(cropped_savename)
     
    # Each size correspond to two variations, apply them both 
    for i, s in enumerate(sizes): 
        
        ## Make letter bigger 
        # Compute factor 
        increase = 1+s/100

        # Resize
        bigger_image = image.resize((int(image.width * increase), int(image.height * increase)), Image.LANCZOS)
    
        # Calculate the coordinates to crop the image to 500x500 from the center
        left = (bigger_image.width - resize_factor) / 2
        top = (bigger_image.height - resize_factor) / 2
        right = (bigger_image.width + resize_factor) / 2
        bottom = (bigger_image.height + resize_factor) / 2
        
        # Crop image
        cropped_image = bigger_image.crop((left, top, right, bottom))
        
        # Adjust filename to note the change
        cropped_savename = f'../../inputs/words/temp_variations/{stim_info}_S{4+i}.png'
        
        # Save image
        cropped_image.save(cropped_savename)
        
        
        ## Make letter smaller 
        # Compute factor 
        decrease = 1-s/100

        # Resize
        smaller_image = image.resize((int(image.width * decrease), int(image.height * decrease)), Image.LANCZOS)
        
        # Create a new white background image with the same size as the original image
        background = Image.new("RGB", (image.width, image.height), (255, 255, 255))
        
        # Calculate the position to paste the resized image onto the center of white background
        x_offset = int(image.width - smaller_image.width) // 2
        y_offset = int(image.height - smaller_image.height) // 2
        
        # Paste the resized image onto the white background
        background.paste(smaller_image, (x_offset, y_offset))

        # Calculate the coordinates to crop the image to 500x500 from the center
        left = (background.width - resize_factor) / 2
        top = (background.height - resize_factor) / 2
        right = (background.width + resize_factor) / 2
        bottom = (background.height + resize_factor) / 2
        
        # Crop image
        cropped_image = background.crop((left, top, right, bottom))
        
        # Adjust filename to note the change
        cropped_savename = f'../../inputs/words/temp_variations/{stim_info}_S{2-i}.png'
        
        # Save image
        cropped_image.save(cropped_savename)

    




### ---------------------------------------------------------------------------
### SUPPORT FUNCTIONS

# Extract script and letter information from filename
# Example: from 'letters/to_variate/br_a' -> script = 'br', letter = 'a'
def parce_filename(image_path):
    
    # Extract the file name from the path
    filename = os.path.basename(image_path)
    
    # Remove the file extension
    filename_no_ext = os.path.splitext(filename)[0]
    
    # Return the whole filename, with script, letter, and variations in size or thicnkess
    return filename_no_ext


        




















