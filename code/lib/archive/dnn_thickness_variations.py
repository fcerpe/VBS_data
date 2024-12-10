#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 11 13:31:33 2024

Create variations in the Braille and Line scripts, in the thickness of the dots and lines

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


# Check if a pixel has the same color as the reference
def is_color(pixel, reference):
    return np.array_equal(pixel, reference)


# Check if in a figure there are still pixels of a given color reference
def has_color_pixels(array, reference):
    return np.any(np.any(array == reference, axis=-1))

# Extract a cut-out of a given size 
# May not work with even numbers
def extract_cutout(array, row, col, side):
    section_size = side
    half_size = section_size // 2

    # Calculate the indices for slicing
    start_row = max(row - half_size, 0)
    end_row = min(row + half_size + 1, array.shape[0])
    
    start_col = max(col - half_size, 0)
    end_col = min(col + half_size + 1, array.shape[1])

    # Extract the section using slicing
    section = array[start_row:end_row, start_col:end_col]

    # Pad the section if necessary
    if section.shape[0] < section_size or section.shape[1] < section_size:
        padded_section = np.full((section_size, section_size, 3), fill_value=0, dtype=np.uint8)
        padded_section[:section.shape[0], :section.shape[1], :] = section
        return padded_section
    else:
        return section


def is_o_corner(array, row, col):
    
    # Returns 
    # - match, whether the corner meatches the pattern
    # - coords, the pixels to exclude relative to the center
    # Larger area checked to control for acute corners
    
    # Check if it's an upper-left acute corner
    white_ref = [(1,0),(2,0),(2,1),(3,0),(3,1),(4,0)]
    black_ref = [(0,0),(0,1),(0,2),(0,3),(0,4),(1,1),(1,2),(1,3),(1,4),(2,2),(2,3),(2,4),(3,2),(3,3),(3,4),(4,1),(4,2),(4,3),(4,4)]

    if all((array[i, j] == [255, 255, 255]).all() for i, j in white_ref) and all((array[i, j] == [0,0,0]).all() for i, j in black_ref):
        
        # Specify which voxels should be eroded
        coords = [(row,col+1), (row+1,col+1)]
        return True, coords
    
    else:
        return False, []
    

def is_acute_corner(array, row, col):
    
    # Returns 
    # - match, whether the corner meatches the pattern
    # - coords, the pixels to exclude relative to the center
    # Larger area checked to control for acute corners
    
    # Check if it's an upper-left acute corner
    white_up_left = [(5,5),(5,6),(5,7),(5,8),(6,5),(6,6),(6,7),(6,8),(7,5),(7,6),(7,7),(7,8),(8,6),(8,7),(8,8)]
    black_up_left = [(0,0),(0,1),(0,2),(0,3),(0,4),(0,5),(0,6),(0,7),(0,8),(1,0),(1,1),(1,2),(1,3),(1,4),(1,5),(1,6),(1,7),(1,8),(2,0),(2,1),(2,2),(2,3),(2,4),
                     (2,5),(2,6),(2,7),(2,8),(3,0),(3,1),(3,2),(3,3),(3,4),(3,5),(3,6),(3,7),(3,8),(4,0),(4,1),(4,2),(4,3),(4,4),(4,5),(4,6),(4,7),(4,8),(5,0),
                     (5,1),(5,2),(5,3),(5,4),(6,0),(6,1),(6,2),(6,3),(6,4),(7,0),(7,1),(7,2),(7,3),(7,4),(8,0),(8,1),(8,2),(8,3),(8,4),(8,5)]

    if all((array[i, j] == [255, 255, 255]).all() for i, j in white_up_left) and all((array[i, j] == [0,0,0]).all() for i, j in black_up_left):
        
        # Specify which voxels should be eroded
        coords = [(row,col-1), (row+1,col-1)]
        return True, coords
    
    # Not UL, rotate reference and check upper-right 
    white_up_right = [(5,0),(5,1),(5,2),(5,3),(6,0),(6,1),(6,2),(6,3),(7,0),(7,1),(7,2),(7,3),(8,0),(8,1),(8,2)]
    black_up_right = [(0,0),(0,1),(0,2),(0,3),(0,4),(0,5),(0,6),(0,7),(0,8),(1,0),(1,1),(1,2),(1,3),(1,4),(1,5),(1,6),(1,7),(1,8),(2,0),(2,1),(2,2),(2,3),(2,4),
                      (2,5),(2,6),(2,7),(2,8),(3,0),(3,1),(3,2),(3,3),(3,4),(3,5),(3,6),(3,7),(3,8),(4,0),(4,1),(4,2),(4,3),(4,4),(4,5),(4,6),(4,7),(4,8),(5,4),
                      (5,5),(5,6),(5,7),(5,8),(6,4),(6,5),(6,6),(6,7),(6,8),(7,4),(7,5),(7,6),(7,7),(7,8),(8,3),(8,4),(8,5),(8,6),(8,7),(8,8)]

    if all((array[i, j] == [255, 255, 255]).all() for i, j in white_up_right) and all((array[i, j] == [0,0,0]).all() for i, j in black_up_right):
        
        # Specify which voxels should be eroded
        coords = [(row,col+1), (row+1,col+1)]
        return True, coords
    
    
    # Neither UR, rotate and check lower-right
    white_low_right = [(0,0),(0,1),(0,2),(1,0),(1,1),(1,2),(1,3),(2,0),(2,1),(2,2),(2,3),(3,0),(3,1),(3,2),(3,3)]
    black_low_right = [(0,3),(0,4),(0,5),(0,6),(0,7),(0,8),(1,4),(1,5),(1,6),(1,7),(1,8),(2,4),(2,5),(2,6),(2,7),(2,8),(3,4),(3,5),(3,6),(3,7),(3,8),(4,0),
                       (4,1),(4,2),(4,3),(4,4),(4,5),(4,6),(4,7),(4,8),(5,0),(5,1),(5,2),(5,3),(5,4),(5,5),(5,6),(5,7),(5,8),(6,0),(6,1),(6,2),(6,3),(6,4),
                       (6,5),(6,6),(6,7),(6,8),(7,0),(7,1),(7,2),(7,3),(7,4),(7,5),(7,6),(7,7),(7,8),(8,0),(8,1),(8,2),(8,3),(8,4),(8,5),(8,6),(8,7),(8,8)]

    if all((array[i, j] == [255, 255, 255]).all() for i, j in white_low_right) and all((array[i, j] == [0,0,0]).all() for i, j in black_low_right):
        
        # Specify which voxels should be eroded
        coords = [(row,col+1), (row-1,col+1)]
        return True, coords
    
    # No acute angle (we don't have BL)
    return False, []



def is_right_corner(array):
    
    # Corner piece isa by-product of the '4Neighbors' algorithm, 
    # it's a triangluar shape that is created in right-acute angles
    # e.g. 
    # _ _ B B B
    # _ _ B B B
    # B B X B B
    # B B B B B
    # B B B B B
    # X will be removed, to avoid said triangles
    
    # Rotate array to have the 2x2 white square on the top-left corner
    reference_orientation = np.array([[[255,255,255],[255,255,255],[0,0,0],[0,0,0],[0,0,0]],
                                      [[255,255,255],[255,255,255],[0,0,0],[0,0,0],[0,0,0]],
                                      [[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0]],
                                      [[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0]],
                                      [[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0]]], dtype=np.uint8)
    
    # Rotate the array until it matches the reference 
    for k in range(4):
        rotated_array = np.rot90(array, k = k)
        
        if np.array_equal(rotated_array, reference_orientation):
            break
    
    # Must provide a 5x5 array, positions are hard-coded
    white_pos = [(0,0), (0,1), (1,0), (1,1)]
    white_check = all((rotated_array[i, j] == [255, 255, 255]).all() for i, j in white_pos)

    black_pos = [(0,2), (0,3), (0,4), (1,2), (1,3), (1,4), (2,0), (2,1), (2,2), (2,3), (2,4), (3,0), (3,1), (3,2), (3,3), (3,4), (4,0), (4,1), (4,2), (4,3), (4,4)]
    black_check = all((rotated_array[i, j] == [0, 0, 0]).all() for i, j in black_pos)
    
    return (white_check and black_check)


def is_acute_8N_corner(array, row, col):
    
    # B B B B B
    # B B B B B
    # _ _ X B B
    # _ _ B B B
    # _ B B B B
    # X will be removed, to avoid residuals
    
    # upper-left corner
    white_UL = [(2,3),(2,4),(3,3),(3,4),(4,4)]
    black_UL = [(0,0),(0,1),(0,2),(0,3),(0,4),(1,0),(1,1),(1,2),(1,3),(1,4),(2,0),(2,1),(2,2),(3,0),(3,1),(3,2),(4,0),(4,1),(4,2),(4,3)]
    
    if all((array[i, j] == [255, 255, 255]).all() for i, j in white_UL) and all((array[i, j] == [0,0,0]).all() for i, j in black_UL):
        # Specify which voxels should be eroded
        coords = [(row-1,col-1), (row,col-1)]
        return True, coords
    
    # upper-right corner
    white_UR = [(2,0),(2,1),(3,0),(3,1),(4,0)]
    black_UR = [(0,0),(0,1),(0,2),(0,3),(0,4),(1,0),(1,1),(1,2),(1,3),(1,4),(2,2),(2,3),(2,4),(3,2),(3,3),(3,4),(4,1),(4,2),(4,3),(4,4)]
    
    if all((array[i, j] == [255, 255, 255]).all() for i, j in white_UR) and all((array[i, j] == [0,0,0]).all() for i, j in black_UR):
        # Specify which voxels should be eroded
        coords = [(row-1,col+1), (row,col+1)]
        return True, coords
    
    # bottom-right corner
    white_BR = [(0,0),(1,0),(1,1),(2,0),(2,1)]
    black_BR = [(0,1),(0,2),(0,3),(0,4),(1,2),(1,3),(1,4),(2,2),(2,3),(2,4),(3,0),(3,1),(3,2),(3,3),(3,4),(4,0),(4,1),(4,2),(4,3),(4,4)]
    
    if all((array[i, j] == [255, 255, 255]).all() for i, j in white_BR) and all((array[i, j] == [0,0,0]).all() for i, j in black_BR):
        # Specify which voxels should be eroded
        coords = [(row,col+1), (row+1,col+1)]
        return True, coords

    # No acute angle (we don't have BL)
    return False, []

    

# Find which pixels are black/white and neighbour a non-black/non-white pixel
def identify_pixels(image_array, ref_flag, ngb):
    
    # Define which target color are we looking 
    if ref_flag == 'white':
        target = [255, 255, 255]
        anti_target = [0, 0, 0]
    elif ref_flag == 'black':
        target = [0, 0, 0]
        anti_target = [255, 255, 255]

    # Get the dimensions of the image
    rows, cols, _ = image_array.shape

    # Create a list to store the coordinates of the pixels
    border_pixels = []

    
    # Check each pixel
    for row in range(rows):
        for col in range(cols):
            
            # If they match the target color
            if has_color_pixels(image_array[row, col], target):
                
                # Check whether the pixel is a corner one, in case delete it
                cutout = extract_cutout(image_array, row, col, 5)
                
                # In case of letter O, taylor the alogrithm
                match, coords = is_o_corner(cutout, row, col)
                if match:
                    for c in coords: border_pixels.append(c)    
                
                # Special case: if the algorithm is 8N, prevent triangle residuals more in depth
                if ngb == '8N': 
                    match, coords = is_acute_8N_corner(cutout, row, col)
                    if match: 
                        for c in coords: border_pixels.append(c)    
                
                # If we encounter a corner
                if is_right_corner(cutout): 
                    border_pixels.append((row, col))
                    
                    # Extract larger cutout to check for acute corners
                    larger_cutout = extract_cutout(image_array, row, col, 9)
                    match, coords = is_acute_corner(larger_cutout, row, col)
                    if match:
                        for c in coords: border_pixels.append(c)
                        
                        

                # List vertical, horizontal, oblique neighbors
                if ngb == '4N': 
                    neighbors = [(row-1, col), 
                                 (row, col-1),                 
                                 (row, col+1),
                                 (row+1, col)]
                    
                else: 
                    neighbors = [(row-1, col), (row-1, col-1), (row-1, col+1),
                                 (row, col-1),                 
                                 (row, col+1),
                                 (row+1, col), (row+1, col-1), (row+1, col+1)]
                
                # Within the neighbors, look for base colors
                for n_row, n_col in neighbors:
                    
                    if 0 <= n_row < rows and 0 <= n_col < cols:
                        
                        if has_color_pixels(image_array[n_row, n_col], anti_target):
                            
                            border_pixels.append((row, col))
                            break
                                                
    return border_pixels




### Main script

# Both scripts are in the same folder now
letter_dir = "letters/to_variate"
letter_paths = glob.glob(os.path.join(letter_dir, '*.png'))

# Set reference colors
ref_white = [255, 255, 255]
ref_black = [0,   0,   0]


for path in letter_paths:
    
    # Open the image and create copies
    letter_image = Image.open(path).convert("RGB")
    letter_array = np.array(letter_image)
    shrink_array = np.array(letter_image)
    expand_array = np.array(letter_image)
    
    # Save a copy as original in the state-of-the-art folder
    script, letter = parce_filename(path)
    copied_image = Image.fromarray(letter_array.astype('uint8'), 'RGB')
    
    savename = 'letters/variations/' + script + '_' + letter + '_original.png'
    copied_image.save(savename)
    
    # Cast non-black pixels as black, to simplify the task
    non_black = np.any(letter_array != ref_white, axis=-1)
    letter_array[non_black] = ref_black
    
    black_image = Image.fromarray(letter_array.astype('uint8'), 'RGB')
    blackname = 'letters/variations/' + script + '_' + letter + '_black.png'
    black_image.save(blackname)
    
    # Create copies to shrink and enlarge lines and dots
    shrink_image = Image.open(path).convert("RGB")
    shrink_array = np.array(shrink_image)
    shrink_array[non_black] = ref_black
    
    expand_image = Image.open(path).convert("RGB")
    expand_array = np.array(expand_image)
    expand_array[non_black] = ref_black
    
    # Set counters to track variations
    minus = 1
    plus = 1
    
    neighbors = '8N'
    
    ## Expand lines and dots until it's all white
    while has_color_pixels(expand_array, ref_white):
    
        # identify which pixels to change
        pixels_to_change = identify_pixels(expand_array, 'white', neighbors)
        
        # Paint them black
        for pixel in pixels_to_change:
            expand_array[pixel[0], pixel[1]] = ref_black
        
        # save image in state-of-the-art folder with change notation
        colored_image = Image.fromarray(expand_array)
        
        colored_savename = f'letters/variations/{script}_{letter}_plus{plus}_{neighbors}.png'
        colored_image.save(colored_savename)
        
        # Update iteration counter
        plus = plus+1
        
        # While shrinking has a natural limit, expansion can go on for a loooong time,
        # much more than necessary. End when the upper bound is reached
        if plus > 12:
            break
        
        
    ## Shrink lines and dots until it's all black
    
    # In Line Braille, diagonal lines are tricky and need to be cared for. 
    # Flag which script and, in case, letter is being processed.
    
    # Braille has the same algorithm
    if script == 'br':
        neighbors = '8N'
        
    # If Line Braille, sitatuion is more complex    
    else:
        if letter in ['n','s','z']:   # 45 degrees lines
            ngb_counter = 0
            sequence = ['4N','4N','8N','4N','8N']
            neighbors = sequence[ngb_counter]

            
        elif letter in ['m','u','x']: # 60 degrees lines
            ngb_counter = 0
            sequence = ['4N','4N','8N']
            neighbors = sequence[ngb_counter]
            
            
        elif letter in ['o']:         # peculiar case
            neighbors = '8N'
            
        else:                         # all the other letters
            neighbors = '4N'
    
    
    while has_color_pixels(shrink_array, ref_black):
    
        # identify which pixels to change
        pixels_to_change = identify_pixels(shrink_array, 'black', neighbors)
        
        # Paint them black
        for pixel in pixels_to_change:
            shrink_array[pixel[0], pixel[1]] = ref_white
        
        # save image in state-of-the-art folder with change notation
        colored_image = Image.fromarray(shrink_array)
        
        colored_savename = f'letters/variations/{script}_{letter}_minus{minus}_{neighbors}.png'
        colored_image.save(colored_savename)
        
        # Update iteration counter
        minus = minus+1
        
        # In case of peculiar case for line stimuli, update the algorithm
        if script == 'ln' and letter in ['n','s','z','m','u','x']:
            
            # 45 degrees
            if letter in ['n','s','z']:
                if ngb_counter == 4:
                    ngb_counter = 0
                else:
                    ngb_counter = ngb_counter +1
                    
                neighbors = sequence[ngb_counter]

            # 60 degrees
            if letter in ['m','u','x']:
                if ngb_counter == 2:
                    ngb_counter = 0
                else:
                    ngb_counter = ngb_counter +1
                    
                neighbors = sequence[ngb_counter]
         
            
    # TBD: Move the right ones to the selection folder

    # # Define your source and destination folders
    # source_folder = 'letters/variations'
    # destination_folder = 'letters/sota'
    
    # # Ensure the destination folder exists
    # os.makedirs(destination_folder, exist_ok = True)
    
    # target_string = f'{script}_{letter}_minus'
    
    # # Iterate over each file in the source folder
    # for filename in os.listdir(source_folder):
        
    #     # Check if the filename follows the pattern "ln_minusX"
    #     if filename.startswith(target_string):
            
    #         try:
    #             # Extract the number X
    #             target_string = filename.replace("_*N", "")
    #             x_value = int(filename.replace(target_string, ""))
                
    #             # If X > 11, move the file
    #             if x_value < 12:
    #                 source_path = os.path.join(source_folder, filename)
    #                 destination_path = os.path.join(destination_folder, filename)
    #                 shutil.move(source_path, destination_path)
    #                 print(f'Moved: {filename}')
    #         except ValueError:
    #             # If the conversion to int fails, skip the file
    #             print(f'Skipped: {filename} (not a valid format)')





