#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 12 09:54:18 2024

Visual Braille Silico - support function for stimuli creation

@author: cerpelloni
"""

import os
import glob

from PIL import Image
import numpy as np



# Extract script and letter information from filename
# Example: from 'letters/to_variate/br_a' -> script = 'br', letter = 'a'
def parce_filename(image_path):
    
    # Extract the file name from the path
    filename = os.path.basename(image_path)
    
    # Remove the file extension
    filename_no_ext = os.path.splitext(filename)[0]
    
    # Return the whole filename, with script, letter, and variations in size or thicnkess
    return filename_no_ext
    

# Check if a given pixel has the same color as a given reference
def is_color(pixel, reference):
    return np.array_equal(pixel, reference)


# Check if in a given figure there are still pixels of a given reference
def has_color_pixels(array, reference):
    return np.any(np.any(array == reference, axis=-1))


# Extract a cut-out of a given size from a given image and coordinates 
# IMPORTANT: it has only been tested on odd sizes (5,7,9). May not work perfectly with even numbers
def extract_cutout(array, row, col, side):
    section_size = side
    half_size = section_size // 2

    # Calculate the indices for slicing
    start_row = max(row - half_size, 0)
    end_row = min(row + half_size + 1, array.shape[0])
    
    start_col = max(col - half_size, 0)
    end_col = min(col + half_size + 1, array.shape[1])

    # Extract the section using slicing
    return array[start_row:end_row, start_col:end_col]


# Identify which pixels correspond to a contour of a given array and a given colour to check
def identify_pixels(image_array, ref_flag, script, ngb):
    
    # Define target color (the one to change) and anti-target (the one to control for)
    # Shrinking uses target = 'black', to then change them to white
    # Expansion uses target = 'white', to change them to black
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
    
    # Speed-up script:
    # We know that letters are placed at the center of the 500x500px image
    # The same applies to the 1430x1430px words 
    # Based on the case (i.e. the sizes of the images), restrict the range in which
    # to look for specific pixels to predifined 250x250 for letters and 300x1430 for words
    if cols == 500: 
        range_rows = range(124,374)
        range_cols = range(124,374)
        
    elif cols == 1859: 
        range_rows = range(784,1055)
        range_cols = range(22,1838)
        

    # Check each pixel (slow algorithm in expansion, there are a lot of white pixels in the border that are not relevant)
    for row in range_rows:
        for col in range_cols:
            
            # If they match the target color
            if has_color_pixels(image_array[row, col], target):
                
                ## Intercept special cases, only if within a reasonable window
                if 1 < row < rows-2 and 1 < col < cols-2:
                    
                    # Extract small cutout 
                    cutout_five = extract_cutout(image_array, row, col, 5)
                       
                    # Special case: LINE SCRIPT LETTER O
                    if script == 'ln': 
                        match, coords = case_line_o(cutout_five, row, col)
                        if match: 
                            for c in coords: border_pixels.append(c)    
                            
                        # Special case: algorithm is 8N, acute angle 
                        if ngb == '8N': 
                            match, coords = case_66deg_8N_corner(cutout_five, row, col)
                            if match: 
                                for c in coords: border_pixels.append(c)    
                            
                        # Special cases: more generally, deal with corners
                        if case_90deg_corner(cutout_five): 
                            border_pixels.append((row, col))
                            
                            # Extract larger cutout to check for acute corners
                            cutout_nine = extract_cutout(image_array, row, col, 9)
                            match, coords = case_66deg_corner(cutout_nine, row, col)
                            if match:
                                for c in coords: border_pixels.append(c)
                
                    # Special cases: LATIN SCRIPT
                    if script == 'lt': 
                        
                        # Check for line-curve corners, like the one in 'b' 
                        cutout_seven = extract_cutout(image_array, row, col, 7)
                        match_round, coords_round = case_round_corner(cutout_seven, row, col)
                        if match_round:
                            for c in coords_round: border_pixels.append(c)
                            
                        # Check for sharp corner
                        match_sharp_y, coords_sharp_y = case_y_corner(cutout_five, row, col)
                        if match_sharp_y: 
                            # that is also not a round corner, otherwise it gets messy
                            if not match_round:
                                for c in coords_sharp_y: border_pixels.append(c)
                         
                        # more generally, deal with corners (especially valid in T and F)
                        if case_90deg_corner(cutout_five): 
                            
                            # Check that it's not actually a sharp corner
                            match_sharp_v, coords_sharp_v = case_v_corner(cutout_seven, row, col)
                            
                            # If it's a sharp corner, overwrite decision and deal with this case
                            # If not, proceed with the round corner
                            if match_sharp_v: 
                                for c in coords_sharp_v: border_pixels.append(c)
                            else:
                                border_pixels.append((row, col))
                            
                    
                # Special cases are accounted for, continue with normal variations
                # Pick the neighbours algorithm: 
                if ngb == '4N': 
                    # 4 directions: up, down, left, right
                    neighbors = [(row-1, col), (row, col-1), (row, col+1), (row+1, col)]
                else: 
                    # 8 directions, include corners
                    neighbors = [(row-1, col), (row-1, col-1), (row-1, col+1),
                                 (row, col-1),                 (row, col+1),
                                 (row+1, col), (row+1, col-1), (row+1, col+1)]
                
                # Within the neighbors, look for anti-target color
                for n_row, n_col in neighbors:
                    
                    if 0 <= n_row < rows and 0 <= n_col < cols:
                        
                        if has_color_pixels(image_array[n_row, n_col], anti_target):
                            
                            border_pixels.append((row, col))
                            break
    return border_pixels


# Shrinking - special case: 
# identify if a 90 degrees corner is present in the imageand artificially remove an extra pixel.
# It prevents the formation of triangles of left-over pixels when shrinking with '4N' algorithm
# IMPORTANT: input array must be 5x5 
def case_90deg_corner(array):
    
    # Look for the following case (_ = white, B = black, X = black pixel target):
    # _ _ B B B
    # _ _ B B B
    # B B X B B
    # B B B B B
    # B B B B B
    # If present, X will be removed
    
    # Create a reference of how the corner is represented in the pixel array 
    # 2x2 white square on the top-left corner is our target
    reference_orientation = np.array([[[255,255,255],[255,255,255],[0,0,0],[0,0,0],[0,0,0]],
                                      [[255,255,255],[255,255,255],[0,0,0],[0,0,0],[0,0,0]],
                                      [[0,0,0],      [0,0,0],      [0,0,0],[0,0,0],[0,0,0]],
                                      [[0,0,0],      [0,0,0],      [0,0,0],[0,0,0],[0,0,0]],
                                      [[0,0,0],      [0,0,0],      [0,0,0],[0,0,0],[0,0,0]]], dtype=np.uint8)
    
    # Rotate the input array until it matches the reference 
    for k in range(4):
        rotated_array = np.rot90(array, k = k)
        
        if np.array_equal(rotated_array, reference_orientation):
            break
    
    # Check that the upper-left corner is a 2x2 white patch AND that the rest is black
    white_pos = [(0,0),(0,1),(1,0),(1,1)]
    white_check = all((rotated_array[i, j] == [255, 255, 255]).all() for i, j in white_pos)

    black_pos = [(0,2),(0,3),(0,4),(1,2),(1,3),(1,4),(2,0),(2,1),(2,2),(2,3),(2,4),(3,0),(3,1),(3,2),(3,3),(3,4),(4,0),(4,1),(4,2),(4,3),(4,4)]
    black_check = all((rotated_array[i, j] == [0, 0, 0]).all() for i, j in black_pos)
    
    return (white_check and black_check)


# Shrinking - special case:
# Identify if a 66° degrees corner is present in a given cutoput and artifically remove extra pixels.
# It prevents the formation of triangles of left-over pixels when shrinking particular letter of the Line Braille script
# IMPORTANT: input array must be 9x9 
def case_66deg_corner(array, row, col):
    
    # This case requires actual coordinates relative to the center of the given cutout. It will not rotate the figure 
    # but elaborate case-by-case the three corner situations (there is no Line Braille letter with a bottom-left corener, will skip that case)
    
    # Example (Xs will be removed): 
    # _ _ _ B B B B B B
    # _ _ _ _ B B B B B
    # _ _ _ _ B B B B B
    # _ _ _ _ B X B B B
    # B B B B C X B B B
    # B B B B B B B B B
    # B B B B B B B B B
    # B B B B B B B B B
    # B B B B B B B B B
    
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


# Shrinking - special case: 
# Identify if a 66° degrees corner is present in a given cutout when using the '8N' algorithm.
# It prevents the formation of triangles of left-over pixels when shrinking particular letter of the Line Braille script
# IMPORTANT: input array must be 5x5 
def case_66deg_8N_corner(array, row, col):
    
    # B B B B B
    # B B B B B
    # _ _ X B B
    # _ _ B B B
    # _ B B B B
    # X will be removed, to avoid residuals
    
    # Same procedure of is_33deg_corner: switch angle based on where there is a white pixel 
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


# Shrinking - special case: 
# Identify if a rotated 90 degrees corner is present in a given cutout
# It prevents the formation of triangles in the letter O in the Line Braille script
# IMPORTANT: input array must be 5x5 
def case_line_o(array, row, col):
    
    # B B B B B 
    # _ B B B B
    # _ _ X B B
    # _ _ X B B
    # _ B B B B
    # Xs will be removed, to avoid residuals
    
    # Check if it's an upper-left acute corner
    white_ref = [(1,0),(2,0),(2,1),(3,0),(3,1),(4,0)]
    black_ref = [(0,0),(0,1),(0,2),(0,3),(0,4),(1,1),(1,2),(1,3),(1,4),(2,2),(2,3),(2,4),(3,2),(3,3),(3,4),(4,1),(4,2),(4,3),(4,4)]

    if all((array[i, j] == [255, 255, 255]).all() for i, j in white_ref) and all((array[i, j] == [0,0,0]).all() for i, j in black_ref):
        
        # Specify which voxels should be eroded
        coords = [(row,col+1), (row+1,col+1)]
        return True, coords
    
    else:
        return False, []


# Latin script - special cases: 
# Identify corners between a straight vertical line and a rounded one (like b)
# Prevent the formation of left-over pixels
# IMPORTANT: input array must be 7x7
def case_round_corner(array, row, col): 
    
    # Looks for the pattern of _ (white), and removes X. ? are ignored, as they change for different cases
    # Example (Xs will be removed): 
    # B B B _ _ ? B 
    # B B B _ ? B B 
    # B B B _ B B B 
    # B B X C B B B 
    # B B X B B B B 
    # B B B B B B B 
    # B B B B B B B 
    
    # Check if it's an upper-left acute corner (like d)
    white_UL = [(0,2),(0,3),(1,3),(2,3)]
    black_UL = [(0,0),(0,4),(0,5),(0,6),(1,0),(1,1),(1,4),(1,5),(1,6),(2,0),(2,1),(2,2),(2,4),(2,5),(2,6),(3,0),(3,1),(3,2),(3,3),(3,4),(3,5),
                (3,6),(4,0),(4,1),(4,2),(4,3),(4,4),(4,5),(4,6),(5,0),(5,1),(5,2),(5,3),(5,4),(5,5),(5,6),(6,0),(6,1),(6,2),(6,3),(6,4),(6,5),(6,6)]

    if all((array[i, j] == [255, 255, 255]).all() for i, j in white_UL) and all((array[i, j] == [0,0,0]).all() for i, j in black_UL):
        # Specify which voxels should be eroded
        return True, [(row,col+1),(row+1,col+1)]
    
    # Not UL, rotate reference and check upper-right 
    white_UR = [(0,3),(0,4),(1,3),(2,3)]
    black_UR = [(0,0),(0,1),(0,2),(0,6),(1,0),(1,1),(1,2),(1,5),(1,6),(2,0),(2,1),(2,2),(2,4),(2,5),(2,6),(3,0),(3,1),(3,2),(3,3),(3,4),(3,5),(3,6),
                (4,0),(4,1),(4,2),(4,3),(4,4),(4,5),(4,6),(5,0),(5,1),(5,2),(5,3),(5,4),(5,5),(5,6),(6,0),(6,1),(6,2),(6,3),(6,4),(6,5),(6,6)]

    if all((array[i, j] == [255, 255, 255]).all() for i, j in white_UR) and all((array[i, j] == [0,0,0]).all() for i, j in black_UR):
        # Specify which voxels should be eroded
        return True, [(row,col-1),(row+1,col-1)]
    
    # Neither UR, rotate and check lower-right
    white_LR = [(4,3),(5,3),(6,3),(6,4)]
    black_LR = [(0,0),(0,1),(0,2),(0,3),(0,4),(0,5),(0,6),(1,0),(1,1),(1,2),(1,3),(1,4),(1,5),(1,6),(2,0),(2,1),(2,2),(2,3),(2,4),(2,5),(2,6),
                (3,0),(3,1),(3,2),(3,3),(3,4),(3,5),(3,6),(4,0),(4,1),(4,2),(4,4),(4,5),(4,6),(5,0),(5,1),(5,2),(5,5),(5,6),(6,0),(6,1),(6,2),(6,6)]

    if all((array[i, j] == [255, 255, 255]).all() for i, j in white_LR) and all((array[i, j] == [0,0,0]).all() for i, j in black_LR):
        # Specify which voxels should be eroded
        return True, [(row,col-1),(row-1,col-1)]
    
    # Check lower-left
    white_LL = [(4,3),(5,3),(6,2),(6,3)]
    black_LL = [(0,0),(0,1),(0,2),(0,3),(0,4),(0,5),(0,6),(1,0),(1,1),(1,2),(1,3),(1,4),(1,5),(1,6),(2,0),(2,1),(2,2),(2,3),(2,4),(2,5),(2,6),
                (3,0),(3,1),(3,2),(3,3),(3,4),(3,5),(3,6),(4,0),(4,1),(4,2),(4,4),(4,5),(4,6),(5,0),(5,1),(5,4),(5,5),(5,6),(6,0),(6,4),(6,5),(6,6)]

    if all((array[i, j] == [255, 255, 255]).all() for i, j in white_LL) and all((array[i, j] == [0,0,0]).all() for i, j in black_LL):
        # Specify which voxels should be eroded
        return True, [(row,col+1),(row-1,col+1)]
    
    # Extra cases: check Z corners
    # Z(Z)-top corner
    white_ZT = [(1,6),(2,5),(2,6),(3,4),(3,5),(3,6)]
    black_ZT = [(0,0),(0,1),(0,2),(0,3),(0,4),(0,5),(1,0),(1,1),(1,2),(1,3),(1,4),(2,0),(2,1),(2,2),(2,3),(2,4),(3,0),(3,1),(3,2),(3,3),
                (4,0),(4,1),(4,2),(4,3),(4,4),(4,5),(4,6),(5,0),(5,1),(5,2),(5,3),(5,4),(5,5),(5,6),(6,0),(6,1),(6,2),(6,3),(6,4),(6,5),(6,6)]

    if all((array[i, j] == [255, 255, 255]).all() for i, j in white_ZT) and all((array[i, j] == [0,0,0]).all() for i, j in black_ZT):
        # Specify which voxels should be eroded
        return True, [(row+1,col-1),(row+1,col)]
    
    # Z-corner
    white_ZB = [(3,0),(3,1),(3,2),(4,0)]
    black_ZB = [(0,0),(0,1),(0,2),(0,3),(0,4),(0,5),(0,6),(1,0),(1,1),(1,2),(1,3),(1,4),(1,5),(1,6),(2,0),(2,1),(2,2),(2,3),(2,4),(2,5),(2,6),
                (3,3),(3,4),(3,5),(3,6),(4,2),(4,3),(4,4),(4,5),(4,6),(5,1),(5,2),(5,3),(5,4),(5,5),(5,6),(6,0),(6,1),(6,2),(6,3),(6,4),(6,5),(6,6)]

    if all((array[i, j] == [255, 255, 255]).all() for i, j in white_ZB) and all((array[i, j] == [0,0,0]).all() for i, j in black_ZB):
        # Specify which voxels should be eroded
        return True, [(row-1,col),(row-1,col+1)]
    
    # Not the case
    return False, []


# Latin script - special cases:
# Identify sharp corners present in Y and in the second and third corners of W. Thin corners 1px wide
def case_y_corner(array, row, col): 
    
    # Looks for the pattern of _ (white), and removes X. Two centers are possible, consider both cases
    # Example (Xs will be removed): 
    # B B _ B B     B B B B B 
    # B B _ B B     B B X B B 
    # B B C B B     B B C B B 
    # B B X B B     B B _ B B
    # B B B B B     B B _ B B
    
    # Check for bottom reference
    white_B = [(0,2),(1,2)]
    black_B = [(0,0),(0,1),(0,3),(0,4),(1,0),(1,1),(1,3),(1,4),(2,0),(2,1),(2,3),(2,4),(3,0),(3,1),(3,2),(3,3),(3,4),(4,0),(4,1),(4,2),(4,3),(4,4)]

    if all((array[i, j] == [255, 255, 255]).all() for i, j in white_B) and all((array[i, j] == [0,0,0]).all() for i, j in black_B):
        # Specify which voxels should be eroded
        return True, [(row+1,col),(row+2,col)]
    
    # Check for top reference
    white_T = [(3,2),(4,2)]
    black_T = [(0,0),(0,1),(0,2),(0,3),(0,4),(1,0),(1,1),(1,2),(1,3),(1,4),(2,0),(2,1),(2,3),(2,4),(3,0),(3,1),(3,3),(3,4),(4,0),(4,1),(4,3),(4,4)]

    if all((array[i, j] == [255, 255, 255]).all() for i, j in white_T) and all((array[i, j] == [0,0,0]).all() for i, j in black_T):
        # Specify which voxels should be eroded
        return True, [(row-1,col),(row-2,col)]
    
    # Not the case
    return False, []
    
    
# Latin script - special cases:
# Identify sharp corners present in V and in the first part of W. Sharp corners 2px thick
def case_v_corner(array, row, col): 
    
    # Looks for the pattern of _ (white), and removes X. Two centers are possible, consider both cases
    # Example (Xs will be removed): 
    # ? ? B B _ _ B     B _ _ B B ? ?
    # ? ? B B _ _ B     B _ _ B B ? ?
    # ? ? B B _ _ B     B _ _ B B ? ?
    # ? ? B C X B B     B B X C B ? ?
    # ? ? B B X B B     B B X B B ? ?
    # ? ? ? ? X ? ?     ? ? ? ? ? ? ?
    # ? ? ? ? ? ? ?     ? ? ? ? ? ? ?
    
    # Check for left reference
    white_L = [(0,4),(0,5),(1,4),(1,5),(2,4),(2,5)]
    black_L = [(0,2),(0,3),(0,6),(1,2),(1,3),(1,6),(2,2),(2,3),(2,6),(3,2),(3,3),(3,4),(3,5),(3,6),(4,2),(4,3),(4,4),(4,5),(4,6)]

    if all((array[i, j] == [255, 255, 255]).all() for i, j in white_L) and all((array[i, j] == [0,0,0]).all() for i, j in black_L):
        # Specify which voxels should be eroded
        return True, [(row,col+1),(row+1,col+1),(row+2,col+1)]
    
    # Check for right reference
    white_R = [(0,1),(0,2),(1,1),(1,2),(2,1),(2,2)]
    black_R = [(0,0),(0,3),(0,4),(1,0),(1,3),(1,4),(2,0),(2,3),(2,4),(3,0),(3,1),(3,2),(3,3),(3,4),(4,0),(4,1),(4,2),(4,3),(4,4)]

    if all((array[i, j] == [255, 255, 255]).all() for i, j in white_R) and all((array[i, j] == [0,0,0]).all() for i, j in black_R):
        # Specify which voxels should be eroded
        return True, [(row,col-1),(row+1,col-1),(row+2,col-1)]
    
    # Not the case
    return False, []

# Correct pixel conversion of latin letters. 
# In some cases (a,e,h,m,v,w) the transformation from affinity to .png resulted in wrong pixels
def correct_latin_letters(array, letter_info):
    
    # A,H,M need an extra pixel on the lower part of the curves, to avoid indents
    # E needs an extra pixel just below the straight line, to avoid indent
    # V needs an extra pixel to avoid being caught in other cases when shrinking
    # W needs extra white pixels to conform the lower corners
    if letter_info.startswith('lt'):
        if letter_info.endswith('a'): array[(210,273)] = [0,0,0]
        elif letter_info.endswith('e'): array[(256,214)] = [0,0,0]
        elif letter_info.endswith('h'): array[(211,271)] = [0,0,0]
        elif letter_info.endswith('m'): array[(205,211),(256,196)] = [0,0,0]
        elif letter_info.endswith('v'): array[(284,249)] = [0,0,0]
        elif letter_info.endswith('w'): array[(279,280),(285,285)] = [255,255,255]
        
    return array
        

# Copy image to shrink and expand
def copy_image(path, image, non_black, letter):
    
    output_image = Image.open(path).convert("RGB")
    output_array = np.array(output_image)
    output_array[non_black] = [0,0,0]
    output_array = correct_latin_letters(output_array, letter)
    
    return output_array


# Resizing function
# From given sizes and image, will create the 5 (actually 4) variations needed
def resize_image(path, image, sizes, stim_info):
    
    # Set desitantion folder
    dest_folder = path.split("/", 1)[0]
    
    # Assuming images passed are squared (they should be), use the original side 
    # to crop enlarged /shrank images
    resize_factor = image.width
    
    ## Make letter the same
    # Just adjust name and save original
    cropped_savename = f'../../input/words/variations/{stim_info}S3.png'
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
        cropped_savename = f'../../input/words/variations/{stim_info}S{4+i}.png'
        
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
        cropped_savename = f'../../input/words/variations/{stim_info}S{2-i}.png'
        
        # Save image
        cropped_image.save(cropped_savename)

    
# Thickening function
# From a given image, will create 4 variations in thickness
def thicken_image(path, image, thicknesses, stim_info):
    
    # Set reference colors
    ref_white = [255, 255, 255]
    ref_black = [0,   0,   0]
    
    # Set desitantion folder
    dest_folder = path.split("/", 1)[0]
    
    # Set input image as array, easier to work with
    image_array = np.array(image)
    
    # Create copies to avoid overwriting information
    shrink_array = np.array(image)
    expand_array = np.array(image)
    
    # Cast non-white pixels as black, to simplify the tasks
    non_black = np.any(image_array != ref_white, axis=-1)
    
    image_array[non_black] = ref_black
    image_array = correct_latin_letters(image_array, stim_info)
    
    # Create copies of the image to avoid overwriting of information
    shrink_array = copy_image(path, image, non_black, stim_info)
    expand_array = copy_image(path, image, non_black, stim_info)
    
    # Save T3 image (mid-thickness, a.k.a. original format)
    black_image = Image.fromarray(image_array.astype('uint8'), 'RGB')
    blackname = f'../../input/words/variations/{stim_info}_T3.png'
    black_image.save(blackname)
    
    
    ## Expansion
    
    # Set counter to track variations
    plus = 1
    
    # Fixed algorithm for expansion
    neighbors = '8N'
    
    for i, t in enumerate(thicknesses):
        
        # until it reaches the desired thickness 
        while plus < t:
        
            # identify which pixels to change
            pixels_to_change = identify_pixels(expand_array, 'white', [], neighbors)
            
            # Paint them black
            for pixel in pixels_to_change: expand_array[pixel[0], pixel[1]] = ref_black
            
            # Update iteration counter
            plus = plus+1
        
        # save image in state-of-the-art folder with change notation
        colored_image = Image.fromarray(expand_array)
        colored_savename = f'../../input/words/variations/{stim_info}_T{i+4}.png'
        colored_image.save(colored_savename)
        
        
    ## Shrinking 
    
    # Set counter
    minus = 1
    
    # Tailor the algorithm to the script and letters shrunk, to limit corner issues
    # Split letter info to extract script and letter
    parts = stim_info.split('_')
    script = parts[0]
    letter = parts[1]
    
    # Braille has the same algorithm
    if script == 'br': neighbors = '8N'
        
    # If Line Braille, sitatuion is more complex. Choose based on letter
    elif script == 'ln':
        if letter in ['n','s','z']:   # 45 degrees lines
            ngb_counter = 0
            sequence = ['4N','4N','8N','4N','8N']
            neighbors = sequence[ngb_counter]

        elif letter in ['m','u','x']: # 66 degrees lines
            ngb_counter = 0
            sequence = ['4N','4N','8N']
            neighbors = sequence[ngb_counter]
            
        elif letter in ['o']: neighbors = '8N' # peculiar case, treat separately
        else: neighbors = '4N'                 # all the orher letters
    
    # Latin script has a standard algorithm        
    else: neighbors = '4N' 
        
    
    for i, t in enumerate(thicknesses): 
        
        # As for expansion, loop until we reach a desired number of steps, then save
        while minus < t:
        
            # identify which pixels to change
            pixels_to_change = identify_pixels(shrink_array, 'black', script, neighbors)
            
            # Paint them white
            for pixel in pixels_to_change: shrink_array[pixel[0], pixel[1]] = ref_white
            
            # Update iteration counter
            minus = minus+1
            
            # In case of peculiar letter in line script, update the neighbor choice
            # Increase counter to move through the sequence, reset it if we reach the end
            if script == 'ln' and letter in ['n','s','z','m','u','x']:
                
                # 45 degrees
                if letter in ['n','s','z']:
                    if ngb_counter == 4: ngb_counter = 0
                    else: ngb_counter = ngb_counter +1
                        
                    neighbors = sequence[ngb_counter]
    
                # 67 degrees
                if letter in ['m','u','x']:
                    if ngb_counter == 2: ngb_counter = 0
                    else: ngb_counter = ngb_counter +1
                        
                    neighbors = sequence[ngb_counter]
        
        # save image in state-of-the-art folder with change notation
        colored_image = Image.fromarray(shrink_array)
        colored_savename = f'../../input/words/variations/{stim_info}_T{2-i}.png'
        colored_image.save(colored_savename)
