#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  9 13:34:00 2024

@author: cerpelloni
"""
import os
import glob
import re

import numpy as np
import scipy.stats as stats

from PIL import Image

## Count amount of black pixels in image
def count_black_pixels(image_path):
    
    # Load the image
    image = Image.open(image_path).convert('RGB')
    np_image = np.array(image)
    
    # A pixel is black if all its RGB components are zero
    black_pixels = np.sum(np.all(np_image == [0, 0, 0], axis = -1))
    
    return black_pixels



### Main script

## Load letters
# Find paths for all scripts
br_dir = "letters/stimuli" 
br_paths = glob.glob(os.path.join(br_dir, 'br_*.png'))
ln_dir = "letters/stimuli"
ln_paths = glob.glob(os.path.join(ln_dir, 'ln_*.png'))
lt_dir = "letters/stimuli"
lt_paths = glob.glob(os.path.join(lt_dir, 'lt_*.png'))


## Across scripts 
# Compute amount of black pixels for each image in the scripts
br_pixels = [count_black_pixels(br) for br in br_paths]
ln_pixels = [count_black_pixels(ln) for ln in ln_paths]
lt_pixels = [count_black_pixels(lt) for lt in lt_paths]

# Compute statistical tests between scripts 
f_stat, p_value = stats.f_oneway(br_pixels, ln_pixels, lt_pixels)

# Visualize test 
print('Difference in the number of black pixels across scripts')
print(f'ANOVA: F = {f_stat}; p = {p_value}')
print('')

# Show additional information
print('Black pixels')

br_mean = np.average(br_pixels)
br_std = np.std(br_pixels)
print(f'Braille: mean = {br_mean}; std = {br_std}')

ln_mean = np.average(ln_pixels)
ln_std = np.std(ln_pixels)
print(f'Line: mean = {ln_mean}; std = {ln_std}')

lt_mean = np.average(lt_pixels)
lt_std = np.std(lt_pixels)
print(f'Latin: mean = {lt_mean}; std = {lt_std}')
print('')

if p_value > 0.05:
    print('no signifcant differences between groups')
else: 
    print('check scripts to ensure similar pixel density')
    

## Across single variations
# Loop thorugh variations and compare the number of pixels in each of them
# e.g. T1S1, T1S2, ...
f_matrix = np.full((5, 5), np.nan)
pval_matrix = np.full((5, 5), np.nan)
br_matrix = np.full((5, 5), np.nan)
ln_matrix = np.full((5, 5), np.nan)
lt_matrix = np.full((5, 5), np.nan)

for t in range(5):
    
    # extract only the images relative to that T (t+1)
    pattern = f'_T{t+1}'
    t_br_paths = [br for br in br_paths if re.search(pattern, br)]
    t_ln_paths = [ln for ln in ln_paths if re.search(pattern, ln)]
    t_lt_paths = [lt for lt in lt_paths if re.search(pattern, lt)]
    
    for s in range(5): 
        
        # extract only the images relative to that S (s+1)
        pattern = f'S{s+1}'
        ts_br_paths = [br for br in t_br_paths if re.search(pattern, br)]
        ts_ln_paths = [ln for ln in t_ln_paths if re.search(pattern, ln)]
        ts_lt_paths = [lt for lt in t_lt_paths if re.search(pattern, lt)]
        
        # Compute amount of black pixels for each image in the scripts
        br_pixels = [count_black_pixels(br) for br in ts_br_paths]
        ln_pixels = [count_black_pixels(ln) for ln in ts_ln_paths]
        lt_pixels = [count_black_pixels(lt) for lt in ts_lt_paths]
        
        # Compute statistical tests between scripts 
        f_stat, p_value = stats.f_oneway(br_pixels, ln_pixels, lt_pixels)
        
        # Visualize test 
        print(f'Difference in the number of black pixels within T{t+1} and S{s+1}')
        print(f'ANOVA: F = {f_stat}; p = {p_value}')
        print('')
        
        # Show additional information
        print('Black pixels')
        
        br_mean = np.average(br_pixels)
        br_std = np.std(br_pixels)
        print(f'Braille: mean = {br_mean}; std = {br_std}')
        
        ln_mean = np.average(ln_pixels)
        ln_std = np.std(ln_pixels)
        print(f'Line: mean = {ln_mean}; std = {ln_std}')
        
        lt_mean = np.average(lt_pixels)
        lt_std = np.std(lt_pixels)
        print(f'Latin: mean = {lt_mean}; std = {lt_std}')
        print('')
        
        if p_value > 0.05:
            print('no signifcant differences between groups')
        else: 
            print('check scripts to ensure similar pixel density')
        print('')
        print('')
            
        # Save information:
        # if there are problems, we need to know where to look
        # if there are no problems, we want to report this 
        f_matrix[t,s] = f_stat
        pval_matrix[t,s] = p_value
        br_matrix[t,s] = br_mean
        ln_matrix[t,s] = ln_mean
        lt_matrix[t,s] = lt_mean
        

# Plot p-values RDM











