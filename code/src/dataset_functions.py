#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 13 16:15:32 2024

@author: cerpelloni
"""

## Create Dataset(s) for training / testing

import os
import zipfile
import glob
import shutil
import pandas as pd
from stimuli_main import *

# Create Dataset (folder structure)for Braille-Latin script associations
# Save a zipped file in input/dataset_Braille.zip
def create_dataset_BR(): 
    
    print("Dataset of Latin-Braille words")
    
    # Classes - word list
    words_path = '../../input/words/nl_wordlist.csv'
    
    # Path to stored images
    stimuli_folder = '../../input/words/stimuli'
    
    print(f"Sorting images from {stimuli_folder} class folders ...")
    
    # Define the scripts to match and copy
    scripts = ['br_', 'lt_']
    
    # Path to dataset structure
    dataset_folder = '../../input/datasets/dataset_BR'
    zip_folder = '../../input/datasets'
    
    # Read the CSV file
    df = pd.read_csv(words_path, header = None, names = ['classes'])
    
    # Ensure destination folder exists
    os.makedirs(dataset_folder, exist_ok = True)
    
    # Iterate over each class name
    for class_name in df['classes']:
        
        # Create a subfolder for each class
        class_folder = os.path.join(dataset_folder, class_name)
        os.makedirs(class_folder, exist_ok = True)
        
        # Find and selectively move images matching the class name and allowed scripts
        for image_name in os.listdir(stimuli_folder):
            
            # Check if the image starts with any allowed prefix and includes the class name
            if any(image_name.startswith(script + class_name) for script in scripts):
                
                # Define source and destination paths
                src_path = os.path.join(stimuli_folder, image_name)
                dest_path = os.path.join(class_folder, image_name)
                    
                # Copy the image to the class folder
                shutil.copy(src_path, dest_path)
    
    print("Images sorted into class folders \n")


# Create Dataset (folder structure) for Latin-Line script
# Save a zipped file in input/dataset_LineBraille.zip
def create_dataset_LN(): 
    
    print("Dataset of Latin-Line words")
    
    # Classes - word list
    words_path = '../../input/words/nl_wordlist.csv'
    
    # Path to stored images
    stimuli_folder = '../../input/words/stimuli'
    
    print(f"Sorting images from {stimuli_folder} class folders ...")
    
    # Define the scripts to match and copy
    scripts = ['ln_', 'lt_']
    
    # Path to dataset structure
    dataset_folder = '../../input/datasets/dataset_LN'
    zip_folder = '../../input/datasets'
    
    # Read the CSV file
    df = pd.read_csv(words_path, header = None, names = ['classes'])
    
    # Ensure destination folder exists
    os.makedirs(dataset_folder, exist_ok = True)
    
    # Iterate over each class name
    for class_name in df['classes']:
        
        # Create a subfolder for each class
        class_folder = os.path.join(dataset_folder, class_name)
        os.makedirs(class_folder, exist_ok = True)
        
        # Find and selectively move images matching the class name and allowed scripts
        for image_name in os.listdir(stimuli_folder):
            
            # Check if the image starts with any allowed prefix and includes the class name
            if any(image_name.startswith(script + class_name) for script in scripts):
                
                # Define source and destination paths
                src_path = os.path.join(stimuli_folder, image_name)
                dest_path = os.path.join(class_folder, image_name)
                    
                # Copy the image to the class folder
                shutil.copy(src_path, dest_path)
        
    print("Images sorted into class folders \n")
    

# Check whether the datasets exists. If no, make them
def do_datasets_exist(): 
    
    # Define where to look for the datasets
    folder_path = "../../input/datasets"

    # Define the names to look for
    file_names = {"dataset_LineBraille.zip", "dataset_Braille.zip"}

    # Check if the files exist in the folder
    existing_files = {file for file in os.listdir(folder_path) if file in file_names}
    
    return existing_files

# Check if the stimuli are created in all their variations 
# (to avoid re-running that long script)
def do_variations_exist():
    
    # Define the folder path
    folder_path = '../../input/words/variations'
    
    # Define the search pattern
    search_pattern = os.path.join(folder_path, '*_T5S5.png')
    
    # Get the list of matching files
    images = glob.glob(search_pattern)
    
    # Count the number of matching files
    count = len(images)
    
    return count == 1000


# Load the datasets 
def load_dataset(dataset): 
    
    # Based on the name of the dataset, load it
    # Full path to the zip file
    path = '../../input/datasets'
    zip_path = os.path.join(path, dataset)

    # Check if the zip file exists
    if not os.path.isfile(zip_path):
        return f"Error: {dataset} does not exist in {path}."
    
    # Check if the file is a zip archive
    if not zipfile.is_zipfile(zip_path):
        return f"Error: {dataset} is not a valid zip file."
    
    try:
        # Unzip the file
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            extract_path = os.path.join(path, dataset.replace('.zip', ''))
            zip_ref.extractall(extract_path)
            
        return f"Successfully unzipped {dataset} to {path}."
    
    except Exception as e:
        return f"An error occurred while unzipping: {e}"


# Make the datasets from scratch
def make_stimuli(): 
    
    if not do_datasets_exist():
        
        print('Found no dataset')
        # if the datasets are not there, two possible options:
        # - variation of stimuli are created and we only need to arrange the stimuli
        #   into folders
        if do_variations_exist():
            
            print('Found images. Organizing them ...')
            # Create the datasets from the images already created
            create_dataset_BR()
            create_dataset_LN()
            
        else:
            
            # We need to create the stimuli from scratch
            # Time consuming script, be aware
            
            print('Found no images. Launching ''stimuli_main'' code to create them ...')
            stimuli_main
            
    else: 
        
        print('Datasets found. No need to create them')

