#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec  8 18:08:33 2024

Set of functions to zip files relative to Visual Braille Silico

Helps managing upload of files to gin through datalad


@author: Filippo Cerpelloni (actually chatGPT generated code)
"""

import os
import csv
import subprocess
import zipfile

## ZIP FILES TOGETHER
# Used in variations, to create zip files containing all the stimuli for a class
# e.g. br_aai_T*S*X*Y* ln_aai_T*S*X*Y* lt_aai_T*S*X*Y* -> aai.zip
def zip_files(folder_path, wordlist):
    
    # Ensure the folder exists
    if not os.path.exists(folder_path):
        print(f"The folder {folder_path} does not exist.")
        return
    
    # Read the list of names from the CSV file
    try:
        with open(wordlist, 'r') as csvfile:
            reader = csv.reader(csvfile)
            names = [row[0].strip() for row in reader]  # Assuming names are in the first column
    except FileNotFoundError:
        print(f"The file {wordlist} does not exist.")
        return

    # Change to the target folder
    os.chdir(folder_path)
    print(f"Changed working directory to: {folder_path}")
    
    # Iterate over each name in the CSV file
    for name in names:
        zip_filename = f"{name}.zip"
        command = f"zip {zip_filename} *_{name}_*"
        
        try:
            # Execute the zip command
            result = subprocess.run(command, shell=True, capture_output=True, text=True)
            
            if result.returncode == 0:
                print(f"Created ZIP: {zip_filename}")
            else:
                print(f"Failed to create ZIP for '{name}'. Error: {result.stderr.strip()}")
        except Exception as e:
            print(f"Error running command for '{name}': {e}")



## ZIP SUBFOLDERS
# Used in dataset_BR and dataset_LN to push to gin
def zip_subfolders(parent_folder):
    
    # Ensure the parent folder exists
    if not os.path.exists(parent_folder):
        print(f"The folder {parent_folder} does not exist.")
        return
    
    # Iterate through each item in the parent folder
    for folder_name in os.listdir(parent_folder):
        subfolder_path = os.path.join(parent_folder, folder_name)
        
        # Process only subfolders
        if os.path.isdir(subfolder_path):
            zip_filename = f"{subfolder_path}.zip"
            with zipfile.ZipFile(zip_filename, 'w', zipfile.ZIP_DEFLATED) as zipf:
                # Walk through the subfolder
                for root, _, files in os.walk(subfolder_path):
                    for file in files:
                        file_path = os.path.join(root, file)
                        # Add files to the zip, preserving the folder structure
                        zipf.write(file_path, os.path.relpath(file_path, subfolder_path))
            
            print(f"Zipped: {zip_filename}")


## UNZIP SUBFOLDERS
# used in datasets to extract data from the zip files downloaded from gin
def unzip_files(parent_folder):
    
    if not os.path.exists(parent_folder):
        print(f"The folder {parent_folder} does not exist.")
        return
    
    
    for file_name in os.listdir(parent_folder):
        if file_name.endswith(".zip"):
            zip_path = os.path.join(parent_folder, file_name)
            extract_folder = os.path.join(parent_folder, file_name[:-4])
            
            with zipfile.ZipFile(zip_path, 'r') as zipf:
                zipf.extractall(extract_folder)
            print(f"Extracted: {zip_path} to {extract_folder}")











