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


# ZIP FOLDER
# used for test sets
def zip_folder(parent_folder):
    
    # Get absolute path
    folder_path = os.path.abspath(parent_folder)  
    
    # Extract folder name and use it to make zipped folder name
    folder_name = os.path.basename(folder_path)  
    output_zip = os.path.join(os.path.dirname(folder_path), f"{folder_name}.zip") 

    # Zip
    with zipfile.ZipFile(output_zip, 'w', zipfile.ZIP_DEFLATED) as zipf:
        
        for root, _, files in os.walk(folder_path):
            
            for file in files:
                
                file_path = os.path.join(root, file)
                
                # Preserve folder structure
                arcname = os.path.relpath(file_path, folder_path)  
                zipf.write(file_path, arcname)
    
    # Once it's done, remove the original folder
    os.rmdir(parent_folder)





