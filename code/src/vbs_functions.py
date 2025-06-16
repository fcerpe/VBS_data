#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 27 15:19:49 2025

Visual Braille Silico - general support functions for the whole repository

@author: Filippo Cerpelloni
"""

### ---------------------------------------------------------------------------
### Imports

import os
import pickle
import zipfile
import shutil
import glob

import pandas as pd

from PIL import Image

### ---------------------------------------------------------------------------
### Options - paths and general settings for saving and loading

def vbs_option():
    """
    Pre-compute paths for important folders in the IODA structure.
    Folders stored:
    - root
    - inputs: inputs, datasets
    - outputs: outputs, figures, weights, logs, results

    Parameters
    ----------
    None
    
    Outputs
    -------
    opt (dict): a dictionary storing the fullpath for folders

    """
    
    # Initialize options dictionary
    opt = {}

    ## PATHS
    
    # IMPORTANT: paths are optimized to run on enuui, adjust if on a different system
    # Enuii: /data/Filippo/  ||  IODA: ../..

    # The directory where the data are located
    opt['dir'] = {}
    opt['dir']['root'] = os.path.join(os.getcwd(), '..', '..')
    
    opt['dir']['inputs'] = os.path.join(opt['dir']['root'], 'inputs')
    # Bypass IODA folder sturcture for training on enuui 
    # change based on the system used 
    # opt['dir']['datasets'] = '/Data/Filippo/inputs/datasets
    opt['dir']['datasets'] = os.path.join(opt['dir']['inputs'], 'datasets')
    
    opt['dir']['outputs'] = os.path.join(opt['dir']['root'], 'outputs')
    opt['dir']['figures'] = os.path.join(opt['dir']['outputs'], 'figures')
    opt['dir']['weights'] = os.path.join(opt['dir']['outputs'], 'weights') 
    opt['dir']['logs'] = os.path.join(opt['dir']['outputs'], 'logs')
    opt['dir']['results'] = os.path.join(opt['dir']['outputs'], 'results')
    
    
    # Information about the script selected
    opt['script'] = {'latin':   {'dataset_spec': 'LT'},
                     'braille': {'dataset_spec': 'LTBR'},
                     'line':    {'dataset_spec': 'LTLN'}}
    

    # List of subjects for full analyses
    opt['subjects'] = [0,1,2,3,4] 
    
    # Information over the important analyses layers given the network name
    opt['layers'] = {'alexnet': ['module.features.1',
                                 'module.features.4',
                                 'module.features.7',
                                 'module.features.9',
                                 'module.features.11',
                                 'module.classifier.2',
                                 'module.classifier.5'],
                     'cornet': ['module.V1.nonlin',
                                'module.V2.nonlin',
                                'module.V4.nonlin',
                                'module.IT.nonlin',
                                'module.decoder.linear']}
                    
    
    # TODO add computer argument to distinguish 'macbook' and 'enuui' and set 
    # paths accordingly
    
    return opt



### ---------------------------------------------------------------------------
### Manage filenames 

def vbs_parce_filename(filename):
    """
    Split a filename into the different BIDS-like tags of the name.
    Saves and returns all of them into a dictionary.
    
    Common tags to all files 
    * model [alexnet, cornet]: model used in the computation
    * sub [0, 1, 2, 3, 4, all]: which network instance (all for averages)
    
    Specific tags relative to the processing step
    * training [LT, LTBR, lTLN]: dataset on which a network is trained
    * test [VBE, VBT]: dataset on which network is tested, refers to human experiments
    * script [LT, BR, LN]: script in the images presented to the network at test
    * layer [ReLU-1..7, TBD]: indication of the layer of the network analysed
    * epoch [1..15]: indication of the epoch of training
    * date [YYYY-MM-DD_hh-mm-ss]: date of training of the network instance
    * plot [description]: type of plot saved
    * data [description]: type of data saved
    * analysis [description]: type of analysis performed on the data

    Parameters
    ----------
    filename (str): the name of the file to be loaded
        
    Outputs
    -------
    tags (dict): the list of elements in the filename, to be used to load and save 
                 results and plots
    """
    
    # Initialize dictionary containing the tags of the filename
    tags = {}
    
    # Ensure that it's just the filename and not a fullpath.
    # If so, first split the path to get the filename only
    path_split = filename.split('/model')
    
    if len(path_split) > 1: 
        tags['path'] = path_split[0]
        filename = f'model-{path_split[1]}'
    
    # Split the filename for '.' to extract the extension
    ext_split = filename.split('.')
    tags['ext'] = ext_split[-1]
    filename = ext_split[0]
    
    # Split the filename for '_' to extract each tag and value
    tag_splits = filename.split('_')
    
    # Iterate through all the tags
    for iT, spl in enumerate(tag_splits):
    
        # Split for '-' to extract the value
        value_split = spl.split('-')
        
        # the first one it the tag, the rest is value
        # if there are more than one value, join together again after split
        # (e.g. analysis: descriptive-anova)
        tags[f'{value_split[0]}'] = value_split[1]
        
    return tags
    


### ---------------------------------------------------------------------------
### Save and load activations

## Save the activations and distance matrices with a predefined filename
def save_extractions(opt, flat, stim, dist, model_name, s, training, test, epoch, method):
    """
    Omni-comprehensive function to save activations extracted and computed 
    from functions in the "activation" step of the pipeline

    Parameters
    ----------
    opt (dict): output of vbs_option() containing the paths of the IODA folder 
    
    flat(dict): activations for every image presented in each layer
        
    stim (dict): stimulus averages of the activations in flat for each layer
        
    dist (dict): distances matrices for each layer 
        
    model_name (str): model used 
        
    s (int): "subject" number, network's iteration
    
    training (str): the expertise of the network
        
    test (str): notation of the experiment tested, either VBE or VBT
    
    epoch (str): when the extraction was performed relative to the training of 
                 the network
                 
    method (str): the method used to compute the distances
        
    Outputs
    -------
    None, but flat, stim, dist will be saved as pickle files in outputs/results
    """

    # Flat activations
    save_single_extraction(opt, flat, model_name, s, training, test, epoch, 
                           'activations', 
                           'flat-activations')
   
    # Stimulus average activations
    save_single_extraction(opt, stim, model_name, s, training, test, epoch, 
                           'activations', 
                           'stimulus-average-activations')

    # Distances
    save_single_extraction(opt, dist, model_name, s, training, test, epoch,
                           'distances', 
                           f'distances_method-{method}')


## Save the activations and distance matrices with a predefined filename
def save_single_extraction(opt, in_dict, model_name, s, training, test, epoch, activation_type, data_info):
    """
    Save one dictionary as pickle. Smaller function compared to save_activations 
    to save only one dictionary computed in any step of the analyses

    Parameters
    ----------
    opt (dict): output of vbs_option() containing the paths of the IODA folder 
    
    in_dict (dict): activations for every image presented in each layer
        
    model_name (str): model used 
        
    s (int): "subject" number, network's iteration
    
    training (str): the expertise of the network
        
    test (str): notation of the experiment, either VBE or VBT
    
    epoch (str): when the extraction was performed relative to the training of 
                 the network
    
    activation_type (str): type of activation to be saved, used to specify if it's 
                           activations or a distances
    
    data_info (str): specification of what to call the saved data
        
    Outputs
    -------
    None, but flat, stim, dist will be saved as pickle files in outputs/results
    """
    
    # Set path and filename 
    filepath = os.path.join(opt['dir']['results'], activation_type)

    ## Flat activations
    filename = f'model-{model_name}_sub-{s}_training-{training}_test-{test}_epoch-{epoch}_data-{data_info}.pkl'
    savename = os.path.join(filepath, filename)

    # Save to file
    with open(savename, "wb") as f:
        pickle.dump(in_dict, f)


## Unpickle activation, performance, or distance dictionaries
def load_extraction(path):
    """
    Load and unpickle activations, performances, distances saved through 
    save_single_extractionactivations

    Parameters
    ----------
    path (str): where to find the pickle to load 
    
    Outputs
    -------
    data (dict): the unpickled dictionary, being it flat, stim, or dist
    """

    # Unpickle the dict
    with open(path, "rb") as f:
        data = pickle.load(f)

    return data



### ---------------------------------------------------------------------------
### Dataset functions 

## Create Dataset folder structure for Latin script stimuli 
def create_dataset_Latin(opt): 
    """
    Organize stimuli of Latin-based alphabet into folder-structure to be 
    compatible with ImageFolder class of pytorch.
    It will be used in training the networks
    
    Parameters
    ----------
    opt (dict): a dictionary storing the fullpath for folders
    
    Outputs
    -------
    None, dataset is saved in inputs/datasets/LT

    """
    
    # Notify the user
    print("Creating the dataset of Latin words only ...\n")
    
    # Get the list of classes / words to create subfolders
    words_path = os.path.join(opt['dir']['inputs'], 'words', 'nl_wordlist.csv')
    words_classes = pd.read_csv(words_path, header = None, names = ['classes'])
    
    # Get the images to sort into folders and the path to the dataset structure
    stimuli_folder = os.path.join(opt['dir']['inputs'], 'words', 'temp_variations')
    dataset_folder = os.path.join(opt['dir']['inputs'], 'datasets', 'LT')
    
    # Define the font IDs
    scripts = ['_F1', '_F2', '_F3', '_F4']
    
    # Ensure destination folder exists
    os.makedirs(dataset_folder, exist_ok = True)
    
    # Iterate over each class name
    for class_name in words_classes['classes']:
        
        # Create a subfolder for each class
        class_folder = os.path.join(dataset_folder, class_name)
        os.makedirs(class_folder, exist_ok = True)
        
        # Find and move the images matching the class and script
        for image_name in os.listdir(stimuli_folder):
            
            # Check if the image starts with any allowed prefix and includes the class name
            if any(image_name.startswith(class_name + script) for script in scripts):
                
                # Define source and destination paths
                src_path = os.path.join(stimuli_folder, image_name)
                dest_path = os.path.join(class_folder, image_name)
                    
                # Copy the image to the class folder
                shutil.copy(src_path, dest_path)
    
    # Notify the user
    print("Images sorted into class folders, dataset is created. Enjoy!\n")


## Create Dataset folder structure for Latin + Braille script stimuli 
def create_dataset_LatinBraille(opt): 
    """
    Organize stimuli of Latin-based alphabet and Braille alphabet into 
    folder-structure to be compatible with ImageFolder class of pytorch.
    It will be used in training the networks
    
    Parameters
    ----------
    opt (dict): a dictionary storing the fullpath for folders
    
    Outputs
    -------
    None, dataset is saved in inputs/datasets/LTBR

    """
    
    # Notify the user
    print("Creating the dataset of Latin words + Braille ...\n")
    
    # Get the list of classes / words to create subfolders
    words_path = os.path.join(opt['dir']['inputs'], 'words', 'nl_wordlist.csv')
    words_classes = pd.read_csv(words_path, header = None, names = ['classes'])
    
    # Get the images to sort into folders and the path to the dataset structure
    stimuli_folder = os.path.join(opt['dir']['inputs'], 'words', 'temp_variations')
    dataset_folder = os.path.join(opt['dir']['inputs'], 'datasets', 'LTBR')
    
    # Define the font IDs
    scripts = ['_F1', '_F2', '_F3', '_F4', '_F5']
    
    # Ensure destination folder exists
    os.makedirs(dataset_folder, exist_ok = True)
    
    # Iterate over each class name
    for class_name in words_classes['classes']:
        
        # Create a subfolder for each class
        class_folder = os.path.join(dataset_folder, class_name)
        os.makedirs(class_folder, exist_ok = True)
        
        # Find and move the images matching the class and script
        for image_name in os.listdir(stimuli_folder):
            
            # Check if the image starts with any allowed prefix and includes the class name
            if any(image_name.startswith(class_name + script) for script in scripts):
                
                # Define source and destination paths
                src_path = os.path.join(stimuli_folder, image_name)
                dest_path = os.path.join(class_folder, image_name)
                    
                # Copy the image to the class folder
                shutil.copy(src_path, dest_path)
    
    # Notify the user
    print("Images sorted into class folders, dataset is created. Enjoy!\n")


## Create Dataset folder structure for Latin + Line Braille script stimuli 
def create_dataset_LatinLine(opt): 
    """
    Organize stimuli of Latin-based alphabet and Line Braille into 
    folder-structure to be compatible with ImageFolder class of pytorch.
    It will be used in training the networks
    
    Parameters
    ----------
    opt (dict): a dictionary storing the fullpath for folders
    
    Outputs
    -------
    None, dataset is saved in inputs/datasets/LTLN

    """
    
    # Notify the user
    print("Creating the dataset of Latin words + Line Braille ...\n")
    
    # Get the list of classes / words to create subfolders
    words_path = os.path.join(opt['dir']['inputs'], 'words', 'nl_wordlist.csv')
    words_classes = pd.read_csv(words_path, header = None, names = ['classes'])
    
    # Get the images to sort into folders and the path to the dataset structure
    stimuli_folder = os.path.join(opt['dir']['inputs'], 'words', 'temp_variations')
    dataset_folder = os.path.join(opt['dir']['inputs'], 'datasets', 'LTLN')
    
    # Define the font IDs
    scripts = ['_F1', '_F2', '_F3', '_F4', '_F6']
    
    # Ensure destination folder exists
    os.makedirs(dataset_folder, exist_ok = True)
    
    # Iterate over each class name
    for class_name in words_classes['classes']:
        
        # Create a subfolder for each class
        class_folder = os.path.join(dataset_folder, class_name)
        os.makedirs(class_folder, exist_ok = True)
        
        # Find and move the images matching the class and script
        for image_name in os.listdir(stimuli_folder):
            
            # Check if the image starts with any allowed prefix and includes the class name
            if any(image_name.startswith(class_name + script) for script in scripts):
                
                # Define source and destination paths
                src_path = os.path.join(stimuli_folder, image_name)
                dest_path = os.path.join(class_folder, image_name)
                    
                # Copy the image to the class folder
                shutil.copy(src_path, dest_path)
    
    # Notify the user
    print("Images sorted into class folders, dataset is created. Enjoy!\n")
    


### ---------------------------------------------------------------------------
### Zipping / unzipping of dataset and subfolders

## Zip the subfolders of a ImageFolder-compatible dataset
# Used in dataset_BR and dataset_LN to push to gin
def zip_subfolders(parent_folder):
    """
    Zip the subfolders of a ImageFolder dataset, to make uploading on GIN and 
    general handling much easier
    
    Parameters
    ----------
    parent_folder (str): the full path of the folder containing the 
                         subfolders to zip
    
    Outputs
    -------
    None, dataset will contain zip files in place of folders

    """
    
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
            
            # Once it's done, remove the original folder
            os.rmdir(subfolder_path)


## Unzip the subfolders of a ImageFolder-compatible dataset
# used in datasets to extract data from the zip files downloaded from gin
def unzip_files(parent_folder):
    """
    Unzip the subfolders of a ImageFolder dataset, to read them during training
    
    Parameters
    ----------
    parent_folder (str): the full path of the folder containing the files to unzip
    
    Outputs
    -------
    None, dataset will contain folders in place of zip files

    """
    
    # CHeck that the folder exists
    if not os.path.exists(parent_folder):
        print(f"The folder {parent_folder} does not exist.")
        return
    
    # browse through each zip file
    for file_name in os.listdir(parent_folder):
        if file_name.endswith(".zip"):
            
            zip_path = os.path.join(parent_folder, file_name)
            
            # Create filename excluding the .zip extension
            extract_folder = os.path.join(parent_folder, file_name[:-4])
            
            # Unzip the file
            with zipfile.ZipFile(zip_path, 'r') as zipf:
                zipf.extractall(extract_folder)
            print(f"Extracted: {zip_path} to {extract_folder}")
            
            
# Zip the entire folder
# used for test sets
def zip_folder(parent_folder):
    """
    Zips the entire folder. Used for test datasets
    
    Parameters
    ----------
    parent_folder (str): the full path of the folder to zip
    
    Outputs
    -------
    None, dataset will be a zip file

    """
    
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



### ---------------------------------------------------------------------------
### Images handling

## Resize word stimuli
def resize_stimuli(target_size, folder, move): 
    """
    Resize images of words to a target size, and if requested place them into
    pre-defined folders. Used when moving images from Matlab scripts to python.
    
    Parameters
    ----------
    target_size (int): the full path of the folder to zip
    
    folder (str): 
        
    move (boolean): whether to move the files or not
    
    Outputs
    -------
    None, files will be overwritten and maybe moved

    """

    # Get the paths of the images to work on
    source_path = folder
    word_paths = glob.glob(os.path.join(source_path, '*.png'))
    
    # If 'move' is requested, store everything in 'matlab_enlarged' and move later
    # Otherwise use the same folder and overwrite images
    if move: dest_path = "../../input/words/matlab_enlarged"
    else:    dest_path = folder
        
    # Define the target dimensions
    background_size = (target_size, target_size)
    
    # Create a white background for resizing
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
        os.makedirs(braille_path, exist_ok =True)
        os.makedirs(line_path, exist_ok = True)
        os.makedirs(latin_path, exist_ok = True)
        
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















