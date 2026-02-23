#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 28 15:01:15 2025

Visual Braille Silico - support functions to extract layer activations 

@author: Filippo Cerpelloni
"""

### ---------------------------------------------------------------------------
### Imports for all the functions 

import sys
sys.path.append('../lib/CORnet')
sys.path.append('../')

# Import custom functions to store paths, extraction of activations
from src.vbs_functions import * 

import os
import glob
import zipfile
import pickle
import torch

import torch.nn as nn
import numpy as np
import pandas as pd

from torchvision import transforms, models
from torchvision.io import read_image
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from natsort import natsorted
from scipy.spatial.distance import pdist, squareform
from cornet import cornet_z
from collections import defaultdict



### ---------------------------------------------------------------------------
### Main extraction functions

# Extract activations for each layer, and for each stimulus
def extract_activations(opt, model_name, subID, training, test, epoch, method):
    """
    Present images (from test) to the network (from model_name, subID, training)
    and extract the relative activations at different processing stages.
    
    Parameters
    ----------
    opt (dict): output of vbs_option() containing the paths of the IODA folder 
    
    model_name (str): the model refernce (whether 'alexnet' or 'cornet') to 
                         load the correct activations and to save the reuslts properly
                         
    subID (int): the number reference of the subject to analyse
    
    training (str): reference of the dataset on which the network was trained, 
                    whether Latin alone (LT), Latin+Braille (LTBR), +Line (LTLN)
                    
    test (str): reference of the human experiment to replicate. Can be 'VBE' for 
                fMRI experiment or VBT for the behavioural experiment
                
    epoch (str): reference of which epoch, which level of training, to consider
    
    method (str): what to use to compute the distances, usually either 
                  'euclidian' or 'correlation'
                         
    Outputs
    -------
    None, saves activations in outputs/results/activations
    """

    # Set all the important specifics of where to find data:
    # - specific test set to load 
    dataset_path, dataset_spec = get_paths(opt, test)
        
    # TODO: Check if the test folder is zipped. If so, unzp it

    # Initialize storage dictionaries
    flat_dict = {}  # Flattened activations
    stim_dict = {}  # Averages across variations for each stimulus
    dist_dict = {}  # Distance between stimuli based on Janini et al. 2022

    # Process a single subject/iteration (otherwise kernel breaks)
    if not subID in opt['subjects']:
        print('Error, we do not have that many iterations of the network.')
        return 0
    else:
        s = subID

    # Inform the user
    print(f'Extracting activations from sub-{s}\n')

    # Load the dataset
    # Batch size is fixed at 55, to process all the size variations of a stimulus together
    # resulting in 5 batches for each stimulus (e.g. one real word in one script)
    dataset, data_loader = load_dataset(dataset_path, dataset_spec)

    # Choose the model
    model = get_weighted_model(opt, model_name, s, training, test, epoch)

    # Load the labels for the individual images and for the classes of stimuli
    image_labels = pd.read_csv(f'../../inputs/words/{dataset_spec}_stimuli.csv')

    # Initialize subject-specific dictionaries
    flat_dict[f'sub-{s}'] = {}
    stim_dict[f'sub-{s}'] = {}
    dist_dict[f'sub-{s}'] = {}

    ## Feed images to the network and extract activations
    for b, batch in enumerate(data_loader):

        # From a list of all the layers of a network, pick the relevant ones
        # for the extraction of activations (e.g. ReLU stages in AlexNet)
        layer_names = name_layers(model)
        layer_names = pick_layers(model_name, layer_names)

        # Extract activations at any layer for the stimuli in the batch
        layer_activations = get_layer_activations(model, layer_names, batch)

        # Store information in a series of dictionaries, to ease stats
        # Common structure to all the dicts is:
        # dict[layer][subject][stimulus][size][x position][y position] = activation for image in layer
        flat_dict = store_layer_activations(layer_activations, layer_names,
                                            image_labels, b, s, flat_dict)

    ## Average activations across all variations
    print('Computing averages and distances for each stimulus ...')

    # Compute averages for each stimulus
    stim_dict[f'sub-{s}'] = average_activations(flat_dict[f'sub-{s}'])

    ## Calculate euclidian distances between letters and their variations
    # Follow janini et al. (2022)'s method:
    # - compare each variation of word A with average of word B
    #   (e.g. BR_FS_1_S*_X*_Y* and BR_NW_1)
    # - average distances to obtain cell of RDM
    # - average across the diagonal (a-b and b-a)
    dist_dict[f'sub-{s}']['LT'], dist_dict[f'sub-{s}']['BR'] = compute_distances_within_scripts(flat_dict[f'sub-{s}'], 
                                                                                                stim_dict[f'sub-{s}'],
                                                                                                method)
    ## Save subject-specifc activations and matrices
    print('Saving subject activations ...\n')

    save_extractions(opt,
                     flat_dict[f'sub-{s}'],
                     stim_dict[f'sub-{s}'],
                     dist_dict[f'sub-{s}'],
                     model_name, s, training, test, epoch, method)

    print('Extraction completed.\n\n')


# Workaround because cornet models crash the kernel
# TODO implent it for alexnet too, seems faster
def cornet_activations(opt, model_name, s, training, test, epoch):
    """
    Same as extract activations, but split the dataset into BR and LT, present
    them separately and join the dictionaries later
    
    Parameters
    ----------
    opt (dict): output of vbs_option() containing the paths of the IODA folder 
    
    model_name (str): the model refernce (whether 'alexnet' or 'cornet') to 
                         load the correct activations and to save the reuslts properly
                         
    subID (int): the number reference of the subject to analyse
    
    training (str): reference of the dataset on which the network was trained, 
                    whether Latin alone (LT), Latin+Braille (LTBR), +Line (LTLN)
                    
    test (str): reference of the human experiment to replicate. Can be 'VBE' for 
                fMRI experiment or VBT for the behavioural experiment
                
    epoch (str): reference of which epoch, which level of training, to consider
                         
    Outputs
    -------
    None, saves activations in outputs/results/activations
    """
    
    ## LATIN
    print(f'Loading Latin subdataset\n')
    
    dataset_path = os.path.join(opt['dir']['datasets'], 'vbe_latin')
    dataset_spec = 'test_vbe_latin'
        
    lt_flat = {}  
    lt_stim = {} 
    br_flat = {}  
    br_stim = {} 
    
    dist_dict = {}

    print(f'Extracting activations from subject\n')

    dataset, data_loader = load_dataset(dataset_path, dataset_spec)
    model = get_weighted_model(opt, model_name, s, training, test, epoch)
    image_labels = pd.read_csv(f'../../inputs/words/{dataset_spec}_stimuli.csv')

    lt_flat[f'sub-{s}'] = {}
    lt_stim[f'sub-{s}'] = {}

    for b, batch in enumerate(data_loader):

        layer_names = name_layers(model)
        layer_names = pick_layers(model_name, layer_names)
        layer_activations = get_layer_activations(model, layer_names, batch)
        flat_dict = store_layer_activations(layer_activations, layer_names,
                                            image_labels, b, s, lt_flat)

    ## BRAILLE
    print(f'Loading Braille subdataset\n')
    
    dataset_path = os.path.join(opt['dir']['datasets'], 'vbe_braille')
    dataset_spec = 'test_vbe_braille'
    dataset, data_loader = load_dataset(dataset_path, dataset_spec)
    image_labels = pd.read_csv(f'../../inputs/words/{dataset_spec}_stimuli.csv')
    
    br_flat[f'sub-{s}'] = {}
    br_stim[f'sub-{s}'] = {}

    for b, batch in enumerate(data_loader):

        layer_names = name_layers(model)
        layer_names = pick_layers(model_name, layer_names)
        layer_activations = get_layer_activations(model, layer_names, batch)
        flat_dict = store_layer_activations(layer_activations, layer_names,
                                            image_labels, b, s, br_flat)
    
    print('Computing averages and distances for each stimulus ...')
    
    dist_dict[f'sub-{s}'] = {}
    
    # TODO join flat and stim
    flat_dict = merge_dictionaries(lt_flat, br_flat)
    stim_dict = merge_dictionaries(lt_stim, br_stim)

    stim_dict[f'sub-{s}'] = average_activations(flat_dict[f'sub-{s}'])

    dist_dict[f'sub-{s}']['LT'], dist_dict[f'sub-{s}']['BR'] = compute_distances_within_scripts(flat_dict[f'sub-{s}'], 
                                                                                                stim_dict[f'sub-{s}'])
    print('Saving subject activations ...\n')

    save_extractions(opt,
                     flat_dict[f'sub-{s}'],
                     stim_dict[f'sub-{s}'],
                     dist_dict[f'sub-{s}'],
                     model_name, s, training, test, epoch)

    print('Extraction completed.\n\n')
    
        
# Extract performance of the network for each layer and each stimulus 
def extract_performances(opt, model_name, subID, training, test, epoch):
    """
    Present images (from test) to the network (from model_name, subID, training)
    and extract the relative performance at classifying those images 
    (result of the last layer)
    
    Parameters
    ----------
    opt (dict): output of vbs_option() containing the paths of the IODA folder 
    
    model_name (str): the model refernce (whether 'alexnet' or 'cornet') to 
                         load the correct activations and to save the reuslts properly
                         
    subID (int): the number reference of the subject to analyse
    
    training (str): reference of the dataset on which the network was trained, 
                    whether Latin alone (LT), Latin+Braille (LTBR), +Line (LTLN)
                    
    test (str): reference of the human experiment to replicate. Can be 'VBE' for 
                fMRI experiment or VBT for the behavioural experiment
                
    epoch (str): reference of which epoch, which level of training, to consider
                         
    Outputs
    -------
    None, saves performances in outputs/results/classifications
    
    """
    
    # TODO make more verbose, specify epochs and scripts
   
    # Process a single subject/iteration (otherwise kernel breaks)
    if not subID in opt['subjects']:
        print('Error, we do not have that many iterations of the network.')
        return 0
    else:
        s = subID

    # Inform the user
    print(f'Extracting performance of network from sub-{s}\n')
    print(f'Loading the model ...\n')

    # Choose the model
    model = get_weighted_model(opt, model_name, s, training, test, epoch)
    
    if test == 'VBT':
        if training == 'LTBR':   scripts = ['LT','BR']
        elif training == 'LTLN': scripts = ['LT','LN']
        else:                    scripts = ['LT']
    
    # Initialize storage dictionaries
    flat_dict = {}  # Flattened activations
    stim_dict = {}  # Averages across variations for each stimulus
    idx_dict = {}  # Distance between stimuli based on Janini et al. 2022
    
    # Initialize subject-specific dictionaries
    flat_dict[f'sub-{s}'] = {}
    stim_dict[f'sub-{s}'] = {}
    idx_dict[f'sub-{s}'] = {}
        
    ## Test scripts independently, doesn't matter the responses will be merged / 
    # compared later
    for scr in scripts: 
    
        # Set all the important specifics of where to find data, like test set to load 
        dataset_path, dataset_spec = get_paths(opt, test)
        
        # Adjust to the script
        dataset_path = dataset_path + f'_{scr}'
        dataset_spec = dataset_spec + f'_{scr}'
            
        # Load the dataset
        print(f'Loading the dataset ...\n')
        
        # Batch size is fixed at 55, to process all the size variations of a stimulus together
        # resulting in 5 batches for each stimulus (e.g. one real word in one script)
        dataset, data_loader = load_dataset(dataset_path, dataset_spec)
    
        # Load the labels for the individual images and for the classes of stimuli
        image_labels = pd.read_csv(f'../../inputs/words/{dataset_spec}_stimuli.csv')
        
        class_labels = pd.read_csv('../../inputs/words/test_vbt_wordlist.csv', header = None)
        
        # Inform the user
        print(f'Testing the network ...\n')
        
        # Cache to save the performances for a full word
        word_idxs = []
        word_id = 0
    
        ## Feed images to the network and extract activations
        for b, batch in enumerate(data_loader):
    
            # Extract performances: let the network process the images and get the
            # final classification for each stimulus
            performances = model(batch)
            performances_idx = torch.argmax(performances, dim = 1).numpy()
            
            # Add to a cahce of the other variations of the same word
            word_idxs.extend(performances_idx)
            
            # Store information in a series of dictionaries, to ease stats
            # Common structure to all the dicts is:
            flat_dict = store_performances(performances, image_labels, b, s, scr, flat_dict)
            idx_dict = store_indices(performances_idx, image_labels, b, s, scr, idx_dict)
            
        # Compute number of correct classifications    
        print('Computing correct classifications ...\n')
                
        stim_dict = correct_performances(stim_dict, class_labels, s, scr, idx_dict)

    ## Save subject-specifc activations and matrices
    print('Saving subject performances ...\n')
    
    save_single_extraction(opt, flat_dict[f'sub-{s}'], 
                           model_name, s, training, 'VBT', epoch,
                           'classifications',
                           'flat-classifications')
    
    save_single_extraction(opt, idx_dict[f'sub-{s}'], 
                           model_name, s, training, 'VBT', epoch,
                           'classifications',
                           'flat-responses')
    
    save_single_extraction(opt, stim_dict[f'sub-{s}'], 
                           model_name, s, training, 'VBT', epoch,
                           'classifications',
                           'corrected-responses')

    print('Extraction completed.\n\n')


## Wrapper to extract performances from one network at given timepoints (epochs)
def extract_timepoint_performances(opt, model_name, subID, training, test, epochs):
    """
    Given all the information to extract performances from one epoch, and a list
    of epochs, call the extraction for all the pochs / timepoints we are interested in.
    Mostly a wrapper to make act_main cleaner
    
    Parameters
    ----------
    opt (dict): output of vbs_option() containing the paths of the IODA folder 
    
    model_name (str): the model refernce (whether 'alexnet' or 'cornet') to 
                         load the correct activations and to save the reuslts properly
                         
    subID (int): the number reference of the subject to analyse
    
    training (str): reference of the dataset on which the network was trained, 
                    whether Latin alone (LT), Latin+Braille (LTBR), +Line (LTLN)
                    
    test (str): reference of the human experiment to replicate. Can be 'VBE' for 
                fMRI experiment or VBT for the behavioural experiment
                
    epochs (list): reference of which epochs to call for extraction, 
                   which timepoints of training to consider 
                         
    Outputs
    -------
    None, saves performances in outputs/results/classifications
    
    """
    
    # Simply call extract_performances multiple times
    for epoch in epochs:
        extract_performances(opt, model_name, subID, training, test, epoch)



### ---------------------------------------------------------------------------
### Miscellaneous

def merge_dictionaries(latin, braille): 
    """
    Merge two dictionaries with the activations of different parts of the test set
    (latin stimuli and braille stimui)
    
    Parameters
    ----------
    latin (dict): the activations for latin script stimuli 
                         
    braille (dict): the activations for braille script stimuli 
                         
    Outputs
    -------
    joint (dict): the activations for all stimuli 
    """
    
    # Create a merged dictionary
    joint = defaultdict(lambda: defaultdict(list))
    
    # Merge latin and init sub-dictionaires
    for sub in latin.keys():
        
        joint[f'{sub}'] = {}
        
        for layer in latin[f'{sub}'].keys():
            
            joint[f'{sub}'][f'{layer}'] = {}
            
            for stim in latin[f'{sub}'][f'{layer}'].keys():
                joint[f'{sub}'][f'{layer}'][f'{stim}'] = latin[f'{sub}'][f'{layer}'][f'{stim}']

    
    # Merge braille - sub-dictionaires already created
    for sub in braille.keys():
        for layer in braille[f'{sub}'].keys():
            for stim in braille[f'{sub}'][f'{layer}'].keys():
                joint[f'{sub}'][f'{layer}'][f'{stim}'] = braille[f'{sub}'][f'{layer}'][f'{stim}']
    
    return joint
    
      
# Get basic paths of the dataset
def get_paths(opt, test):
    """
    From the test experiment, get which notation to find in the paths and which
    dataset to extract.
    
    Parameters
    ----------
    opt (dict): output of vbs_option() containing the paths of the IODA folder 
                    
    test (str): reference of the human experiment to replicate. Can be 'VBE' for 
                fMRI experiment or VBT for the behavioural experiment
                         
    Outputs
    -------
    dataset_path (str): the path of the folder where the test images are stored
    
    dataset_spec (str): notation of the experiment, to load image labels 
    """

    # Test-sepcific directory: VBE or VBT
    if test == 'VBE':   dataset_spec = 'test_vbe'
    elif test == 'VBT': dataset_spec = 'test_vbt'
    else:               dataset_spec = test

    # Full dataset path: quick workaround f'{} error
    dataset_dir = opt['dir']['datasets']
    dataset_path = f"{dataset_dir}/{dataset_spec}"

    return dataset_path, dataset_spec


# Compute RDMs between stimuli 
def make_rdm(layer, activations, averages, stim_names, method):
    """
    For a given layer, use image activations and stimuli averages to compute the
    distance between two stimuli, using the methods from janini et al. 2022
    
    Parameters
    ----------
    layer (str): the name of the layer in which to compute the RDM
    
    activations (dict): the activations for single images
    
    averages (dict): the activations for single stimuli, averaged across the images
                    
    stim_names (array): the names of the stimuli, to be used as key in the output matrix
    
    method (str): what to use to compute the distances, usually either 
                  'euclidian' or 'correlation'
                         
    Outputs
    -------
    dataset_path (str): the path of the folder where the test images are stored
    
    dataset_spec (str): notation of the experiment, to load image labels 
    """

    # Get the number of stimuli
    stim_nb = len(stim_names)

    # Initiate a matrix of len(nb of stimuli) to store the RDM values
    matrix = np.full((stim_nb, stim_nb), np.nan)

    # Browse through stim A
    for r, stim_a in enumerate(stim_names):
        
        # Iniitialize array with the different variations
        stim_a_variations = []

        stimuli = activations[f'{layer}']
        # Extract activation for all the variants in a np list (like for assignment in storing)
        var = 0

        # Go into each variation and extract the activation array
        for size in stimuli[f'{stim_a}'].keys():
            for x in stimuli[f'{stim_a}'][f'{size}'].keys():
                for y in stimuli[f'{stim_a}'][f'{size}'][f'{x}'].keys():

                    # Append the values to a numpy list, to ease averaging
                    # stim_a_variations.append(np.array(y))
                    stim_a_variations.append(np.array(stimuli[f'{stim_a}'][f'{size}'][f'{x}'][f'{y}']))
                    var = var +1

        # Browse through stim B
        for c, stim_b in enumerate(stim_names):

            # Extract average activation
            stim_b_average = averages[f'{layer}'][f'{stim_b}']

            # Concatenate stimulus B average to stimulus A variations
            ab_concatenated = np.vstack([stim_b_average, stim_a_variations])

            # Compute distances between all the elements in the list,
            # then keep only the ones referring to stimuls B average and stimulus A variations
            ab_distances = pdist(ab_concatenated, method)[:var]

            # Average the distances
            ab_average = np.average(ab_distances)

            # assign to matrix
            matrix[r,c] = ab_average

    # Average distances across the diagonal
    for r, row in enumerate(stim_names):
        for c, col in enumerate(stim_names):
            average = (matrix[r,c] + matrix[c,r])/2
            matrix[r,c] = average
            matrix[c,r] = average

    return matrix


### ---------------------------------------------------------------------------
### Network functions

## Pick the right model and the right weights
def get_weighted_model(opt, model_name, s, tr, te, ep):
    """
    From the inputs given, return the model selected with the weights applied 
    
    Parameters
    ----------
    opt (dict): output of vbs_option() containing the paths of the IODA folder 
        
    model_name (str): the refernece of the model to load
                    
    s (int): the subject number ID
    
    tr (str): the training preformed by the network
    
    te (str): reference of the human experiment to replicate. Can be 'VBE' for 
              fMRI experiment or VBT for the behavioural experiment
              
    ep (str): reference of which training epoch to consider fro classification 
              or activations
                         
    Outputs
    -------
    model: the model selected with the weights selected 
    """

    # AlexNet
    if model_name == 'alexnet':
        print('Loading AlexNet model ...\n')

        # Get model
        model = models.alexnet()
        model = nn.DataParallel(model)
        
        # Define epoch from user input
        if ep == 'last': ep = 20
        else: ep = int(ep)
        
        # Quick fix for unknown problem: f'{} does not accept opt dictionary
        # assign it to avoid problems.
        weights = opt['dir']['weights']

        # Apply expertise weights
        saved_weights_path = f'{weights}/literate/{tr}/model-{model_name}_sub-{s}_training-{tr}_epoch-{ep}.pth'
        saved_weights = torch.load(saved_weights_path, map_location = torch.device('cpu'))
        model.load_state_dict(saved_weights)

        # Evaluate the model, just to check
        model.eval()

    # CORnet
    elif model_name == 'cornet':
        print('Loading CORnet Z model ...\n')

        # Get model
        model = cornet_z()
        
        # Define epoch from user input
        if ep == 'last': ep = 15
        else: ep = int(ep)
        
        # Quick fix for unknown problem: f'{} does not accept opt dictionary
        # assign it to avoid problems.
        weights = opt['dir']['weights']

        # Apply weights of expert network
        saved_weights_path = f'{weights}/literate/{tr}/model-{model_name}_sub-{s}_training-{tr}_epoch-{ep}.pth'
        saved_weights = torch.load(saved_weights_path, map_location = torch.device('cpu'))
        model.load_state_dict(saved_weights)

        # Evaluate the model, just to check
        model.eval()

    else:

        # Close the function - very rudimental but works
        print('\nERROR: no compatible model specified, stopping here.')
        return 0

    return model


### ---------------------------------------------------------------------------
### Dataset functions

## Define a custom class for the test datasets
class TestDataset(Dataset):

    def __init__(self, annotations, images, transform = None):
        self.image_labels = pd.read_csv(annotations)
        self.images = images
        self.transform = transform

    def __len__(self):
        return len(self.image_labels)

    def __getitem__(self, index):
        image = self.images[index]
        if self.transform:
            image = self.transform(image)

        return image

    def __getlabel__(self, index):
        label = self.image_labels.iloc[index,0]

        return label


## Import a dataset and the wordlist (always the same) from a given path
def load_dataset(path, dataset_spec):
    """
    Load a dataset from a path 
    
    Parameters
    ----------
    path (str): the fullpath of the dataset to load of vbs_option() containing the paths of the IODA folder 
        
    dataset_spec (str): notation of the experiment, to load image labels 
                    
    Outputs
    -------
    dataset (TestDataset): the dataset 
    
    dataset_loader (DataLoader): the dataset loaded and divided into batches
    """

    # From the path, get the list of stimuli to load
    stim_paths = glob.glob(os.path.join(path, '*.png'))

    # Order stimuli paths and labels in the same manner,
    # to make sure that the image we load corresponds to the label we give it
    stim_paths = order_stimuli(stim_paths, dataset_spec)

    # Load the images in memory
    images = [Image.open(stim_path).convert('RGB') for stim_path in stim_paths]

    # Define transformations that will be applied to images
    # - resize to 224x224 (alexnet input)
    # - convert to tensor
    # - normalize with ImageNet stats
    transform = transforms.Compose([transforms.Resize((224, 224)),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean = [0.485, 0.456, 0.406],
                                                         std = [0.229, 0.224, 0.225]) ])

    # Create the Dataset and DataLoader
    dataset = TestDataset(annotations = f'../../inputs/words/{dataset_spec}_stimuli.csv', 
                          images = images, transform = transform)

    # Use DataLoader to create batches
    dataset_loader = DataLoader(dataset, batch_size = 55, shuffle = False)

    return dataset, dataset_loader


## Order a list of stimuli in a natrual order
def order_stimuli(in_paths, dataset_spec):
    """
    Order a list of stimuli in a natural alphabetical order and save a .csv file
    with the list, to be used as labels in the dataset loader 
    
    Parameters
    ----------
    in_paths (array): list of strings to be ordered
    
    dataset_spec (str): notation of the experiment, to load image labels 
    
    Outputs
    -------
    out_paths (array): list of stimuli in order, also saved as .csv
    """

    # Sort the paths in a natural order: alphabetical considering numbers
    out_paths = natsorted(in_paths)

    # If the test is VBE, there is a more strict naming. 
    # Implement a custom modification to have our own order:
    # - LT before BR;
    # - RW, PW, NW, and then FS
    if dataset_spec.startswith('test_vbe'): 
        num_parts = 8
        chunk_size = len(out_paths) // num_parts
        chunks = [out_paths[i * chunk_size:(i + 1) * chunk_size] for i in range(num_parts)]
        out_paths = sum(chunks[::-1], [])
        

    # Create set to store
    out_set = set()

    for s in out_paths:
        parts = s.split('/')
        stim_info = parts[-1].split('.')[0]
        out_set.add(f"{stim_info}")

    out_list = natsorted(list(out_set))

    # Rearrange again. Why loop does not maintain order?
    if dataset_spec.startswith('test_vbe'):     
        chunk_size = len(out_list) // num_parts
        chunks = [out_list[i * chunk_size:(i + 1) * chunk_size] for i in range(num_parts)]
        out_list = sum(chunks[::-1], [])

    stimuli = pd.DataFrame(out_list, columns = ['Stimulus'])
    stimuli.to_csv(f'../../inputs/words/{dataset_spec}_stimuli.csv', index = False)

    return out_paths



### ---------------------------------------------------------------------------
### Layer functions

# Get the names of the layers of a network, from Andrea Costantino
def name_layers(model):
    """
    Extract the names of all last-level layers in a PyTorch neural network.
    
    Parameters
    ----------
    model (torch.nn.Module): the model from which to extract the layer names
    
    Outputs
    -------
    layers (list): a list containing the names of all last-level layers in the model.
    """
    
    layers = []
    for name, module in model.named_modules():

        # Check if the module is a leaf module (no children)
        if not list(module.children()):

            # Exclude the top-level module (the model itself) which is always a leaf
            if name: layers.append(name)

    return layers


## Get important layers for activations, based on the network's architecture
def pick_layers(model_name, layer_names):
    """
    From a given network, and a list of layers, make a selection of the 
    important ones.
    
    Parameters
    ----------
    model (torch.nn.Module): the model from which to extract the layer names
    
    Outputs
    -------
    layers (list): a list containing the names of all last-level layers in the model.
    """

    layers = []

    if model_name == 'alexnet':

        # Pick all the ReLU stage
        layers = ['module.features.1',
                  'module.features.4',
                  'module.features.7',
                  'module.features.9',
                  'module.features.11',
                  'module.classifier.2',
                  'module.classifier.5',
                  'module.classifier.6']

    elif model_name == 'cornet':

        # Pick layers as previous studies did 
        layers = ['module.V1.output',
                  'module.V2.output',
                  'module.V4.output',
                  'module.IT.output',
                  'module.decoder.flatten',
                  'module.decoder.output']

    else:
        print('Cannot decide which layers to pick, do not know this network.')
        return 0

    return layers


### ---------------------------------------------------------------------------
### Activation functions

## Extract activations for a layer
def get_layer_activations(model, layer_names, batch):
    """
    Get the activations of specified layers in response to input data and 
    concatenating the results across batches. The activations are detached from 
    the computation graph and moved to the CPU before storage.

    Parameters
    ----------
    model (torch.nn.Module): the model from which to extract activations
    
    layer_names (list): from which layer to extract the activations
    
    batch (torch.Tensor): batch of images to feed through the model

    Outputs
    ----------
    activations (dict): the dictionary with the activations. Layer names are keys 
                         and activations are values 
    """
    
    # Ensure layer_names is a list
    if not isinstance(layer_names, list):

        layer_names = [layer_names]

    activations = {name: [] for name in layer_names}
    hooks = []

    def get_activation(name):
        def hook(model, input, output):

            # Detach activations from computation graph and move to CPU
            activations[name].append(output.detach().cpu())

        return hook

    # Register hooks for each specified layer
    for name in layer_names:
        layer = dict([*model.named_modules()])[name]
        hook = layer.register_forward_hook(get_activation(name))
        hooks.append(hook)

    model(batch)

    # Concatenate the activations for each layer across all batches
    for name in activations:
        activations[name] = torch.cat(activations[name], dim = 0)

    # Remove hooks after completion
    for hook in hooks:
        hook.remove()

    return activations


## Store the activations in nested dictionaries
def store_layer_activations(activations, layers, labels, b, s, flat):
    """
    Store the activations in a orderly manner that considers the layer, the subject,
    the type of activation to save, the stimulus

    Parameters
    ----------
    activations: the activations in a layer in response of the images
    
    layers (list): the list of layer names
    
    labels (list): the list of image names
    
    b (int): the number of batch processed, to compute whether to create 
             new sub-dictionaries
    
    s (int): the subject number
        
    flat (dict): the dictionary storing all the flat activations

    Outputs
    ----------
    raw (dict): the dictionary storing all the raw activations, now updated
                with the activations from this batch
    
    flat (dict): the dictionary storing all the flat activations now updated
                 with the activations from this batch
    """

    # Get the label of the stimulus presented, with size variation
    example = labels['Stimulus'][b*55]
    parts = example.split('_')
    scr = parts[0]
    cat = parts[1]
    wrd = parts[2]
    size = parts[3]
    stim = f'{scr}_{cat}_{wrd}'

    # If it's the first batch of this stimulus, initialize the dictionaries
    if b == 0:  init_layer = True
    else: init_layer = False

    if size == 'S1': init_size = True
    else: init_size = False

    # For each layer / stage
    for i, layer in enumerate(layers, start = 1):

        # Get the activations
        stage = activations[layers[i-1]]

        # Initialize stage dict and stim dict, if needed
        if init_layer:
            flat[f'sub-{s}'][f'{layer}'] = {}

        if init_size:
            flat[f'sub-{s}'][f'{layer}'][f'{stim}'] = {}

        # Size will always be a new entry
        flat[f'sub-{s}'][f'{layer}'][f'{stim}'][f'{size}'] = {}

        # Assign the 55 rows of each batch to the corresponding X and Y positions
        # Define the labels to assign
        x_labels = [f'X{i}' for i in range(1, 12)]  # 1-11
        y_labels = [f'Y{i}' for i in range(1, 6)]   # 1-5

        # Flatten the activations to a single vector for each image
        # e.g. layer 5: 256x13x13 -> 256x1, sum of the 13x13 activation maps across the features
        if stage.dim() > 2:
            stage = torch.sum(stage, (3,2))

        # Move flattened tensor to numpy (again)
        stage = stage.numpy()

        # Save 'flattened' activations
        for idx, (x, y) in enumerate([(x, y) for x in x_labels for y in y_labels]):
            flat[f'sub-{s}'][f'{layer}'][f'{stim}'][f'{size}'].setdefault(x, {})[y] = stage[idx]

    return flat


## Organize activations in a script, layer, subject hierarchy to ease the stats 
def regorganize_activations(in_dict):
    """
    Rearrange dictionaries from sub-script-layer to script-layer-sub nesting

    Parameters
    ----------
    in_dict (dict): the dictionary to rearrange

    Outputs
    ----------
    out_dict (dict): the dictionary rearranged
    """
    
    # Reorganize the distances from sub-script-layer to script-layer-sub
    reorganized = defaultdict(lambda: defaultdict(dict))
    
    for sub, scripts in in_dict.items():
        for script, layers in scripts.items():
            for layer, value in layers.items():
                reorganized[script][layer][sub] = value
                
    out_dict = reorganized
    
    return out_dict


## Compute averages across stimuli variations to obtain stimulus-identity activation
def average_activations(flat_dict):
    """
    Compute the average activation for one stimulus across all the variations 
    in size, position in which it is presented to the network
    
    Parameters
    ----------
    flat_dict (dict): the dictionary with arrays of activations for each 
                      stimulus variation

    Outputs
    ----------
    out_dict (dict): the dictionary containing the averages, one array for one 
                     stimulus across all it's variations
    """

    # Copy the structure of the flat dictionary
    # Scroll through every stimulus and get all the values out and averaged
    out_dict = {}

    for layer, layer_dict in flat_dict.items():

        # Initialize layer dictionary
        out_dict[layer] = {}

        for stim, stim_dict in layer_dict.items():
            activations = []

            # go into each variation and extract the activation array
            for size in stim_dict.values():
                for x in size.values():
                    for y in x.values():

                        # Append the values to a numpy list, to ease averaging
                        activations.append(np.array(y))

            out_dict[layer][stim] = np.mean(activations, axis = 0)

    return out_dict


## Compute distances between the stimuli and within each network
def compute_distances_within_scripts(activations, averages, method):
    """
    Compute the distance between stimuli of the same script
    
    Parameters
    ----------
    activations (dict): the activations for single images
    
    averages (dict): the activations for single stimuli, averaged across the images
    
    method (str): what to use to compute the distances, usually either 
                  'euclidian' or 'correlation'

    Outputs
    ----------
    distances_lt (dict): the dictionary containing the distances between 
                         stimuli in the latin script
    
    distances_br (dict): the dictionary containing the distances between 
                         stimuli in the braille script
    """

    # Initiate a dictionary to contain the distances between elements in a given layer
    distances_lt = {}
    distances_br = {}

    # Extract from the dictionary the number and name of each layer
    layer_names = list(averages.keys())

    # Latin alphabet - get the stimuli in each layer
    stim_names = list(averages[f'{layer_names[0]}'].keys())
    lt_names = [item for item in stim_names if item.startswith('LT_')]
    
    # Order names, just in case
    lt_names = natsorted(lt_names)

    # Loop through layers of the datasets
    for i, layer in enumerate(layer_names):
        distances_lt[f'{layer}'] = make_rdm(layer, activations, averages, lt_names, method)

    # Braille alphabet - get the stimuli in each layer
    br_names = [item for item in stim_names if item.startswith('BR_')]
    
    # Order names, just in case
    br_names = natsorted(br_names)

    # Loop through layers of the datasets
    for i, layer in enumerate(layer_names):
        distances_br[f'{layer}'] = make_rdm(layer, activations, averages, br_names, method)

    return distances_lt, distances_br



## Compute distance from activations across subjects
def compute_distance_subjects(opt, model_name, training, test, epoch, method):
    """
    Compute the distance between stimuli of the same script, loading the flat 
    and stim dictionaries autonomously 
    
    Parameters
    ----------
    opt (dict): output of vbs_option() containing the paths of the IODA folder 
    
    model_name (str): the model refernce (whether 'alexnet' or 'cornet') to 
                         load the correct activations and to save the reuslts properly
                        
    training (str): reference of the dataset on which the network was trained, 
                    whether Latin alone (LT), Latin+Braille (LTBR), +Line (LTLN)
                    
    test (str): reference of the human experiment to replicate. Can be 'VBE' for 
                fMRI experiment or VBT for the behavioural experiment
                
    epoch (str): reference of which epoch, which level of training, to consider
    
    method (str): what to use to compute the distances, usually either 
                  'euclidian' or 'correlation'
                         
    Outputs
    -------
    None, saves activations in outputs/results/activations
    """
    
    # Initialize storage dictionaries
    dist = {}  # Distance between stimuli based on Janini et al. 2022

    # Browse each subject
    for s, subject in enumerate(opt['subjects']):
        
        # Load flat (activations) and stim (averages) dictionaries
        flatname = os.path.join(opt['dir']['results'], 'activations', 
                                f'model-{model_name}_sub-{s}_training-{training}_test-{test}_epoch-last_data-flat-activations.pkl')
        stimname = os.path.join(opt['dir']['results'], 'activations', 
                                f'model-{model_name}_sub-{s}_training-{training}_test-{test}_epoch-last_data-stimulus-average-activations.pkl')
        
        flat = load_extraction(flatname)
        stim = load_extraction(stimname)
        
        print(f'Computing distances for model {model_name} and sub {s} ...\n')
        
        # Call distance function
        dist['LT'], dist['BR'] = compute_distances_within_scripts(flat, stim, method)
    
        # Save it
        save_single_extraction(opt, dist, model_name, s, training, test, epoch, 
                               'distances', 
                               f'distances_method-{method}')


### ---------------------------------------------------------------------------
### Classification functions

## Store the activations in nested dictionaries
def store_performances(performances, labels, b, s, scr, flat):
    """
    Store the performances in a orderly manner that considers subject and stimulus

    Parameters
    ----------
    performances (torch.nn.Module): the classification of images in a batch
    
    labels (list): the list of image names
    
    b (int): the number of batch processed, to create new sub-dictionaries
    
    s (int): the subject number
    
    flat (dict): the dictionary storing all the flat activations

    Outputs
    ----------
    flat (dict): the dictionary storing all the flat activations now updated
                 with the activations from this batch
    """

    # Get the label of the stimulus presented, with size variation
    example = labels['Stimulus'][b*55]
    parts = example.split('_')
    wrd = parts[1]
    size = parts[3]
            
    # Compose stimulus name
    stim = f'{wrd}'

    # If it's the first batch of this stimulus, initialize the dictionary for 
    # the specific stimulus 
    if b == 0: flat[f'sub-{s}'][scr] = {}
    if size == 'S1': flat[f'sub-{s}'][scr][stim] = {}

    # Size will always be a new entry
    flat[f'sub-{s}'][scr][stim][size] = {}

    # Assign the 55 rows of each batch to the corresponding X and Y positions
    # Define the labels to assign
    x_labels = [f'X{i}' for i in range(1, 12)]  # 1-11
    y_labels = [f'Y{i}' for i in range(1, 6)]   # 1-5

    # Move flattened tensor to numpy (again)
    performances = performances.detach().numpy()

    # Save 'flattened' activations
    for idx, (x, y) in enumerate([(x, y) for x in x_labels for y in y_labels]):
        flat[f'sub-{s}'][scr][stim][size].setdefault(x, {})[y] = performances[idx]

    return flat 


## Store the activations indices in nested dictionaries
def store_indices(performances, labels, b, s, scr, idx):
    """
    Store the performances in a orderly manner that considers subject and stimulus

    Parameters
    ----------
    performances (torch.nn.Module): the classification of images in a batch
    
    labels (list): the list of image names
    
    b (int): the number of batch processed, to create new sub-dictionaries
    
    s (int): the subject number
    
    flat (dict): the dictionary storing all the flat activations

    Outputs
    ----------
    flat (dict): the dictionary storing all the flat activations now updated
                 with the activations from this batch
    """

    # Get the label of the stimulus presented, with size variation
    example = labels['Stimulus'][b*55]
    parts = example.split('_')
    wrd = parts[1]
    size = parts[3]
            
    # Compose stimulus name
    stim = f'{wrd}'

    # If it's the first batch of this stimulus, initialize the dictionary for 
    # the specific stimulus 
    if b == 0: idx[f'sub-{s}'][f'{scr}'] = {}
    if size == 'S1': idx[f'sub-{s}'][f'{scr}'][f'{stim}'] = {}

    # Size will always be a new entry
    idx[f'sub-{s}'][f'{scr}'][f'{stim}'][f'{size}'] = performances

    return idx 


## Compute averages across stimuli variations to obtain stimulus-identity performance
def correct_performances(stim_dict, class_labels, s, scr, idx_dict): 
    """
    Compute the average performance for one stimulus across all the variations 
    in size, position in which it is presented to the network
    
    Parameters
    ----------
    idx_dict (dict): the dictionary with arrays of activations for each 
                      stimulus variation
    
    Outputs
    ----------
    stim_dict (dict): the dictionary containing the averages, one array for one 
                     stimulus across all it's variations
    """
    
    responses = {}
    stim_dict[f'sub-{s}'][f'{scr}'] = {}
    
    # Concatenate responses to different sizes of the same word
    for w, word in idx_dict[f'sub-{s}'][scr].items(): 
        arrays = []
        
        for sz in idx_dict[f'sub-{s}'][scr][w].values():
            arrays.append(sz)
    
        responses[w] = np.concatenate([a.ravel() for a in arrays])

        # Extract correct label from the wordlist, given a response 
        label_id = class_labels.loc[class_labels[0] == w, 1].values
    
        # compute correct answers 
        stim_dict[f'sub-{s}'][f'{scr}'][w] = np.sum(responses[w] == label_id) / len(responses[w])
        
    return stim_dict



