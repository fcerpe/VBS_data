#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 28 15:01:15 2025

Support functions to extract layer activations relative to the VBS project

@author: Filippo Cerpelloni
"""
import os
import glob
import sys
sys.path.append('../lib/CORnet')

import torch
import torch.nn as nn
from torchvision import transforms, models
from torchvision.io import read_image
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from natsort import natsorted
from scipy.spatial.distance import pdist, squareform
from cornet import cornet_z

import numpy as np
import pandas as pd
import zipfile

import pickle


### MAIN EXTRACTION FUNCTIONS 

# ! TEMPORARLY DEAL ONLY WITH BRAILLE AND ALEXNET
# Extract activations 
# - for each layer of the specified network 
# - for each stimulus class presented 
def extract_activations(opt, model_name, nSub, experiment, script):
    
    # Set all the important specifics of where to find data:
    # - paths (weights, figures, dataets) 
    # - script shortname
    # - specific test set
    weights_dir, figures_dir, scriptID, dataset_path, dataset_spec = get_paths(opt, script, experiment)
    
    
    # TODO: Check if the folder is zipped. If so, unzp it
    # if os.path.exists(dataset_path +'.zip'):
    #     zipf.extractall(dataset_path +'.zip')
    
    # Initialize storage dictionaries 
    raw_dict = {}       # Raw data
    flat_dict = {}      # Flattened activations
    stim_dict = {}      # Averages across variations for each stimulus
    dist_dict = {}      # Distance between stimuli based on Janini et al. 2022
    
    
    # Iterate through all the subjects (up to five) requested
    for s in range(nSub): 
        
        # Load the dataset 
        # Batch size is fixed at 55, to process all the size variations of a stimulus together
        # resulting in 5 batches for each stimulus (e.g. one real word in one script)
        dataset, data_loader = load_dataset(dataset_path)
        
        ## Choose the model  
        model = get_weighted_model(model_name, weights_dir, script, scriptID, s)
        
        
        ## Load the labels for the individual images and for the classes of stimuli 
        image_labels = pd.read_csv(f'../../inputs/words/{dataset_spec}_stimuli.csv')
        stim_labels = pd.read_csv(f'../../inputs/words/{dataset_spec}_wordlist.csv', header = None)
        
        # Initialize subject-specific dictionaries 
        raw_dict[f'sub-{s}'] = {} 
        flat_dict[f'sub-{s}'] = {}
        stim_dict[f'sub-{s}'] = {}   
        dist_dict[f'sub-{s}'] = {}    


        ## Feed images to the network and extract activations
        for b, batch in enumerate(data_loader):
            
            # From a list of all the layers of a network, pick the relevant ones
            # for the extraction of activations
            # (e.g. ReLU stages in AlexNet)
            layer_names = name_layers(model)
            layer_names = pick_layers(model_name, layer_names)
            
            # Extract activations at any layer for the stimuli in the batch
            layer_activations = get_layer_activations(model, layer_names, batch)
        
            # Store information in a series of dictionaries, to ease stats
            # Common structure to all the dicts is:
            # dict[layer][subject][stimulus][size][x position][y position] = activation for image in layer
            raw_dict, flat_dict = store_layer_activations(layer_activations, layer_names,
                                                          image_labels, b, s, 
                                                          raw_dict, flat_dict)

        # All the data is extrcted
        print('Extracted data from batches\n')
        
        
        ## Average activations across all variations
        print('Computing averages and distances for each stimulus ...\n')
        
        # Compute averages for each stimulus
        stim_dict[f'sub-{s}'] = average_activations(flat_dict[f'sub-{s}'])
    
    
        ## Calculate euclidian distances between letters and their variations
        # Follow janini et al. (2022)'s method:
        # - compare each variation of word A with average of word B
        #   (e.g. BR_FS_1_S*_X*_Y* and BR_NW_1)
        # - average distances to obtain cell of RDM
        # - average across the diagonal (a-b and b-a)
        dist_dict[f'sub-{s}'] = compute_distances(flat_dict[f'sub-{s}'], stim_dict[f'sub-{s}'])
        
        ## Save subject-specifc activations and matrices 
        save_activations(raw_dict[f'sub-{s}'], flat_dict[f'sub-{s}'], stim_dict[f'sub-{s}'], dist_dict[f'sub-{s}'])
        
        
            


# Extract performance of the network  
# - for each layer of the specified network 
# - for each stimulus class presented 
def extract_performance(model_name, script, nSub): 
    
    # empty for now
    return 0


### ---------------------------------------------------------------------------
### MISC FUNCTIONS

def get_paths(opt, script, experiment):
    
    # Path where to save the weights
    weights_dir = opt['dir']['weights']
    
    # Path where to save the leraning curve
    figures_dir = opt['dir']['figures']
    
    ## Switch test set based on which experiment we refer to
    # IMPORTANT: different sets than training
    
    # Get the abbreviation of the script used in the bids-like filename
    if script == 'braille':
        scriptID = 'BR'
    elif script == 'line':
        scriptID = 'LN'
    else:
        scriptID = 'LT'
    
    # General dataset directory
    dataset_dir = opt['dir']['datasets']
    
    # Test-sepcific directory: VBE or VBT 
    if experiment == 'VBE': 
        dataset_spec = 'test_vbe'
    elif experiment == 'VBT': 
        dataset_spec = 'text_vbt'
    else:
        print('ERROR: dataset not found')
            
    # Full dataset path
    dataset_path = f"{dataset_dir}/{dataset_spec}"
    
    return weights_dir, figures_dir, scriptID, dataset_path, dataset_spec



### ---------------------------------------------------------------------------
### NETWORK FUNCTIONS

# Pick the right model and the right weights 
def get_weighted_model(model_name, weights_dir, script, scriptID, s):
    
    # AlexNet
    if model_name == 'alexnet': 
        print('\nExtracting activations from AlexNet models ...')
        
        # Get model
        model = models.alexnet()
        model = nn.DataParallel(model)
        
        # Apply expertise weights
        saved_weights_path = f'{weights_dir}/literate/{script}/model-{model_name}_sub-{s}_data-{scriptID}_epoch-10.pth'
        state_dict = torch.load(saved_weights_path, map_location = torch.device('cpu'))
        model.load_state_dict(state_dict)
        
        # Evaluate the model, just to check
        model.eval()
        
    # CORnet
    elif model_name == 'cornet':
        print('\nExtracting activations from CORnet Z models ...')
        
        # Get model
        model = cornet_z()
        
        # Apply weights of expert network 
        saved_weights_path = '../../outputs/weights/literate/{script}/model-{model_name}_sub-{s}_data-{scriptID}_epoch-10.pth'
        saved_weights = torch.load(saved_weights_path)
        model.load_state_dict(saved_weights['state_dict'])
        
        # Evaluate the model, just to check
        model.eval()

    else:
        
        # Close the function - very rudimental but works
        print('\nERROR: no compatible model specified, stopping here.')
        return 0
    

### ---------------------------------------------------------------------------
### DATASET FUNCTIONS

# Define a custom dataset for the test sets
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
        

# Import a dataset and the wordlist (always the same)
# Needs speification of dataset, with path
def load_dataset(path): 
    
    # From the path, get the list of stimuli to load 
    stim_paths = glob.glob(os.path.join(path, 'BR_FS_3*.png'))

    # Order stimuli paths and labels in the same manner, 
    # to make sure that the image we load corresponds to the label we give it
    stim_paths = order_stimuli(stim_paths)
    
    
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
    dataset = TestDataset(annotations = '../../inputs/words/test_vbe_stimuli.csv', images = images, transform = transform)
    
    # Use DataLoader to create batches
    dataset_loader = DataLoader(dataset, batch_size = 55, shuffle = False)

    return dataset, dataset_loader


# Order the lsit of stimuli in a natural alphabetical order and save a csv with
# the list of stimuli, to be used as labels in the dataset loader
def order_stimuli(in_paths):
    
    # Sort the paths
    out_paths = natsorted(in_paths)
    
    # Create set to store 
    out_set = set()

    for s in out_paths:
        parts = s.split('/')
        stim_info = parts[-1].split('.')[0]  
        out_set.add(f"{stim_info}")

    out_list = natsorted(list(out_set))
    
    stimuli = pd.DataFrame(out_list, columns = ["Stimulus"])
    stimuli.to_csv("../../inputs/words/test_vbe_stimuli.csv", index = False)

    return out_paths

### ---------------------------------------------------------------------------
### LAYER FUNCTIONS

# Get the names of the layers of a network, from Andrea Costantino
def name_layers(model):
    """
    Extract the names of all last-level layers in a PyTorch neural network.

    Args: model (torch.nn.Module): The PyTorch model.

    Returns: list: A list containing the names of all last-level layers in the model.
    """
    layers = []
    for name, module in model.named_modules():
        
        # Check if the module is a leaf module (no children)
        if not list(module.children()):
            
            # Exclude the top-level module (the model itself) which is always a leaf
            if name: layers.append(name)

    return layers


# Get the important layers for the extraction of activations, based on the network's architecture
def pick_layers(model_name, layer_names):
    
    layers = []
    
    if model_name == 'alexnet': 
        
        # Pick all the ReLU stage
        layers = [layer_names[1], 
                  layer_names[4], 
                  layer_names[7], 
                  layer_names[9], 
                  layer_names[11],
                  layer_names[16],
                  layer_names[19]]
        
    elif model_name == 'cornet':
        
        # Pick ???
        layers = [layer_names[1]]
        
    else: 
        print('Cannot decide which layers to pick, do not know this network.')
        return 0

    return layers


### ---------------------------------------------------------------------------
### ACTIVATIONS FUNCTIONS

# Extract activations 
def get_layer_activations(model, layer_names, batch):
    """
    Get the activations of specified layers in response to input data, handling large batches by
    splitting them into smaller batches of size 100, and concatenating the results. The activations
    are detached from the computation graph and moved to the CPU before storage.

    Args:
        model (torch.nn.Module): The neural network model to probe.
        layer_names (list): List of names of the layers to probe.
        image_tensor (torch.Tensor): Batch of images to feed through the model.

    Returns:
        dict: A dictionary where keys are layer names and values are concatenated activations for all batches,
              with each tensor detached and moved to CPU.
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


# From layer activation, store the activations for each stimulus of the batch 
# in the correct place in the correct dictionary
def store_layer_activations(activations, layers, labels, b, s, raw, flat): 
   
    # Get the label of the stimulus presented, with size variation
    example = labels['Stimulus'][b*55]
    parts = example.split('_')
    scr = parts[0]
    cat = parts[1]
    wrd = parts[2]
    size = parts[3]
    stim = f'{scr}_{cat}_{wrd}'
    
    # If it's the first batch of this stimulus, initialize the dictionaries
    if size == 'S1':
        init_dict = True
    else: 
        init_dict = False
   
    # For each layer / stage
    for i, layer in enumerate(layers, start = 1):
        
        # Get the activations
        stage = activations[layers[i-1]]
        
        # Initialize stage dict on all its levels, if needed
        if init_dict: 
            # Stage
            raw[f'sub-{s}'][f'stage-{i}'] = {}
            flat[f'sub-{s}'][f'stage-{i}'] = {}
            
            # Stimulus 
            raw[f'sub-{s}'][f'stage-{i}'][f'{stim}'] = {}
            flat[f'sub-{s}'][f'stage-{i}'][f'{stim}'] = {}
            
        # Size will always be a new entry
        raw[f'sub-{s}'][f'stage-{i}'][f'{stim}'][f'{size}'] = {}
        flat[f'sub-{s}'][f'stage-{i}'][f'{stim}'][f'{size}'] = {}
        
        ## Assign the 55 rows of each batch to the corresponding X and Y positions
        
        # Define the labels to assign
        x_labels = [f'X{i}' for i in range(1, 12)]  # 1-11
        y_labels = [f'Y{i}' for i in range(1, 6)]   # 1-5
        
        # Assign values to the dictionary
        for idx, (x, y) in enumerate([(x, y) for x in x_labels for y in y_labels]):
            raw[f'sub-{s}'][f'stage-{i}'][f'{stim}'][f'{size}'].setdefault(x, {})[y] = stage[idx]

        
        ## Flatten the activations to a single vector for each image
        # e.g. layer 5: 256x13x13 -> 256x1, sum of the 13x13 activation maps across the features
        if stage.dim() > 2:
            stage = torch.sum(stage, (3,2))
        
        # Average activations from all the position variations into one
        # and move from tensor to np array
        stage = stage.numpy()
        
        # Save 'flattened' activations
        for idx, (x, y) in enumerate([(x, y) for x in x_labels for y in y_labels]):
            flat[f'sub-{s}'][f'stage-{i}'][f'{stim}'][f'{size}'].setdefault(x, {})[y] = stage[idx]
        
    return raw, flat


# Compute averages across stimuli variations to obtain a value representing the whole stimulus
# within a subject
def average_activations(flat_dict):
    
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


# Compute distances between the stimuli and within each network
def compute_distances(activations, averages):
    
    # Initiate a dictionary to contain the distances between elements in a given layer
    distances = {}
    
    # Extract from the dictionary:
    # - number and name of each layer
    layer_names = list(averages.keys())
    
    # - the stimuli in each layer
    stim_names = list(averages[f'{layer_names[0]}'].keys())
    
    # - the number of stimuli
    stim_nb = len(averages[f'{layer_names[0]}'])

    # Loop through layers of the datasets 
    for i, layer in enumerate(layer_names, start = 1):
        
        # Initiate a matrix of len(nb of stimuli) to store the RDM values
        matrix = np.full((stim_nb, stim_nb), np.nan)
        
        # Browse through stim A
        for r, stim_a in enumerate(stim_names):
            
            stim_a_variations = []
        
            stimuli = activations[f'{layer_names[0]}']
            # Extract activation for all the variants in a np list (like for assignment in storing)
            var = 0
            
            # TODO stim_a_variations = 0
            # go into each variation and extract the activation array
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
                stim_b_average = averages[f'{layer_names[0]}'][f'{stim_b}']
        
                # Concatenate stimulus B average to stimulus A variations
                ab_concatenated = np.vstack([stim_b_average, stim_a_variations])
        
                # Compute distances between all the elements in the list, 
                # then keep only the ones referring to stimuls B average and stimulus A variations
                ab_distances = pdist(ab_concatenated, 'euclidean')[:var]
                  
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
        
        # Save distances    
        distances[f'{layer_names[0]}'] = matrix   

    return distances


# Save the activations and distance matrices with a predefined filename
def save_activations(opt, raw, flat, stim, dist):
    
    # Set path and filename for raw, flat, stim activations
    filepath = os.path.join(opt['dir']['results'], 'activations')
    filename = f'model-{model_name}_sub-{s}_script-{scriptID}_data-activations'
    
    # Save the activations 
    
    
    # Set path and filename for distances
    filepath = os.path.join(opt['dir']['results'], 'distances')
    filename = f'model-{model_name}_sub-{s}_script-{scriptID}_data-distances'
    
    # Save the distances
    
    
    
    return 0 







