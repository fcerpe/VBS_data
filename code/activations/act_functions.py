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
sys.path.append('../')

from lib.CORnet.cornet import *

import torch
import torch.nn as nn
from torchvision import transforms, models
from torchvision.io import read_image
from torch.utils.data import Dataset, DataLoader
from PIL import Image

import pandas as pd
import zipfile
from natsort import natsorted

### MAIN EXTRACTION FUNCTIONS 

# ! TEMPORARLY DEAL ONLY WITH BRAILLE AND ALEXNET
# Extract activations 
# - for each layer of the specified network 
# - for each stimulus class presented 
def extract_activations(model_name, nSub, experiment, script):
    
    # Set all the important paths (weights, figures, dataets) and the specifics
    # of the experiment and script
    
    # Path where to save the weights
    weights_dir = opt['dir']['weights']
    weights_path = f"{weights_dir}/literate/{script}/"  
    
    # Path where to save the leraning curve
    figures_dir = opt['dir']['figures']
    figures_path = f"{figures_dir}/literate/{script}/"  
    
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
    
    # TO-DO: Check if the folder is zipped. If so, unzp it
    # if os.path.exists(dataset_path +'.zip'):
    #     zipf.extractall(dataset_path +'.zip')
    
    
    # Iterate through all the subjects (up to five) requested
    for s in range(nSub): 
        
        # Load the dataset 
        dataset, data_loader = load_dataset(dataset_path)
        
        ## Choose the model  
        # can become: model = get_network_with_weights(opt, model_name, script, epochNb)
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
            
        # elif model_name == 'cornet':
        #     print('\nExtracting activations from CORnet Z models ...')
            
        #     # Get model
        #     model = cornet_z()
            
        #     # Apply weights of expert network 
        #     saved_weights_path = '../../outputs/weights/literate/{script}/model-{model_name}_sub-{s}_data-{scriptID}_epoch-10.pth'
        #     saved_weights = torch.load(saved_weights_path)
        #     model.load_state_dict(saved_weights['state_dict'])
                
        #     # Evaluate the model, just to check
        #     model.eval()
    
                
        else:
            
            # Close the function - very rudimental but works
            print('\nERROR: no compatible model specified, stopping here.')
            return 0
        

        ## Feed images to the network and extract activations
        for b, batch in enumerate(data_loader):
            
            # From a list of all the layers of a network, pick the relevant ones
            # for the extraction of activations
            # (e.g. ReLU stages in AlexNet)
            layer_names = name_layers(model)
            layer_names = pick_layers(model_name, layer_names)
            
            # Extract activations at any layer for the stimuli in the batch
            layer_activations = get_layer_activations(model, layer_names, batch)
        
            # Store information in a organized way
    
            # Process activations 
    
    return 0


# Extract performance of the network  
# - for each layer of the specified network 
# - for each stimulus class presented 
def extract_performance(model_name, script, nSub): 
    
    # empty for now
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








