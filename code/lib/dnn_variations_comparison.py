#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 26 16:01:35 2024

Simple script to get familiar with DNNs

Uses AlexNet to replicate, with stimuli from Visual Braille Training (VBT), the 
results of Janini et al. 2022 ()

Steps:
    - loads AlexNet
    - loads letter stimuli from VBT (Braille and Line Braille) and from Latin letters
    - extract activations at given layers for all the stimuli together
    - calculates euclidian distance between stimuli 
    - plots the corresponding RDMs
    
Based on Janini et al. 2022 and on Andrea Costantino's script 

@author: Filippo Cerpelloni
"""
import os
import glob
import json
import urllib
import torch
import torch.nn as nn
import torchvision.models as models
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from scipy.spatial.distance import pdist, squareform
from activation_extraction import * 


## Load AlexNet and relative wizardry 

# List all the models avaliable through pytorch
all_models = models.list_models()

# Take IMAGENET1K_V1 == DEFAULT
alex = models.alexnet(weights = 'IMAGENET1K_V1')
alex = nn.DataParallel(alex)
alex.eval()

# Load ImageNet class names 
# - for classification purposes, not relevant at the moment
url = "https://raw.githubusercontent.com/anishathalye/imagenet-simple-labels/master/imagenet-simple-labels.json"
imagenet_classes = json.loads(urllib.request.urlopen(url).read().decode())


## Load stimuli and their variations
# Assumes that all the stimuli are balanced in terms of pixel density across scripts, sizes, thicknesses

# Find paths for all scripts
stim_dir = "letters/stimuli" 
stim_paths = glob.glob(os.path.join(stim_dir, '*.png'))

# Get a list of alphabet + letter, to order the batches 
letters_set = set()

for s in stim_paths:
    parts = s.split('_')
    prefix = parts[0].split('/')[-1]  
    letter = parts[1]  
    letters_set.add(f"{prefix}_{letter}")

letters_list = sorted(list(letters_set))

# Load the images in memory
# Assumes that the pixel density of the different stimuli correspond
# To double-check, run 'dnn_letters_homogeneity.py'
images = [Image.open(stim_path).convert('RGB') for stim_path in stim_paths]


# Define transformations that will be applied to images 
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize to the desired size
    transforms.ToTensor(),  # Convert to tensor
    transforms.Normalize(mean = [0.485, 0.456, 0.406], 
                         std = [0.229, 0.224, 0.225])  # Normalize with ImageNet stats
    ])

# Create the Dataset and DataLoader
dataset = ImageDataset(images, transform = transform)

# batch size of 20 means that we process one letter and its variations at the time
dataloader = DataLoader(dataset, batch_size = 25, shuffle = False)

# Store data in dictionaries to then average, distance, plot activations
raw_activations_dict = {}
activations_dict = {}
averages_dict = {}


## Feed images to the network and extract activations

# For each batch
for b, batch in enumerate(dataloader):
    
    # Tell the user which letter is being processed
    print(f"working on: {letters_list[b]}")
    
    # Before passing the images to the network, we need to register a forward hook
    # The forward  hook is executed by the dnn  while the images are passed, 
    # and return the activations from selected layers
    
    # Here we get all the layer names we can extract activations from
    all_layer_names = get_last_level_layer_names(alex)
    print(all_layer_names)
    
    # Add in the list below all the layers we want to extract
    # Janini et al use all the ReLU stages 
    layers_list = [all_layer_names[1], 
                   all_layer_names[4], 
                   all_layer_names[7], 
                   all_layer_names[9], 
                   all_layer_names[11],
                   all_layer_names[16],
                   all_layer_names[19]]
    
    # Get the activations
    relu_activations = get_layer_activation_alexnet(alex, layers_list, batch)
    
    ## Average activations for one letter at different stages
    
    # Create a dictionary entry for each letter
    raw_activations_dict[f"{letters_list[b]}"] = {}
    activations_dict[f"{letters_list[b]}"] = {}
    averages_dict[f"{letters_list[b]}"] = {}
    
    # For each layer we are interested in: 
    for i, layer in enumerate(layers_list, start = 1):
        
        # Tell the user which layer are we working on
        print(f"ReLU stage {i} - {layer}")
        
        # Get the layer's activations
        stage_act = relu_activations[layers_list[i-1]]
        
        # Save activations and the layer's name
        raw_activations_dict[f"{letters_list[b]}"][f"stage{i}"] = stage_act
        
        # Compute sum of each activation map into a single value, to obtain a single vector for each image
        # e.g. layer 5: 256x13x13 -> 256x1, sum of the 13x13 activation maps across the features
        if stage_act.dim() > 2:
            stage_act = torch.sum(stage_act, (3,2))
        
        # Average activations from 25 images into one
        # Move from tensor to np array
        stage_np = stage_act.numpy()
        
        # Save 'flattened' activations
        activations_dict[f"{letters_list[b]}"][f"stage{i}"] = stage_np
        
        # Compute average across images presented
        stage_avg = np.average(stage_np, 0)
        
        # Save average
        averages_dict[f"{letters_list[b]}"][f"stage{i}"] = stage_avg    
            
    

## Calculate euclidian distances between letters and their variations

# Follow janini et al. (2022)'s method:
# - average activations across variations (done for each layer in batch) 
# - compute distance by comparing each letter's activation to the average of each letters (e.g. br_a_TxSx - br_b_AVG)
# - average distances to obtain cell of RDM
# - average across the diagonal (a-b and b-a)

# Initiate a dictionary to contain the distances between elements in a given layer
distances_dict = {}

# Loop through layers (again but inevitable)
for i, layer in enumerate(layers_list, start = 1):
    
    # Initiate a dictionary to contain the distances between elements in a given layer
    distance_matrix = np.full((78, 78), np.nan)
    
    # Extract the average activations across variations for each letter, previously computed
    letters_avg = np.vstack([averages_dict[f"{letter}"][f"stage{i}"] for letter in letters_list])
    
    # For each letter(1):
    for j, let1 in enumerate(letters_list): 
        
        # Get letter(1) activations 
        let1_variations = activations_dict[f'{let1}'][f'stage{i}']
        
        # For each letter(2) (again):
        for k, let2 in enumerate(letters_list): 
                
            # Get letter(2) average activation
            let2_avg = letters_avg[k]
            
            # Concatenate letter(2) average to all of letter(1)'s single activation
            # to use pdist and avoid looping through those single activations
            let1let2_activations = np.vstack([let2_avg, let1_variations])

            # Compute distance between letter(1)'s specific variation and letter(2)'s average
            let1let2_distances_array = pdist(let1let2_activations, 'euclidean')[:25]
                
            # Compute average of distances
            let1let2_distance = np.average(let1let2_distances_array)
            
            # Place in the final matrix for the layer
            distance_matrix[j,k] = let1let2_distance
            
    # Average distances of comparisons on same letter (e.g. (A-B + B-A) / 2 )
    for l, let in enumerate(letters_list):
        for m, let in enumerate(letters_list):
            average = (distance_matrix[l,m] + distance_matrix[m,l]) / 2
            distance_matrix[l,m] = average
            distance_matrix[m,l] = average
    
    # Save distances    
    distances_dict[f"stage{i}"] = distance_matrix   


    ## Plot distances as Representational Dissimilarity Matrices (RDMs)
    # Assumes distance matrix as 'dnn_simple_comparison'

    plt.figure()

    base_labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 
                    'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
    repeated_labels = base_labels * 3

    # Create new labels with sequence indicators
    scripts = ['Braille', 'Line', 'Latin']
    new_labels = []
    for l in range(3):
        new_labels.extend([label for label in base_labels])
            
    ax = sns.heatmap(distance_matrix, 
                      cmap = 'viridis', 
                      annot = False, 
                      xticklabels = False, 
                      yticklabels = False)

    # Customize the heatmap
    title = f'ReLU stage {i}'
    ax.set_title(title, fontsize = 15)

    # Add sequence indicators as subtitles for the axis
    for j, script in enumerate(scripts):
        plt.text(-8, 26 * j + 13, script, rotation = 90, fontsize = 12, verticalalignment = 'center')
        plt.text(26 * j + 13, 85, script, rotation = 0, fontsize = 12, horizontalalignment = 'center')

    ax.yaxis.set_tick_params(rotation = 0)
    for label in ax.get_yticklabels():
        label.set_verticalalignment('center')

    ax.set_aspect('equal')

    # Save plot
    savename = f'vbs_alexnet-letters_stage-{i}_plot-rdm-variations_TS.png'
    savepath = os.path.join('figures', savename)
    plt.savefig(savepath, dpi = 600)

    # Show the plot
    plt.show()  

    

        

