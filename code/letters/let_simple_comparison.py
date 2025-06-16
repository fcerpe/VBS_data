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

import matplotlib.patches as patches


import sys
sys.path.append('../../')
from lib.activation_extraction import * 


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


## Load letters and check their pixels density 

# Find paths for all scripts
letters_dir = "../../inputs/letters" 
br_paths = glob.glob(os.path.join(letters_dir, '*_F5.png'))
ln_paths = glob.glob(os.path.join(letters_dir, '*_F6.png'))
lt_paths = glob.glob(os.path.join(letters_dir, '*_F1.png'))

image_paths = br_paths + ln_paths + lt_paths

# Load the images in memory
# Assumes that the pixel density of the different stimuli correspond
# To double-check, run 'dnn_letters_homogeneity.py'
images = [Image.open(image_path).convert('RGB') for image_path in image_paths]


# Define transformations that will be applied to images 
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize to the desired size
    transforms.ToTensor(),  # Convert to tensor
    transforms.Normalize(mean = [0.485, 0.456, 0.406], 
                         std = [0.229, 0.224, 0.225])  # Normalize with ImageNet stats
    ])

# Create the Dataset and DataLoader
dataset = ImageDataset(images, transform = transform)
dataloader = DataLoader(dataset, batch_size = 78, shuffle = False)


## Feed images to the network and extract activations

# For each batch - technically only one batch
for b, batch in enumerate(dataloader):
    
    # Before passing the images to the network, we need to register a forward hook
    # The forward  hook is executed by the dnn  while the images are passed, 
    # and return the activations from selected layers
    
    # Here we get all the layer names we can extract activations from
    all_layer_names = get_last_level_layer_names(alex)
    
    # Add in the list below all the layers we want to extract
    # Janini et al use all the ReLU stages 
    layers_list = ['module.features.1',
                   'module.features.4',
                   'module.features.7',
                   'module.features.9',
                   'module.features.11',
                   'module.classifier.2',
                   'module.classifier.5',
                   'module.classifier.6']

    
    # Get the activations
    relu_activations = get_layer_activation_alexnet(alex, layers_list, batch)
    
    ## Compute distances between stimuli at different stages
    
    # Store activations and distances in dictionaries
    activations_dict = {}
    distances_dict = {}
    layers_dict = {}
    
    # For each layer we are interested in: 
    for i, layer in enumerate(layers_list, start = 1):
        
        # Tell the user which layer are we working on
        print(f"ReLU stage {i} - {layer}")
        
        # Get the layer's activations
        stage_act = relu_activations[layers_list[i-1]]
        
        # Save activations and the layer's name
        activations_dict[f"stage{i}"] = stage_act
        layers_dict[f"stage{i}"] = layer
        
        # Compute sum of each activation map into a single value, to obtain a single vector for each image
        # e.g. layer 5: 256x13x13 -> 256x1, sum of the 13x13 activation maps across the features
        if stage_act.dim() > 2:
            stage_act = torch.sum(stage_act, (3,2))
        
        # Compute euclidian distance
        
        # Move from tensor to np array
        stage_np = stage_act.numpy()
        
        # Compute pairwise Euclidean distances
        dist_matrix = squareform(pdist(stage_np, 'euclidean'))
        
        # Save distances    
        distances_dict[f"stage{i}"] = dist_matrix    
            
    
        ## Plot distances as Representational Dissimilarity Matrices (RDMs)
        
        # Create a heatmap with letters labels (in english to make it easier to read)
        plt.figure()
        
        base_labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 
                       'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
        
        repeated_labels = base_labels * 3
        
        # Create new labels with sequence indicators
        scripts = ['Braille', 'Line', 'Latin']
        new_labels = []
        for l in range(3):
            new_labels.extend([label for label in base_labels])
                
        ax = sns.heatmap(dist_matrix, 
                         cmap='viridis', 
                         annot=False, 
                         xticklabels = False, 
                         yticklabels = False)
        
        # Customize the heatmap
        if b == 0:
            script = 'BR'
        else: 
            script = 'LN'
        
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
        
        # Add squares to separate clusters
        num_clusters = len(scripts)  # Assuming clusters correspond to the classes
        cluster_size = 26
        for k in range(num_clusters):
            for j in range(num_clusters):
                rect = patches.Rectangle((j * cluster_size, k * cluster_size),  # (x, y) starting point
                                          cluster_size,  # Width
                                          cluster_size,  # Height
                                          linewidth = 1, 
                                          edgecolor = 'white', 
                                          facecolor = 'none') 
                ax.add_patch(rect)
        
        # Save plot
        savename = f'vbs_alexnet-letters_stage-{i}_plot-rdm-all-scripts.png'
        savepath = os.path.join('../../outputs/figures/letters', savename)
        plt.savefig(savepath, dpi = 600)
        
        # Show the plot
        plt.show()
        
        # labels = ['A','B','C','D','E','F','G','H','I','J','K','L','M', 
        #           'N','O','P','Q','R','S','T','U','V','W','X','Y','Z'] 
        
        # ax = sns.heatmap(dist_matrix, 
        #                  cmap = 'viridis', 
        #                  annot = False, 
        #                  xticklabels = labels, 
        #                  yticklabels = labels)
        
        # # Customize the heatmap
        # if b == 0:
        #     script = 'BR'
        # else: 
        #     script = 'LN'
            
        # title = (f'Batch {script} - ReLU stage {i}')
        
        # ax.set_title(title, fontsize = 15)
        
        # ax.yaxis.set_tick_params(rotation = 0)
        # for label in ax.get_yticklabels():
        #     label.set_verticalalignment('center')
        
        # ax.set_aspect('equal')
        
        # # # Save plot
        # savename = (f'vbs_alexnet-letters_batch-{script}_stage-{i}_resize-squared.png')
        # savepath = os.path.join('figures', savename)
        # plt.savefig(savepath, dpi = 600)
        
        # # Show the plot
        # plt.show()

# Soon



