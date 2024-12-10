#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 26 16:01:35 2024

Simple script to get familiar with DNNs

Uses AlexNet to replicate, with stimuli from Visual Braille Training (VBT), the 
results of Janini et al. 2022 ()

Steps:
    - loads AlexNet
    - loads stimuli from VBT
    - extract activations at given layers for given stimuli
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


## Load images

# Find paths for both scripts
br_dir = "letters/braille" 
br_paths = glob.glob(os.path.join(br_dir, '*_squared.png'))
ln_dir = "letters/line"
ln_paths = glob.glob(os.path.join(ln_dir, '*_squared.png'))

image_paths = br_paths + ln_paths

# Load the images in memory
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
dataloader = DataLoader(dataset, batch_size = 26, shuffle = False)


## Feed images to the network and extract activations

# For each batch
for b, batch in enumerate(dataloader):
    
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
    
    # # Visualization of activations 
    # # select the activations for the first layer in the list
    # features_0_act = layers_activations[layers_list[0]]
    # # select the activations for the first image, and flatten to a vector
    # features_0_act_0 = features_0_act[0].flatten()
    
    # # Classification stuff
    # probabilities = alex(batch)
    # # To get the predicted label for each image, need to transform probabilities 
    # # into a single number. To do this, get the index of the highest probability class
    # predictions_idx = torch.argmax(probabilities, dim = 1)
    # # Now, we need to map these indices to the actual ImageNet class labels
    # predicted_classes = [imagenet_classes[idx] for idx in predictions_idx]
    # print(predicted_classes)


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
        
        labels = ['A','B','C','D','E','F','G','H','I','J','K','L','M', 
                  'N','O','P','Q','R','S','T','U','V','W','X','Y','Z'] 
        
        ax = sns.heatmap(dist_matrix, 
                         cmap = 'viridis', 
                         annot = False, 
                         xticklabels = labels, 
                         yticklabels = labels)
        
        # Customize the heatmap
        if b == 0:
            script = 'BR'
        else: 
            script = 'LN'
            
        title = (f'Batch {script} - ReLU stage {i}')
        
        ax.set_title(title, fontsize = 15)
        
        ax.yaxis.set_tick_params(rotation = 0)
        for label in ax.get_yticklabels():
            label.set_verticalalignment('center')
        
        ax.set_aspect('equal')
        
        # # Save plot
        savename = (f'vbs_alexnet-letters_batch-{script}_stage-{i}_resize-squared.png')
        savepath = os.path.join('figures', savename)
        plt.savefig(savepath, dpi = 600)
        
        # Show the plot
        plt.show()

# Soon



