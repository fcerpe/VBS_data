#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 15 16:15:36 2024

@author: cerpelloni
"""

import torch
import torchvision.models as models
import torch.nn as nn
import torchvision
import matplotlib.pyplot as plt
import numpy as np

# Helper function for sanity_check_dataset
def visualize_dataset_imgs(img, one_channel = False):
    
    if one_channel:
        img = img.mean(dim=0)
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    if one_channel:
        plt.imshow(npimg, cmap="Greys")
    else:
        plt.imshow(np.transpose(npimg, (1, 2, 0)))

# Visualize some transformed stimuli from the train set, to make sure everything is in order
def sanity_check_dataset(train_loader): 
    
    dataiter = iter(train_loader)
    images, labels = next(dataiter)

    # Create a grid from the images and show them
    img_grid = torchvision.utils.make_grid(images)
    visualize_dataset_imgs(img_grid, one_channel = True)
    
    
# Replace the last layer of alexnet with a "clean" new one
def reset_last_layer(model, num_classes):
    
    # Freeze all layers except the last one - just to be cautious
    for param in model.parameters():
        param.requires_grad = False
    
    # Reset the last layer
    # Get the number of features in the original layer
    num_features = model.module.classifier[6].in_features  
    
    # Replace the last layer
    # In VBS we deal with the same number of classes (1000), we just want to overwrite them 
    model.module.classifier[6] = nn.Linear(num_features, num_classes)
    
    # Unfreeze the layers
    for param in model.parameters():
        param.requires_grad = False
    
    

# Save the weights at a give epoch
def save_epoch(): 
    
    return 0
    

# Visualize training progress over the number of batches 
# Need to extract train_counter, test_counter which are train total and val total
def visualize_training_progress(train_counter, train_loss, val_counter, val_loss): 
    
    fig = plt.figure()
    plt.plot(train_counter, train_loss, color = 'cornflowerblue', linewidth = 1.5)
    plt.scatter(val_counter, val_loss, zorder = 5, color = 'darkred')         
    plt.legend(['Train Loss', 'Validation Loss'], loc = 'upper right')
    plt.xlabel('number of training examples seen')
    plt.ylabel('negative log likelihood loss')
    plt.show()
    
    
    
    
    