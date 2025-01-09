#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 12 13:16:24 2024

Network training script

@author: cerpelloni
"""

# Import everything, even other folders
import sys

sys.path.append('../')

from src.activation_extraction import *
from network_functions import *

import os, glob, json, urllib

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

from datetime import datetime

import subprocess


### ---------------------------------------------------------------------------
### DATASET 

# Get the dataset from the image structure
# (also returns 1000 word categories)
latin, word_classes = import_dataset('../../inputs/datasets/LT/')


### ---------------------------------------------------------------------------
### ITERATIONS 

# To create variability, get multiple instances of the network
# Batches, training, will slightly differ between networks 

# Number of iterations
subjects = 5 

## Define hyperparameters
# Epochs: number of times that a network passes through training
epochs = 10

# Learning rate: to which extent the network parameters are updated for each batch / epoch
learning_rate = 1e-3

# Loss function: different functions available in pytorch
loss_fn = nn.CrossEntropyLoss() 

# Momentum: nudge the optimezer in towards strongest gradient ove multiple steps
momentum = .9

# Optimizer: different functions in pytorch
optimizer = torch.optim.SGD(alexnet.parameters(), lr = learning_rate, momentum = momentum)



for s in range(subjects): 
    
    ## ------------------------------------------------------------------------
    ## CREATE BATCHES
    
    # Split dataset into training and validation and create batches through DataLoader
    train_loader, val_loader = load_dataset(latin)
    
    # Dataset sanity check: visualize some stimuli
    sanity_check_dataset(train_loader)
    
    ## ------------------------------------------------------------------------
    ## GET NETWORK

    # List all the models avaliable through pytorch
    all_models = models.list_models()
    
    # Get AlexNet with Imagenet weights
    alexnet = models.alexnet(weights = 'IMAGENET1K_V1')
    alexnet = nn.DataParallel(alexnet)
    
    # evaluate the model, just to check
    alexnet.eval()
    model_name = 'alexnet'

    # Reset last layer (classifier) for new training
    alexnet = reset_last_layer(alexnet, len(word_classes))

    ## ---------------------------------------------------------------------------
    ## TRAIN THE NETWORK

    # Trackers to monitor training progression
    train_losses = []
    train_counter = []
    val_losses = []
    val_counter = []
    
    # Create filename to identify the iteration
    filename = f"model-{model_name}_sub-{s}_data-LT"

    # Check your device - GPU (a.k.a. 'cuda') is faster, use it if available
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # Notify the user 
    print(f"Using {device} device")
    print("Training started at:", datetime.now())

    for e in range(epochs):
    
        # Train one epoch
        # Obtain: total number of batches ran, losses, accuracy
        train_total, train_loss, train_accuracy = train(alexnet, train_loader, optimizer, loss_fn, device)    
        
        # Track loss progression, for visualization
        train_losses.append(train_loss)
        train_counter.append(e)
        
        # Validation
        # Obtain: number of batches ran, losses, accuracy
        val_total, val_loss, val_accuracy = validate(alexnet, val_loader, loss_fn, device)
        
        # Track loss progression, for visualization
        val_losses.append(val_loss)
        val_counter.append(e)
        
        # Print report
        print(f"Epoch {e+1}")
        print(f"Training - Loss: {train_loss:.4f}, Accuracy: {train_accuracy:.4f}")
        print(f"Validation - Loss: {val_loss:.4f}, Accuracy: {val_accuracy:.4f}")
        
        # Save current state of the training
        torch.save(alexnet.state_dict(), f"{filename}_epoch-{e+1}.pth")
        
        print(f"Epoch {e+1} ended at: ", datetime.now())
    
    
    ## ------------------------------------------------------------------------
    ## VISUALZIE TRAINING
    
    visualize_training_progress(train_counter, train_losses, 
                                val_counter, val_losses, 
                                filename)    







