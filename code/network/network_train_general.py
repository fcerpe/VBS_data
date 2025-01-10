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

# Define transformations that will be applied to images 
# - resize to 224x224 (alexnet input)
# - convert to tensor
# - normalize with ImageNet stats
transform = transforms.Compose([transforms.Resize((224, 224)),  
                                transforms.ToTensor(),
                                transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225]) ])


# Load the Dataset from a structure of folders - BRAILLE 
dataset = torchvision.datasets.ImageFolder(root = '../../inputs/datasets/LT/', transform = transform)

# Get the classes / labels for each word
word_classes = pd.read_csv('../../inputs/words/nl_wordlist.csv', header = None).values.tolist()

# Split into training and validation
train_size = int(0.7 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

# Use DataLoader to create batches
train_loader = DataLoader(train_dataset, batch_size = 64, shuffle = True)
val_loader = DataLoader(val_dataset, batch_size = 64, shuffle = False)

# Dataset sanity check: visualize some stimuli
sanity_check_dataset(train_loader)


### ---------------------------------------------------------------------------
### MODEL 

# AlexNet with Imagenet weights

# List all the models avaliable through pytorch
all_models = models.list_models()

# IMAGENET1K_V1 equals DEFAULT
alexnet = models.alexnet(weights = 'IMAGENET1K_V1')
alexnet = nn.DataParallel(alexnet)

# evaluate the model, just to check
alexnet.eval()
model_name = 'alexnet'

# Reset last layer (classifier) for new training
# Does not work, deletes the network
alexnet = reset_last_layer(alexnet, len(word_classes))


### ---------------------------------------------------------------------------
### TRAINING, VALIDATION FUNCTIONS

## Train the network for one epoch
# Take as input:
# - the network
# - the data
# - hyperparameters: optimizer and loss function
# - device, where to run the training (GPU is possible)
def train(model, loader, optimizer, loss_fn, device):
    
    # Setting the network to training mode (net.train) allows the gradients to be computed,
    # and activates training-specific features (dropout, batch normalization) to prevent overfitting
    model.train()
    
    # Initialize parameters to compute loss and accuracy
    running_loss = 0.0
    correct = 0
    total = 0

    # For all the batches
    for inputs, labels in loader:
    
        inputs, labels = inputs.to(device), labels.to(device)

        # Zero the gradients for this batch
        optimizer.zero_grad()
        
        # Make prediction of class based on image
        outputs = model(inputs)
        
        # Compute loss and gradients for this batch
        loss = loss_fn(outputs, labels)
        loss.backward()
        
        # Adjust learning weigths
        optimizer.step()

        ## Collect data on the batch
        # Add this batch's loss to the total loss for the epoch
        running_loss += loss.item() * inputs.size(0)
        
        # Store predicted classes
        _, predicted = outputs.max(1)
        
        # Add this batch's predictions to the total ones
        correct += predicted.eq(labels).sum().item()
        
        # Accumulate total samples 
        total += labels.size(0)

    # Compute the loss and the accuracy for the whole epoch
    epoch_loss = running_loss / total
    accuracy = correct / total
    
    return total, epoch_loss, accuracy
            
        
## Validate the learning at a given epoch
# Needs as input the same parameters of 'train' but the optimizer, as there is 
# no adjustment of the weights
def validate(model, loader, loss_fn, device):
    
    # Set the model to the evaluation mode
    model.eval()
    
    # Initialize parameters to compute loss and accuracy
    running_loss = 0.0
    correct = 0
    total = 0

    # Disable gradient tracking, to avoid changes in the state of the training 
    with torch.no_grad():
        
        for inputs, labels in loader:
            
            inputs, labels = inputs.to(device), labels.to(device)

            # Predict the image classes and update the loss for this batch
            outputs = model(inputs)
            loss = loss_fn(outputs, labels)

            # Keep track of running loss and prediction accuracy 
            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)

    # Compute the loss and the accuracy for the whole epoch
    epoch_loss = running_loss / total
    accuracy = correct / total
    
    return total, epoch_loss, accuracy



### ---------------------------------------------------------------------------
### NETWORK TRAINING


## Define hyperparameters
# Epochs: number of times that a network passes through training
epochs = 10

# Learning rate: to which extent the network parameters are updated for each batch / epoch
learning_rate = 1e-4

# Loss function: different functions available in pytorch
loss_fn = nn.CrossEntropyLoss() 

# Momentum: nudge the optimezer in towards strongest gradient ove multiple steps
momentum = .5

# Optimizer: different functions in pytorch
optimizer = torch.optim.SGD(alexnet.parameters(), lr = learning_rate, momentum = momentum)

# Trackers to monitor training progression
train_losses = []
train_counter = []
val_losses = []
val_counter = []


## Check your device
# Training on GPU (a.k.a. 'cuda') is faster, use it if available
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Let the user know
print(f"Using {device} device")
print("Training started at:", datetime.now())

## Train the network
# It will (hopefully) learn to associate Latin and Braille words based on their "semantic" meaning
for e in range(epochs):
    
    # Train the network for one epoch
    # Obtain:
    # - total number of batches ran (for visualization)
    # - total losses
    # - total accuracy
    train_total, train_loss, train_accuracy = train(alexnet, train_loader, optimizer, loss_fn, device)    
    
    # Track loss progression, for visualization
    train_losses.append(train_loss)
    train_counter.append(e)
    
    
    # Validation
    # Obtain:
    # - total number of batches ran (for visualization)
    # - total losses
    # - total accuracy
    val_total, val_loss, val_accuracy = validate(alexnet, val_loader, loss_fn, device)
    
    # Track loss progression, for visualization
    val_losses.append(val_loss)
    val_counter.append(e)
    
    
    # Print report
    print(f"Epoch {e+1}")
    print(f"Training - Loss: {train_loss:.4f}, Accuracy: {train_accuracy:.4f}")
    print(f"Validation - Loss: {val_loss:.4f}, Accuracy: {val_accuracy:.4f}")
    
    # Save current state of the training
    filename = f"model-{model_name}_lr-{learning_rate}_mom-{momentum}_bsize-64"
    torch.save(alexnet.state_dict(), f"{filename}_epoch-{e+1}.pth")
    
    print(f"Epoch {e+1} ended at: ", datetime.now())
    
    
# Visualize how the training went
visualize_training_progress(train_counter, 
                            train_losses, 
                            val_counter, 
                            val_losses, 
                            filename)    







