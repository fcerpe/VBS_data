#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 15 16:15:36 2024

Visual Braille Silico - training and support functions for the network training

@author: Filippo Cerpelloni
"""
import sys
sys.path.append('../lib/CORnet')
from cornet import *

import os
import glob
import json
import urllib
import csv
import subprocess
import torch
import torchvision

import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from PIL import Image
from datetime import datetime
from collections import OrderedDict



### ---------------------------------------------------------------------------
### Main training functions

## Train AlexNets
def network_train_alexnets(opt, script, nSub, epochs, lr, tr, bt): 
    """
    Train AlexNet models based on specified parameters. 
    
    Parameters
    ----------
    opt (dict): output of vbs_option() containing the paths of the IODA folder 
    
    script (str): the script to learn (i.e. the dataset to load)
    
    nSub (int): the number reference of the subject to analyse
    
    epochs (int): the number of epochs for which to train the network 
                  (in epxeriment: 10)
        
    lr (float): learning rate (in epxeriment: 1e-4)
        
    tr (float: training size, to split the dataset into training and validation
               (in epxeriment: 0.7)
        
    bt (int): the batch size, how many images to present in each batch during 
              training (in epxeriment: 100)
    
    Outputs
    -------
    Saves weights for each epoch in outputs/weights/literate
    Saves training reports for each instance in outputs/logs
    Plots loss function for each instance
    
    """
    
    ## Preliminary operations 
    # Based on the 'script' variable, decide paths and dataset notations
    
    # Get the specifics of this script to define the dataset 
    # and the type of training
    script_values = opt['script'].get(script)
    
    # Specific paths to dataset, weights
    dataset_dir = opt['dir']['datasets']
    dataset_spec = script_values['dataset_spec']
    dataset_path = f"{dataset_dir}/{dataset_spec}"
    
    weights_dir = opt['dir']['weights']
    weights_path = f"{weights_dir}/literate/{dataset_spec}/"  
        
    # Start the logging 
    log_dir = opt['dir']['logs']
    log = init_log(log_dir, dataset_spec, nSub, 'alexnet')
    
    ## Dataset
    # Get the dataset from the image structure and the classes from csv file.
    # Loading of the dataset and division in training and validation sets is 
    # done in the itreation loop, to randomize batches
    # IMPORTANT: paths may change based on the computer used (macbook or enuui)
    latin, word_classes = import_dataset(dataset_path, 
                                         '../../inputs/words/nl_wordlist.csv')
    
    ## Iterations / subjects
    # Get multiple instances of the network: batches and training will differ 
    # slightly between networks 
    
    # Number of iterations
    subjects = nSub
    
    # Define general hyperparameters (epochs and training are specific to the networks' weights)
    # Learning rate: to which extent the network parameters are updated for each batch / epoch
    learning_rate = lr
    
    # Loss function: different functions available in pytorch
    loss_fn = nn.CrossEntropyLoss() 

    for s in range(subjects): 
        
        ## Create batches
        # Split dataset into training and validation and create batches through DataLoader
        # Specify dataset, size of the training set, size of the batch
        train_loader, val_loader = load_dataset(latin, tr, bt)
        
        # Dataset sanity check: visualize some stimuli
        sanity_check_dataset(train_loader)
        
        ## Get the network and the appropriate weights
        # If we're training the network on Dutch only (latin bsed script),
        # take alexnet trained on imagenet and reset the last layer
        if script == 'latin':
        
            # Get AlexNet with Imagenet weights
            alexnet = models.alexnet(weights = 'IMAGENET1K_V1')
            alexnet = nn.DataParallel(alexnet)
            
            # Evaluate the model, just to check
            alexnet.eval()
            model_name = 'alexnet'
        
            # Reset last layer (classifier) for new training
            alexnet = reset_last_layer(alexnet, 'alexnet', len(word_classes))
            
            # Epochs: illiterate network is trained from epoch 0 to epoch desired
            epochs = range(epochs)
            
        # In the other cases (training on latin-based AND experimental conditions),
        # take alexnet and apply weights of the previous training
        else: 
            
            # Load alexnet without weights
            alexnet = models.alexnet()
            alexnet = nn.DataParallel(alexnet)
            
            # Load the weights for that given subject 
            # (it assumes that we are asking for subjects 0 to 4)
            saved_weights_path = f'{weights_dir}/literate/latin/model-alexnet_sub-{s}_training-LT_epoch-10.pth'
            state_dict = torch.load(saved_weights_path)
            
            # Apply the weights to the model
            alexnet.load_state_dict(state_dict)
            
            # Evaluate the model, just to check
            alexnet.eval()
            model_name = 'alexnet'
            
            # Epochs: literate network is trained starting from epoch 10
            epochs = range(epochs, epochs*2)
            
        # Optimizer: different functions in pytorch
        optimizer = torch.optim.SGD(alexnet.parameters(), lr =  learning_rate)

        ## Train the network
        # Trackers to monitor training progression
        train_losses = []
        train_counter = []
        val_losses = []
        val_counter = []
        
        # Create filename to identify the iteration
        filename = f"model-{model_name}_sub-{s}_training-{dataset_spec}"
    
        # Check your device - GPU (a.k.a. 'cuda') is faster, use it if available
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        alexnet.to(device)
    
        # Notify the user 
        print(f"\nTraining on {dataset_spec} dataset\n")
        print(f"Using {device} device")
        print("Training started at:", datetime.now())
        print("\n")
        print(f"Subject {s+1}\n")
    
        for e in epochs:
            
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
            
            # Also save to csv
            log_entry(log, dataset_spec, s, e+1, train_loss, train_accuracy, val_loss, val_accuracy)
            
            # Save current state of the training
            fullpath = f"{weights_path}/{filename}"
            torch.save(alexnet.state_dict(), f"{fullpath}_epoch-{e+1}.pth")
            
            print(f"Epoch {e+1} ended at: ", datetime.now())
        
        ## Visualize training        
        visualize_training_progress(train_counter,
                                    train_losses, 
                                    val_counter, 
                                    val_losses, 
                                    filename)  


## Train CORnets
def network_train_cornets(opt, script, nSub, epochs, lr, tr, bt): 
    """
    Train CORnet models based on specified parameters. 
    
    Parameters
    ----------
    opt (dict): output of vbs_option() containing the paths of the IODA folder 
    
    script (str): the script to learn (i.e. the dataset to load)
    
    nSub (int): the number reference of the subject to analyse
    
    epochs (int): the number of epochs for which to train the network 
                  (in epxeriment: 15)
        
    lr (float): learning rate (in epxeriment: 1e-4)
        
    tr (float: training size, to split the dataset into training and validation
               (in epxeriment: 0.7)
        
    bt (int): the batch size, how many images to present in each batch during 
              training (in epxeriment: 100)
    
    Outputs
    -------
    Saves weights for each epoch in outputs/weights/literate
    Saves training reports for each instance in outputs/logs
    Plots loss function for each instance
    
    """
    
    ## Preliminary operations 
    # Based on the 'script' variable, decide paths and dataset notations
    
    # Get the specifics of this script to define the dataset 
    # and the type of training
    script_values = opt['script'].get(script)
    
    # Specific paths to dataset, weights
    dataset_dir = opt['dir']['datasets']
    dataset_spec = script_values['dataset_spec']
    dataset_path = f"{dataset_dir}/{dataset_spec}"
    
    weights_dir = opt['dir']['weights']
    weights_path = f"{weights_dir}/literate/{dataset_spec}/"  
    
    # Start the logging 
    log_dir = opt['dir']['logs']
    log = init_log(log_dir, dataset_spec, nSub, 'cornet')
    
    ## Dataset
    # Get the dataset from the image structure and the classes from csv file.
    # Loading of the dataset and division in training and validation sets is 
    # done in the itreation loop, to randomize batches
    # IMPORTANT: paths may change based on the computer used (macbook or enuui)
    latin, word_classes = import_dataset(dataset_path, 
                                         '../../inputs/words/nl_wordlist.csv')
    
    ## Iterations / subjects
    # Get multiple instances of the network: batches and training will differ 
    # slightly between networks 
    
    # Number of iterations
    subjects = nSub
    
    # Define general hyperparameters (epochs and training are specific to the networks' weights)
    # Learning rate: to which extent the network parameters are updated for each batch / epoch
    learning_rate = lr
    
    # Epochs: train from epoch 0 to epoch desired
    epochs = range(epochs)
    
    # Loss function: different functions available in pytorch
    loss_fn = nn.CrossEntropyLoss() 

    for s in range(subjects): 
        
        ## Create batches
        # Split dataset into training and validation and create batches through DataLoader
        # Specify dataset, size of the training set, size of the batch
        train_loader, val_loader = load_dataset(latin, tr, bt)
        
        # Dataset sanity check: visualize some stimuli
        sanity_check_dataset(train_loader)

        ## Get the network        
        # Load CORnet Z (from dicarlolab's github)
        cornet = cornet_z()

        # Always take literate network instance and present it with the desired
        # dataset, whichever it is 
        
        # Quick workaround: Agrawal and Dehaene have 2000 categories (1000 ImageNet + 1000 words)
        # We need to modifiy our netwrok to fit the weights. It will be reset to 1000 later anyway
        cornet = reset_last_layer(cornet, 'cornet', 2000)
        
        # Apply the weights corresponding to the literate network in French
        # (Latin script)
        saved_weights_path = f'../../outputs/weights/literate/cornet/french/save_lit_fr_rep{s}.pth.tar'
        saved_weights = torch.load(saved_weights_path)
        
        saved_state_dict = reconcile_cornet_literate_french(saved_weights['state_dict'])
        cornet.load_state_dict(saved_state_dict)
        
        # Evaluate the model, just to check
        cornet.eval()
        model_name = 'cornet'
        
        # Reset last layer (classifier) for new training
        cornet = reset_last_layer(cornet, 'cornet', len(word_classes))
        
        # Optimizer: different functions in pytorch
        optimizer = torch.optim.SGD(cornet.parameters(), lr =  learning_rate)
    
        ## Train the network
        # Trackers to monitor training progression
        train_losses = []
        train_counter = []
        val_losses = []
        val_counter = []
        
        # Create filename to identify the iteration
        filename = f"model-{model_name}_sub-{s}_training-{dataset_spec}"
    
        # Check your device - GPU (a.k.a. 'cuda') is faster, use it if available
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        cornet.to(device)
    
        # Notify the user 
        print(f"\nTraining on {dataset_spec} dataset\n")
        print(f"Using {device} device")
        print("Training started at:", datetime.now())
        
        print("\n")
        print(f"Subject {s+1}\n")
    
        for e in epochs:
            
            # Train one epoch
            # Obtain: total number of batches ran, losses, accuracy
            train_total, train_loss, train_accuracy = train(cornet, train_loader, optimizer, loss_fn, device)    
            
            # Track loss progression, for visualization
            train_losses.append(train_loss)
            train_counter.append(e)
            
            # Validation
            # Obtain: number of batches ran, losses, accuracy
            val_total, val_loss, val_accuracy = validate(cornet, val_loader, loss_fn, device)
            
            # Track loss progression, for visualization
            val_losses.append(val_loss)
            val_counter.append(e)
            
            # Print report
            print(f"Epoch {e+1}")
            print(f"Training - Loss: {train_loss:.4f}, Accuracy: {train_accuracy:.4f}")
            print(f"Validation - Loss: {val_loss:.4f}, Accuracy: {val_accuracy:.4f}")
            
            # Also save to csv, to avoid ctrl-c cmd-v between computers
            log_entry(log, dataset_spec, s, e+1, train_loss, train_accuracy, val_loss, val_accuracy)
            
            # Save current state of the training
            fullpath = f"{weights_path}/{filename}"
            torch.save(cornet.state_dict(), f"{fullpath}_epoch-{e+1}.pth")
            
            print(f"Epoch {e+1} ended at: ", datetime.now())
        
        ## Visualize training
        visualize_training_progress(train_counter,
                                    train_losses, 
                                    val_counter, 
                                    val_losses, 
                                    filename)  



### ---------------------------------------------------------------------------
### DATASET FUNCTIONS

## Import a dataset and the wordlist
def import_dataset(path, classes): 
    """
    Import a ImageFolder dataset and the list of classes, applying the image transforms
    
    Parameters
    ----------
    path (str): the fullpath to the dataset  
    
    classes (str): the path to the .csv file containing the list of classes to 
                   use in training
    
    Outputs
    -------
    dataset (ImageFolder): the dataset of images arranged in one subfolder per class
        
    word_classes (list): the list of output classes to train the network
    
    """

    # Define transformations that will be applied to images 
    # - resize to 224x224 (alexnet input)
    # - convert to tensor
    # - normalize with ImageNet stats
    transform = transforms.Compose([transforms.Resize((224, 224)),  
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225]) ])
    
    
    # Load the Dataset from a structure of folders 
    dataset = torchvision.datasets.ImageFolder(root = path, transform = transform)

    # Get the classes / labels for each word
    word_classes = pd.read_csv(classes, header = None).values.tolist()

    return dataset, word_classes


## Split the dataset into training and validation and in batches
def load_dataset(dataset, tr_size, bt_size): 
    """
    Split the dataset into training and validation sets and load them using 
    DataLoader 
    
    Parameters
    ----------
    dataset (ImageFolder): the dataset to load  
    
    tr_size (float): the training size, expressed in decimals (1 is 100%)
        
    bt_size (int): the batch size
    
    Outputs
    -------
    train_loader (DataLoader): the training set loaded and divided randomly in 
                               batches of bt_size
        
    val_loader (DataLoader): the validation set loaded and divided randomly in 
                             batches of bt_size
    
    """

    # Split into training and validation
    train_size = int(tr_size * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    # Use DataLoader to create batches
    train_loader = DataLoader(train_dataset, batch_size = bt_size, shuffle = True)
    val_loader = DataLoader(val_dataset, batch_size = bt_size, shuffle = False)

    return train_loader, val_loader


## Helper function for sanity_check_dataset
def visualize_dataset_imgs(img, one_channel = False):
    """
    Plot a grid of images coming from the loaded dataset, to check visually that 
    everything is in order
    
    Parameters
    ----------
    img (array): the images to plot  
    
    Outputs
    -------
    Grid-plot of stimuli images
    
    """
    
    if one_channel:
        img = img.mean(dim = 0)
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    if one_channel:
        plt.imshow(npimg, cmap = "Greys")
    else:
        plt.imshow(np.transpose(npimg, (1, 2, 0)))


## Visualize some transformed stimuli from the train set, to make sure everything is in order
def sanity_check_dataset(train_loader): 
    """
    Control that the dataset has been loaded properly and plot images from the 
    stimuli
    
    Parameters
    ----------
    train_loader (DataLoader): the dataset to control
    
    Outputs
    -------
    Grid-plot of stimuli images (from visualize_dataset_imgs)
    
    """
    
    dataiter = iter(train_loader)
    images, labels = next(dataiter)

    # Create a grid from the images and show them
    img_grid = torchvision.utils.make_grid(images)
    visualize_dataset_imgs(img_grid, one_channel = True)



### ---------------------------------------------------------------------------
### Training and validation functions

## Train the network for one epoch
def train(model, loader, optimizer, loss_fn, device):
    """
    Train the model specified for one epoch  
    
    Parameters
    ----------
    model (torch.nn.Module): the model to train (in experiment: AlexNet or CORnet Z)  
    
    loader (DataLoader): the loaded dataset for training
                         
    optimizer (function): the optimizer function
    
    loss_fn (function): the loss function
        
    device (str): the computing unit (GPU / cuda or CPU)
    
    Outputs
    -------
    total (int): the number of samples
    
    epoch_loss (float): the loss of the training's epoch
        
    accuracy (float): the accuracy at training for this epoch
    
    """
    
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
def validate(model, loader, loss_fn, device):
    """
    Train the model specified for one epoch  
    
    Parameters
    ----------
    model (torch.nn.Module): the model to train (in experiment: AlexNet or CORnet Z)  
    
    loader (DataLoader): the loaded dataset for validation
                             
    loss_fn (function): the loss function
        
    device (str): the computing unit (GPU / cuda or CPU)
    
    Outputs
    -------
    total (int): the number of samples
    
    epoch_loss (float): the loss of the validation epoch
        
    accuracy (float): the accuracy at validation for this epoch
    
    """
    
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
### Network functions

# Replace the last layer of alexnet with a "clean" new one
def reset_last_layer(model, model_name, num_classes):
    """
    Reset the last layer to clean the classes. If CORnet, also reduce the number 
    of classes to the ones of this experiment 
    
    Parameters
    ----------
    model (torch.nn.Module): the model to modify 
    
    model_name (str): the name of the model, to distinguish operations 
                      for 'alexnet' and 'cornet'
                         
    num_classes (int): how many classes do we want in the output layer
    
    Outputs
    -------
    model (torch.nn.Module): the model with the updated layer
    
    """
    
    # Freeze all layers except the last one - just to be cautious
    for param in model.parameters():
        param.requires_grad = False
    
    # Reset the last layer, depending on the network
    # Same procedure, different specifics: we deal with the same number of classes (1000), we just want to overwrite them 
    if model_name == 'alexnet':
        
        # Get the number of features in the original layer
        num_features = model.module.classifier[6].in_features  
        
        # Replace the last layer
        model.module.classifier[6] = nn.Linear(num_features, num_classes)
        
    elif model_name == 'cornet':
            
        # Get the number of features in the original layer
        num_features = model.module.decoder.linear.in_features  
        
        # Replace the last layer
        model.module.decoder.linear = nn.Linear(num_features, num_classes)
        
    # Unfreeze the layers
    for param in model.parameters():
        param.requires_grad = True
        
    return model


    
### ---------------------------------------------------------------------------
### Logging

## Open document and set header
def init_log(log_dir, script, nSub, model_name): 
    """
    Start a loggin file for a training on one instance
    
    Parameters
    ----------
    log_dir (str): the path to the directory where the loggings are saved
        
    script (str): the script (i.e. dataset) trained 
        
    nSub (int): the number of the subject (i.e. instance) logged
    
    model_name (DataLoader): the name of the model either 'alexnet' or 'cornet'
                         
    Outputs
    -------
    csv_file (file): the log file
    
    """
    
    # get date time
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    
    # Create the filename
    csv_file = f"{log_dir}/model-{model_name}_sub-{nSub}_training-{script}_date-{timestamp}.csv"

    # Headers for the CSV file
    headers = ["Script", "subject", "Epoch", "Train_Loss", "Train_Accuracy", "Val_Loss", "Val_Accuracy"]

    # Write headers to the CSV file
    with open(csv_file, mode = "w", newline = "") as file:
        writer = csv.writer(file)
        writer.writerow(headers)
    
    return csv_file


## Add entry to log file
def log_entry(csv_file, script, subject, epoch, tr_loss, tr_acc, val_loss, val_acc): 
    """
    Add all information about one epoch to the log file
    
    Parameters
    ----------
    csv_file (file): the log file
            
    script (str): the script trained 
        
    subject (int): the number of the subject (i.e. instance) logged
    
    epoch (int): the number of the epoch (will be +1 in the file, to make it more human)
    
    tr_loss (float): the loss at training
        
    tr_acc (float): the accuracy at training
        
    val_loss (float): the loss at validation
        
    val_acc (float): the accuracy at validation
        
    Outputs
    -------
    None, updates the log file
    
    """
    
    # Appen to the csv
    with open(csv_file, mode = "a", newline = "") as file:
        writer = csv.writer(file)
        writer.writerow([script, subject, epoch+1, tr_loss, tr_acc, val_loss, val_acc])


    
### ---------------------------------------------------------------------------
### Miscellaneous
    

## Visualize training progress over the number of batches 
def visualize_training_progress(train_counter, train_loss, val_counter, val_loss, filename): 
    """
    After training the specified number of epochs, plot losses
    
    Parameters
    ----------
    train_counter (int): the number of batches trained
    
    train_loss (float): the loss at training
    
    val_counter (int): the number of batches validated
    
    val_loss (float): the loss at validation
        
    filename (str): how to save the plot
        
    Outputs
    -------
    None, updates the log file
    
    """
    
    path = '../../outputs/figures/literate/latin/'
    fullpath = f"{path}{filename}"
    
    fig = plt.figure()
    plt.plot(train_counter, train_loss, color = 'cornflowerblue', linewidth = 1.5)
    plt.scatter(val_counter, val_loss, zorder = 5, color = 'darkred')         
    plt.legend(['Train Loss', 'Validation Loss'], loc = 'upper right')
    plt.xlabel('number of training examples seen')
    plt.ylabel('negative log likelihood loss')
    
    plt.savefig(f"{fullpath}.png", dpi = 300, bbox_inches='tight')  

    plt.show()
 
    
## Reconcile 'state_dict' by modifying the keys names
def reconcile_cornet_imagenet(in_dict):
    """
    When loading the weights for ImageNet in CORnet, there is an issue of naming
    in the keys of the weights dictionary. Change names of modules keys, to 
    circumvent this problem. Rename some of the weights dictionaires keys, 
    to delete excessive 'module.' naming
    
    Parameters
    ----------
    in_dict (dict): the dictionary containing the weights to load on the network
        
    Outputs
    -------
    out_dict (dict): the same dictionary containing the weights renamed
    
    """
    
    # load_state_dict does not like the format of the ImageNet weights
    # Looking at the errors, turns out that 'state_dict' has keys in the format
    # "module.{area}.etc" instead of "module.module.{area}.etc"
    
    # Not a hack, just a renaming
    out_dict = OrderedDict((f"module.{key}", value) for key, value in in_dict.items())

    return out_dict
    

## Similarly, reconcile 'state_dict' from Agrawal and Dehaene 
def reconcile_cornet_literate_french(in_dict):
    """
    When loading the weights for french literates in CORnet, there is another 
    issue of naming in the keys of the weights dictionary. Change names of 
    modules keys, to circumvent this problem. Rename some of the weights 
    dictionaires keys, to delete excessive 'module.' naming
    
    Parameters
    ----------
    in_dict (dict): the dictionary containing the weights to load on the network
        
    Outputs
    -------
    out_dict (dict): the same dictionary containing the weights renamed
    
    """
    
    # load_state_dict does not like the format of the ImageNet weights
    # Looking at the errors, turns out that 'state_dict' has keys in the format
    # "module.{area}.etc" instead of "module.module.{area}.etc"
    
    # Not a hack, just a renaming
    # Create a new OrderedDict with modified keys
    out_dict = OrderedDict()
    for key, value in in_dict.items():
        
        # Add "module." to all the entries
        new_key = f"{key}"  
        
        # If it's a "linear" layer, specify "decoder."
        if "linear" in new_key:  
            new_key = new_key.replace("module.", "module.decoder.", 1)
            
        # Add new proper key with corresponding state
        out_dict[new_key] = value

    return out_dict
    
    