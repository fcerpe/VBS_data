#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 26 14:48:02 2024

@author: costantino_ai
"""

import os
import glob
from PIL import Image
import torch
import torch.nn as nn
from torchvision import transforms, models
from torch.utils.data import Dataset, DataLoader


## Setup functions to extract activations from DNN layers 

# Define a custom Dataset
class ImageDataset(Dataset):
    def __init__(self, images, transform=None):
        self.images = images
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        if self.transform:
            image = self.transform(image)
        return image



def get_layer_activation_alexnet(model, layer_names, batch):
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



def get_last_level_layer_names(model):
    """
    Extract the names of all last-level layers in a PyTorch neural network.

    Args:
        model (torch.nn.Module): The PyTorch model.

    Returns:
        list: A list containing the names of all last-level layers in the model.
    """
    last_level_layers = []
    for name, module in model.named_modules():
        # Check if the module is a leaf module (no children)
        if not list(module.children()):
            # Exclude the top-level module (the model itself) which is always a leaf
            if name:
                last_level_layers.append(name)

    return last_level_layers







