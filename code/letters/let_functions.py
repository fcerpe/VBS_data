#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 24 16:39:12 2025

Visual Braille Silico - support functions to investigate letters

@author: Filippo Cerpelloni
"""

import sys
sys.path.append('../')

from src.vbs_functions import *
from activations.act_functions import * 

import os
import re
import glob
import json
import urllib
import torch

import torch.nn as nn
import torchvision.models as models
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
import pandas as pd

from torchvision import transforms, models
from torchvision.io import read_image
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from scipy.spatial.distance import pdist, squareform
from natsort import natsorted


from scipy.stats import sem




### ---------------------------------------------------------------------------
### Main functions: create stimuli, get activations, do stats, plot

## Make T and S variations for selected images, then make into a dataset
def create_letters_variations(opt, fonts, t_var, s_var): 
    """
    From starting letters, make the thickness and size variations for all the 
    fonts specified. Temporarly save the images in inputs/letters/tmp_variations
    and use that folder to make a dataset, then delete the folder
    
    Parameters
    ----------
    opt (dict): output of vbs_option() containing the paths of the IODA folder 
    
    fonts (list): an array of strings containing the tags of the fonts to consider
                  F1: Arial;  F2: Times New Roman; F3: American Typewriter
                  F4: Futura; F5: Braille;  F6: Line Braille
                  
    t_var (array): the variations in thickness, expressed in number of pixels 
                   to add to / remove from the perimeter
                   
    s_var (array): the variations in size, expressed in percentages of increase
                   or decrease of the image
        
    Outputs
    -------
    A 'letters' dataset in inputs/datasets
    
    """
    
    # Extract the letters corresponding to the selected fonts 
    fonts = ['F1', 'F5', 'F6']
    t_var = [3,6]
    s_var = [15,30]
    
    # Get general path to the letters directory
    letters_dir = os.path.join(opt['inputs'], 'letters')
    letter_paths = []
    
    # Concatenate all the paths into one list 
    for f in fonts:
        f_paths = glob.glob(os.path.join(letters_dir, f'*_{f}.png'))
        letter_paths = letter_paths + f_paths
    
    # Make temporary directory to store T and S variations
    variations_dir = os.path.join(opt['inputs'], 'letters', 'tmp_variations')
    os.makedirs(variations_dir, exist_ok = True)
    
    
    # Create variations of letter thickness (T)
    for path in letter_paths: 
        
        # Extract letter information (e.g. a_F1)
        letter_info = parce_filename(path)
        
        # Open the image and create array to modify
        img = Image.open(path).convert("RGB")
        
        # Enlarge and shrink the images and save them in 'letters/variations'. Here, we create the remaining 20/25 stimuli needed
        make_T_variations(opt, path, img, t_var, letter_info)
        
    # Create size (S) variation building on thickness (T) ones
    # Redirect to temporary folder with thickness variations
    letter_paths = glob.glob(os.path.join(variations_dir, '*.png'))
    
    # Loop through all the new letters to create size (S) variations
    for path in letter_paths:
        
        # Extract script and letter to save the new image with the correct name
        letter_info = parce_filename(path)
        
        # Open the image 
        img = Image.open(path).convert("RGB")
        
        # Resize the images and save them in 'letters/variations'
        make_S_variations(img, s_var, letter_info)

    # Look inside tmp_variations and delete 'just T' variations
    # Regex pattern for files ending in T1.png to T5.png
    pattern = re.compile(r'T[1-5]\.png$')

    # Loop through files in the folder and remove those that match the pattern
    for filename in os.listdir(variations_dir):
        
        if pattern.search(filename):
            file_path = os.path.join(variations_dir, filename)
            os.remove(file_path)
    
    # Store letter variations as dataset 
    zip_folder(variations_dir)
    zip_file = variations_dir + '.zip'
    
    # Move the zipped folder
    shutil.move(zip_file, os.path.join(opt['datasets'], os.path.basename(zip_file)))


## Feed letters to the network and extract the activations 
def extract_letters_activations(opt):
    
    # Load AlexNet with ImageNet weights 
    model = models.alexnet(weights = 'IMAGENET1K_V1')
    model = nn.DataParallel(model)
    model.eval()

    # Load images from letters dataset
    letters_paths = natsorted(glob.glob(os.path.join(opt['dir']['datasets'], 'letters', '*.png')))

    # Load the images in memory
    images = [Image.open(letter).convert('RGB') for letter in letters_paths]
    
    # Extract list of just the stimuli names and save it as .csv for annotations
    labels = [path.split('/')[-1].split('.')[0] for path in letters_paths]
    labels_path = os.path.join(opt['dir']['inputs'], 'letters', 'letters_stimuli.csv')
    labels = pd.DataFrame(labels, columns = ['Stimulus'])
    labels.to_csv(labels_path, index = False)

    # Define transformations that will be applied to images 
    transform = transforms.Compose([transforms.Resize((224, 224)),  
                                    transforms.ToTensor(),  
                                    transforms.Normalize(mean = [0.485, 0.456, 0.406], 
                                                         std = [0.229, 0.224, 0.225])])

    # Create the Dataset and DataLoader
    # Shuffle is set to False and batch to 25 to ensure that we process one 
    # letter in all its variations at once
    dataset = TestDataset(annotations = os.path.join(opt['dir']['inputs'], 'letters', 'letters_stimuli.csv'), 
                          images = images, 
                          transform = transform)
    dataloader = DataLoader(dataset, batch_size = 25, shuffle = False)

    # Store activations and distances in dictionaries
    flat = {}
    stim = {}

    # Feed each batch (one letter, 25 variations of it) to the network
    for b, batch in enumerate(dataloader):
        
        # From a list of all the layers of alexnet, pick the ReLU stages
        layer_names = name_layers(model)
        layer_names = pick_layers('alexnet', layer_names)
        
        # Extract activations at any layer for the stimuli in the batch
        layer_activations = get_layer_activations(model, layer_names, batch)
        
        # Store information in a series of dictionaries, to ease stats
        # Common structure to all the dicts is:
        # dict[layer][subject][stimulus][size][x position][y position] = activation for image in layer
        flat = store_letters_activations(layer_activations, layer_names,
                                         labels, b, flat)
        
    # Compute averages for each stimulus
    stim = average_letters(flat)
    
    # Tell the user 
    print(f'Computing distances ...\n')
    
    ## Calculate euclidian distances between letters and their variations
    # Follow janini et al. (2022)'s method:
    # - compare each variation of word A with average of word B
    #   (e.g. BR_FS_1_S*_X*_Y* and BR_NW_1)
    # - average distances to obtain cell of RDM
    # - average across the diagonal (a-b and b-a)
    dist = compute_letters_distances(flat, stim)
    
    ## Save subject-specifc activations and matrices
    print('Saving activations ...\n')
    
    save_single_extraction(opt, flat, 'alexnet', 0, 'imagenet', 'letters', 'none', 
                           'letters', 
                           'flat-activations')
    
    save_single_extraction(opt, stim, 'alexnet', 0, 'imagenet', 'letters', 'none', 
                           'letters', 
                           'stimulus-average-activations')
    
    save_single_extraction(opt, dist, 'alexnet', 0, 'imagenet', 'letters', 'none', 
                           'letters', 
                           'distances_method-euclidean')
    
            
## Plot the distance matrices of a model at different stages
def plot_letters_representations(opt):
    """
    Plot the distances between letters at each layer
    
    Parameters
    ----------
    opt (dict): output of vbs_option() containing the paths of the IODA folder
    
    Outputs
    -------
    Figures for each script and layer, also saved in outputs/figures/distances
    
    """
    
    # Load the distances 
    dist = load_extraction(os.path.join(opt['dir']['results'], 'letters',
                                        'model-alexnet_sub-0_training-imagenet_test-letters_epoch-none_data-distances_method-euclidean.pkl'))
    
    # Load flat dictionary to get the keys, the labels of the stimuli
    flat = load_extraction(os.path.join(opt['dir']['results'], 'letters', 
                                        'model-alexnet_sub-0_training-imagenet_test-letters_epoch-none_data-flat-activations.pkl'))
    
    layers = list(flat.keys())
    stim_labels = list(flat[layers[0]].keys())
        
    # Iterate through the layers computed
    for p, layer in enumerate(flat.keys()):
    
        # Start plotting 
        plt.figure()
        
        # Letters labels
        labels = ['A','B','C','D','E','F','G','H','I','J','K','L','M', 
                  'N','O','P','Q','R','S','T','U','V','W','X','Y','Z'] 
        
        # Data to plot
        matrix = dist[layer]
        
        # Modify so order of fonts is Braille - Line - Latin
        

        # Set specific and class labels
        repeated_labels = stim_labels * 3

        # Create new labels with sequence indicators
        classes = ['LT', 'BR', 'LN']
        new_labels = []
        for l in range(1):
            new_labels.extend([label for label in stim_labels])
                
        ax = sns.heatmap(matrix, 
                         cmap = 'viridis', 
                         annot = False, 
                         xticklabels = False, 
                         yticklabels = False)

        # Customize the heatmap
        title = f'layer-{l}_{layer}'
        ax.set_title(title, fontsize = 15, pad = 20)

        # Add sequence indicators as subtitles for the axis
        for j, cla in enumerate(classes):
            
            plt.text(-8, j*26 +13, cla, rotation = 0, fontsize = 12, verticalalignment = 'center')
            plt.text(j*26 +13, 84, cla, rotation = 0, fontsize = 12, horizontalalignment = 'center')

        ax.yaxis.set_tick_params(rotation = 0)
        for label in ax.get_yticklabels():
            label.set_verticalalignment('center')

        ax.set_aspect('equal')
        
        # Add squares to separate clusters
        num_clusters = len(classes)  # Assuming clusters correspond to the classes
        cluster_size = 26
        for i in range(num_clusters):
            for j in range(num_clusters):
                rect = patches.Rectangle((j * cluster_size, i * cluster_size),  # (x, y) starting point
                                          cluster_size,  # Width
                                          cluster_size,  # Height
                                          linewidth = 1, 
                                          edgecolor = 'white', 
                                          facecolor = 'none') 
                ax.add_patch(rect)

        # Save plot
        savename = f'model-alexnet_test-letters_layer-{p}_plot-rdm.png'
        savepath = os.path.join(opt['dir']['figures'], 'letters', savename)
        plt.savefig(savepath, dpi = 600)

        # Show the plot
        plt.show()  

        

### ---------------------------------------------------------------------------
### Extraction functions

## Store the activations in nested dictionaries
def store_letters_activations(activations, layers, labels, b, flat):
    """
    Store the activations in a orderly manner that considers the layer, the subject,
    the type of activation to save, the stimulus

    Parameters
    ----------
    activations: the activations in a layer in response of the images
    
    layers (list): the list of layer names
    
    labels (list): the list of image names
    
    b (int): the number of batch processed, to compute whether to create 
             new sub-dictionaries
    
    flat (dict): the dictionary storing all the flat activations

    Outputs
    ----------
    raw (dict): the dictionary storing all the raw activations, now updated
                with the activations from this batch
    
    flat (dict): the dictionary storing all the flat activations now updated
                 with the activations from this batch
    """

    # Get the label of the stimulus presented, with size variation
    letter = labels['Stimulus'][b*25]
    parts = letter.split('_')
    let = parts[0]
    font = parts[1]
    stim = f'{let}_{font}'

    # If it's the first batch of this stimulus, initialize the dictionaries
    if b == 0:  init_layer = True
    else: init_layer = False
    
    # For each layer / stage
    for l, layer in enumerate(layers):

        # Get the activations
        stage = activations[layers[l]]

        # First batch is a_F1, always. Initialize layer key and Latin key
        if init_layer: 
            flat[layer] = {}
        
        # Letter will always be a new entry
        flat[layer][stim] = {}

        # Assign the 25 rows of each batch to the corresponding variations
        t_labels = [f'T{i}' for i in range(1, 6)]  # T1 to T5
        s_labels = [f'S{i}' for i in range(1, 6)]   # S1 to S5

        # Flatten the activations to a single vector for each image
        # e.g. layer 5: 256x13x13 -> 256x1, sum of the 13x13 activation maps across the features
        if stage.dim() > 2:
            stage = torch.sum(stage, (3,2))

        # Move flattened tensor to numpy (again)
        stage = stage.numpy()

        # Save 'flattened' activations
        for idx, (t, s) in enumerate([(t, s) for t in t_labels for s in s_labels]):
            flat[layer][stim].setdefault(t, {})[s] = stage[idx]

    return flat


## Custom class for the test datasets
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


## Average letters activations to ease distances
def average_letters(flat):
    """
    Compute the average activation for one stimulus across all the variations 
    in size, position in which it is presented to the network
    
    Parameters
    ----------
    flat (dict): the dictionary with arrays of activations for each 
                 stimulus variation

    Outputs
    ----------
    out (dict): the dictionary containing the averages, one array for one 
                stimulus across all it's variations
    """

    # Copy the structure of the flat dictionary
    # Scroll through every stimulus and get all the values out and averaged
    out = {}

    for layer in flat.keys():

        # Initialize layer dictionary
        out[layer] = {}
        
        # go into each variation and extract the activation array
        for letter in flat[layer].keys():
            
            for t in flat[layer][letter].keys():
                activations = []
                
                for s in flat[layer][letter][t].values():

                    # Append the values to a numpy list, to ease averaging
                    activations.append(np.array(s))

            out[layer][letter] = np.mean(activations, axis = 0)

    return out


# Compute distances between the letters
def compute_letters_distances(activations, averages):
    """
    Compute the distance between stimuli of the same script
    
    Parameters
    ----------
    activations (dict): the activations for single images
    
    averages (dict): the activations for single stimuli, averaged across the images
    
    Outputs
    ----------
    distances (dict): the dictionary containing the distances between stimuli
    
    """

    # Initiate a dictionary to contain the distances between elements in a given layer
    dist = {}

    # Extract from the dictionary the number and name of each layer
    layer_names = list(averages.keys())

    # Latin alphabet - get the stimuli in each layer
    stim_names = list(averages[layer_names[0]].keys())
    
    # Order names, just in case
    stim_names = natsorted(stim_names)

    # Loop through layers of the datasets
    for i, layer in enumerate(layer_names):
        
        # Get the number of stimuli
        stim_nb = len(stim_names)

        # Initiate a matrix of len(nb of stimuli) to store the RDM values
        matrix = np.full((stim_nb, stim_nb), np.nan)
        
        # Get activations
        stimuli = activations[layer]

        # Browse through stim A
        for r, stim_a in enumerate(stim_names):
            
            # Iniitialize array with the different variations
            stim_a_variations = []
            
            # Extract activation for all the variants in a np list (like for assignment in storing)
            var = 0

            # Go into each variation and extract the activation array
            for t in stimuli[stim_a].keys():
                for s in stimuli[stim_a][t].keys():
        
                    # Append the values to a numpy list, to ease averaging
                    # stim_a_variations.append(np.array(y))
                    stim_a_variations.append(np.array(stimuli[stim_a][t][s]))
                    var = var +1

            # Browse through stim B
            for c, stim_b in enumerate(stim_names):

                # Extract average activation
                stim_b_average = averages[layer][stim_b]

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
                
        dist[layer] = matrix

    return dist



### ---------------------------------------------------------------------------
### Stimuli creation functions

## Make thickness variations
def make_T_variations(opt, path, image, thicknesses, stim_info):
    """
    Make specified variations in thickness (number of pixels to add to / remove
    from the perimeter) for all the images specified.
    IMPORTANT: the function is specific to the experiment and works if only 2 
               variations are given as input

    Parameters
    ----------
    opt (dict): output of vbs_option() containing the paths of the IODA folder 
    
    path (str): the path to the specific image to manipulate
    
    image (array): the image loaded as an array of pixels
    
    thicknesses (array): the list of thickness variations to consider
    
    stim_info (str): the identity of the letter (e.g. a_F1)
        
    Outputs
    -------
    Images variated in thickness, saved in inputs/letters/tmp_variations
    
    """
    
    # Only made to create 5 stimuli from [X,Y] t_var array
    # T1 = -Y steps; T2 = -X steps; T3 = original; T4 = +X steps; T5 = +Y steps
    
    # Set reference colors
    ref_white = [255, 255, 255]
    ref_black = [0,   0,   0]
    
    # Set desitantion folder
    dest_folder = os.path.join(opt['inputs'], 'letters', 'tmp_variations')
    
    # Set input image as array, easier to work with
    image_array = np.array(image)
    
    # Create copies to avoid overwriting information
    shrink_array = np.array(image)
    expand_array = np.array(image)
    
    # Cast non-white pixels as black, to simplify the tasks
    non_black = np.any(image_array != ref_white, axis=-1)
    
    image_array[non_black] = ref_black
    image_array = correct_latin_letters(image_array, stim_info)
    
    # Create copies of the image to avoid overwriting of information
    shrink_array = copy_image(path, image, non_black, stim_info)
    expand_array = copy_image(path, image, non_black, stim_info)
    
    # Save T3 image (mid-thickness, a.k.a. original format)
    black_image = Image.fromarray(image_array.astype('uint8'), 'RGB')
    blackname = os.path.join(dest_folder, f'/{stim_info}_T3.png')
    black_image.save(blackname)
    
    
    # Expansion
    # Set counter to track variations
    plus = 1
    
    # Set neighbours algorithm: 
    # - 8N will consider the side and diagonal neighbours
    # - 4N will only consider the sides
    # Algorithm changes throughout the expansion and shirking of stimuli and 
    # is based on extensive piloting on the effects of either on the images
    neighbors = '8N'
    
    # Iterate through the T variations
    for i, t in enumerate(thicknesses):
        
        # until it reaches the desired thickness 
        while plus < t:
        
            # identify which pixels to change
            pixels_to_change = identify_pixels(expand_array, 'white', [], neighbors)
            
            # Paint them black
            for pixel in pixels_to_change: expand_array[pixel[0], pixel[1]] = ref_black
            
            # Update counter
            plus = plus+1
        
        # save image in state-of-the-art folder with change notation
        colored_image = Image.fromarray(expand_array)
        colored_savename = os.path.join(dest_folder, f'/{stim_info}_T{i+4}.png')
        colored_image.save(colored_savename)
        
        
    # Reduction 
    # Set counter
    minus = 1
    
    # There are a lot of issues with corners when we reduce the perimeter. 
    # Tailor the algorithm to the script and letters reduced
    parts = stim_info.split('_')
    font = parts[1]
    letter = parts[0]
    
    # Braille has always the same algorithm
    if font == 'F5': neighbors = '8N'
        
    # Line Braille changes based on the letter
    elif font == 'F6':
        
        # special 45 degrees angles: will alternate between 4N and 8N
        if letter in ['n','s','z']:   
            ngb_counter = 0
            sequence = ['4N','4N','8N','4N','8N']
            neighbors = sequence[ngb_counter]

        # special 66 degrees angles: : will alternate between 4N and 8N
        elif letter in ['m','u','x']: 
            ngb_counter = 0 
            sequence = ['4N','4N','8N']
            neighbors = sequence[ngb_counter]
        
        # special 90 degre angle, titled
        elif letter == 'o': 
            neighbors = '8N' 
        
        # all the remaining letters
        else: neighbors = '4N' 
    
    # Latin script has a standard algorithm        
    else: neighbors = '4N' 
        
    # Iterate through the T variations
    for i, t in enumerate(thicknesses): 
        
        # Loop until we reach a desired number of steps, then save
        while minus < t:
        
            # identify which pixels to change
            pixels_to_change = identify_pixels(shrink_array, 'black', font, neighbors)
            
            # Paint them white
            for pixel in pixels_to_change: shrink_array[pixel[0], pixel[1]] = ref_white
            
            # Update iteration counter
            minus = minus+1
            
            # For peculiar letters in line script, update the neighbor choice
            if font == 'F6' and letter in ['n','s','z','m','u','x']:
                
                # 45 degrees angles
                if letter in ['n','s','z']:
                    if ngb_counter == 4: ngb_counter = 0
                    else: ngb_counter = ngb_counter +1
                        
                    neighbors = sequence[ngb_counter]
    
                # 66 degrees angles
                if letter in ['m','u','x']:
                    if ngb_counter == 2: ngb_counter = 0
                    else: ngb_counter = ngb_counter +1
                        
                    neighbors = sequence[ngb_counter]
        
        # save image in state-of-the-art folder with change notation
        colored_image = Image.fromarray(shrink_array)
        colored_savename = os.path.join(dest_folder, f'/{stim_info}_T{2-i}.png')
        colored_image.save(colored_savename)


# Create the size variations
def make_S_variations(path, image, sizes, stim_info):
    """
    Make specified variations in size (percentage of increase / decrease) for 
    all the images specified.
    IMPORTANT: the function is specific to the experiment and works if only 2 
               variations are given as input

    Parameters
    ----------
    opt (dict): output of vbs_option() containing the paths of the IODA folder 
    
    path (str): the path to the specific image to manipulate
    
    image (array): the image loaded as an array of pixels
    
    sizes (array): the list of size variations to consider
    
    stim_info (str): the identity of the letter (e.g. a_F1)
        
    Outputs
    -------
    Images variated in thickness, saved in inputs/letters/tmp_variations
    
    """
    
    # Define size variations 
    # S1 = - Y%; S2 = - X%; S3 = original; S4 = + X%; S5 = + Y%
    
    # Set desitantion folder
    dest_folder = os.path.join(opt['inputs'], 'letters', 'tmp_variations')
    
    # Assuming images passed are squared (they should be), use the original side 
    # to crop enlarged /shrank images
    resize_factor = image.width
    
    # Just adjust name and save original
    cropped_savename = f'{dest_folder}/{stim_info}_S3.png'
    image.save(cropped_savename)
    
    # Each size correspond to two variations, apply them both 
    for i, s in enumerate(sizes): 
        
        # Make letter bigger by the increase factor
        increase = 1+s/100

        # Resize image
        bigger_image = image.resize((int(image.width * increase), 
                                     int(image.height * increase)), 
                                    Image.LANCZOS)
    
        # Calculate the cropping coordinates from the center, to get 500x500px 
        left = (bigger_image.width - resize_factor) / 2
        top = (bigger_image.height - resize_factor) / 2
        right = (bigger_image.width + resize_factor) / 2
        bottom = (bigger_image.height + resize_factor) / 2
        
        # Crop image
        cropped_image = bigger_image.crop((left, top, right, bottom))
        
        # Adjust filename to note the change
        cropped_savename = f'{dest_folder}/{stim_info}_S{4+i}.png'
        
        # Save image
        cropped_image.save(cropped_savename)
        
        
        # Make letter smaller by a decrease factor 
        decrease = 1-s/100

        # Resize
        smaller_image = image.resize((int(image.width * decrease), 
                                      int(image.height * decrease)), 
                                     Image.LANCZOS)
        
        # Create new white background with the same size as the original image
        background = Image.new("RGB", (image.width, image.height), (255, 255, 255))
        
        # Calculate where to paste the resized image to be at the center of the background
        x_offset = int(image.width - smaller_image.width) // 2
        y_offset = int(image.height - smaller_image.height) // 2
        
        # Paste the resized image onto the white background
        background.paste(smaller_image, (x_offset, y_offset))

        # Calculate the coordinates to crop the image to 500x500 from the center
        left = (background.width - resize_factor) / 2
        top = (background.height - resize_factor) / 2
        right = (background.width + resize_factor) / 2
        bottom = (background.height + resize_factor) / 2
        
        # Crop image
        cropped_image = background.crop((left, top, right, bottom))
        
        # Adjust filename to note the change
        cropped_savename = f'{dest_folder}/{stim_info}_S{2-i}.png'
        
        # Save image
        cropped_image.save(cropped_savename)


# TODO document functions from here to bottom (move todo as checkpoint)

# Identify which pixels correspond to a contour of a given array and a given colour to check
def identify_pixels(image_array, ref_flag, script, ngb):
    
    # Define target color (the one to change) and anti-target (the one to control for)
    # Shrinking uses target = 'black', to then change them to white
    # Expansion uses target = 'white', to change them to black
    if ref_flag == 'white':
        target = [255, 255, 255]
        anti_target = [0, 0, 0]
    elif ref_flag == 'black':
        target = [0, 0, 0]
        anti_target = [255, 255, 255]

    # Get the dimensions of the image
    rows, cols, _ = image_array.shape

    # Create a list to store the coordinates of the pixels
    border_pixels = []
    
    # Speed-up script:
    # We know that letters are placed at the center of the 500x500px image
    # The same applies to the 1430x1430px words 
    # Based on the case (i.e. the sizes of the images), restrict the range in which
    # to look for specific pixels to predifined 250x250 for letters and 300x1430 for words
    if cols == 500: 
        range_rows = range(124,374)
        range_cols = range(124,374)
        
    elif cols == 1859: 
        range_rows = range(784,1055)
        range_cols = range(22,1838)
        

    # Check each pixel (slow algorithm in expansion, there are a lot of white pixels in the border that are not relevant)
    for row in range_rows:
        for col in range_cols:
            
            # If they match the target color
            if has_color_pixels(image_array[row, col], target):
                
                ## Intercept special cases, only if within a reasonable window
                if 1 < row < rows-2 and 1 < col < cols-2:
                    
                    # Extract small cutout 
                    cutout_five = extract_cutout(image_array, row, col, 5)
                       
                    # Special case: LINE SCRIPT LETTER O
                    if script == 'ln': 
                        match, coords = case_line_o(cutout_five, row, col)
                        if match: 
                            for c in coords: border_pixels.append(c)    
                            
                        # Special case: algorithm is 8N, acute angle 
                        if ngb == '8N': 
                            match, coords = case_66deg_8N_corner(cutout_five, row, col)
                            if match: 
                                for c in coords: border_pixels.append(c)    
                            
                        # Special cases: more generally, deal with corners
                        if case_90deg_corner(cutout_five): 
                            border_pixels.append((row, col))
                            
                            # Extract larger cutout to check for acute corners
                            cutout_nine = extract_cutout(image_array, row, col, 9)
                            match, coords = case_66deg_corner(cutout_nine, row, col)
                            if match:
                                for c in coords: border_pixels.append(c)
                
                    # Special cases: LATIN SCRIPT
                    if script in ['F1', 'F2', 'F3', 'F4']: 
                        
                        # Check for line-curve corners, like the one in 'b' 
                        cutout_seven = extract_cutout(image_array, row, col, 7)
                        match_round, coords_round = case_round_corner(cutout_seven, row, col)
                        if match_round:
                            for c in coords_round: border_pixels.append(c)
                            
                        # Check for sharp corner
                        match_sharp_y, coords_sharp_y = case_y_corner(cutout_five, row, col)
                        if match_sharp_y: 
                            # that is also not a round corner, otherwise it gets messy
                            if not match_round:
                                for c in coords_sharp_y: border_pixels.append(c)
                         
                        # more generally, deal with corners (especially valid in T and F)
                        if case_90deg_corner(cutout_five): 
                            
                            # Check that it's not actually a sharp corner
                            match_sharp_v, coords_sharp_v = case_v_corner(cutout_seven, row, col)
                            
                            # If it's a sharp corner, overwrite decision and deal with this case
                            # If not, proceed with the round corner
                            if match_sharp_v: 
                                for c in coords_sharp_v: border_pixels.append(c)
                            else:
                                border_pixels.append((row, col))
                            
                    
                # Special cases are accounted for, continue with normal variations
                # Pick the neighbours algorithm: 
                if ngb == '4N': 
                    # 4 directions: up, down, left, right
                    neighbors = [(row-1, col), (row, col-1), (row, col+1), (row+1, col)]
                else: 
                    # 8 directions, include corners
                    neighbors = [(row-1, col), (row-1, col-1), (row-1, col+1),
                                 (row, col-1),                 (row, col+1),
                                 (row+1, col), (row+1, col-1), (row+1, col+1)]
                
                # Within the neighbors, look for anti-target color
                for n_row, n_col in neighbors:
                    
                    if 0 <= n_row < rows and 0 <= n_col < cols:
                        
                        if has_color_pixels(image_array[n_row, n_col], anti_target):
                            
                            border_pixels.append((row, col))
                            break
    return border_pixels


# Correct pixel conversion of latin letters. 
# In some cases (a,e,h,m,v,w) the transformation from affinity to .png resulted in wrong pixels
def correct_latin_letters(array, letter_info):
    
    # A,H,M need an extra pixel on the lower part of the curves, to avoid indents
    # E needs an extra pixel just below the straight line, to avoid indent
    # V needs an extra pixel to avoid being caught in other cases when shrinking
    # W needs extra white pixels to conform the lower corners
    if letter_info.endswith('_F6'):
        if letter_info.startswith('a'): array[(210,273)] = [0,0,0]
        elif letter_info.startswith('e'): array[(256,214)] = [0,0,0]
        elif letter_info.startswith('h'): array[(211,271)] = [0,0,0]
        elif letter_info.startswith('m'): array[(205,211),(256,196)] = [0,0,0]
        elif letter_info.startswith('v'): array[(284,249)] = [0,0,0]
        elif letter_info.startswith('w'): array[(279,280),(285,285)] = [255,255,255]
        
    return array
        

# Copy image to shrink and expand
def copy_image(path, image, non_black, letter):
    
    output_image = Image.open(path).convert("RGB")
    output_array = np.array(output_image)
    output_array[non_black] = [0,0,0]
    output_array = correct_latin_letters(output_array, letter)
    
    return output_array


# Special case of thickness variations
## Identify a 90 degrees corner and remove an extra pixel
def case_90deg_corner(array):
    
    # IMPORTANT: input array must be 5x5 
    # Look for the following case (_ = white, B = black, X = black pixel target):
    # _ _ B B B
    # _ _ B B B
    # B B X B B
    # B B B B B
    # B B B B B
    # If present, X will be removed
    
    # Create a reference of how the corner is represented in the pixel array 
    # 2x2 white square on the top-left corner is our target
    reference_orientation = np.array([[[255,255,255],[255,255,255],[0,0,0],[0,0,0],[0,0,0]],
                                      [[255,255,255],[255,255,255],[0,0,0],[0,0,0],[0,0,0]],
                                      [[0,0,0],      [0,0,0],      [0,0,0],[0,0,0],[0,0,0]],
                                      [[0,0,0],      [0,0,0],      [0,0,0],[0,0,0],[0,0,0]],
                                      [[0,0,0],      [0,0,0],      [0,0,0],[0,0,0],[0,0,0]]], dtype=np.uint8)
    
    # Rotate the input array until it matches the reference 
    for k in range(4):
        rotated_array = np.rot90(array, k = k)
        
        if np.array_equal(rotated_array, reference_orientation):
            break
    
    # Check that the upper-left corner is a 2x2 white patch AND that the rest is black
    white_pos = [(0,0),(0,1),(1,0),(1,1)]
    white_check = all((rotated_array[i, j] == [255, 255, 255]).all() for i, j in white_pos)

    black_pos = [(0,2),(0,3),(0,4),(1,2),(1,3),(1,4),(2,0),(2,1),(2,2),(2,3),(2,4),(3,0),(3,1),(3,2),(3,3),(3,4),(4,0),(4,1),(4,2),(4,3),(4,4)]
    black_check = all((rotated_array[i, j] == [0, 0, 0]).all() for i, j in black_pos)
    
    return (white_check and black_check)


# Shrinking - special case:
# Identify if a 66° degrees corner is present in a given cutoput and artifically remove extra pixels.
# It prevents the formation of triangles of left-over pixels when shrinking particular letter of the Line Braille script
# IMPORTANT: input array must be 9x9 
def case_66deg_corner(array, row, col):
    
    # This case requires actual coordinates relative to the center of the given cutout. It will not rotate the figure 
    # but elaborate case-by-case the three corner situations (there is no Line Braille letter with a bottom-left corener, will skip that case)
    
    # Example (Xs will be removed): 
    # _ _ _ B B B B B B
    # _ _ _ _ B B B B B
    # _ _ _ _ B B B B B
    # _ _ _ _ B X B B B
    # B B B B C X B B B
    # B B B B B B B B B
    # B B B B B B B B B
    # B B B B B B B B B
    # B B B B B B B B B
    
    # Check if it's an upper-left acute corner
    white_up_left = [(5,5),(5,6),(5,7),(5,8),(6,5),(6,6),(6,7),(6,8),(7,5),(7,6),(7,7),(7,8),(8,6),(8,7),(8,8)]
    black_up_left = [(0,0),(0,1),(0,2),(0,3),(0,4),(0,5),(0,6),(0,7),(0,8),(1,0),(1,1),(1,2),(1,3),(1,4),(1,5),(1,6),(1,7),(1,8),(2,0),(2,1),(2,2),(2,3),(2,4),
                     (2,5),(2,6),(2,7),(2,8),(3,0),(3,1),(3,2),(3,3),(3,4),(3,5),(3,6),(3,7),(3,8),(4,0),(4,1),(4,2),(4,3),(4,4),(4,5),(4,6),(4,7),(4,8),(5,0),
                     (5,1),(5,2),(5,3),(5,4),(6,0),(6,1),(6,2),(6,3),(6,4),(7,0),(7,1),(7,2),(7,3),(7,4),(8,0),(8,1),(8,2),(8,3),(8,4),(8,5)]

    if all((array[i, j] == [255, 255, 255]).all() for i, j in white_up_left) and all((array[i, j] == [0,0,0]).all() for i, j in black_up_left):
        # Specify which voxels should be eroded
        coords = [(row,col-1), (row+1,col-1)]
        return True, coords
    
    # Not UL, rotate reference and check upper-right 
    white_up_right = [(5,0),(5,1),(5,2),(5,3),(6,0),(6,1),(6,2),(6,3),(7,0),(7,1),(7,2),(7,3),(8,0),(8,1),(8,2)]
    black_up_right = [(0,0),(0,1),(0,2),(0,3),(0,4),(0,5),(0,6),(0,7),(0,8),(1,0),(1,1),(1,2),(1,3),(1,4),(1,5),(1,6),(1,7),(1,8),(2,0),(2,1),(2,2),(2,3),(2,4),
                      (2,5),(2,6),(2,7),(2,8),(3,0),(3,1),(3,2),(3,3),(3,4),(3,5),(3,6),(3,7),(3,8),(4,0),(4,1),(4,2),(4,3),(4,4),(4,5),(4,6),(4,7),(4,8),(5,4),
                      (5,5),(5,6),(5,7),(5,8),(6,4),(6,5),(6,6),(6,7),(6,8),(7,4),(7,5),(7,6),(7,7),(7,8),(8,3),(8,4),(8,5),(8,6),(8,7),(8,8)]

    if all((array[i, j] == [255, 255, 255]).all() for i, j in white_up_right) and all((array[i, j] == [0,0,0]).all() for i, j in black_up_right):
        # Specify which voxels should be eroded
        coords = [(row,col+1), (row+1,col+1)]
        return True, coords
    
    # Neither UR, rotate and check lower-right
    white_low_right = [(0,0),(0,1),(0,2),(1,0),(1,1),(1,2),(1,3),(2,0),(2,1),(2,2),(2,3),(3,0),(3,1),(3,2),(3,3)]
    black_low_right = [(0,3),(0,4),(0,5),(0,6),(0,7),(0,8),(1,4),(1,5),(1,6),(1,7),(1,8),(2,4),(2,5),(2,6),(2,7),(2,8),(3,4),(3,5),(3,6),(3,7),(3,8),(4,0),
                       (4,1),(4,2),(4,3),(4,4),(4,5),(4,6),(4,7),(4,8),(5,0),(5,1),(5,2),(5,3),(5,4),(5,5),(5,6),(5,7),(5,8),(6,0),(6,1),(6,2),(6,3),(6,4),
                       (6,5),(6,6),(6,7),(6,8),(7,0),(7,1),(7,2),(7,3),(7,4),(7,5),(7,6),(7,7),(7,8),(8,0),(8,1),(8,2),(8,3),(8,4),(8,5),(8,6),(8,7),(8,8)]

    if all((array[i, j] == [255, 255, 255]).all() for i, j in white_low_right) and all((array[i, j] == [0,0,0]).all() for i, j in black_low_right):
        # Specify which voxels should be eroded
        coords = [(row,col+1), (row-1,col+1)]
        return True, coords
    
    # No acute angle (we don't have BL)
    return False, []


# Shrinking - special case: 
# Identify if a 66° degrees corner is present in a given cutout when using the '8N' algorithm.
# It prevents the formation of triangles of left-over pixels when shrinking particular letter of the Line Braille script
# IMPORTANT: input array must be 5x5 
def case_66deg_8N_corner(array, row, col):
    
    # B B B B B
    # B B B B B
    # _ _ X B B
    # _ _ B B B
    # _ B B B B
    # X will be removed, to avoid residuals
    
    # Same procedure of is_33deg_corner: switch angle based on where there is a white pixel 
    # upper-left corner
    white_UL = [(2,3),(2,4),(3,3),(3,4),(4,4)]
    black_UL = [(0,0),(0,1),(0,2),(0,3),(0,4),(1,0),(1,1),(1,2),(1,3),(1,4),(2,0),(2,1),(2,2),(3,0),(3,1),(3,2),(4,0),(4,1),(4,2),(4,3)]
    
    if all((array[i, j] == [255, 255, 255]).all() for i, j in white_UL) and all((array[i, j] == [0,0,0]).all() for i, j in black_UL):
        # Specify which voxels should be eroded
        coords = [(row-1,col-1), (row,col-1)]
        return True, coords
    
    # upper-right corner
    white_UR = [(2,0),(2,1),(3,0),(3,1),(4,0)]
    black_UR = [(0,0),(0,1),(0,2),(0,3),(0,4),(1,0),(1,1),(1,2),(1,3),(1,4),(2,2),(2,3),(2,4),(3,2),(3,3),(3,4),(4,1),(4,2),(4,3),(4,4)]
    
    if all((array[i, j] == [255, 255, 255]).all() for i, j in white_UR) and all((array[i, j] == [0,0,0]).all() for i, j in black_UR):
        # Specify which voxels should be eroded
        coords = [(row-1,col+1), (row,col+1)]
        return True, coords
    
    # bottom-right corner
    white_BR = [(0,0),(1,0),(1,1),(2,0),(2,1)]
    black_BR = [(0,1),(0,2),(0,3),(0,4),(1,2),(1,3),(1,4),(2,2),(2,3),(2,4),(3,0),(3,1),(3,2),(3,3),(3,4),(4,0),(4,1),(4,2),(4,3),(4,4)]
    
    if all((array[i, j] == [255, 255, 255]).all() for i, j in white_BR) and all((array[i, j] == [0,0,0]).all() for i, j in black_BR):
        # Specify which voxels should be eroded
        coords = [(row,col+1), (row+1,col+1)]
        return True, coords

    # No acute angle (we don't have BL)
    return False, []


# Shrinking - special case: 
# Identify if a rotated 90 degrees corner is present in a given cutout
# It prevents the formation of triangles in the letter O in the Line Braille script
# IMPORTANT: input array must be 5x5 
def case_line_o(array, row, col):
    
    # B B B B B 
    # _ B B B B
    # _ _ X B B
    # _ _ X B B
    # _ B B B B
    # Xs will be removed, to avoid residuals
    
    # Check if it's an upper-left acute corner
    white_ref = [(1,0),(2,0),(2,1),(3,0),(3,1),(4,0)]
    black_ref = [(0,0),(0,1),(0,2),(0,3),(0,4),(1,1),(1,2),(1,3),(1,4),(2,2),(2,3),(2,4),(3,2),(3,3),(3,4),(4,1),(4,2),(4,3),(4,4)]

    if all((array[i, j] == [255, 255, 255]).all() for i, j in white_ref) and all((array[i, j] == [0,0,0]).all() for i, j in black_ref):
        
        # Specify which voxels should be eroded
        coords = [(row,col+1), (row+1,col+1)]
        return True, coords
    
    else:
        return False, []


# Latin script - special cases: 
# Identify corners between a straight vertical line and a rounded one (like b)
# Prevent the formation of left-over pixels
# IMPORTANT: input array must be 7x7
def case_round_corner(array, row, col): 
    
    # Looks for the pattern of _ (white), and removes X. ? are ignored, as they change for different cases
    # Example (Xs will be removed): 
    # B B B _ _ ? B 
    # B B B _ ? B B 
    # B B B _ B B B 
    # B B X C B B B 
    # B B X B B B B 
    # B B B B B B B 
    # B B B B B B B 
    
    # Check if it's an upper-left acute corner (like d)
    white_UL = [(0,2),(0,3),(1,3),(2,3)]
    black_UL = [(0,0),(0,4),(0,5),(0,6),(1,0),(1,1),(1,4),(1,5),(1,6),(2,0),(2,1),(2,2),(2,4),(2,5),(2,6),(3,0),(3,1),(3,2),(3,3),(3,4),(3,5),
                (3,6),(4,0),(4,1),(4,2),(4,3),(4,4),(4,5),(4,6),(5,0),(5,1),(5,2),(5,3),(5,4),(5,5),(5,6),(6,0),(6,1),(6,2),(6,3),(6,4),(6,5),(6,6)]

    if all((array[i, j] == [255, 255, 255]).all() for i, j in white_UL) and all((array[i, j] == [0,0,0]).all() for i, j in black_UL):
        # Specify which voxels should be eroded
        return True, [(row,col+1),(row+1,col+1)]
    
    # Not UL, rotate reference and check upper-right 
    white_UR = [(0,3),(0,4),(1,3),(2,3)]
    black_UR = [(0,0),(0,1),(0,2),(0,6),(1,0),(1,1),(1,2),(1,5),(1,6),(2,0),(2,1),(2,2),(2,4),(2,5),(2,6),(3,0),(3,1),(3,2),(3,3),(3,4),(3,5),(3,6),
                (4,0),(4,1),(4,2),(4,3),(4,4),(4,5),(4,6),(5,0),(5,1),(5,2),(5,3),(5,4),(5,5),(5,6),(6,0),(6,1),(6,2),(6,3),(6,4),(6,5),(6,6)]

    if all((array[i, j] == [255, 255, 255]).all() for i, j in white_UR) and all((array[i, j] == [0,0,0]).all() for i, j in black_UR):
        # Specify which voxels should be eroded
        return True, [(row,col-1),(row+1,col-1)]
    
    # Neither UR, rotate and check lower-right
    white_LR = [(4,3),(5,3),(6,3),(6,4)]
    black_LR = [(0,0),(0,1),(0,2),(0,3),(0,4),(0,5),(0,6),(1,0),(1,1),(1,2),(1,3),(1,4),(1,5),(1,6),(2,0),(2,1),(2,2),(2,3),(2,4),(2,5),(2,6),
                (3,0),(3,1),(3,2),(3,3),(3,4),(3,5),(3,6),(4,0),(4,1),(4,2),(4,4),(4,5),(4,6),(5,0),(5,1),(5,2),(5,5),(5,6),(6,0),(6,1),(6,2),(6,6)]

    if all((array[i, j] == [255, 255, 255]).all() for i, j in white_LR) and all((array[i, j] == [0,0,0]).all() for i, j in black_LR):
        # Specify which voxels should be eroded
        return True, [(row,col-1),(row-1,col-1)]
    
    # Check lower-left
    white_LL = [(4,3),(5,3),(6,2),(6,3)]
    black_LL = [(0,0),(0,1),(0,2),(0,3),(0,4),(0,5),(0,6),(1,0),(1,1),(1,2),(1,3),(1,4),(1,5),(1,6),(2,0),(2,1),(2,2),(2,3),(2,4),(2,5),(2,6),
                (3,0),(3,1),(3,2),(3,3),(3,4),(3,5),(3,6),(4,0),(4,1),(4,2),(4,4),(4,5),(4,6),(5,0),(5,1),(5,4),(5,5),(5,6),(6,0),(6,4),(6,5),(6,6)]

    if all((array[i, j] == [255, 255, 255]).all() for i, j in white_LL) and all((array[i, j] == [0,0,0]).all() for i, j in black_LL):
        # Specify which voxels should be eroded
        return True, [(row,col+1),(row-1,col+1)]
    
    # Extra cases: check Z corners
    # Z(Z)-top corner
    white_ZT = [(1,6),(2,5),(2,6),(3,4),(3,5),(3,6)]
    black_ZT = [(0,0),(0,1),(0,2),(0,3),(0,4),(0,5),(1,0),(1,1),(1,2),(1,3),(1,4),(2,0),(2,1),(2,2),(2,3),(2,4),(3,0),(3,1),(3,2),(3,3),
                (4,0),(4,1),(4,2),(4,3),(4,4),(4,5),(4,6),(5,0),(5,1),(5,2),(5,3),(5,4),(5,5),(5,6),(6,0),(6,1),(6,2),(6,3),(6,4),(6,5),(6,6)]

    if all((array[i, j] == [255, 255, 255]).all() for i, j in white_ZT) and all((array[i, j] == [0,0,0]).all() for i, j in black_ZT):
        # Specify which voxels should be eroded
        return True, [(row+1,col-1),(row+1,col)]
    
    # Z-corner
    white_ZB = [(3,0),(3,1),(3,2),(4,0)]
    black_ZB = [(0,0),(0,1),(0,2),(0,3),(0,4),(0,5),(0,6),(1,0),(1,1),(1,2),(1,3),(1,4),(1,5),(1,6),(2,0),(2,1),(2,2),(2,3),(2,4),(2,5),(2,6),
                (3,3),(3,4),(3,5),(3,6),(4,2),(4,3),(4,4),(4,5),(4,6),(5,1),(5,2),(5,3),(5,4),(5,5),(5,6),(6,0),(6,1),(6,2),(6,3),(6,4),(6,5),(6,6)]

    if all((array[i, j] == [255, 255, 255]).all() for i, j in white_ZB) and all((array[i, j] == [0,0,0]).all() for i, j in black_ZB):
        # Specify which voxels should be eroded
        return True, [(row-1,col),(row-1,col+1)]
    
    # Not the case
    return False, []


# Latin script - special cases:
# Identify sharp corners present in Y and in the second and third corners of W. Thin corners 1px wide
def case_y_corner(array, row, col): 
    
    # Looks for the pattern of _ (white), and removes X. Two centers are possible, consider both cases
    # Example (Xs will be removed): 
    # B B _ B B     B B B B B 
    # B B _ B B     B B X B B 
    # B B C B B     B B C B B 
    # B B X B B     B B _ B B
    # B B B B B     B B _ B B
    
    # Check for bottom reference
    white_B = [(0,2),(1,2)]
    black_B = [(0,0),(0,1),(0,3),(0,4),(1,0),(1,1),(1,3),(1,4),(2,0),(2,1),(2,3),(2,4),(3,0),(3,1),(3,2),(3,3),(3,4),(4,0),(4,1),(4,2),(4,3),(4,4)]

    if all((array[i, j] == [255, 255, 255]).all() for i, j in white_B) and all((array[i, j] == [0,0,0]).all() for i, j in black_B):
        # Specify which voxels should be eroded
        return True, [(row+1,col),(row+2,col)]
    
    # Check for top reference
    white_T = [(3,2),(4,2)]
    black_T = [(0,0),(0,1),(0,2),(0,3),(0,4),(1,0),(1,1),(1,2),(1,3),(1,4),(2,0),(2,1),(2,3),(2,4),(3,0),(3,1),(3,3),(3,4),(4,0),(4,1),(4,3),(4,4)]

    if all((array[i, j] == [255, 255, 255]).all() for i, j in white_T) and all((array[i, j] == [0,0,0]).all() for i, j in black_T):
        # Specify which voxels should be eroded
        return True, [(row-1,col),(row-2,col)]
    
    # Not the case
    return False, []
    
    
# Latin script - special cases:
# Identify sharp corners present in V and in the first part of W. Sharp corners 2px thick
def case_v_corner(array, row, col): 
    
    # Looks for the pattern of _ (white), and removes X. Two centers are possible, consider both cases
    # Example (Xs will be removed): 
    # ? ? B B _ _ B     B _ _ B B ? ?
    # ? ? B B _ _ B     B _ _ B B ? ?
    # ? ? B B _ _ B     B _ _ B B ? ?
    # ? ? B C X B B     B B X C B ? ?
    # ? ? B B X B B     B B X B B ? ?
    # ? ? ? ? X ? ?     ? ? ? ? ? ? ?
    # ? ? ? ? ? ? ?     ? ? ? ? ? ? ?
    
    # Check for left reference
    white_L = [(0,4),(0,5),(1,4),(1,5),(2,4),(2,5)]
    black_L = [(0,2),(0,3),(0,6),(1,2),(1,3),(1,6),(2,2),(2,3),(2,6),(3,2),(3,3),(3,4),(3,5),(3,6),(4,2),(4,3),(4,4),(4,5),(4,6)]

    if all((array[i, j] == [255, 255, 255]).all() for i, j in white_L) and all((array[i, j] == [0,0,0]).all() for i, j in black_L):
        # Specify which voxels should be eroded
        return True, [(row,col+1),(row+1,col+1),(row+2,col+1)]
    
    # Check for right reference
    white_R = [(0,1),(0,2),(1,1),(1,2),(2,1),(2,2)]
    black_R = [(0,0),(0,3),(0,4),(1,0),(1,3),(1,4),(2,0),(2,3),(2,4),(3,0),(3,1),(3,2),(3,3),(3,4),(4,0),(4,1),(4,2),(4,3),(4,4)]

    if all((array[i, j] == [255, 255, 255]).all() for i, j in white_R) and all((array[i, j] == [0,0,0]).all() for i, j in black_R):
        # Specify which voxels should be eroded
        return True, [(row,col-1),(row+1,col-1),(row+2,col-1)]
    
    # Not the case
    return False, []



### ---------------------------------------------------------------------------
### Miscellaneous

## Check if a pixel has the same color as a reference
def is_color(pixel, reference):
    return np.array_equal(pixel, reference)


## Check if in an image there are still pixels of a reference
def has_color_pixels(array, reference):
    return np.any(np.any(array == reference, axis=-1))


## Extract a cut-out of an image given size and coordinates 
# IMPORTANT: it has only been tested on odd sizes (5,7,9)
def extract_cutout(array, row, col, side):
    
    section_size = side
    half_size = section_size // 2

    # Calculate the indices for slicing
    start_row = max(row - half_size, 0)
    end_row = min(row + half_size + 1, array.shape[0])
    
    start_col = max(col - half_size, 0)
    end_col = min(col + half_size + 1, array.shape[1])

    # Extract the section using slicing
    return array[start_row:end_row, start_col:end_col]


## Parce the filename to extract script and letter information
def parce_filename(image_path):
    # Extract the file name from the path
    filename = os.path.basename(image_path)
    
    # Remove the file extension
    filename_no_ext = filename.split('.')[0]
    
    return filename_no_ext


## Count amount of black pixels in image
def count_black_pixels(image_path):
    
    # Load the image
    image = Image.open(image_path).convert('RGB')
    np_image = np.array(image)
    
    # A pixel is black if all its RGB components are zero
    black_pixels = np.sum(np.all(np_image == [0, 0, 0], axis = -1))
    
    return black_pixels








