#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Created on Mon Mar 10 15:37:32 2025

Visual Braille Silico - visualization functions

@author: Filippo Cerpelloni
'''
# Import personal functions, then all libraries are coming from there
import sys
sys.path.append('../')

# Import custom functions to store paths, extraction of activations
from src.vbs_functions import *

import os
import glob
import re

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns

from scipy.stats import sem
from matplotlib.colors import LinearSegmentedColormap

sys.path.append('../lib/CORnet')

# bad practice, do better
sys.path.append('/Users/cerpelloni/Documents/GitHub/PlotNeuralNet/')
from pycore.tikzeng import *


import torch
import torch.nn as nn
import numpy as np

from torchvision import transforms, models
from torchvision.io import read_image
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from cornet import cornet_z



### ---------------------------------------------------------------------------
### Plot training stats

# Learning curves specific for novel scripts
def plot_anova_training(opt, results_path):
    
    # Load the results 
    results = pd.read_csv(os.path.join(opt['dir']['results'], 'classifications', results_path))
    
    # From file path, extract model name and information for plotting
    filename = results_path.split('_sub')[0]
    model_name = filename.split('-')[1]
    
    # Consider only novel scripts, not LT 
    # If AlexNet, we already know that performance is perferct. We mostly care 
    # about the novel scripts. 
    # Still TBD in CORnet
    # results = results[(results['model_name'] == model_name) & (results['script'] != 'LT')]
    
    # Specify scripts colours
    custom_palette = {
        'LT': '#4C75B3', # Naive participant blue
        'BR': '#FF9E4A', # CPP Orange
        'LN': '#fb92ff'  # New pink
    }
    dodge_val = 0.2
    jitter_val = 0.1

    # Average and SD bars
    point_plot = sns.pointplot(x = 'epoch', 
                               y = 'score', 
                               hue = 'script', 
                               data = results, 
                               dodge = dodge_val, 
                               markers = 'o', 
                               capsize = 0, 
                               palette = custom_palette,
                               markersize = 9, linewidth = 4,
                               errorbar = 'sd',
                               legend = False,
                               zorder = 2)
    
    # Individual data points
    strip_plot = sns.stripplot(x = 'epoch', 
                               y = 'score', 
                               hue = 'script', 
                               data = results, 
                               dodge = dodge_val, 
                               jitter = jitter_val, 
                               alpha = 0.4, 
                               palette = custom_palette,
                               legend = False,
                               zorder = 1)


    # Adjust dodge by separating the hues manually
    for i, artist in enumerate(strip_plot.collections):
        if i % 2 == 0: offset = -0.12  
        else:          offset = 0.12
        artist.set_offsets(artist.get_offsets() - np.array([offset, 0]))

    # Customize plot
    if model_name == 'alexnet': 
        plt.xticks([0, 4, 9], ['11', '15', '20'], fontname = 'Avenir', fontsize = 12)
    else:
        plt.xticks([0, 4, 9, 14], ['1', '5', '10', '15'], fontname = 'Avenir', fontsize = 12)
    plt.yticks(fontname = 'Avenir', fontsize = 12)
    
    # Add axes labels and title
    plt.xlabel('Epoch', fontname = 'Avenir', fontsize = 16)
    plt.ylabel('Accuracy', fontname = 'Avenir', fontsize = 16)
    # plt.title(f'{model_name} - accuracies across epochs')

    # Customize x-axis ticks, accoid half-days


    # Save plot
    plt.savefig(os.path.join(opt['dir']['figures'], f'{filename}_plot-learning-comparison-novel-scripts.png'), 
                dpi = 400)
    
    # Display the plot
    plt.show()
    

            

### ---------------------------------------------------------------------------
### Learning curves

## Obsolete: plot the curves for literacy and Braille expertise for AlexNet
def plot_braille_expertise(opt, lt_training, br_training): 
    '''
    Plot the learning curves for Latin script alone and Latin + Braille scripts
    (used in Figure 2 of CCN 2025 Extended Abstract). It's a static figure so it 
    requires fewer arguments (e.g. it's for AlexNet alone for sure)
    
    Parameters
    ----------    
    opt (dict): output of vbs_option() containing the paths of the IODA folder
    
    lt_training (string): the name of the file relative to the training on 
                          Latin script 
    
    br_training (string): the name of the file relative to the training on 
                          Latin + Braille script 
    
    Outputs
    -------
    Figure, saved also in outputs/figures
    '''
    
    ##  Open literacy and expertise logs
    lt = pd.read_csv(os.path.join(opt['dir']['logs'], lt_training))
    br = pd.read_csv(os.path.join(opt['dir']['logs'], br_training))
    
    # Arrange data
    all_training = pd.concat([lt, br], ignore_index = True)    
    
    ## Plot data
    # Compute Mean and Confidence Interval (CI)
    grouped = all_training.groupby(['Script', 'Epoch']).agg(
        mean_accuracy=('Val_accuracy', 'mean'),
        sem_accuracy=('Val_accuracy', sem)
    ).reset_index()
    
    # Shift BR epochs
    grouped['Epoch'] = grouped['Epoch'] + 9 * (grouped['Script'] == 'BR')
    
    # Compute 95% Confidence Interval
    grouped['ci'] = 1.96 * grouped['sem_accuracy']
    grouped['upper'] = grouped['mean_accuracy'] + grouped['ci']
    grouped['lower'] = grouped['mean_accuracy'] - grouped['ci']

    # Define colors
    colors = {'LT': '#699AE5', 'BR': '#FF9E4A'}
    labels = {'LT': 'Latin',   'BR': 'Latin + Braille'}
    
    # Plot
    plt.figure(figsize=(8, 6))
    
    for script in grouped['Script'].unique():
        subset = grouped[grouped['Script'] == script]
    
        # Line plot for mean accuracy
        plt.plot(subset['Epoch'], subset['mean_accuracy'], marker='', linewidth = 5,
                 label = labels[script], color = colors[script])
    
        # Confidence interval as a shaded area (halo)
        plt.fill_between(subset['Epoch'], subset['lower'], subset['upper'], 
                         color = colors[script], alpha = 0.2)

    # Add vertical dashed line to spearate trainings
    plt.axvline(x = 10, color = 'black', linestyle = '--', linewidth = 3)

    # Adjust ticks and labels
    plt.xticks([5, 10, 15, 20], [5, 10, 5, 10])
    plt.xlabel('Epochs', font = 'Avenir', fontsize = 15, labelpad = 20)
    plt.ylabel('Accuracy', font = 'Avenir', fontsize = 15)
    plt.legend(loc = 'lower right', title = 'Script learnt')
    plt.grid(False)
    
    # Add text labels under x-axis
    plt.text(5, plt.ylim()[0] - 0.12, 'Literacy acquisition', 
             ha = 'center', fontsize = 15, fontname='Avenir')

    plt.text(15, plt.ylim()[0] - 0.12, 'Expertise acquisition', 
             ha = 'center', fontsize = 15, fontname = 'Avenir')
    
    # Save the plot
    plt.savefig('../../outputs/figures/ccn-abstract_fig-2_model-alexnet_plot-joint-learning-braille.png', 
                dpi = 600, bbox_inches = 'tight')
    
    # Show the plot
    plt.show()


## Plot AlexNet trainings on literacy and expertise
def plot_alexnet_training(opt, lt_training, br_training, ln_training): 
    '''
    Plot the learning curves of Latin alphabet alone, plus the addition of 
    Latin + Braille and Latin + Line datasets.
    For AlexNet only. 
    
    Parameters
    ----------    
    opt (dict): output of vbs_option() containing the paths of the IODA folder
    
    lt_training (str): the name of the file relative to the training on 
                       Latin script 
    
    br_training (str): the name of the file relative to the training on 
                       Latin + Braille script 
                          
    ln_training (str): the name of the file relative to the training on 
                       Latin + Line script 
    
    Outputs
    -------
    Figure, saved in outputs/figures
    
    '''
    
    ##  Open literacy and expertise logs
    lt = pd.read_csv(os.path.join(opt['dir']['logs'], lt_training))
    br = pd.read_csv(os.path.join(opt['dir']['logs'], br_training))
    ln = pd.read_csv(os.path.join(opt['dir']['logs'], ln_training))
    
    # Arrange data
    all_training = pd.concat([lt, br, ln], ignore_index = True)    
    
    # Compute Mean and Confidence Interval (CI)
    grouped = all_training.groupby(['Script', 'Epoch']).agg(mean_accuracy=('Val_accuracy', 'mean'),
                                                            sem_accuracy=('Val_accuracy', sem)).reset_index()
    
    # ??? still needed? 
    # Shift epochs of Braille and Line by 10
    # grouped['Epoch'] = grouped['Epoch'] + 10 * (grouped['Script'] == 'BR')
    # grouped['Epoch'] = grouped['Epoch'] + 10 * (grouped['Script'] == 'LN')
    
    # Compute 95% Confidence Interval
    grouped['ci'] = 1.96 * grouped['sem_accuracy']
    grouped['upper'] = grouped['mean_accuracy'] + grouped['ci']
    grouped['lower'] = grouped['mean_accuracy'] - grouped['ci']

    # Define colours of training
    # Pastel blue for Latin, CPP orange for Braille, CPP green for Line Braille
    colors = {'LT': '#4C75B3', 'LTBR': '#FF9E4A',         'LTLN': '#69B5A2'}
    labels = {'LT': 'Latin',   'LTBR': 'Latin + Braille', 'LTLN': 'Latin + Line Braille'}
    
    # Figure 
    plt.figure(figsize=(7, 5))
    
    # Plot one script at the time
    for script in grouped['Script'].unique():
        subset = grouped[grouped['Script'] == script]
    
        # Line plot for mean accuracy
        plt.plot(subset['Epoch'], subset['mean_accuracy'], 
                 marker='', linewidth = 4.5,
                 label = labels[script], color = colors[script])
    
        # Confidence interval as a shaded area (halo)
        plt.fill_between(subset['Epoch'], subset['lower'], subset['upper'], 
                         color = colors[script], alpha = 0.2)

    # Add vertical dashed line to spearate trainings
    plt.axvline(x = 10.9, color = 'black', linestyle = '--', linewidth = 2)

    # Adjust ticks and labels
    plt.xticks([5, 10, 15, 20], font = 'Avenir', fontsize = 12)
    plt.xlabel('Epoch', font = 'Avenir', fontsize = 15)
    plt.yticks([0.2, 0.4, 0.6, 0.8, 1], font = 'Avenir', fontsize = 12)
    plt.ylabel('Accuracy', font = 'Avenir', fontsize = 15)
    plt.grid(False)
    
    # Save the plot
    plt.savefig('../../outputs/figures/model-alexnet_plot-learning-curves.png', 
                dpi = 600, bbox_inches = 'tight')
    
    # Show the plot
    plt.show()

    
## Plot CORnet trainings on literacy and expertise
def plot_cornet_training(opt, lt_training, br_training, ln_training): 
    '''
    Plot the learning curves of Latin alphabet alone, plus the addition of 
    Latin + Braille and Latin + Line datasets.
    For CORnet only. 
    
    Parameters
    ----------    
    opt (dict): output of vbs_option() containing the paths of the IODA folder
    
    lt_training (str): the name of the file relative to the training on 
                       Latin script 
    
    br_training (str): the name of the file relative to the training on 
                       Latin + Braille script 
                          
    ln_training (str): the name of the file relative to the training on 
                       Latin + Line script 
    
    Outputs
    -------
    Figure, saved in outputs/figures
    
    '''
    ##  Open literacy and expertise logs
    lt = pd.read_csv(os.path.join(opt['dir']['logs'], lt_training))
    br = pd.read_csv(os.path.join(opt['dir']['logs'], br_training))
    ln = pd.read_csv(os.path.join(opt['dir']['logs'], ln_training))
    
    # Arrange data
    all_training = pd.concat([lt, br, ln], ignore_index = True)    
    
    # Compute Mean and Confidence Interval (CI)
    grouped = all_training.groupby(['Script', 'Epoch']).agg(mean_accuracy=('Val_Accuracy', 'mean'),
                                                            sem_accuracy=('Val_Accuracy', sem)).reset_index()
    
    # Compute 95% Confidence Interval
    grouped['ci'] = 1.96 * grouped['sem_accuracy']
    grouped['upper'] = grouped['mean_accuracy'] + grouped['ci']
    grouped['lower'] = grouped['mean_accuracy'] - grouped['ci']

    # Define colours of training
    # Pastel blue for Latin, CPP orange for Braille, CPP green for Line Braille
    colors = {'LT': '#4C75B3', 'LTBR': '#FF9E4A',         'LTLN': '#FFDB42'}
    labels = {'LT': 'Latin',   'LTBR': 'Latin + Braille', 'LTLN': 'Latin + Line Braille'}
    
    # Figure 
    plt.figure(figsize=(7,5))
    
    # Plot one script at the time
    for script in grouped['Script'].unique():
        subset = grouped[grouped['Script'] == script]
    
        # Line plot for mean accuracy
        plt.plot(subset['Epoch'], subset['mean_accuracy'], 
                 marker='', linewidth = 4.5,
                 label = labels[script], color = colors[script])
    
        # Confidence interval as a shaded area (halo)
        plt.fill_between(subset['Epoch'], subset['lower'], subset['upper'], 
                         color = colors[script], alpha = 0.2)

    # Adjust ticks and labels
    plt.xticks([5, 10, 15], font = 'Avenir', fontsize = 12)
    plt.xlabel('Epoch', font = 'Avenir', fontsize = 15)
    plt.yticks([0.2, 0.4, 0.6, 0.8, 1], font = 'Avenir', fontsize = 12)
    plt.ylabel('Accuracy', font = 'Avenir', fontsize = 15)
    plt.grid(False)
    
    # Save the plot
    plt.savefig('../../outputs/figures/model-cornet_plot-learning-curves.png', 
                dpi = 600, bbox_inches = 'tight')
    
    # Show the plot
    plt.show()
    
    

### ---------------------------------------------------------------------------
### Clustering
       
## Plot clustering evolution across layers 
def plot_clustering(opt, model_name, exp_clusters, ctr_clusters):
    '''
    Plot the clustering curve for two curves for Latin script, Latin + Braille 
    and Latin + Line trainings
    
    Parameters
    ----------    
    opt (dict): output of vbs_option() containing the paths of the IODA folder
    
    model_name (str): the name of the model plotted, to get layer names and
                      save accordingly 
                      
    exp_clusters (str): the name of the clustering file at the matrix level
                        (one value for each subject/script/layer) of the expert
                        network
                        
    ctr_clusters (str): the name of the clustering file at the matrix level
                        (one value for each subject/script/layer) of the naive
                        network
    
    Outputs
    -------
    Figure, saved also in outputs/figures
    
    '''
    
    ##  Open clustering results
    exp_clu = pd.read_csv(os.path.join(opt['dir']['results'], 'clustering', exp_clusters))
    ctr_clu = pd.read_csv(os.path.join(opt['dir']['results'], 'clustering', ctr_clusters))
    
    # Ensure that layer order is maintained
    layer_order = exp_clu['layer'].drop_duplicates().tolist()
    exp_clu['layer'] = pd.Categorical(exp_clu['layer'], categories = layer_order, ordered = True)
    
    layer_order = ctr_clu['layer'].drop_duplicates().tolist()
    ctr_clu['layer'] = pd.Categorical(ctr_clu['layer'], categories = layer_order, ordered = True)

    ## Plot data
    # Compute Confidence Interval (CI)
    exp_grouped = exp_clu.groupby(['expertise', 'script', 'layer'], sort = False).agg(mean_clustering = ('clustering', 'mean'),
                                                                    sem_clustering = ('clustering', sem)
                                                                    ).reset_index()
    ctr_grouped = ctr_clu.groupby(['expertise', 'script', 'layer'], sort = False).agg(mean_clustering = ('clustering', 'mean'),
                                                                    sem_clustering = ('clustering', sem)
                                                                    ).reset_index()
    
    # Compute 95% Confidence Interval
    exp_grouped['ci'] = 1.96 * exp_grouped['sem_clustering']
    exp_grouped['upper'] = exp_grouped['mean_clustering'] + exp_grouped['ci']
    exp_grouped['lower'] = exp_grouped['mean_clustering'] - exp_grouped['ci']
    
    ctr_grouped['ci'] = 1.96 * ctr_grouped['sem_clustering']
    ctr_grouped['upper'] = ctr_grouped['mean_clustering'] + ctr_grouped['ci']
    ctr_grouped['lower'] = ctr_grouped['mean_clustering'] - ctr_grouped['ci']
    
    # Join expert and naive data 
    grouped = pd.concat([exp_grouped, ctr_grouped])
    grouped['group'] = grouped['expertise'] + '-' + grouped['script']
    
    # Define colours
    colors = {'braille_expert-BR': '#FF9E4A',        
              'braille_expert-LT': '#69B5A2',
              'naive-BR': '#DA5F49',        
              'naive-LT': '#4C75B3'}
    labels = {'braille_expert-BR': 'Expert network - Braille script', 
              'braille_expert-LT': 'Expert network - Latin script',
              'naive-BR':  'Naive network - Braille script', 
              'naive-LT':  'Naive network - Latin script'}
    
    # Define layer names
    if model_name == 'alexnet': 
        xticks = [0, 1, 2, 3, 4, 5, 6, 7]
        xlabels = ['1', '2', '3', '4', '5', '6', '7', 'OUT']
    elif model_name == 'cornet':
        xticks = [0, 1, 2, 3, 4, 5]
        xlabels = ['V1', 'V2', 'V4', 'IT', 'AvgIT', 'OUT']  
        
    # Plot
    plt.figure(figsize=(7, 5))
    
    for grp in grouped['group'].unique():
        subset = grouped[grouped['group'] == grp]
    
        # Line plot for mean accuracy
        plt.plot(subset['layer'], subset['mean_clustering'], 
                 marker = '', linewidth = 5,
                 label = labels[grp], color = colors[grp])
    
        # Confidence interval as a shaded area (halo)
        plt.fill_between(subset['layer'], subset['lower'], subset['upper'], 
                         color = colors[grp], alpha = 0.2)
    
    # Adjust tickes for both axes, mentioning names of stages based on network
    if model_name == 'alexnet': plt.xticks([0, 1, 2, 3, 4, 5, 6], opt['layers'][f'{model_name}'], font = 'Avenir')
    else:                       plt.xticks([0, 1, 2, 3, 4], opt['layers'][f'{model_name}'], font = 'Avenir')
    
    plt.xlabel('Layer', font = 'Avenir', fontsize = 15)
    plt.xticks(xticks, xlabels, font = 'Avenir', fontsize = 12)
    plt.yticks([0, 0.1, 0.2], font = 'Avenir' , fontsize = 12)
    plt.ylabel('Clustering', font = 'Avenir', fontsize = 15)
    plt.ylim([0, 0.21])
    plt.grid(False)
    
    # Save the plot
    plt.savefig(f'../../outputs/figures/model-{model_name}_plot-clustering-across-training.png', 
                dpi = 600, bbox_inches = 'tight')
    
    # Show the plot
    plt.show()
    
    
       
# Plot clustering evolution across layers, for one script only
def plot_clustering_script(opt, model_name, script, exp_clusters, ctr_clusters):
    '''
    Plot the clustering curve for two curves for Latin script, Latin + Braille 
    and Latin + Line trainings
    
    Parameters
    ----------    
    opt (dict): output of vbs_option() containing the paths of the IODA folder
    
    model_name (str): the name of the model plotted, to get layer names and
                      save accordingly 
                      
    script (str): to which script limit the plotting 
                      
    exp_clusters (str): the name of the clustering file at the matrix level
                        (one value for each subject/script/layer) of the expert
                        network
                        
    ctr_clusters (str): the name of the clustering file at the matrix level
                        (one value for each subject/script/layer) of the naive
                        network
    
    Outputs
    -------
    Figure, saved also in outputs/figures
    
    '''
    
    ##  Open clustering results
    exp_clu = pd.read_csv(os.path.join(opt['dir']['results'], 'clustering', exp_clusters))
    ctr_clu = pd.read_csv(os.path.join(opt['dir']['results'], 'clustering', ctr_clusters))
    
    ## Plot data
    # Compute Confidence Interval (CI)
    exp_grouped = exp_clu.groupby(['expertise', 'script', 'layer']).agg(mean_clustering = ('clustering', 'mean'),
                                                                    sem_clustering = ('clustering', sem)
                                                                    ).reset_index()
    ctr_grouped = ctr_clu.groupby(['expertise', 'script', 'layer']).agg(mean_clustering = ('clustering', 'mean'),
                                                                    sem_clustering = ('clustering', sem)
                                                                    ).reset_index()
    
    # Compute 95% Confidence Interval
    exp_grouped['ci'] = 1.96 * exp_grouped['sem_clustering']
    exp_grouped['upper'] = exp_grouped['mean_clustering'] + exp_grouped['ci']
    exp_grouped['lower'] = exp_grouped['mean_clustering'] - exp_grouped['ci']
    
    ctr_grouped['ci'] = 1.96 * ctr_grouped['sem_clustering']
    ctr_grouped['upper'] = ctr_grouped['mean_clustering'] + ctr_grouped['ci']
    ctr_grouped['lower'] = ctr_grouped['mean_clustering'] - ctr_grouped['ci']
    
    # Join expert and naive data 
    grouped = pd.concat([exp_grouped, ctr_grouped])
    grouped['group'] = grouped['expertise'] + '-' + grouped['script']
    
    # Keep only selected script
    grouped = grouped[grouped['script'] == script]
    
    # Define colours
    colors = {'expert-BR': '#FF9E4A',        
              'expert-LT': '#69B5A2',
              'naive-BR': '#DA5F49',        
              'naive-LT': '#4C75B3'}
    labels = {'expert-BR': 'Expert network - Braille script', 
              'expert-LT': 'Expert network - Latin script',
              'naive-BR':  'Naive network - Braille script', 
              'naive-LT':  'Naive network - Latin script'}
    
    # Define layer names
    if model_name == 'alexnet': 
        xticks = [0, 1, 2, 3, 4, 5, 6]
        xlabels = ['1', '2', '2', '4', '5', '6', '7']
    elif model_name == 'cornet':
        xticks = [0, 1, 2, 3, 4]
        xlabels = ['V1', 'V2', 'V4', 'IT', 'AvgIT']        
    
    # Plot
    plt.figure(figsize=(8, 6))
    
    for grp in grouped['group'].unique():
        subset = grouped[grouped['group'] == grp]
    
        # Line plot for mean accuracy
        plt.plot(subset['layer'], subset['mean_clustering'], 
                 marker = '', linewidth = 5,
                 label = labels[grp], color = colors[grp])
    
        # Confidence interval as a shaded area (halo)
        plt.fill_between(subset['layer'], subset['lower'], subset['upper'], 
                         color = colors[grp], alpha = 0.2)
    
    # Adjust tickes for both axes, mentioning names of stages based on network
    if model_name == 'alexnet': plt.xticks([0, 1, 2, 3, 4, 5, 6], opt['layers'][f'{model_name}'], font = 'Avenir')
    else:                       plt.xticks([0, 1, 2, 3, 4], opt['layers'][f'{model_name}'], font = 'Avenir')
    
    plt.xlabel('Layer', font = 'Avenir', fontsize = 15)
    plt.xticks(xticks, xlabels, font = 'Avenir')
    plt.yticks([0, 0.1, 0.2])
    plt.ylabel('Clustering', font = 'Avenir', fontsize = 15)
    plt.grid(False)
    
    # Save the plot
    plt.savefig(f'../../outputs/figures/model-{model_name}_plot-clustering-{script}.png', 
                dpi = 600, bbox_inches = 'tight')
    
    # Show the plot
    plt.show()
    
    

### ---------------------------------------------------------------------------
### Dissimilarity scores 

## Plot mean dissimilarity across layers 
def plot_mean_dissimilarity(opt, model_name, exp_scores, ctr_scores):
    '''
    Plot the clustering curve for two curves for Latin script, Latin + Braille 
    and Latin + Line trainings
    
    Parameters
    ----------    
    opt (dict): output of vbs_option() containing the paths of the IODA folder
    
    model_name (str): the name of the model plotted, to get layer names and
                      save accordingly 
                      
    exp_scores (str): the name of the dissmilarity scores (one value for each 
                      subject/script/layer) of the expert network
                        
    ctr_scores (str): the name of the dissmilarity scores (one value for each 
                      subject/script/layer) of the naive network
    
    Outputs
    -------
    Figure, saved also in outputs/figures
    
    '''
    
    ##  Open dissmilarity scores
    exp_diss = pd.read_csv(os.path.join(opt['dir']['results'], 'distances', exp_scores))
    ctr_diss = pd.read_csv(os.path.join(opt['dir']['results'], 'distances', ctr_scores))
    
    # Ensure that layer order is maintained
    layer_order = exp_diss['layer'].drop_duplicates().tolist()
    exp_diss['layer'] = pd.Categorical(exp_diss['layer'], categories = layer_order, ordered = True)
    
    layer_order = ctr_diss['layer'].drop_duplicates().tolist()
    ctr_diss['layer'] = pd.Categorical(ctr_diss['layer'], categories = layer_order, ordered = True)

    ## Plot data
    # Compute Confidence Interval (CI)
    exp_grouped = exp_diss.groupby(['expertise', 'script', 'layer'], sort = False).agg(mean_diss = ('dissimilarity', 'mean'),
                                                                    sem_diss = ('dissimilarity', sem)
                                                                    ).reset_index()
    ctr_grouped = ctr_diss.groupby(['expertise', 'script', 'layer'], sort = False).agg(mean_diss = ('dissimilarity', 'mean'),
                                                                    sem_diss = ('dissimilarity', sem)
                                                                    ).reset_index()
    
    # Compute 95% Confidence Interval
    exp_grouped['ci'] = 1.96 * exp_grouped['sem_diss']
    exp_grouped['upper'] = exp_grouped['mean_diss'] + exp_grouped['ci']
    exp_grouped['lower'] = exp_grouped['mean_diss'] - exp_grouped['ci']
    
    ctr_grouped['ci'] = 1.96 * ctr_grouped['sem_diss']
    ctr_grouped['upper'] = ctr_grouped['mean_diss'] + ctr_grouped['ci']
    ctr_grouped['lower'] = ctr_grouped['mean_diss'] - ctr_grouped['ci']
    
    # Join expert and naive data 
    grouped = pd.concat([exp_grouped, ctr_grouped])
    grouped['group'] = grouped['expertise'] + '-' + grouped['script']
    
    # Define colours
    colors = {'expert-BR': '#FF9E4A',        
              'expert-LT': '#69B5A2',
              'naive-BR': '#DA5F49',        
              'naive-LT': '#4C75B3'}
    labels = {'expert-BR': 'Expert network - Braille script', 
              'expert-LT': 'Expert network - Latin script',
              'naive-BR':  'Naive network - Braille script', 
              'naive-LT':  'Naive network - Latin script'}
    
    # Define layer names
    if model_name == 'alexnet': 
        xticks = [0, 1, 2, 3, 4, 5, 6, 7]
        xlabels = ['1', '2', '3', '4', '5', '6', '7', 'OUT']
    elif model_name == 'cornet':
        xticks = [0, 1, 2, 3, 4, 5]
        xlabels = ['V1', 'V2', 'V4', 'IT', 'AvgIT', 'OUT']  
        
    # Plot
    plt.figure(figsize=(7, 5))
    
    for grp in grouped['group'].unique():
        subset = grouped[grouped['group'] == grp]
    
        # Line plot for mean accuracy
        plt.plot(subset['layer'], subset['mean_diss'], 
                 marker = '', linewidth = 5,
                 label = labels[grp], color = colors[grp])
    
        # Confidence interval as a shaded area (halo)
        plt.fill_between(subset['layer'], subset['lower'], subset['upper'], 
                         color = colors[grp], alpha = 0.2)
    
    # Adjust tickes for both axes, mentioning names of stages based on network
    if model_name == 'alexnet': plt.xticks([0, 1, 2, 3, 4, 5, 6], opt['layers'][f'{model_name}'], font = 'Avenir')
    else:                       plt.xticks([0, 1, 2, 3, 4], opt['layers'][f'{model_name}'], font = 'Avenir')
    
    plt.xlabel('Layer', font = 'Avenir', fontsize = 15)
    plt.xticks(xticks, xlabels, font = 'Avenir', fontsize = 12)
    plt.yticks(font = 'Avenir' , fontsize = 12)
    plt.ylabel('Dissimilarity', font = 'Avenir', fontsize = 15)
    plt.ylim([0, 0.85])
    plt.grid(False)
    
    # Save the plot
    plt.savefig(f'../../outputs/figures/model-{model_name}_plot-dissimilarity-scores.png', 
                dpi = 600, bbox_inches = 'tight')
    
    # Show the plot
    plt.show()


### ---------------------------------------------------------------------------
### Representational Dissimilarity Matrices (RDMs)

## Plot the distance matrices of a model at different stages
def plot_stimuli_distances(opt, distances):
    '''
    Plot the distances between stimuli for each layer (single subject)
    TODO expand to all the subjects
    
    Parameters
    ----------
    opt (dict): output of vbs_option() containing the paths of the IODA folder

    distances (string): the path to the distances to plot, saved as pickle file
    
    Outputs
    -------
    Figures for each script and layer, also saved in outputs/figures/distances
    
    '''

    # Load the distances 
    dist = load_activations(os.path.join(opt['dir']['results'], 'distances', distances))
    
    # Parce the filename to extract sub and model references
    fileinfo = distances.split('_data')[0]
    
    # Load flat dictionary to get the keys, the labels of the stimuli
    # TODO pre-save labels in vbs_option() to load them more easily
    flat = load_activations(os.path.join(opt['dir']['results'], 'activations', fileinfo + '_data-flat-activations.pkl'))
    
    stim_labels = list(flat['stage-1'].keys())
    lt_labels = [item for item in stim_labels if item.startswith('LT_')]
    br_labels = [item for item in stim_labels if item.startswith('BR_')]
    
    layer_list = flat.keys()
    
    # Iterate through the layers computed
    for i, layer in enumerate(layer_list):
    
        # Plot the distances for the Latin script
        lt_distance = dist['LT'][f'{layer}']
        plot_rdm_stimuli(opt, lt_distance, lt_labels, fileinfo, layer, 'latin')
        
        # Plot the distances for the Braille script
        br_distance = dist['BR'][f'{layer}']
        plot_rdm_stimuli(opt, br_distance, br_labels, fileinfo, layer, 'braille')
        
    
## Plot average matrix across  
def plot_category_distances(opt, model_name, training, activations): 
    '''
    Plot the distances between categories for each layer and script, averaged across subjects
    
    Parameters
    ----------
    opt (dict): output of vbs_option() containing the paths of the IODA folder
    
    model_name (string): the model refernce (whether 'alexnet' or 'cornet') to 
                         load the correct activations and to save the reuslts properly
                         
    training (str): the reference of the network's expertise
                         
    activations (string): the path to the activation file to load
    
    Outputs
    -------
    Figures for each script and layer, also saved in outputs/figures/distances
    
    '''
    
    # Determine color of the RDM based on expertise
    if training == 'LT': exp = False
    else: exp = True

    # Load activations
    acts = load_extraction(os.path.join(opt['dir']['results'], 'distances', activations))
    
    # Iterate through each subscript and layer 
    # to plot the corresponding categorical RDM
    for s, scr in enumerate(acts.keys()):
        
        # Determine color of the RDM based on script
        if exp: 
            if scr == 'LT': color_name = 'green'
            else: color_name = 'orange'
        else:
            if scr == 'LT': color_name = 'blue'
            else: color_name = 'red'
        
        # For each layer 
        for l, layer in enumerate(acts[f'{scr}'].keys()):
            
            # Plot the RDM with the group colour
            plot_rdm_categories(opt, acts[f'{scr}'][f'{layer}'], color_name,              
                                f'model-{model_name}_sub-all_training-{training}_test-VBE_script-{scr}_plot-rdm-categories-layer-{l+1}')


## Plot the RDM of a given matrix of dissimilarities between stimuli 
# TODO: improve function, like plot_rdm_categories
def plot_rdm_stimuli(opt, matrix, stim_labels, fileinfo, layer, script):
    '''
    Plot a 48-by-48 RDM: one script, one layer, all the stimuli (averaged across variations)
    
    Parameters
    ----------    
    opt (dict): output of vbs_option() containing the paths of the IODA folder
    
    matrix (array): the matrix to plot
    
    stim_labels (array): the labels with all the stimuli noted (e.g. BR_RW_1)
    
    fileinfo (string): string with essential information of the bids-like name
    
    layer (string): the name of the layer
            
    script (string): 'latin' or 'braille', information to save the figure 
    
    Outputs
    -------
    Figure for the requested RDM, also saved in outputs/figures/distances
    
    '''
    
    # Start plotting 
    plt.figure()

    # Set specific and class labels
    repeated_labels = stim_labels * 4

    # Create new labels with sequence indicators
    classes = ['RW', 'PW', 'NW', 'FS']
    new_labels = []
    for l in range(1):
        new_labels.extend([label for label in stim_labels])
            
    ax = sns.heatmap(matrix, 
                      cmap = 'viridis', 
                      annot = False, 
                      xticklabels = False, 
                      yticklabels = False)

    # Customize the heatmap
    title = f'{fileinfo}_layer-{layer}_plot-rdm-{script}'
    ax.set_title(title, fontsize = 15, pad = 20)

    # Add sequence indicators as subtitles for the axis
    for j, cla in enumerate(classes):
        plt.text(-5, j*12 +6, cla, rotation = 90, fontsize = 12, verticalalignment = 'center')
        plt.text(j*12 +6, 53, cla, rotation = 0, fontsize = 12, horizontalalignment = 'center')

    ax.yaxis.set_tick_params(rotation = 0)
    for label in ax.get_yticklabels():
        label.set_verticalalignment('center')

    ax.set_aspect('equal')
    
    # Add squares to separate clusters
    num_clusters = len(classes)  # Assuming clusters correspond to the classes
    cluster_size = 12  # Each cluster is 12x12
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
    savename = f'{fileinfo}_layer-{layer}_plot-rdm-{script}.png'
    savepath = os.path.join(opt['dir']['figures'], 'distances', savename)
    plt.savefig(savepath, dpi = 600)

    # Show the plot
    plt.show()  


## Plot the RDM of a given matrix of dissimilarities between categories of stimuli 
def plot_rdm_categories(opt, matrix, color_name, savename):
    '''
    Plot a 4-by-4 RDM: one script, one layer, averages of within-category distances.
    
    Parameters
    ----------
    opt (dict): output of vbs_option() containing the paths of the IODA folder
    
    matrix (array): the matrix to plot
    
    color_name (str): the common name of the code wanted for a given network-script
    
    savename (string): string with all the information of the bids-like name 
    
    Outputs
    -------
    Figure for the requested RDM, also saved in outputs/figures/distances
    
    '''
    # Select colour
    colors = {'orange': '#FF9E4A',        
              'green': '#69B5A2',
              'red': '#DA5F49',        
              'blue': '#4C75B3',
              'basic': '#000000'}
    
    # Create a custom colormap from white to the custom color
    custom_cmap = LinearSegmentedColormap.from_list('custom_cmap', ['white', colors[f'{color_name}']])

    
    # Start plotting 
    plt.figure()

    # Labels
    classes = ['RW', 'PW', 'NW', 'FS']
    ticks = [0.5, 1.5, 2.5, 3.5]
            
    ax = sns.heatmap(matrix, 
                      cmap = custom_cmap, 
                      annot = False, 
                      xticklabels = False, 
                      yticklabels = False)

    # Customize the heatmap
    title = savename
    ax.set_title(title, fontsize = 15, pad = 20)

    # Add sequence indicators as subtitles for the axis
    ax.set_aspect('equal')
    
    # Set axes
    ax.set_xticks(ticks)
    ax.set_xticklabels(classes, fontsize = 20)
    ax.tick_params(axis = 'x', length = 0)
    
    ax.set_yticks(ticks)
    ax.set_yticklabels(classes, fontsize = 20)
    ax.tick_params(axis = 'y', length = 0)
    
    # Save plot
    savepath = os.path.join(opt['dir']['figures'], 'distances', savename)
    plt.savefig(savepath, dpi = 600)

    # Show the plot
    plt.show()  



### ---------------------------------------------------------------------------
### Utils for plotting and data management

## Extract relevant logs from query
def logs_query(opt, model_name, script_trained):
    '''
    From the logs folder, extract the learning reports that match the 
    model and script trained
    
    Parameters
    ----------    
    opt (dict): output of vbs_option() containing the paths of the IODA folder
    
    model_name (string): the name of the model used (either 'alxenet' or 'cornet')
    
    script_trained (string): which type of training the network underwent
                             (either 'LT', 'BR', 'LN')
    
    Outputs
    -------
    trainings (list): list of paths to the files correspondingto the query
    
    '''
    
    # Match pattern
    pattern = f'model-{model_name}_training-{script_trained}_*'
    trainings = [os.path.basename(f) for f in glob.glob(os.path.join(opt['dir']['logs'], pattern))]
    
    # Sort files based on the timestamp at the end
    trainings.sort(key = lambda x: re.search(r'\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}', x).group() 
                   if re.search(r'\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}', x) else '')
    
    return trainings


## Concatenate multiple log files into one
def concatenate_trainings(opt, training_list):
    '''
    Concatenate a list of training logs, to get a single file for all the subjects
    
    Parameters
    ----------    
    opt (dict): output of vbs_option() containing the paths of the IODA folder
    
    training_list (list): list of paths to the trainings of indivudal subjects
    
    Outputs
    -------
    table in csv file containing the concatenated training logs
    
    '''
    
    # Savename for the concatenated trainings: take the last training and rename it
    savename = tr_list[-1]
    
    # Replace 'subjects-1' with 'all'
    savename = savename.replace('subjects-1', 'subjects-all')
    
    # Replace the last 8 characters with '00_00_00'
    savename = savename[:-12] + '00-00-00.csv'
    
    # Concatenate the logs of this training to the one of the full training
    trainings = [pd.read_csv(os.path.join(opt['dir']['logs'], tr)) for tr in training_list]
    total_training = pd.concat(trainings, ignore_index = True)
    total_training.to_csv(os.path.join(opt['dir']['logs'], savename), index = False)
        

## Differentiate subject numbers 
def reassing_subject(opt, training_list):
    '''
    Fix for a bug in the early logging of files: 
    function changes the subject number within the log to avoid mistaking networks
    
    Parameters
    ----------    
    opt (dict): output of vbs_option() containing the paths of the IODA folder
    
    training_list (list): list of paths to the trainings of indivudal subjects
    
    Outputs
    -------
    table in csv file containing the training logs with the correct subject ID
    
    '''
    
    # Open each file in the list of trainings
    for i, tr in enumerate(training_list):
        
        # Read the training
        training = pd.read_csv(os.path.join(opt['dir']['logs'], tr))
        
        # Change the subject ID to the corresponding of the iteration (i.e. from 0 to 4)
        if 'subject' in training.columns:
            
            training['subject'] = i
            
            # Save the log with the new subject ID
            training.to_csv(os.path.join(opt['dir']['logs'], tr), index = False)


