#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 10 15:37:32 2025

@author: cerpelloni
"""

import os, glob, re
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import sem

# Options (just paths for now)
def viz_option(): 
    
    # Initialize options dictionary
    opt = {}

    # PATHS
    # IMPORTANT: paths are optimized to run on enuui, adjust if on a different system
    # Enuii: /data/Filippo/  ||  IODA: ../..

    # The directory where the data are located
    opt['dir'] = {}
    opt['dir']['root'] = os.path.join(os.getcwd(), '..', '..')
    opt['dir']['inputs'] = os.path.join(opt['dir']['root'], 'inputs')
    opt['dir']['outputs'] = os.path.join(opt['dir']['root'], 'outputs')
    
    # Bypass IODA folder sturcture for training on enuui
    # opt['dir']['datasets'] = os.path.join(opt['dir']['root'], 'inputs', 'datasets')
    opt['dir']['inputs'] = '../../inputs'
    opt['dir']['datasets'] = '../../inputs/datasets'
    
    opt['dir']['figures'] = os.path.join(opt['dir']['outputs'], 'figures')
    opt['dir']['weights'] = os.path.join(opt['dir']['outputs'], 'weights') 
    opt['dir']['logs'] = os.path.join(opt['dir']['outputs'], 'logs')

    return opt


# Extract relevant logs from query
def logs_query(model, training, opt):
    
    pattern = f"model-{model}_training-{training}_*"
    trainings = [os.path.basename(f) for f in glob.glob(os.path.join(opt['dir']['logs'], pattern))]
    
    # Sort files based on the timestamp at the end
    trainings.sort(key = lambda x: re.search(r'\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}', x).group() 
                   if re.search(r'\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}', x) else "")
    
    return trainings


# Concatenate multiple log files into one
# 
# I often ran training on BR and LN individually (not in batches of 5 iterations)
# so the log files are split. This script combines multiple log files into one, 
# to ease plotting
def concat_trainings(tr_list, opt):
    
    # Savename for the concatenated trainings: take the last training and rename it
    savename = tr_list[-1]
    
    # Replace 'subjects-1' with 'subjects-5'
    savename = savename.replace('subjects-1', 'subjects-5')
    
    # Replace the last 8 characters with '00_00_00'
    savename = savename[:-12] + '00-00-00.csv'
    
    
    # Concatenate the logs of this training to the one of the full training
    trainings = [pd.read_csv(os.path.join(opt['dir']['logs'], tr)) for tr in tr_list]
    total_training = pd.concat(trainings, ignore_index = True)
    total_training.to_csv(os.path.join(opt['dir']['logs'], savename), index = False)
        

# 
def reassing_subject(tr_list, opt):
    
    # Open each file in the list of trainings
    for i, tr in enumerate(tr_list):
        
        # Read the training
        training = pd.read_csv(os.path.join(opt['dir']['logs'], tr))
        
        # Change the subject ID to the corresponding of the iteration (i.e. from 0 to 4)
        if "subject" in training.columns:
            
            training["subject"] = i
            
            # Save the log with the new subject ID
            training.to_csv(os.path.join(opt['dir']['logs'], tr), index = False)
            
            

# Plot just the training on Latin alphabet (literacy acquisition)
def plot_training_literacy(lt_training, model):
    
    return 0

# Plot the training on Latin alphabet (literacy) + the novel alphabet (expertise)
def plot_training_expertise(lt_training, br_training, ln_training, model, opt): 
    
    ##  Open literacy and expertise logs
    lt = pd.read_csv(os.path.join(opt['dir']['logs'], lt_training))
    br = pd.read_csv(os.path.join(opt['dir']['logs'], br_training))
    ln = pd.read_csv(os.path.join(opt['dir']['logs'], ln_training))
    
    # Arrange data
    all_training = pd.concat([lt, br, ln], ignore_index = True)    
    
    ## Plot data
    
    # Compute Mean and Confidence Interval (CI)
    grouped = all_training.groupby(['Script', 'Epoch']).agg(
        mean_accuracy=('Val_accuracy', 'mean'),
        sem_accuracy=('Val_accuracy', sem)
    ).reset_index()
    
    # Shift BR epochs
    grouped['Epoch'] = grouped['Epoch'] + 9 * (grouped['Script'] == 'BR')
    grouped['Epoch'] = grouped['Epoch'] + 9 * (grouped['Script'] == 'LN')
    
    # Compute 95% Confidence Interval
    grouped['ci'] = 1.96 * grouped['sem_accuracy']
    grouped['upper'] = grouped['mean_accuracy'] + grouped['ci']
    grouped['lower'] = grouped['mean_accuracy'] - grouped['ci']

    
    # Define colors
    colors = {'LT': '#699AE5', 
              'BR': '#FF9E4A', 
              'LN': '#69B5A2'}
    labels = {'LT': 'Latin', 
              'BR': 'Latin + Braille', 
              'LN': 'Latin + Line Braille'}
    
    # Plot
    plt.figure(figsize=(8, 6))
    
    for script in grouped['Script'].unique():
        subset = grouped[grouped['Script'] == script]
    
        # Line plot for mean accuracy
        plt.plot(subset['Epoch'], subset['mean_accuracy'], 
                 marker='', linewidth = 5,
                 label = labels[script], color = colors[script])
    
        # Confidence interval as a shaded area (halo)
        plt.fill_between(subset['Epoch'], subset['lower'], subset['upper'], 
                         color = colors[script], alpha = 0.2)

    # Add vertical dashed line to spearate trainings
    plt.axvline(x = 10, color = 'black', linestyle = '--', linewidth = 3)

    plt.xticks([5, 10, 14, 19], [5, 10, 15, 20])
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Mean Accuracy over literacy and expertise')
    plt.legend(loc = 'lower right', title = 'Script learnt')
    plt.grid(True)
    
    # Save the plot
    plt.savefig("../../outputs/figures/model-alexnet_plot-joint-learning.png", 
                dpi = 600, bbox_inches = 'tight')
    
    # Show the plot
    plt.show()
    
    
    return 0









