#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 27 10:49:27 2025

Visual Braille Silico - main function to run statistical analyses

@author: Filippo Cerpelloni
"""

# Import personal functions, then all libraries are coming from there
import sys
sys.path.append('../')

# Import custom functions to store paths, extraction of activations
from src.vbs_functions import * 
from stats_functions import *



### ---------------------------------------------------------------------------
### Load options 

opt = vbs_option()

# stats_correlate_distances(opt, 
#                           'model-alexnet_sub-all_training-LTBR_test-VBE_epoch-last_data-distances_average-categories_method-euclidean.pkl',
#                           'model-alexnet_sub-all_training-LT_test-VBE_epoch-last_data-distances_average-categories_method-euclidean.pkl')

# stats_correlate_distances(opt, 
#                           'model-cornet_sub-all_training-LTBR_test-VBE_epoch-last_data-distances_average-categories_method-euclidean.pkl',
#                           'model-cornet_sub-all_training-LT_test-VBE_epoch-last_data-distances_average-categories_method-euclidean.pkl')

stats_correlate_model(opt, 
                          'model-cornet_sub-all_training-LTBR_test-VBE_epoch-last_data-distances_average-categories_method-euclidean.pkl',
                          'model-cornet_sub-all_training-LT_test-VBE_epoch-last_data-distances_average-categories_method-euclidean.pkl')

stats_correlate_model(opt, 
                          'model-alexnet_sub-all_training-LTBR_test-VBE_epoch-last_data-distances_average-categories_method-euclidean.pkl',
                          'model-alexnet_sub-all_training-LT_test-VBE_epoch-last_data-distances_average-categories_method-euclidean.pkl')


### ---------------------------------------------------------------------------
### Clustering

# Measure clustering for each dissimilairty matrix of layers, groups, scripts. 
# Compute averages across categories (from single stimulus to categories) and 
# across subjects (from individual instances to network average). 
# Compute descriptive statistics and join the two clustering tables as one, to 
# facilitate rmANOVA steps. 
#
# IMPORTANT, please read. 
# rmANOVA is done in stats.Rproj, because of issues in computing it 
# through python and in using python to launch R. The whole project should be 
# standalone and should require only launching the main script to perfrom rmANOVAs


## AlexNet models

# Latin script training: naive networks
stats_clustering(opt, 'alexnet', 'LT', 'VBE', 'euclidean')
stats_average_categories(opt, 'alexnet', 'LT', 'VBE', 'euclidean')
stats_average_subjects(opt, 'alexnet', 'LT', 'VBE', 'euclidean')

# # Latin+Braille training: expert networks
stats_clustering(opt, 'alexnet', 'LTBR', 'VBE', 'euclidean')
stats_average_categories(opt, 'alexnet', 'LTBR', 'VBE', 'euclidean')
stats_average_subjects(opt, 'alexnet', 'LTBR', 'VBE', 'euclidean')

# Descriptive statistics of the two groups joint
stats_descriptive_clustering(opt, 
                             'model-alexnet_training-LT_test-VBE_data-clustering-matrices_method-euclidean.csv', 
                             'model-alexnet_training-LTBR_test-VBE_data-clustering-matrices_method-euclidean.csv')


## CORnet Z models

# Latin script training: naive networks
stats_clustering(opt, 'cornet', 'LT', 'VBE', 'euclidean')
stats_average_categories(opt, 'cornet', 'LT', 'VBE', 'euclidean')
stats_average_subjects(opt, 'cornet', 'LT', 'VBE', 'euclidean')

# Latin+Braille training: expert networks
stats_clustering(opt, 'cornet', 'LTBR', 'VBE', 'euclidean')
stats_average_categories(opt, 'cornet', 'LTBR', 'VBE', 'euclidean')
stats_average_subjects(opt, 'cornet', 'LTBR', 'VBE', 'euclidean')

# Descriptive statistics of the two groups joint
stats_descriptive_clustering(opt, 
                             'model-cornet_training-LT_test-VBE_data-clustering-matrices_method-euclidean.csv', 
                             'model-cornet_training-LTBR_test-VBE_data-clustering-matrices_method-euclidean.csv')



### ---------------------------------------------------------------------------
### Dissimilarity 

## AlexNet models 

stats_dissimilarity_scores(opt, 'alexnet', 'LT', 'VBE', 'correlation')
stats_dissimilarity_scores(opt, 'alexnet', 'LT', 'VBE', 'euclidean')
stats_dissimilarity_scores(opt, 'alexnet', 'LTBR', 'VBE', 'correlation')
stats_dissimilarity_scores(opt, 'alexnet', 'LTBR', 'VBE', 'euclidean')


## CORnet models 

stats_dissimilarity_scores(opt, 'cornet', 'LT', 'VBE', 'correlation')
stats_dissimilarity_scores(opt, 'cornet', 'LT', 'VBE', 'euclidean')
stats_dissimilarity_scores(opt, 'cornet', 'LTBR', 'VBE', 'correlation')
stats_dissimilarity_scores(opt, 'cornet', 'LTBR', 'VBE', 'euclidean')













