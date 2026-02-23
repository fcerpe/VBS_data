#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  3 16:55:31 2025

Visual Braille Silico - main script to command visualization functions

All figures are to be considered building blocks for what is found in 
manuscripts / submissions. I usually refine figures in Affinity Designer, e.g. 
to change the font of a legend.

@author: Filippo Cerpelloni
"""

# Import personal functions, then all libraries are coming from there
import sys
sys.path.append('../')

# Import custom functions to store paths, extraction of activations
from src.vbs_functions import * 
from viz_functions import *



### ---------------------------------------------------------------------------
### Load options 

opt = vbs_option()



# ### ---------------------------------------------------------------------------
# ### Figures for thesis manuscript

# ## Figure 1 - experimental design
# # TBD


# ## Figure 2 - Letters representations in AlexNet
# # TBD


# ## Figure 3 - training paradigms and outcomes for both networks

# # Learning curves for AlexNet models
# # Output: outputs/figures/model-alexnet_plot-learning-curves.png
# plot_alexnet_training(opt, 
#                       'model-alexnet_sub-all_training-LT_date-2025-04-18_00-00-00.csv', 
#                       'model-alexnet_sub-all_training-LTBR_date-2025-02-02_00-00-00.csv', 
#                       'model-alexnet_sub-all_training-LTLN_date-2025-02-01_00-00-00.csv')

# # Learning curves for CORnet Z models
# # Output: outputs/figures/model-cornet_plot-learning-curves.png
# plot_cornet_training(opt, 
#                       'model-cornet_sub-all_training-LT_date-2025-04-22_00-00-00.csv', 
#                       'model-cornet_sub-all_training-LTBR_date-2025-03-29_00-00-00.csv', 
#                       'model-cornet_sub-all_training-LTLN_date-2025-04-11_00-00-00.csv')


# ## Figure 4 - clustering (and RDMs?) of word representations for both networks

# # Clustering plot for AlexNet models
# # Output: outputs/figures/model-alexnet_plot-clustering-across-training.png
# plot_clustering(opt, 'alexnet', 
#                 'model-alexnet_training-LTBR_test-VBE_data-clustering-matrices_method-euclidean.csv',
#                 'model-alexnet_training-LT_test-VBE_data-clustering-matrices_method-euclidean.csv')

# # Clustering plot for CORnet Z models
# # Output: outputs/figures/model-cornet_plot-clustering-across-training.png
# plot_clustering(opt, 'cornet', 
#                 'model-cornet_training-LTBR_test-VBE_data-clustering-matrices_method-euclidean.csv',
#                 'model-cornet_training-LT_test-VBE_data-clustering-matrices_method-euclidean.csv')

# # RDMs for each script and network at each layer
# plot_category_distances(opt, 'alexnet', 'LT',
#                         'model-alexnet_sub-all_training-LT_test-VBE_epoch-last_data-distances_average-subjects.pkl')

# plot_category_distances(opt, 'alexnet', 'LTBR', 
#                         'model-alexnet_sub-all_training-LTBR_test-VBE_epoch-last_data-distances_average-subjects.pkl')

# plot_category_distances(opt, 'cornet', 'LT', 
#                         'model-cornet_sub-all_training-LT_test-VBE_epoch-last_data-distances_average-subjects.pkl')

# plot_category_distances(opt, 'cornet', 'LTBR', 
#                         'model-cornet_sub-all_training-LTBR_test-VBE_epoch-last_data-distances_average-subjects.pkl')

# Potential extension of figure with dissimilarity scores 
plot_mean_dissimilarity(opt, 'alexnet',
                        'model-alexnet_training-LT_test-VBE_data-dissimilarity-scores-correlation.csv',
                        'model-alexnet_training-LTBR_test-VBE_data-dissimilarity-scores-correlation.csv')

plot_mean_dissimilarity(opt, 'cornet',
                        'model-cornet_training-LT_test-VBE_data-dissimilarity-scores-correlation.csv',
                        'model-cornet_training-LTBR_test-VBE_data-dissimilarity-scores-correlation.csv')


# ### ---------------------------------------------------------------------------
# ### Figures for CCN 2025 

# ## Extended abstact submission

# ## Figure 2 - AlexNet clustering of Braille script and RDMs of selected layers
# # Building blocks, then combined and polished in Affinity Designer

# # Clustering of distance matrices for Braille script 
# # Output: outputs/figures/model-alexnet_plot-clustering-braille.png
# plot_clustering(opt, 
#                 'alexnet',
#                 'braille',
#                 'model-alexnet_training-LTBR_test-VBE_data-clustering-matrices.csv',
#                 'model-alexnet_training-LT_test-VBE_data-clustering-matrices.csv')

# # RDMs at final features and classifier layers for both scripts
# # Output: outputs/figures/model-alexnet_plot-clustering.png
# plot_category_distances(opt, 
#                         'model-alexnet_sub-all_training-LT_test-VBE_data-distances_average-categories.pkl', 
#                         'alexnet',
#                         'expert')


# ## Poster

# # Figure X - learning curves for Braille and Line in the two networks
# plot_anova_training(opt, 'model-alexnet_sub-all_training-all_test-VBT_epoch-all_data-averaged-responses.csv')

# plot_anova_training(opt, 'model-cornet_sub-all_training-all_test-VBT_epoch-all_data-averaged-responses.csv')





