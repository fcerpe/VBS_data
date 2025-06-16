#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 27 11:08:10 2025

Visual Braille Silico - support functions for statistical analyses

@author: Filippo Cerpelloni
"""

### ---------------------------------------------------------------------------
### Imports for all the functions 

import sys
sys.path.append('../')

import os

import pandas as pd
import numpy as np
import pingouin as pg

from scipy.stats import pearsonr
from collections import defaultdict
from statsmodels.stats.multitest import fdrcorrection

import rpy2.robjects as robjects
from rpy2.robjects import pandas2ri
from rpy2.robjects.packages import importr

# Most generic support functions, like vbs_option(), can be found here
from src.vbs_functions import *



### ---------------------------------------------------------------------------
### Main statistical analyses 

## Measure clustering across RDMs of a network and script 
def stats_clustering(opt, model_name, training, test, method):
    """
    Compute clustering of the activation dissimilarity matrices across layers of
    a given network for a given script. 
    - load the activations
    - iterate through the layers 
    - obtain a general measure of clustering across layers 
    - save results to csv file
    
    Parameters
    ----------
    opt (dict): output of vbs_option() containing the paths of the IODA folder 
    
    model_name (str): the model refernce (whether 'alexnet' or 'cornet') to 
                         load the correct activations and to save the reuslts properly
    
    training (str): whether the network was trained on Latin+Brialle dataset, 
                    on Latin+Line, or Latin alone. Determines expertise 
                         
    test (str): the expertiment that is being analysed (VBE or VBT)
    
    Outputs
    -------
    None, but saves results using save_clustering
    
    """
    
    # Specify type of experiment and relative stimuli
    if test == 'VBE':   stimuli_spec = 'test_vbe'
    elif test == 'VBT': stimuli_spec = 'text_vbt'
    else: print('ERROR: experiment not found')
    
    # From expertise, introduce a notation more user-friendly
    if training == 'LT': expertise = 'naive'
    elif training == 'LTBR': expertise = 'braille_expert'
    elif training == 'LTLN': expertise = 'line_expert'
    else: print('ERROR: training not found')
    
    # Load labels (missing from the distance matrix) 
    labels = pd.read_csv(os.path.join(opt['dir']['inputs'], 'words', f'{stimuli_spec}_labels.csv'))
    
    # Extract classes (unique initial parts of the labels, e.g. RW - PW - NW - FS)
    classes = list(dict.fromkeys(labels["Labels"].str.split("_").str[0]))
    
    # Initialize storage of clustering values for each script and layer
    cluster_values = pd.DataFrame(columns = ['sub', 'expertise', 'script', 'layer', 'cat_i', 'cat_j', 'clustering'])
    cluster_matrix_averages = pd.DataFrame(columns = ['sub', 'expertise', 'script', 'layer', 'clustering'])
    cluster_subject_averages = pd.DataFrame(columns = ['expertise', 'script', 'layer', 'clustering'])
    
    # Browse each subject
    for s, subject in enumerate(opt['subjects']):
        
        # Specify the file to extract and load it 
        filename = f'model-{model_name}_sub-{s}_training-{training}_test-{test}_epoch-last_data-distances_method-{method}.pkl'
        filepath = os.path.join(opt['dir']['results'], 'distances', filename)
    
        activations = load_extraction(filepath)
        
        # Browse each script 
        # (should be LT and BR, but who knows where this project will end)
        for c, scr in enumerate(activations.keys()):
        
            # Browse each layer of the network
            for l, layer in enumerate(activations[f'{scr}'].keys()):
                
                # Get the matrix for easier coding 
                matrix = activations[f'{scr}'][f'{layer}']
                
                # Store the values in a separate table for easier averages
                matrix_values = pd.DataFrame(columns = ['sub', 'expertise', 'script', 'layer', 'cat_i', 'cat_j', 'clustering'])
        
                # Iterate through the categories of stimuli for which we compute clustering
                for i, i_cat in enumerate(classes): 
                    
                    # Find the indexes of the labels corresponding to this class
                    i_idxs = labels.index[labels["Labels"].str.startswith(f'{i_cat}')].tolist()
                    i_min = i_idxs[0]
                    i_max = i_idxs[-1] +1
                    
                    for j, j_cat in enumerate(classes): 
                                        
                        # Only compute clustering if the two labels, classes, categories 
                        # are different
                        if i != j: 
                            
                            # Find the indexes of the labels corresponding to this class
                            j_idxs = labels.index[labels["Labels"].str.startswith(f'{j_cat}')].tolist()
                            j_min = j_idxs[0]
                            j_max = j_idxs[-1] +1
                            
                            # Extract groups of distances based on class indexes
                            # i = within-class i distances 
                            # j = within-class j distances 
                            # ij = distances between classes i,j
                            cluster_i = matrix[i_min:i_max, i_min:i_max]
                            cluster_j = matrix[j_min:j_max, j_min:j_max]
                            cluster_ij = matrix[j_min:j_max, i_min:i_max]
                            
                            # Clustering = (mean(i,j) - mean(i, j)) / (mean(i,j) + mean(i, j))
                            score_ij = compute_clustering(cluster_i, cluster_j, cluster_ij)
            
                            # Add to table, together with information about script, layer, classes
                            entry = pd.DataFrame([[s, expertise, scr, layer, i_cat, j_cat, score_ij]], 
                                                 columns = cluster_values.columns)
                            
                            # Add to: all the values, to the single matrix
                            cluster_values = pd.concat([cluster_values, entry], ignore_index = True)
                            matrix_values = pd.concat([matrix_values, entry], ignore_index = True)
                
                # Average pairwise clustering into a matrix-wide measure
                matrix_average = np.mean(matrix_values['clustering'])
                matrix_entry = pd.DataFrame([[s, expertise, scr, layer, matrix_average]], 
                                            columns = cluster_matrix_averages.columns)
                cluster_matrix_averages = pd.concat([cluster_matrix_averages, matrix_entry])
                
    # Average across subjects for a single layer
    cluster_subject_averages = cluster_matrix_averages.groupby(['layer','script'])[['clustering']].mean().reset_index()
    
    # Save individual and averaged data
    save_clustering(opt, cluster_values, cluster_matrix_averages, cluster_subject_averages, 
                    model_name, training, test, method)


## Compute ANOVA on clustering, for features layers 
def stats_descriptive_clustering(opt, clustering_naives, clustering_experts): 
    """
    Compute descriptive statistics and rmANOVA for the clustering across 
    subjets and layers
    
    Parameters
    ----------
    opt (dict): output of vbs_option() containing the paths of the IODA folder 
    
    clustering_naives (str): filename of the clustering values for each subject, 
                             layer (..._data-clustering-matrices.csv) ofthe naive
                             group
                            
    clustering_experts (str): filename of the clustering values for each subject, 
                              layer (..._data-clustering-matrices.csv) of the expert
                              group

    Outputs
    -------
    None, but stats csv files are created in outputs/results/clustering

    """

    # Import clustering data
    naives = pd.read_csv(os.path.join(opt['dir']['results'], 'clustering', clustering_naives))
    experts = pd.read_csv(os.path.join(opt['dir']['results'], 'clustering', clustering_experts))

    # Concatenate data into one big table
    clusters = pd.concat([naives, experts])
    
    # Get descriptive stats
    clusters_desc = clusters.groupby(['expertise', 'script', 'layer']).agg(n = ('clustering', 'count'),
                                                                           mean = ('clustering', 'mean'),
                                                                           std = ('clustering', 'std')).reset_index()
    
    # Unfortunately, I am not (yet) able to compute a rmANOVA (two within factors
    # and one between) in python. I tried pingouin, statsmodel, rpy2. 
    
    # Refer to stats.Rproj and it will create the rmANOVAs for both models
    # (alexnet and cornet) and save them in the right folders and with the correct
    # naming, so that 'viz' can read them
    
    # Parcel matrices filename to extract model and experiment, to construct the name
    # with which to save the stats
    filename = clustering_naives.split('/')[-1]
    parts = filename.split('_')
    model = parts[0]
    test = parts[2]
    
    savename = f'{model}_training-all_{test}_data-clustering_analysis'
    cluster_savename = f'{model}_training-all_{test}_data-clustering-matrices'
    
    # Save results
    clusters.to_csv(os.path.join(opt['dir']['results'], 'clustering', f'{cluster_savename}.csv'), index = False)
    clusters_desc.to_csv(os.path.join(opt['dir']['results'], 'clustering', f'{savename}-descriptives-clusters.csv'), index = False)
    # clusters_anova.to_csv(os.path.join(opt['dir']['results'], 'clustering', f'{savename}-anova-clusters.csv'), index = False)


## Correlate output layer representations between expertise and script
def stats_correlate_distances(opt, distances_experts, distances_controls):
    """
    Correlate the distance matrices of subjects between each other, performing
    non-parametric stats (permutations). Should get 'average-categories' as input
    
    Parameters
    ----------
    opt (dict): output of vbs_option() containing the paths of the IODA folder 
    
    distances_experts (str): name of the distances to load, refering to the 
                             expert group (categories-level)
                            
    distances_controls (str): name of the distances to load, refering to the 
                              control group (categories-level)
    
    Outputs
    -------
    None, but stats csv files are created in outputs/results/clustering

    """
    
    # Load distances for each subject and group
    experts = load_extraction(os.path.join(opt['dir']['results'], 'distances', distances_experts))
    controls = load_extraction(os.path.join(opt['dir']['results'], 'distances', distances_controls))
    
    # Get filename information (bad code, I know)
    fileinfo = distances_experts.split('_training')[0]
    savename = f'{fileinfo}_test-VBE_epoch-last_data-distances_analysis-last-layer-correlations.csv'
    
    # Extract groups and pairs to compare, to ease iterations 
    matrices = {}
    matrices['exp_lt'] = {}
    matrices['exp_br'] = {}
    matrices['ctr_lt'] = {}
    matrices['ctr_br'] = {}
    
    last_layer = list(experts['sub-0']['LT'].keys())[-1]
    
    for s in opt['subjects']:
        matrices['exp_lt'][f'sub-{s}'] = experts[f'sub-{s}']['LT'][last_layer]
        matrices['exp_br'][f'sub-{s}'] = experts[f'sub-{s}']['BR'][last_layer]
        matrices['ctr_lt'][f'sub-{s}'] = controls[f'sub-{s}']['LT'][last_layer]
        matrices['ctr_br'][f'sub-{s}'] = controls[f'sub-{s}']['BR'][last_layer]
        
    # define the comparisons to make
    comparisons = [['exp_lt', 'exp_br'], 
                   ['exp_lt', 'ctr_lt'], 
                   ['exp_lt', 'ctr_br'], 
                   ['exp_br', 'ctr_lt'], 
                   ['exp_br', 'ctr_br'], 
                   ['ctr_lt', 'ctr_br']]
    
    # Initialize results to store observed correlations and p values
    results = []
    
    # Correlate distances, using permutations 
    # Iterate through the comparisons
    for c, comp in enumerate(comparisons):
        
        # Pick the conditions
        g1name = comp[0]
        g2name = comp[1]
        
        g1 = matrices[g1name]
        g2 = matrices[g2name]
        
        # Init arrays for groups correlations and null distributions
        s1g2corr = np.zeros([len(opt['subjects']), 1])
        s2g1corr = np.zeros([len(opt['subjects']), 1])
        
        s1g2null = np.zeros([len(opt['subjects']), 10000])
        s2g1null = np.zeros([len(opt['subjects']), 10000])
        
        # Compare each subject of group 1 against the average of group 2
        for s1, sub1 in enumerate(g1.keys()):
            
            # Get subject matrix
            s1mat = g1[sub1]
            
            # Concatenate the matrices to average.
            # If we are comparing the smae network (group), exclude the subject's 
            # data from the average 
            if g1name[0:3] == g2name[0:3]: subs = [mat for key, mat in g2.items() if key != f'sub-{s1}']
            else: subs = list(g2.values())
                
            # Get average of group 2 (without subject 1, if same network)
            g2mat = np.mean(subs, axis = 0)
            
            # Extract offdiagonal for both subject and group
            s1tri = s1mat[np.tril_indices(s1mat.shape[0], k = -1)]
            g2tri = g2mat[np.tril_indices(g2mat.shape[0], k = -1)]
            
            # Compute correlation and store it 
            s1g2corr[s1] = np.array(pearsonr(s1tri, g2tri)[0])
            
            # Create null distribution
            s1g2null[s1,:] = compute_permutations(s1tri, g2tri, 10000)
            
        # Compare each subject of group 1 against the average of group 2
        for s2, sub2 in enumerate(g2.keys()):
            
            # Get subject matrix
            s2mat = g2[sub2]
            
            # Concatenate the matrices to average
            if g1name[0:3] == g2name[0:3]: subs = [mat for key, mat in g1.items() if key != f'sub-{s2}']
            else: subs = list(g2.values())
              
            # Get group average from selected subjects
            g1mat = np.mean(subs, axis = 0)
            
            # Extract offdiagonal for both subject and group
            s2tri = s2mat[np.tril_indices(s2mat.shape[0], k = -1)]
            g1tri = g1mat[np.tril_indices(g1mat.shape[0], k = -1)]
            
            # Compute correlation and store it 
            s2g1corr[s2] = np.array(pearsonr(s2tri, g1tri)[0])
            
            # Create null distribution
            s2g1null[s2,:] = compute_permutations(s2tri, g1tri, 10000)
            
        # Make grand averages of correlations and of null distributions
        g1g2corr = np.mean([np.mean(s1g2corr), 
                           np.mean(s2g1corr)])
        g1g2null = np.mean([np.mean(s1g2null, axis = 0), 
                           np.mean(s2g1null, axis = 0)], axis = 0)
        
        # Compute p-value
        g1g2pval = np.sum(g1g2corr <= g1g2null) / len(g1g2null)
    
        # Add results to table
        results.append({'group1': g1name,
                        'group2': g2name,
                        'correlation': g1g2corr,
                        'pval_uncorr': g1g2pval})
        
    # Convert the results to a DataFrame
    results = pd.DataFrame(results)
    
    # Correct p-values for False Detection Rate (FDR-corr)
    _, corrected = fdrcorrection(results['pval_uncorr'], alpha = 0.05, method = 'indep')
    results['pval_fdr'] = corrected

    # Save results 
    results.to_csv(os.path.join(opt['dir']['results'], 'distances', savename), index = False)



## Correlate clusters with language model
def stats_correlate_model(opt, distances_experts, distances_controls): 
    """
    Correlate the distance matrices of subjects with language model, performing
    non-parametric stats (permutations). Should get 'average-categories' as input
    
    Parameters
    ----------
    opt (dict): output of vbs_option() containing the paths of the IODA folder 
    
    distances_filename (str): name of the distances pickle to load, the average 
                              on the categories-level with the individual network data
    
    Outputs
    -------
    None, but stats csv files are created in outputs/results/clustering

    """
    
    # Load distances for each subject
    experts = load_extraction(os.path.join(opt['dir']['results'], 'distances', distances_experts))
    controls = load_extraction(os.path.join(opt['dir']['results'], 'distances', distances_controls))
    
    # Initialize language model 
    language = np.array([[0, 1, 2, 3], 
                         [1, 0, 1, 2], 
                         [2, 1, 0, 1], 
                         [3, 2, 1, 0]])
    
    # Extract off-diagonal (unnecessary step)
    language = language[np.tril_indices(language.shape[0], k = -1)]
    
    # Initialize results to store observed correlations and p values
    results = []
    
    # TODO Parce filename to extract relevant information over the file
    # At the moment savename is fixed (bad practice I know)
    fileinfo = distances_experts.split('_training')[0]
    
    # Reorganize the distances from sub-script-layer to script-layer-sub
    reorganized = defaultdict(lambda: defaultdict(dict))
    
    for sub, scripts in experts.items():
        for script, layers in scripts.items():
            for layer, value in layers.items():
                reorganized[script][layer][sub] = value
                
    experts = reorganized
    
    reorganized = defaultdict(lambda: defaultdict(dict))
    
    for sub, scripts in controls.items():
        for script, layers in scripts.items():
            for layer, value in layers.items():
                reorganized[script][layer][sub] = value
                
    controls = reorganized
    
    # Experts
    # Correlate them with model, need to use permutations? 
    # Iterate through each subject, script, layer
    for c, scr in enumerate(experts.keys()):
        
        clu = f'exp-{scr}'
        
        last_layer = list(experts[scr].keys())[-1]
            
        # Initialize storage of correlations and permutations
        # Dimensions are nb of subejcts, nb of values per subject
        correlations = np.zeros([len(experts[f'{scr}'][last_layer].keys()), 1])
        permutations = np.zeros([len(experts[f'{scr}'][last_layer].keys()), 10000])
        
        for s, sub in enumerate(experts[f'{scr}'][last_layer].keys()): 
            
            # From overall distance matrix, extract the lower off-diagonal
            distance = experts[f'{scr}'][last_layer][f'{sub}']
            offdiagonal = distance[np.tril_indices(distance.shape[0], k = -1)]
            
            # Compute correlateion between specific distance matrix and language model 
            correlations[s] = np.array(pearsonr(offdiagonal, language)[0])
            
            # Compute permutations to create null distribution
            permutations[s,:] = compute_permutations(offdiagonal, language, 10000)
            
        # Average observed correlations and null distributions across subjects
        correlation = np.mean(correlations)
        permutation = np.mean(permutations, 0)
        
        # Compute significance of the averaged correlations
        pvalue = np.sum(correlation <= permutation) / len(permutation)
        
        # Add the script, layer, observed correlation, p-value uncorrected
        # to the results
        results.append({
            'cluster': clu,
            'layer': last_layer,
            'correlation': correlation,
            'pval_uncorr': pvalue})
            
            
    # Controls
    # Correlate them with model, need to use permutations? 
    # Iterate through each subject, script, layer
    for c, scr in enumerate(controls.keys()):
        
        clu = f'ctr-{scr}'
        
        last_layer = list(experts[scr].keys())[-1]
        
        # Initialize storage of correlations and permutations
        # Dimensions are nb of subejcts, nb of values per subject
        correlations = np.zeros([len(controls[f'{scr}'][last_layer].keys()), 1])
        permutations = np.zeros([len(controls[f'{scr}'][last_layer].keys()), 10000])
        
        for s, sub in enumerate(controls[f'{scr}'][last_layer].keys()): 
            
            # From overall distance matrix, extract the lower off-diagonal
            distance = controls[f'{scr}'][last_layer][f'{sub}']
            offdiagonal = distance[np.tril_indices(distance.shape[0], k = -1)]
            
            # Compute correlateion between specific distance matrix and language model 
            correlations[s] = np.array(pearsonr(offdiagonal, language)[0])
            
            # Compute permutations to create null distribution
            permutations[s,:] = compute_permutations(offdiagonal, language, 10000)
            
        # Average observed correlations and null distributions across subjects
        correlation = np.mean(correlations)
        permutation = np.mean(permutations, 0)
        
        # Compute significance of the averaged correlations
        pvalue = np.sum(correlation <= permutation) / len(permutation)
        
        # Add the script, layer, observed correlation, p-value uncorrected
        # to the results
        results.append({
            'cluster': clu,
            'layer': last_layer,
            'correlation': correlation,
            'pval_uncorr': pvalue})
                    
    # Convert the results to a DataFrame
    results = pd.DataFrame(results)
    
    # Correct p-values for False Detection Rate (FDR-corr)
    _, corrected = fdrcorrection(results['pval_uncorr'], alpha = 0.05, method = 'indep')
    results['pval_fdr'] = corrected

    # Create filename (incorporate filename parcing)
    savename = f'{fileinfo}_test-VBE_epoch-last_data-distances_analysis-language-model-correlations.csv'
    
    # Save results 
    results.to_csv(os.path.join(opt['dir']['results'], 'distances', savename), index = False)
    

## Compute one value of dissimilarity for the whole layer
def stats_dissimilarity_scores(opt, model_name, training, test, method):
    """
    Compute one value of dissimilarity for each subject / script / layer of a 
    given network architecture
    
    Parameters
    ----------
    opt (dict): output of vbs_option() containing the paths of the IODA folder 
    
    model_name (str): the model refernce (whether 'alexnet' or 'cornet') to 
                         load the correct activations and to save the reuslts properly
    
    training (str): whether the network was trained on Latin+Brialle dataset, 
                    on Latin+Line, or Latin alone. Determines expertise 
                         
    test (str): the expertiment that is being analysed (VBE or VBT)
    
    method (str): the method used to compute distances, usually either 'euclidian'
                  or 'correlation'
    
    Outputs
    -------
    None, but saves results in outputs/results/distances 
    
    """
    
    # Convert training into more-readable experitse
    if training == 'LT': expertise = 'naive'
    else: expertise = 'expert' 
    
    # Initialize storage of dissimilarity scores for each script and layer
    dissimilarities = pd.DataFrame(columns = ['sub', 'expertise', 'script', 'layer', 'dissimilarity'])
    
    # Browse each subject
    for s, subject in enumerate(opt['subjects']):
        
        # Specify the file to extract and load it 
        filename = f'model-{model_name}_sub-{s}_training-{training}_test-{test}_epoch-last_data-distances_method-{method}.pkl'
        filepath = os.path.join(opt['dir']['results'], 'distances', filename)
    
        distances = load_extraction(filepath)
        
        # Browse scripts and layers 
        for c, scr in enumerate(distances.keys()):
            
            for l, layer in enumerate(distances[f'{scr}'].keys()):
                
                # Average distances within a layer and script into a single value
                diss = np.mean(distances[f'{scr}'][f'{layer}'])
                
                # Combine information into one entry for the output csv file
                entry = pd.DataFrame([[s, expertise, scr, layer, diss]], 
                                     columns = dissimilarities.columns)
                
                # Concatenate the entry to the full table
                dissimilarities = pd.concat([dissimilarities, entry], ignore_index = True)
            
    # Set path and save the dissimilarity scores
    filepath = os.path.join(opt['dir']['results'], 'distances', 
                            f'model-{model_name}_training-{training}_test-{test}_data-dissimilarity-scores-{method}.csv')
    
    dissimilarities.to_csv(filepath, index = False)



### ---------------------------------------------------------------------------
### Support functions - computations

## From the clusters of a matrix, compute the clustering measure
def compute_clustering(i, j, ij): 
    """
    Compute the clustering measure (normalized) between two clusters i and j, and 
    their intersection ij 
    
    Parameters
    ----------
    i (array): the first cluster
    
    j (array): the second cluster
    
    ij (array): the intersection of the two clusters
    
    Outputs
    -------
    clustering (float): the resulting difference, between 0 and 1
    
    """
    
    # Clustering = (mean(i,j) - mean(i, j)) / (mean(i,j) + mean(i, j))
    mean_ij = np.mean(ij)
    mean_i = np.mean(i)
    mean_j = np.mean(j)
    mean_i_j = (mean_i + mean_j)/2
    
    cl = (mean_ij - mean_i_j) / np.mean([mean_i, mean_j, mean_ij])
    
    return cl


## From activations, make averages for visualization porpuses
def stats_average_categories(opt, model_name, training, test, method):
    """
    Extract the average activation for each category from each dissimilarity 
    matrix across layers and subjects. 
    - load the activations
    - iterate through the layers 
    - average activations for a given "block" in the RDM 
    - save results to csv file
    
    Parameters
    ----------
    opt (dict): output of vbs_option() containing the paths of the IODA folder 
    
    model_name (string): the model refernce (whether 'alexnet' or 'cornet') to 
                         load the correct activations and to save the reuslts properly
    
    training (str): whether the network was trained on Latin+Brialle dataset, 
                    on Latin+Line, or Latin alone. Determines expertise 
                         
    test (str): the expertiment that is being analysed (VBE or VBT)
    
    Outputs
    -------
    None
    
    """
    
    # Specify type of experiment and relative stimuli
    if test == 'VBE':   stimuli_spec = 'test_vbe'
    elif test == 'VBT': stimuli_spec = 'text_vbt'
    else: print('ERROR: experiment not found')
    
    # Load labels (missing from the distance matrix) 
    labels = pd.read_csv(os.path.join(opt['dir']['inputs'], 'words', f'{stimuli_spec}_labels.csv'))
    
    # Extract classes (unique initial parts of the labels, e.g. RW - PW - NW - FS)
    classes = list(dict.fromkeys(labels["Labels"].str.split("_").str[0]))
    
    # Initialize storage of clustering values for each script and layer
    averages_dict = {}
    
    # Browse each subject
    for s, subject in enumerate(opt['subjects']):
        
        # Initialize subject-specific storage
        averages_dict[f'sub-{s}'] = {}
        
        # Specify the file to extract and load it 
        filename = f'model-{model_name}_sub-{s}_training-{training}_test-{test}_epoch-last_data-distances_method-{method}.pkl'
        filepath = os.path.join(opt['dir']['results'], 'distances', filename)
    
        activations = load_extraction(filepath)
        
        # Browse each script 
        # (should be LT and BR, but who knows where this project will end)
        for c, scr in enumerate(activations.keys()):
            
            # Initialize script-specific storage
            averages_dict[f'sub-{s}'][f'{scr}'] = {}
        
            # Browse each layer of the network
            for l, layer in enumerate(activations[f'{scr}'].keys()):
                                
                # Get the matrix for easier coding 
                matrix = activations[f'{scr}'][f'{layer}']
                
                # Initialize matrix with average results 
                averages = np.zeros((4, 4), dtype = float)
                        
                # Iterate through the categories of stimuli for which we compute clustering
                for i, i_cat in enumerate(classes): 
                    
                    # Find the indexes of the labels corresponding to this class
                    i_idxs = labels.index[labels["Labels"].str.startswith(f'{i_cat}')].tolist()
                    i_min = i_idxs[0]
                    i_max = i_idxs[-1] +1
                    
                    for j, j_cat in enumerate(classes): 
                                        
                        # Find the indexes of the labels corresponding to this class
                        j_idxs = labels.index[labels["Labels"].str.startswith(f'{j_cat}')].tolist()
                        j_min = j_idxs[0]
                        j_max = j_idxs[-1] +1
                        
                        # Extract groups of distances based on class indexes
                        cluster_ij = matrix[j_min:j_max, i_min:i_max]
                        
                        # Average the group
                        avg_ij = np.mean(cluster_ij)
        
                        # Add to matrix
                        averages[i,j] = avg_ij 
                    
                # Assign average matrix to the corresponding dictionary entry 
                averages_dict[f'sub-{s}'][f'{scr}'][f'{layer}'] = averages
                
    # Save individual and averaged data
    save_single_extraction(opt, 
                           averages_dict, 
                           model_name, 
                           'all', 
                           training, 
                           test,
                           'last',
                           'distances', 
                           f'distances_average-categories_method-{method}')

       
## Average the classes RDMs across subjects 
def stats_average_subjects(opt, model_name, training, test, method):
    """
    Average the category-level matrices of activation dissimilarity across subjects
    - load the activations
    - iterate through the layers 
    - average activations for a given "block" in the RDM 
    - save results to csv file
    
    Parameters
    ----------
    opt (dict): output of vbs_option() containing the paths of the IODA folder 
    
    model_name (string): the model refernce (whether 'alexnet' or 'cornet') to 
                         load the correct activations and to save the reuslts properly
    
    training (str): whether the network was trained on Latin+Brialle dataset, 
                    on Latin+Line, or Latin alone. Determines expertise 
                         
    test (str): the expertiment that is being analysed (VBE or VBT)
    
    Outputs
    -------
    None
    
    """
    
    # Load the category averages 
    # Specify the file to extract and load it 
    filename = f'model-{model_name}_sub-all_training-{training}_test-{test}_epoch-last_data-distances_average-categories_method-{method}.pkl'
    filepath = os.path.join(opt['dir']['results'], 'distances', filename)

    averages = load_extraction(filepath)
    
    # Create script dictionary with averages across subjects
    sub_averages = {'LT': {}, 'BR': {}}
    
    # Browse each script 
    # (should be LT and BR, but who knows where this project will end)
    for c, scr in enumerate(averages['sub-0'].keys()):
    
        # For each layer 
        # Browse each layer of the network
        for l, layer in enumerate(averages['sub-0'][f'{scr}'].keys()):
        
            # Latin script: extract matrices for every subject at a given layer
            matrices = [averages[f'sub-{s}'][f'{scr}'][f'{layer}'] for s, sub in enumerate(opt['subjects'])]
        
            # Convert to numpy and average
            array = np.array(matrices)  
            average = np.mean(array, axis = 0)
            
            # Assign to dictionary
            sub_averages[f'{scr}'][f'{layer}'] = average
            
    # Save averaged data
    save_single_extraction(opt, 
                           sub_averages, 
                           model_name, 
                           'all', 
                           training, 
                           test,
                           'last',
                           'distances', 
                           f'distances_average-subjects_method-{method}')


## Compute permutations on correlationsbetween distances and model
def compute_permutations(subject, model, permutations):
    """
    Shuffles the subject data and computes permutations for the correlation 
    between a single (shuffled) subject and a model

    Parameters
    ----------
    subject (array): vector of the subject's distances
    
    model (array): vector of the model's theoretical activations 
                              
    permutations (int): number of cycles

    Outputs
    -------
    null_distribution (array): correlations for every permutation
    """
    
    # Set up random generator
    rng = np.random.default_rng(None)

    # Ensure data is in numpy format
    subject = np.asarray(subject)
    model = np.asarray(model)

    # Generate null distribution by permuting the subject
    null_distribution = np.empty(permutations)
    
    for i in range(permutations):
        
        # Randomly permute the subject vector (keeping model fixed)
        subject_permuted = rng.permutation(subject)

        # Compute correlation between subject and permuted model
        null_distribution[i], _ = pearsonr(subject_permuted, model)

    return null_distribution



### ---------------------------------------------------------------------------
### Saving functions

## Save clustering values as csv
def save_clustering(opt, values, matrix, subject, model_name, training, test, method):
    """
    Save clustering compputed for individual pairs of conditions, for full matrices, 
    for layer averages across subjects 
    
    Parameters
    ----------
    opt (dict): output of vbs_option() containing the paths of the IODA folder 
    
    values (DataFrame): the scores for single pairs of classes
    
    matrix (DataFrame): the scores averaged across pairs, one value per RDM
    
    subject (DataFrame): the scores averaged across pairs and subjects, one
                         value per script and layer
                         
    model_name (str): the model reference (whether 'alexnet' or 'cornet')
    
    training (str): whether the network was trained on Latin+Brialle dataset, 
                    on Latin+Line, or Latin alone. Determines expertise 
                         
    test (str): the expertiment that is being analysed (VBE or VBT)
    
    Outputs
    -------
    None, but data-clustering csv files are created in outputs/results/clustering

    """
    
    # Set path and common filename
    filename = f'model-{model_name}_training-{training}_test-{test}_data-clustering'
    filepath = os.path.join(opt['dir']['results'], 'clustering', filename)
    
    # Save single comparisons
    values.to_csv(f'{filepath}-values_method-{method}.csv', index = False)
    
    # Save averages across comparisons
    matrix.to_csv(f'{filepath}-matrices_method-{method}.csv', index = False)
    
    # Save averages across subjects
    subject.to_csv(f'{filepath}-averages_method-{method}.csv', index = False)
    


    
    
    
    
    
    
    
    
    
    
    
    
    
    
    