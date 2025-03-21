#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 28 14:42:49 2025

@author: Filippo Cerpelloni
"""

import os

def act_option():
    
    # Initialize options dictionary
    opt = {}

    ## PATHS
    
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
    
    # opt['dir']['extracted'] = os.path.join(opt['dir']['root'], 'outputs', 'derivatives', 'extracted-data')
    # opt['dir']['stats'] = os.path.join(opt['dir']['root'], 'outputs', 'derivatives', 'stats')
    
    opt['dir']['figures'] = os.path.join(opt['dir']['outputs'], 'figures')
    opt['dir']['weights'] = os.path.join(opt['dir']['outputs'], 'weights') 
    opt['dir']['logs'] = os.path.join(opt['dir']['outputs'], 'logs')
    
    opt['script'] = {'latin': {'dataset_spec': 'LT/', 
                               'notation': 'LT'},
                     'braille': {'dataset_spec': 'BR_LT', 
                                 'notation': 'BR'},
                     'line': {'dataset_spec': 'LN_LT', 
                              'notation': 'LN'}}

    return opt
