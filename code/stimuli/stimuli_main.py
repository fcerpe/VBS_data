#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 12 09:45:17 2024

Visual Braille Silico - main script to create stimuli 

@author: cerpelloni
"""
# Recap of steps 
import stim_create_letters_variations
import stim_create_words_variations
import stim_functions

# vbs_main

## stimuli_main
# From letters in each script (Latin Arial, Braille, Line Braille), create 
# visual variations in the size and thickness of the single letters. 
# 
# Saves everthing in: letters/variations
stim_create_letters_variations 

### Create word variations
# From the starting letters in Braille and Line Braille, create the words to be 
# used in the training of Braille 
# stim_create_words

# From the original words, implement the same variations made on the single letters
stim_create_words_variations

### Something else? Like checking pixel densities etc

## network_main
### Take istances of alexnet
### Test on letters without training
### Train on words

## analyses_main
### RDMs and correlations between chunks of letters 
### Decoding and whatever on words 

## visualization_main 
### Plot cool stuff


