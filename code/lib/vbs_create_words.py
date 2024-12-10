#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  7 15:59:50 2024

WORK IN PROGRESS TRANSLATION FROM MATLAB

@author: cerpelloni
"""

import os
import pygame
import pandas as pd
import scipy.io
import numpy as np
from PIL import Image

# Load list of words from source/nl_wordslist.csv

data = scipy.io.loadmat("/Users/cerpelloni/Documents/GitHub/VisualBraille_backstageCode/visbra_training/vbdt/dutch_manual_selection.mat")

br_words = {}
cb_words = {}

# Initialize pygame
pygame.init()
num_displays = pygame.display.get_num_displays()

# Use the first display (index 0)
# Set the SDL environment variable to force the window to the second screen
os.environ["SDL_VIDEO_WINDOW_POS"] = "%d, %d" % (pygame.display.Info().current_w, 1)

# Open the screen in fullscreen mode on the second display
screen = pygame.display.set_mode((1512, 982), pygame.FULLSCREEN)


# Set positions for letters based on letter count
positions = {
    4: [(626, 454), (696, 454), (766, 454), (836, 454)],
    5: [(591, 454), (661, 454), (731, 454), (801, 454), (871, 454)],
    6: [(556, 454), (626, 454), (696, 454), (766, 454), (836, 454), (906, 454)],
    7: [(521, 454), (591, 454), (661, 454), (731, 454), (801, 454), (871, 454), (941, 454)],
    8: [(486, 454), (556, 454), (626, 454), (696, 454), (766, 454), (836, 454), (906, 454), (976, 454)]
}

# Iterate through word lengths
for n_letters in range(4, 9):
    curr_word_length = data[f'words_{n_letters}_letters']
    curr_pseudo_length = data[f'pseudowords_{n_letters}_letters']
    curr_positions = positions[n_letters]

    # Process words and pseudowords
    for word_list in [curr_word_length, curr_pseudo_length]:
        for word in word_list:
            textures_br = []
            textures_cb = []

            # Process each letter in the word
            for i_letter, letter in enumerate(word):
                br_image = Image.open(f'/Users/cerpelloni/Documents/GitHub/VisualBraille_backstageCode/visbra_training/vbdt/input/letters/br_{letter}.png')
                cb_image = Image.open(f'/Users/cerpelloni/Documents/GitHub/VisualBraille_backstageCode/visbra_training/vbdt/input/letters/cb_{letter}.png')

                # Convert images to pygame surfaces
                br_texture = pygame.image.fromstring(br_image.tobytes(), br_image.size, br_image.mode)
                cb_texture = pygame.image.fromstring(cb_image.tobytes(), cb_image.size, cb_image.mode)

                textures_br.append(br_texture)
                textures_cb.append(cb_texture)

            # Display and save braille word image
            for i, texture in enumerate(textures_br):
                screen.blit(texture, curr_positions[i])
            pygame.display.flip()

            pygame.image.save(screen, "temp_br_screenshot.png")
            br_cropped_image = Image.open("temp_br_screenshot.png").crop((813, 883, 2212, 1082))
            br_words[word] = np.array(br_cropped_image)
            br_cropped_image.save(f'output/images/br_{word}.png')

            # Display and save connected braille word image
            for i, texture in enumerate(textures_cb):
                screen.blit(texture, curr_positions[i])
            pygame.display.flip()

            pygame.image.save(screen, "temp_cb_screenshot.png")
            cb_cropped_image = Image.open("temp_cb_screenshot.png").crop((813, 883, 2212, 1082))
            cb_words[word] = np.array(cb_cropped_image)
            cb_cropped_image.save(f'output/images/cb_{word}.png')

            # Clear screen
            screen.fill((0, 0, 0))
            pygame.display.flip()

# Save results
scipy.io.savemat('dutch_processed_stimuli.mat', {
    'br_words': br_words,
    'cb_words': cb_words,
    **{f'words_{n_letters}_letters': data[f'words_{n_letters}_letters'] for n_letters in range(4, 9)},
    **{f'pseudowords_{n_letters}_letters': data[f'pseudowords_{n_letters}_letters'] for n_letters in range(4, 9)}
})

pygame.quit()
