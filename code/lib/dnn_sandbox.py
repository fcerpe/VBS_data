#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 11 14:31:32 2024

@author: cerpelloni
"""

import cv2
import numpy as np

# Load the image
image = cv2.imread('letters/to_variate/ln_z_black.png')

# Convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply binary threshold
_, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)

# Create a kernel for erosion
kernel_size = 
kernel = np.ones((kernel_size, kernel_size), np.uint8)

# Number of iterations for erosion
iterations = 10

# Perform iterative erosion
for i in range(iterations):
    # Find contours
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Draw the contours on the image
    cv2.drawContours(image, contours, -1, (255, 255, 255), thickness=cv2.FILLED)
    
    # Erode the image
    binary = cv2.erode(binary, kernel, iterations=1)

    # Show intermediate result (for visualization purposes)
    cv2.imshow(f'Erosion Iteration {i+1}', binary)
    cv2.waitKey(500)  # Wait for 500ms between iterations

# Final result
cv2.imshow('Final Result', binary)
cv2.waitKey(10)
cv2.destroyAllWindows()
