#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 24 14:51:05 2024

Get confident with pytorch

@author: cerpelloni
"""
# All imports needed for the full tutorial
import torch
import numpy as np
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt



# tensor from data
data = [[1, 2],[3, 4]]
x_data = torch.tensor(data)

# tensor from np array
np_array = np.array(data)
x_np = torch.from_numpy(np_array)

# tensor from another tensor
x_ones = torch.ones_like(x_data) # retains the properties of x_data
print(f"Ones Tensor: \n {x_ones} \n")
x_rand = torch.rand_like(x_data, dtype = torch.float) # overrides the datatype of x_data
print(f"Random Tensor: \n {x_rand} \n")

# tensor with random or constant values
shape = (2,3,)
rand_tensor = torch.rand(shape)
ones_tensor = torch.ones(shape)
zeros_tensor = torch.zeros(shape)
print(f"Random Tensor: \n {rand_tensor} \n")
print(f"Ones Tensor: \n {ones_tensor} \n")
print(f"Zeros Tensor: \n {zeros_tensor}")


## Attributes of tensors

tensor = torch.rand(3,4)
print(f"Shape of tensor: {tensor.shape}")
print(f"Datatype of tensor: {tensor.dtype}")
print(f"Device tensor is stored on: {tensor.device}")


## Operations on tensors

# Move on GPU (will it be a problem with a mac?)
if torch.cuda.is_available():
    tensor = tensor.to("cuda")
    
# matrix multiplication between two tensors. y1, y2, y3 will have the same value
# tensor.T returns the transpose of a tensor
y1 = tensor @ tensor.T
y2 = tensor.matmul(tensor.T)
y3 = torch.rand_like(y1)
torch.matmul(tensor, tensor.T, out = y3)

# element-wise product. z1, z2, z3 will have the same value
z1 = tensor * tensor
z2 = tensor.mul(tensor)
z3 = torch.rand_like(tensor)
torch.mul(tensor, tensor, out = z3)

# aggregation
agg = tensor.sum()
agg_item = agg.item()
print(agg_item, type(agg_item))

# in-place operations change the operand
# all have '_' as suffix, e.g. x.t_() and x.copy_(y)
print(f"{tensor} \n")
tensor.add_(5)
print(tensor)

# move to numpy, changes to tensor will also modify numpy associated to it. 
# Bidirectional link, they share the same memory
n = tensor.numpy()


## Datasets and dataloaders

# Load the dataset
training_data = datasets.FashionMNIST(root = "data",
                                      train = True,
                                      download = True,
                                      transform = ToTensor())

test_data = datasets.FashionMNIST(root = "data",
                                  train = False,
                                  download = True,
                                  transform = ToTensor())

# Visualize the dataset
labels_map = {
    0: "T-Shirt",
    1: "Trouser",
    2: "Pullover",
    3: "Dress",
    4: "Coat",
    5: "Sandal",
    6: "Shirt",
    7: "Sneaker",
    8: "Bag",
    9: "Ankle Boot",
}
figure = plt.figure(figsize=(8, 8))
cols, rows = 3, 3
for i in range(1, cols * rows + 1):
    sample_idx = torch.randint(len(training_data), size=(1,)).item()
    img, label = training_data[sample_idx]
    figure.add_subplot(rows, cols, i)
    plt.title(labels_map[label])
    plt.axis("off")
    plt.imshow(img.squeeze(), cmap="gray")
plt.show()










