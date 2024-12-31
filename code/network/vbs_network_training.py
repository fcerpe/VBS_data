#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  8 10:56:10 2024

From Tom van Hogen's jupyter notebook.
Thank you Tom

@author: cerpelloni
"""

import os
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms 


### ---------------------------------------------------------------------------
### 1 - PRE-PROCESSING

## Data transformation and retrieval

# Use MNIST dataset 

# Define transformations to apply to the input images
# In the transforms.Compose function you can define other transformations (resize, centre or normalize). 
# e.g., pretrained AlexNet architecture needs the data to fit the input of the network
transformations = transforms.Compose([
    
    # Convert image to PyTorch tensor
    # effectively changes the value range from 0-255 to 0-1 
    transforms.ToTensor()
])

# Download the data using the specified transformations
# Specify location to store MNIST data
download_path = os.path.join("source/data")

# Create the folder if doesn't exist
if not os.path.exists(download_path):
    os.makedirs(download_path)

# Create a dataset for training
# Root is where pytorch is going to look for the data
train_data = datasets.MNIST(
    root = download_path, 
    train = True,
    download = True,
    transform = transformations
)

# Create a dataset for testing (train = False!)
test_data = datasets.MNIST(
    root = download_path,
    train = False,
    download = True,
    transform = transformations
)


## PyTorch DataLoader

# this module helps with sampling and iterating over the Dataset object.
# Also helps with sampling batches of images instead of single ones. 

# Specify the batch size
batch_size = 64

# Create dataloaders for training and test datasets (shuffle to make new batches each time)
train_loader = DataLoader(dataset = train_data, 
                          batch_size = batch_size, 
                          shuffle = True)

test_loader = DataLoader(dataset = test_data, 
                         batch_size = batch_size, 
                         shuffle = True)

# DataLoader is a generator, does not hold objects in memory. 
# Printing the dataloader object directly won't show its contents. 
# Instead you see where the object itself is stored in memory
print(train_loader)

# Adjust print output length for tensors
torch.set_printoptions(threshold = 20, edgeitems = 2)

sample = next(iter(train_loader))
sample

# DataLoader returns both images and corresponding labels
# Split them up:
sample_images, sample_labels = next(iter(train_loader))

# Both have 64 elements (batch size). Image has 1 channel (RGB has 3)
print(f"Sample_data shape: {sample_images.shape}")
print(f"Sample_labels shape: {sample_labels.shape}")

# Look at first image:
plt.imshow(sample_images[0].permute((1,2,0)), cmap = 'gray')
plt.title(f"Image with label: {sample_labels[0]}")
plt.show()


### ---------------------------------------------------------------------------
### 2 - CREATE THE NETWORK

## Check your device
# Training on GPU is faster, ysed through 'cuda'
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu') # Configures to cuda if it is available.
print(f"Using {device} device")

## Network architecture
# Create network by defining a python class that specifies:
# - the amount and types of layers
# - the order of their arrangement
# - how many input and output nodes are for each layer
# - activation functions that make the network non-linear
class SimpleNeuralNetwork(nn.Module):
    
    def __init__(self):
        
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )

    # define transformation flow from inout to output
    # name must be 'forward' as it's used internally by pytorch
    def forward(self, x):
        
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


### ---------------------------------------------------------------------------
### 3 - PREPRARE TRAINING AND TESTING OF THE NETWORK

# Choose the network
net = SimpleNeuralNetwork().to(device)
print(net)

## Define function to train the network

# Function takes as input:
# - DataLoader
# - network architecture
# - loss function: dissimilarity between network's prediction (pred) and actual answer (y)
# - optimizer functions: gradient based on losses
# - current epoch, to track network's performance
def train_loop(dataloader, net, loss_fn, optimizer, epoch):
    
    size = len(dataloader.dataset)
    
    # Setting the network to training mode (net.train) allows the gradients to be computed,
    # and activates training-specific features (dropout, batch normalization) to prevent overfitting
    net.train()
    
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device) # Send to device that is available (either GPU or CPU)
        
        # Compute prediction and loss
        pred = net(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
        # track the losses for visualization
        if batch % 10 == 0:
            train_losses.append(loss.item())
            train_counter.append((batch * 64) + (epoch*len(dataloader.dataset)))

        #track progress of the training
        if batch % 100 == 0:
            loss, current = loss.item(), batch * batch_size + len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


## Define function to test the network

# Function takes as input:
# - DataLoader
# - network architecture
# - loss function
def test_loop(dataloader, net, loss_fn):
    
    net.eval()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            
            X, y = X.to(device), y.to(device) # Send to device that is available (either GPU or CPU)
            pred = net(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    test_losses.append(test_loss)
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")



### ---------------------------------------------------------------------------
### 4 - TRAIN THE NETWORK

## Define hyperparameters

# epochs: number of times that a network passes through training
epochs = 10

# learning rate: to which extent the network parameters are updated for each batch / epoch
learning_rate = 1e-3

# loss function: different functions available in pytorch
loss_fn = nn.CrossEntropyLoss() 

# optimizer: different functions in pytorch
optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate)

# arrays to store counters and losses during training and testing
train_losses = []
train_counter = []
test_losses = []
test_counter = [i*len(train_loader.dataset) for i in range(epochs + 1)]


## Test the untrained network, should be at chance
test_loop(test_loader, net, loss_fn)


## Train the network
for t in range(epochs):
    
    print(f"Epoch {t+1}\n-------------------------------")
    train_loop(train_loader, net, loss_fn, optimizer, t)
    test_loop(test_loader, net, loss_fn)
print("Training complete!")

# Visualize network's progress
import matplotlib.pyplot as plt

fig = plt.figure()
plt.plot(train_counter, train_losses, color = 'cornflowerblue', linewidth = 1.5)
plt.scatter(test_counter, test_losses, zorder = 5, color = 'darkred')         
plt.legend(['Train Loss', 'Test Loss'], loc = 'upper right')
plt.xlabel('number of training examples seen')
plt.ylabel('negative log likelihood loss')
plt.show()



### ---------------------------------------------------------------------------
### 5 - SAVE THE NETWORK

# Different options

## Save (and load) parameters
# Recommended option, allows for more flexibility. 
# When loading the network, it requires you to create an instance of the original network architecture

# Store the network parameters
torch.save(net.state_dict(), 'netMNIST_pars.pth')

# Load parameters into an instance of the network
loaded_net = SimpleNeuralNetwork()
loaded_net.load_state_dict(torch.load("netMNIST_pars.pth", weights_only=True))

# Put the network in evaluation mode
loaded_net.eval()


## Save the entire network

#save the network
torch.save(net, 'netMNIST.pth')

# Load it
net = torch.load('netMNIST.pth')









