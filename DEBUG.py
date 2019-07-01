import torch
import pandas as pd
import numpy as np
import torchvision
from torchvision import transforms
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

import torch.nn as nn
import torch.nn.functional as F

import torch.optim as optim


from Network import *
from TorchMalaria import *

import matplotlib.pyplot as plt
from skimage.transform import resize


data = DataMalaria("list_of_imagenames_with_label.csv", transform=torchvision.transforms.ToTensor())
dataloader = DataLoader(data, batch_size = 8, shuffle = True)

# initialize network and optimizer
network = Network()
optimizer = optim.Adam(network.parameters(), lr = 0.01)

batch = next(iter(dataloader))
images, labels = batch

# forward pass + calculate loss
preds = network(images)
loss = F.cross_entropy(preds, labels)

# backward propagation + updating weights
loss.backward()
optimizer.step()

################################

print("Loss 1: {}".format(loss.item()))
preds = network(images)
loss = F.cross_entropy(preds, labels)
print("Loss 2: {}".format(loss.item()))