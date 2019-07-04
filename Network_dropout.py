import torch
import pandas as pd
import numpy as np
import torchvision
from torchvision import transforms
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

import torch.nn as nn
import torch.nn.functional as F

class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        # Convolutional transformations
        # Zorg dat het aantal out_channels meer en meer wordt
        self.conv1 = nn.Conv2d(in_channels = 3, out_channels = 32, kernel_size = 5, stride = 1, padding = 0)
        self.conv2 = nn.Conv2d(in_channels = 32, out_channels = 64, kernel_size = 5, stride = 1, padding = 0)
        
        # Fully Connected layers
        # Zorg dat het aantal out_features minder en minder wordt
        self.fc1Input = 64*9*9
        self.fc1 = nn.Linear(in_features = self.fc1Input, out_features = 120)
        self.fc2 = nn.Linear(in_features = 120, out_features = 60)
        self.out = nn.Linear(in_features = 60, out_features = 2)
        
        # Dropout layers
        # Helpt bij het filteren van onnodige features. Dit zorgt voor een robuster model.
        self.drop1 = nn.Dropout(p = 0.3)
        self.drop2 = nn.Dropout(p = 0.3)
        
    def forward(self, t):
        # Implementation of layers
        # (1) input layer
        t = t
        
        # (2) hidden conv layer 1
        t = self.conv1(t)
        t = F.max_pool2d(t, kernel_size = 2, stride = 2)
        t = F.relu(t)
        
        # (3) hidden conv layer 2
        t = self.conv2(t)
        t = F.max_pool2d(t, kernel_size = 2, stride = 2)
        t = F.relu(t)
        
        # (4) hidden linear layer 1
        t = t.reshape(-1, self.fc1Input)
        t = self.drop1(t)
        t = self.fc1(t)
        t = F.relu(t)
        
        # (5) hidden linear layer 2
        t = self.drop2(t)
        t = self.fc2(t)
        t = F.relu(t)
        
        # (6) output linear layer
        t = self.out(t)
        # Normaal zou je bij de output layer een softmax uitvoeren na een serie van relu operaties
        # De loss/cost functie die we gaan toepassen maakt al impliciet gebruik van de softmax 
        #t = F.softmax(t, dim = 1)             
        return t