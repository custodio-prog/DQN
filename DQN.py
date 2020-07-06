import gym 
import math 
import random 
import numpy as np
import matplotlib
import matplotlib.pyplot as plt  
from collections import namedtuple
from itertools import count
from PIL import Image
import torch.nn as nn 
import torch.optim as optim 
import torch.nn.functional as F 
import torchvision.transforms as T 


class DQN(nn.Module):
    def __init__(self, imag_height, img_width):
        super().__init__()

        self.fc1 = nn.Linear(in_features=imag_height*img_width*3, out_features=24)
        self.fc2 = nn.Linear(in_features=24, out_features=32)
        self.out = nn.Linear(in_features=32, out_features=2)

    def forward(self,t):
        t = t.flatten(start_dim=1)
        t = F.relu(self.fc1(t))
        t = F.relu(self.fc2(t))
        t = self.out(t)
        return t