import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as Data
from torch.autograd import Variable
import torchvision
from torchvision import transforms
from torchvision.utils import save_image
from torchvision import datasets
import matplotlib.pyplot as plt
import imageio
import itertools
import numpy as np
import struct
from layers import GraphConvolution

# adjust aitivation func: relu, leaky_relu, tanh, sigmoid...
# consider batch_normalization, dropout
# adjust hid, hid/2, ...
# if the last activation is tanh in G, the input pixel value should be normalized to [-1, 1]?


class Generator(nn.Module):
    def __init__(self, infeat, hid, outfeat):
        super(Generator, self).__init__()
        # infeat = init_dimension + condition_num
        # outfeat = 1
        self.gconv1 = GraphConvolution(infeat, hid)
        self.bn1 = nn.BatchNorm1d(100)
        self.gconv2 = GraphConvolution(hid, int(hid / 2))
        self.bn2 = nn.BatchNorm1d(100)
        self.gconv3 = GraphConvolution(int(hid / 2), int(hid / 4))
        self.bn3 = nn.BatchNorm1d(100)
        self.gconv4 = GraphConvolution(int(hid / 4), outfeat)

        #self.bn4 = nn.BatchNorm1d(100)
        #self.fc5 = nn.Linear(100, 100)

    def forward(self, x, adj, c):
        # input noise x <- (batch_size, pixel_num, init_dimension)
        # input adjacency matrix <- (batch_size, region_width * region_length, region_width * region_length)
        # input condition c <- (batch_size, pixel_num, condition_num)
        x = torch.cat((x, c), dim=2)
        x = F.relu(self.bn1(self.gconv1(x, adj)))
        x = F.relu(self.bn2(self.gconv2(x, adj)))
        x = F.relu(self.bn3(self.gconv3(x, adj)))
        x = torch.tanh(self.gconv4(x, adj))
        #x = F.relu(self.bn4(self.gconv4(x, adj)))
        #x = x.view(list(x.size())[0], -1)
        #x = torch.tanh(self.fc5(x))
        #x = x.view(list(x.size())[0], -1, 1)

        return x


class Discriminator(nn.Module):
    def __init__(self, infeat, hid, outfeat):
        super(Discriminator, self).__init__()

        self.gconv1 = GraphConvolution(infeat, hid)
        self.bn1 = nn.BatchNorm1d(100)
        self.gconv2 = GraphConvolution(hid, hid * 2)
        self.bn2 = nn.BatchNorm1d(100)
        self.gconv3 = GraphConvolution(hid * 2, hid * 4)
        self.bn3 = nn.BatchNorm1d(100)
        self.gconv4 = GraphConvolution(hid * 4, outfeat)
        self.bn4 = nn.BatchNorm1d(100)
        self.fc5 = nn.Linear(100, 1)         # adjust 100 if region size changes

    def forward(self, x, adj, c):
        # input region x <- (batch_size, pixel_num, feature_num)
        # input adjacency matrix <- (batch_size, region_width * region_length, region_width * region_length)
        # input condition c <- (batch_size, pixel_num, condition_num)
        x = torch.cat((x, c), dim=2)
        x = F.leaky_relu(self.bn1(self.gconv1(x, adj)), 0.2)
        x = F.leaky_relu(self.bn2(self.gconv2(x, adj)), 0.2)
        x = F.leaky_relu(self.bn3(self.gconv3(x, adj)), 0.2)
        x = F.leaky_relu(self.bn4(self.gconv4(x, adj)), 0.2)
        x = x.view(list(x.size())[0], -1)
        x = torch.sigmoid(self.fc5(x))

        return x
