"""Models used for the Signal Perceptron Paper"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from signal_perceptron import *

#Functional Space Models
##############################################################################################################################
class MLP_pytorch(nn.Module):
    def __init__(self,n,k,heads=1):
        super(MLP_pytorch, self).__init__()
        self.hidden_layer = nn.Linear(k, n)
        self.output_layer = nn.Linear(n, heads)
    #print("frecuency matrix",arrw.shape)
    def forward(self,x):
         x = torch.sigmoid(self.hidden_layer(x))
         x = torch.sigmoid(self.output_layer(x))
         return x

#MNIST Models 
##############################################################################################################################
class FSP_mnist(nn.Module):
    def __init__(self,signals=512):
        super(FSP_mnist, self).__init__()
        self.flatten = nn.Flatten()
        self.FSP =FSP_pytorch(signals,28*28,10)

    def forward(self, x):
        x = self.flatten(x)
        logits = self.FSP(x)
        return logits

class MLP1_mnist(nn.Module):
    def __init__(self):
        super(MLP1_mnist, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
            nn.ReLU()
        )
    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

class MLP2_mnist(nn.Module):
    def __init__(self):
        super(MLP2_mnist, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
            nn.ReLU()
        )
    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

#CIFAR Models
##############################################################################################################################
class FSP_cifar(nn.Module):
    def __init__(self):
        super(FSP_cifar, self).__init__()
        self.flatten = nn.Flatten()
        self.FSP =FSP_pytorch(1024,32*32*3,10)

    def forward(self, x):
        x = self.flatten(x)
        logits = self.FSP(x)
        return logits





#NUMPY MODELS DOESNT WORK 
#def sigmoid(x):
#    return 1/(1 + np.exp(-x))

#class Linear_numpy(object):
#    def __init__(self,n,k):
#        pass

#class MLP_numpy(object):
#    def __init__(self,n,k,heads=1):
#        self.hidden_layer = Linear_numpy(k, n)
#        self.output_layer = Linear_numpy(n, heads)
    #print("frecuency matrix",arrw.shape)
#    def forward(self,x):
#         x = sigmoid(self.hidden_layer(x))
#         x = sigmoid(self.output_layer(x))
#         return x
