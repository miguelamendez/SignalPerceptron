import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def sigmoid(x):
    return 1/(1 + np.exp(-x))

class Linear_numpy(object):
    def __init__(self,n,k):
        pass

class MLP_numpy(object):
    def __init__(self,n,k):
        self.hidden_layer = Linear_numpy(k, n)
        self.output_layer = Linear_numpy(n, 1)
    #print("frecuency matrix",arrw.shape)
    def forward(self,x):
         x = sigmoid(self.hidden_layer(x))
         x = sigmoid(self.output_layer(x))
         return x
         

class MLP_pytorch(nn.Module):
    def __init__(self,n,k):
        super(MLP_pytorch, self).__init__()
        self.hidden_layer = nn.Linear(k, n)
        self.output_layer = nn.Linear(n, 1)
    #print("frecuency matrix",arrw.shape)
    def forward(self,x):
         x = torch.sigmoid(self.hidden_layer(x))
         x = torch.sigmoid(self.output_layer(x))
         return x
