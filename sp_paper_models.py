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

class Prod(nn.Module):
    def __init__(self,k):
        super(Prod, self).__init__()
        self.weights=nn.Parameter(torch.randn(k),requires_grad=True)
        self.bias=nn.Parameter(torch.randn(1),requires_grad=True)
    def forward(self,x):
         x = x*self.weights
         x = torch.prod(x,1)*self.bias
         return x
#Example
#input=torch.randn(3,2)
#ob_prod = Prod(2)
#print(input,"\n",ob_prod.weights,"\n",ob_prod.bias)
#print(ob_prod(input))

class GN_pytorch(nn.Module):
    def __init__(self,k,heads=1):
        super(GN_pytorch, self).__init__()
        self.sum_layer = nn.Linear(k, 1)
        self.prod_layer = Prod(k)
        self.lambda_sum=nn.Parameter(torch.randn(1),requires_grad=True)
        self.lambda_prod=nn.Parameter(torch.randn(1),requires_grad=True)
        self.gamma=nn.Parameter(torch.randn(1),requires_grad=True)
    #print("frecuency matrix",arrw.shape)
    def forward(self,x):
        x_sum=torch.sigmoid(self.lambda_sum*self.sum_layer(x))
        x_prod=torch.exp(self.lambda_prod*self.sum_layer(x))
        x=self.gamma*x_sum+(1-self.gamma)*x_prod
        return x

#Example
#input=torch.randn(3,3)
#gn = GN_pytorch(3)
#print(input)
#print(gn(input))
#Amount of parameters
#params=filter(lambda p: p.requires_grad, gn.parameters())
#print(sum([np.prod(p.size()) for p in params]))

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






