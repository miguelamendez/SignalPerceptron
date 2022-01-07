#Internal libraries
import os
import sys
full_path = os.path.dirname(os.path.realpath(__file__))
print("[archs][sp]:sp_practical.py:",full_path)
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(full_path))))
#print(os.path.dirname(os.path.dirname(os.path.dirname(full_path))))
#print("Internal libraries:")
from ml.archs.sp import  baselines as sp
#External libraries
import numpy as np
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter, UninitializedParameter
import math

#Multilayer Perceptrons for comparition:
class ML_arch_v1(nn.Module):
    def __init__(self, inout,parameters):
        input_size ,output_size =inout
        layers = parameters
        super(ML_arch_v1,self).__init__()
        all_layers = []
        for i in layers:
            all_layers.append(nn.Linear(input_size, i))
            all_layers.append(nn.Sigmoid())
            input_size = i
        all_layers.append(nn.Linear(layers[-1], output_size))
        all_layers.append(nn.Sigmoid())
        self.layers = nn.Sequential(*all_layers)

    def forward(self,x):
        x = self.layers(x)
        return x
#Example:
#arch=ML_arch_v1((5,1),([200,100,50]))
#print(arch)

class ML_arch_v2(nn.Module):
    def __init__(self, inout,parameters):
        input_size ,output_size =inout
        layers , p = parameters
        super(ML_arch_v2,self).__init__()
        all_layers = []
        for i in layers:
            all_layers.append(nn.Linear(input_size, i))
            all_layers.append(nn.ReLU(inplace=True))
            all_layers.append(nn.BatchNorm1d(i))
            all_layers.append(nn.Dropout(p))
            input_size = i
        all_layers.append(nn.Linear(layers[-1], output_size))
        all_layers.append(nn.ReLU())
        self.layers = nn.Sequential(*all_layers)

    def forward(self, x):
        x = self.layers(x)
        return x
#Example:
#arch=ML_arch_v2((5,1),([200,100,50],.4))
#print(arch)



#For Image datasets
class MNIST_arch(nn.Module):
    def __init__(self,inout,parameters):
        network, net_params=parameters
        super(MNIST_arch, self).__init__()
        self.flat=nn.Flatten()
        self.arch= network(inout,net_params)
    def forward(self, x):
        x=self.flat(x)
        return self.arch(x)
#Example
#fake_image_batch=torch.randn(5,1,28,28)
#nn1=MNIST_arch((784,10),(ML_arch_v1,[256]))
#nn2=MNIST_arch((784,10),(ML_arch_v2,([256],.4)))
#fsp=MNIST_arch((784,10),(sp.FSP_pytorch,256))
#lsp=MNIST_arch((784,10),(sp.LSP_pytorch,256))
#arch=nn1
#print(arch)
#print(arch(fake_image_batch))

#For Tabular datasets:
class Tabular_arch(nn.Module):
    #Model based on https://stackabuse.com/introduction-to-pytorch-for-classification/
    def __init__(self, inout,parameters):
        embedding_size, num_numerical_cols, output_size =inout
        network,net_params=parameters
        super(Tabular_arch,self).__init__()
        #Embeddings#
        self.all_embeddings = nn.ModuleList([nn.Embedding(ni, nf) for ni, nf in embedding_size])
        self.batch_norm_num = nn.BatchNorm1d(num_numerical_cols)
        num_categorical_cols = sum((nf for ni, nf in embedding_size))
        input_size = num_categorical_cols + num_numerical_cols
        #The used network
        self.arch= network((input_size,output_size),net_params)

    def forward(self, x_categorical, x_numerical):
        embeddings = []
        for i,e in enumerate(self.all_embeddings):
            embeddings.append(e(x_categorical[:,i]))
        x = torch.cat(embeddings,1)
        x_numerical = self.batch_norm_num(x_numerical)
        x = torch.cat([x, x_numerical], 1)
        x = self.arch(x)
        return x
#Example
#from data.utils.preprocessing import *
#dataset = Tabular("Churn_Modelling.csv")
#categorical_columns = ['Geography', 'Gender', 'HasCrCard', 'IsActiveMember']
#numerical_columns = ['CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'EstimatedSalary']
#outputs = ['Exited']
#dataset.process(categorical_columns,numerical_columns,outputs)
#nn1=Tabular_arch((dataset.cat_emb,dataset.num_data.shape[1],1),(ML_arch_v1,[256]))
#nn2=Tabular_arch((dataset.cat_emb,dataset.num_data.shape[1],1),(ML_arch_v2,([256],.4)))
#fsp=Tabular_arch((dataset.cat_emb,dataset.num_data.shape[1],1),(sp.FSP_pytorch,256))
#lsp=Tabular_arch((dataset.cat_emb,dataset.num_data.shape[1],1),(sp.LSP_pytorch,256))
#arch=lsp
#print(arch)
#print(arch(dataset.cat_data[:5],dataset.num_data[:5]))
