"""
    Description: Main file for training/testing archs.
    Currently this file is usless/haven't found any application yet
    """
import os
import sys
full_path = os.path.dirname(os.path.realpath(__file__))
print("main.py",full_path)
sys.path.append(os.path.dirname(os.path.dirname(full_path)))

#Internal libraries
#ML libraries
from ml.archs.build import *
from ml.archs.sp.baselines import *
from ml.archs.sp.asp_baselines import *
from ml.archs.sp.sp_practical import *
#Data libraries
#from data
#External libraries
import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
#import torchvision.transforms as transforms


def work(data_mngr,model):
    #Initial Batch
    data_x, done =data_mngr.reset()
    pred=model.reset(data_x)
    loss_history=[]
    while not done:
        data_x,data_y,meta_data,done=data_mngr.forward(data=pred)
        pred,loss=model.forward(data_x,data_y)
        if loss is not None:
            loss_history.append(loss)
def runs(num_runs=1):
    runs_info=[]
    for i in range(0,num_runs):
        train_info=work()
        runs_info.append(train_info)
    return runs_info 

#Example
#arch =MNIST_arch((784,10),(sp.FSP_pytorch,256))
#optimizer =torch.optim.Adam(arch.parameters(), lr=1e-3)
#import torchvision.datasets as datasets
#mnist_trainset = datasets.MNIST(root='home/misha/Projects/AI/data/datasets/image/MNIST', train=True, download=True, transform=None)
#image ,out =mnist_trainset[1]
#print(image)
#print(torch.from_numpy(np.array(image)))
