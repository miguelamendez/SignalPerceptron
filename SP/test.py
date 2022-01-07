"""
    Description: Main file for training/testing archs.
    Currently this file is usless/haven't found any application yet
    """
import os
import sys
full_path = os.path.dirname(os.path.realpath(__file__))
print("main.py",full_path)
sys.path.append(os.path.dirname(os.path.dirname(full_path)))

#Data libraries
#from data
#External libraries
import numpy as np
import torch

print("hola_mundo")
x=torch.randn(1,3)
print(x)
