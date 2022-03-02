#This experiments show that we can learn the parameters of the signal perceptron and Real signal perceptron using system of linear equations.
from signal_perceptron import *
from sp_paper_models import *
from data_load import *
from train import *
import os
from utils import *
import sys
import numpy as np
np.set_printoptions(threshold=sys.maxsize)
def m_k_functional_space_analysis(m,k,samples=[]):
    #Binary Boolean Function space 
    m=m
    k=k
    samples=samples
    x_train,y_train=data_gen(m,k,samples)
    #RSP matrix form using walshMatrix generator:
    rsp_matrix= RSP_Matrix(m,k)
    alphas_rsp = train_linear_numpy(y_train,rsp_matrix)
    #SP using SP_Matrix generator 
    sp_matrix= SP_Matrix(m,k)
    alphas_sp = train_linear_numpy(y_train,sp_matrix)
    rsp_model = RSP_numpy(m,k)
    sp_model = SP_numpy(m,k)
    print("SP results:")
    for i in range(0,len(y_train)):
        print("Function: ",y_train[i],"\n Parameters: ",np.round(alphas_sp[i],4))
    print("RealSP results:")
    for i in range(0,len(y_train)):
        print("Function: ",y_train[i],"\n Parameters: ",np.round(alphas_rsp[i],4)) 

print("This experiment is gona be run ",sys.argv[-1], " times:")
n= int(sys.argv[-1])
for i in range(0,n):
    orig_stdout = sys.stdout
    subfolder="run"+str(i+1)+"/"
    subname="functional analisys"
    out="data/experiments/exp3/"+subfolder+subname+".txt"
    f = open(out, 'w+')
    sys.stdout = f
    m_k_functional_space_analysis(2,2)
    m_k_functional_space_analysis(2,3)
    m_k_functional_space_analysis(3,1)
    m_k_functional_space_analysis(5,2,10)
    m_k_functional_space_analysis(7,2,10)
    m_k_functional_space_analysis(11,2,10)
    sys.stdout = orig_stdout
    f.close()

