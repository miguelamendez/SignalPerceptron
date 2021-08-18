#This experiments show that we can learn the parameters of the signal perceptron and Real signal perceptron using system of linear equations.
from signal_perceptron import *
from sp_paper_models import *
from data_load import *
from train import *
import os
from utils import *

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
    total_loss_rsp = test_linear_numpy(x_train,y_train,rsp_model,alphas_rsp,MSE_Loss)
    total_loss_sp = test_linear_numpy(x_train,y_train,sp_model,alphas_sp,MSE_Loss)
    print("SP results:")
    for i in range(0,len(y_train)):
        print("Function: ",y_train[i],"\n Loss: ",total_loss_sp[i],"\n Parameters: ",alphas_sp[i])
    print("RSP results:")
    for i in range(0,len(y_train)):
        print("Function: ",y_train[i],"\n Loss: ",total_loss_rsp[i],"\n Parameters: ",alphas_rsp[i])

import sys
n=5
for i in range(0,n):
    orig_stdout = sys.stdout
    subname="functional_analysis"
    out="data/experiments/exp3/"+subname+"_"+str(i)+".txt"
    f = open(out, 'w')
    sys.stdout = f
    m_k_functional_space_analysis(2,2)
    m_k_functional_space_analysis(3,1)
    m_k_functional_space_analysis(5,2,10)
    m_k_functional_space_analysis(7,2,10)
    m_k_functional_space_analysis(11,2,10)
    sys.stdout = orig_stdout
    f.close()

