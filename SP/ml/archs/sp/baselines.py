"""
    Description: This library contains all the base functions used for defining the signal perceptron
    Please refer to each of the functions/classes for a full description of what they do.

    Functions (True are the implemented Functions):
        SP_Matrix:
        SP_r_Matrix:
        frequencies_gen:
        GD_MSE_SP_step:
        MSELoss:

    Classes (True are the implemented classes):
        SP_numpy:True. Implementation of the signal perceptron using the "numpy" library (USED IN PAPER EXPERIMENTS)
        SP_r_numpy:True. Implementation of the real signal perceptron using the "numpy" library (USED IN PAPER EXPERIMENTS)
        SP_pytorch:False. Implementation of the signal perceptron using the "pytorch" library (Not functional as pytorch has problems to calculating complex gradients with respect of complex numbers)
        RSP_pytorch:True. Implementation of the real signal perceptron using the "pytorch" library (USED IN PAPER EXPERIMENTS)
        FSP_pytorch:True. Implementation of the real signal perceptron with learnable frequencies and fixed ammount of signals (USED IN THE PAPER EXPERIMENTS)
           """

#Internal libraries
import os
import sys
full_path = os.path.dirname(os.path.realpath(__file__))
print("[archs][sp]:baselines.py:",full_path)
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(full_path))))
#print(os.path.dirname(os.path.dirname(os.path.dirname(full_path))))
#
from utils.matrices import *


#External libraries
import numpy as np
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter, UninitializedParameter
import math


def freq_gen_sp(m,k):
    """
    Description: This function creates the frecuency array used by the signals of the signal perceptron
    is_Implemented:
        True
    Args:
        (m:int): The domain size , the amount of possible variables that each variable can take
        (k:int): The arity, the amount of variables that each signal can recieve

    Shape:
        - Input: integers that define the functional space
        - Output: an array of size :math:`m*k`

    Examples::
        frequencies = freq_gen_sp(2,2)
        print(frequencies)
        [[0,0],[0,1],[1,0],[1,1]]
           """
    wki=[]
    aiw=np.zeros([k]);
    for i in range(0,m**k,1):
        kw=i;
        for j in range(0,k,1):
            aiw[j]= int ( kw % m );
            kw=int(kw/m);
        w=[]
        for l in aiw:
            w.append(l)
        wki.append(w)
    arrw = np.asarray(wki,dtype=np.float32)
    return arrw

def SP_Matrix(m,k):
    aix=np.zeros([k]); #Array of indexes (to order them)
    aiw=np.zeros([k]); #Array of indexes (to order them)
    ni=m**k   #Number of Iterations
    n=k  #No. of variables
    nn=m**n #|m^k| domain space
    nnn=m**nn #|Delta|=|m^m^k| function space
    #Matrix
    A=np.zeros([nn,nn],dtype=complex) 
    divfrec=m-1
    i=0; j=0
    v=0; 
    for xi in range(0,ni,1):
        kx=xi;
        for xj in range(0,k,1): 
            aix[xj]= int ( kx % m ); 
            kx=int(kx/m); 
            #print("aix=",aix)
            j=0;
        #First Inner nested loop that generates all combinations of w for a signal
        for wi in range(0,ni,1):
            kw=wi;
            for wj in range(0,k,1): #Generamos los índices
                aiw[wj]= int ( kw % m ) ; #Lo metemos en array 
                kw=int(kw/m); #siguientes índices
                #print(i,j,A[i,j],"|",end='')
            exponente=0
            #Seconf Inner loop that  multiplies and sums
            for ii in range(0,k,1):
                exponente=exponente + aix[ii]*aiw[ii]
                exponente=int(exponente)
            #print("exponente=",exponente)
            exponente=1j*np.pi*exponente/divfrec
            #print(exponente)
            #print(np.exp(exponente))
            A[i,j]=np.exp(exponente)
            #print(A[i,j])
            j=j+1
            #print("aiw=",aiw,"j=",j)
            #for aj in range(0,nc,1):
            #	print(i,j,A[i,j],"|",end='')
            #	print()
        i=i+1
        
    return A 

def RSP_Matrix(m,k):
    aix=np.zeros([k]); #Array of indexes (to order them)
    aiw=np.zeros([k]); #Array of indexes (to order them)
    ni=m**k   #Number of Iterations
    n=k  #No. of variables
    nn=m**n #|m^k| domain space
    nnn=m**nn #|Delta|=|m^m^k| function space
    # Matrix
    A=np.zeros([nn,nn],dtype=np.float32) 
    divfrec=m-1
    i=0; j=0
    v=0; 
    for xi in range(0,ni,1):
        kx=xi;
        for xj in range(0,k,1): 
            aix[xj]= int ( kx % m ); 
            kx=int(kx/m); 
            #print("aix=",aix)
            j=0;
        #First Inner nested loop that generates all combinations of w for a signal
        for wi in range(0,ni,1):
            kw=wi;
            for wj in range(0,k,1): #Generamos los índices
                aiw[wj]= int ( kw % m ) ; #Lo metemos en array 
                kw=int(kw/m); #siguientes índices
                #print(i,j,A[i,j],"|",end='')
            exponente=0
            #Seconf Inner loop that  multiplies and sums
            for ii in range(0,k,1):
                exponente=exponente + aix[ii]*aiw[ii]
                exponente=int(exponente)
            #print("exponente=",exponente)
            exponente=np.pi*exponente/divfrec
            #print(exponente)
            #print(np.exp(exponente))
            A[i,j]=np.cos(exponente)
            #print(A[i,j])
            j=j+1
            #print("aiw=",aiw,"j=",j)
            #for aj in range(0,nc,1):
            #	print(i,j,A[i,j],"|",end='')
            #	print()
        i=i+1
    return A


def GD_MSE_SP_step(Y, X, model,lr):
    N=len(X)
    #Calculate the gradient
    pred ,m_exp= model.forward(X)
    #print("pred",pred,"real",Y)
    gradient= -2/N*np.dot((Y-pred),m_exp)
    #print("grad",gradient)
    #Update parameters
    model.alphas = model.alphas - lr * gradient


    
#Signal Perceptron Classes:
#amplitude*e^(freq*input)
#amplitude=funcxsignals 
#freq=signalsxvars
#input=batchxvars
class SP_numpy(object):
    def __init__(self,m,k,heads=1):
        self.m=m
        self.freq=freq_gen_sp(m,k)
        self.init_alphas=.5 * np.random.randn(heads, m**k)
        self.alphas=self.init_alphas.copy()
    #print("frecuency matrix",arrw.shape)
    def __call__(self,x):
        return self.forward(x)

    def forward(self,x):
        #print("freq",self.freq.shape)
        #print("x",x.shape)
        self.freq = np.transpose(self.freq)
        #print("freq trans",self.freq.shape)
        exp=np.dot(x,self.freq)
        #print("exponent",exp.shape)
        o_sp=np.exp((1j*np.pi/(self.m-1))*exp)
        #print("after exponential \n",o_sp.shape,self.alphas.shape)
        y_sp=np.dot(o_sp,np.transpose(self.alphas))
        #print("result",y_sp)
        return y_sp , o_sp
    def count(self):
        return self.alphas.size
    def reset_params(self):
        self.alphas=self.init_alphas
    def load_params(self,alphas):
        self.alphas=alphas
#Exmaple:
#x=np.array([[0,0],[1,0],[0,1],[1,1]])
#print("Input",x)
#arch=SP_numpy(2,2,1)
#arch(x)

class LSP_numpy(object):
    def __init__(self,m,k,heads=1):
        self.m=m
        self.freq=m
        self.init_alphas=.5 * np.random.randn(heads, m**k)
        self.alphas_real=self.init_alphas.copy()
        self.alphas_imag=self.init_alphas.copy()
    #print("frecuency matrix",arrw.shape)
    def __call__(self,x):
        return self.forward(x)

    def forward(self,x):
        print("freq",self.freq.shape)
        print("x",x.shape)
        self.freq = np.transpose(self.freq)
        print("x trans",x.shape)
        exp=np.dot(x,self.freq)
        print("exponent",exp.shape)
        o_sp=np.exp((1j*np.pi/(self.m-1))*exp)
        print("after exponential \n",o_sp)
        y_sp=np.dot(self.alphas,o_sp)
        print("result",y_sp)
        return y_sp , o_sp
    def count(self):
        return self.alphas.size
    def reset_params(self):
        self.alphas=self.init_alphas
    def load_params(self,alphas):
        self.alphas=alphas
#Exmaple:
#x=np.array([[0,0],[1,0],[0,1],[1,1]])
#print("Input",x)
#arch=SP_numpy(2,2,1)
#arch(x)

class RSP_numpy(object):
    def __init__(self,m,k,heads=1):
        self.m=m
        self.freq=freq_gen_sp(m,k)
        self.init_alphas=.5 * np.random.randn(heads, m**k)
        self.alphas=self.init_alphas.copy()
    #print("frecuency matrix",arrw.shape)
    def forward(self,x):
        #print("x",x.shape)
        x = np.transpose(x)
        #print("x trans",x.shape)
        exp=np.dot(self.freq,x)
        #print("exponent",exp.shape)
        o_sp=np.cos((np.pi/(self.m-1))*exp)
        #print("after exponential",o_sp)
        #print("theta vector",theta.shape)
        y_sp=np.dot(self.alphas,o_sp)
        #print("result",y_sp)
        return y_sp , o_sp
    def count(self):
        return self.alphas.size
    def reset_params(self):
        self.alphas=self.init_alphas
    def load_params(self,alphas):
        self.alphas=alphas

#Signal Perceptron pytorch implementation.
class SP_pytorch(nn.Module):
    def __init__(self,inout, parameters=None):
        m,k, heads = inout 
        super(SP_pytorch, self).__init__()
        self.m = m
        if parameters is None:
            self.imag_params = torch.from_numpy(freq_gen_sp(m,k))
            self.real_params = torch.zeros(self.imag_params.size())
        else:
            self.real_params,self.imag_params=parameters
        self.freq_real = nn.Linear(k, m**k,bias=False)
        self.freq_imag = nn.Linear(k, m**k,bias=False)
        self.freq_real.weight = torch.nn.Parameter(self.real_params)
        self.freq_imag.weight = torch.nn.Parameter(self.imag_params)
        for param in self.freq_real.parameters():
            param.requires_grad = False
        for param in self.freq_imag.parameters():
            param.requires_grad = False
        self.alphas_real= nn.Linear(m**k, heads,bias=False)
        self.alphas_imag= nn.Linear(m**k, heads,bias=False)
        
    def forward(self, x):
        if torch.is_complex(x):
            x_real=x.real
            x_imag=x.imag
        else:
            x_real=x.type(torch.float32)
            x_imag=torch.zeros(x.size()).type(torch.float32)  
        f_r_r=self.freq_real(x_real)
        f_r_i=self.freq_real(x_imag)
        f_i_r=self.freq_imag(x_real)
        f_i_i=self.freq_imag(x_imag)
        exp_val=f_r_r-f_i_i
        freq_val=f_r_i+f_i_r
        dot_prod_r1=torch.matmul(torch.exp(exp_val),torch.cos((torch.tensor(math.pi)/(self.m-1))*freq_val))
        dot_prod_r2=torch.matmul(torch.exp(exp_val),(torch.tensor(math.pi)/(self.m-1))*torch.sin(freq_val))
        dot_prod_i1=torch.matmul(torch.exp(exp_val),torch.cos((torch.tensor(math.pi)/(self.m-1))*freq_val))
        dot_prod_i2=torch.matmul(torch.exp(exp_val),(torch.tensor(math.pi)/(self.m-1))*torch.sin(freq_val))
        signal_real=self.alphas_real(dot_prod_r1)-self.alphas_imag(dot_prod_r2)
        signal_imag=self.alphas_imag(dot_prod_i1)+self.alphas_real(dot_prod_i1)
        #real=torch.exp(exp_val)*signal_real
        #imag=torch.exp(exp_val)*signal_imag
        #x=torch.stack((real,imag),-1)
        #z=torch.view_as_complex(x)
        return signal_real,signal_imag

#Example
#arch=SP_pytorch((2,2,16))
#x=torch.randint(0,2,(4,2)).type(torch.float32)
#print(arch(x))


#Laplace signal perceptron:Warning forward function may not be properly implemented.
#The forward function should do :
#amplitude*e^(freq*input)= e^(a x) (g cos(a y + b x) - d sin(a y + b x)) + i e^(a x) (d cos(a y + b x) + g sin(a y + b x))
#prod_r1=e^(a x - b y)cos(a y + b x)
#prod_r2=e^(a x - b y)sin(a y + b x)
#dprod_i1=e^(a x - b y)cos(a y + b x)
#prod_i2=e^(a x - b y)sin(a y + b x)
#amplitude*e^(freq*input)=  (g dot_prod_r1 - d dot_prod_r2) + i (d dot_prod_i1 + g dot_prod_i2)
#amplitude= g+id
#freq= a+ib
#input= x+iy

class LSP_pytorch(nn.Module):
    def __init__(self,inout,parameters):
        k,heads=inout
        n = parameters
        super(LSP_pytorch, self).__init__()
        self.m = n
        self.k = k
        self.freq_real = nn.Linear(k, n,bias=False)
        self.freq_imag = nn.Linear(k, n,bias=False)
        self.alphas_real =nn.Linear(n, heads,bias=False)
        self.alphas_imag =nn.Linear(n, heads,bias=False)

    def forward(self, x):
        if torch.is_complex(x):
            print("complex")
            x_real=x.real
            x_imag=x.imag
        else:
            x_real=x.type(torch.float32)
            x_imag=torch.zeros(x.size()).type(torch.float32)  
        f_r_r=self.freq_real(x_real)
        f_r_i=self.freq_real(x_imag)
        f_i_r=self.freq_imag(x_real)
        f_i_i=self.freq_imag(x_imag)
        exp_val=torch.exp(f_r_r-f_i_i)
        #trans_exp=torch.transpose(exp_val,0,1)
        freq_val=f_r_i+f_i_r
        #dot_prod_r1=torch.matmul(trans_exp,torch.cos(freq_val))
        #dot_prod_r2=torch.matmul(trans_exp,torch.sin(freq_val))
        #dot_prod_i1=torch.matmul(trans_exp,torch.cos(freq_val))
        #dot_prod_i2=torch.matmul(trans_exp,torch.sin(freq_val))
        dot_prod_r1=exp_val*torch.cos(freq_val)
        dot_prod_r2=exp_val*torch.sin(freq_val)
        dot_prod_i1=exp_val*torch.cos(freq_val)
        dot_prod_i2=exp_val*torch.sin(freq_val)
        signal_real=self.alphas_real(dot_prod_r1)-self.alphas_imag(dot_prod_r2)
        signal_imag=self.alphas_imag(dot_prod_i1)+self.alphas_real(dot_prod_i1)
        #real=torch.exp(exp_val)*signal_real
        #imag=torch.exp(exp_val)*signal_imag
        #x=torch.stack((real,imag),-1)
        #z=torch.view_as_complex(x)
        return signal_real,signal_imag
#Example
#y=torch.tensor([[1,2],[2,3]])
#print(torch.exp(y))
#x=torch.randint(0,2,(4,2)).type(torch.float32)
#input=torch.tensor([[0,1],[1,1]])
#print("Input;",input)
#arch=LSP_pytorch((2,10),(256))
#print(arch(input))

#Real Signal Perceptron (RSP) implementation using pytorch. This version is defined in the signal perceptron paper
class RSP_pytorch(nn.Module):
    def __init__(self,inout,parameters=None):
        m, k, heads = inout
        super(RSP_pytorch, self).__init__()
        self.m = m
        if parameters is None:
            self.params = torch.from_numpy(freq_gen_sp(m,k))
        else:
            self.params=parameters
        self.freq = nn.Linear(k, m**k,bias=False)
        self.freq.weight = torch.nn.Parameter(self.params)
        for param in self.freq.parameters():
            param.requires_grad = False
        self.alphas= nn.Linear(m**k, heads,bias=False)

    def forward(self, x):
        freq=self.freq(x)
        signals=torch.cos((torch.tensor(math.pi)/(self.m-1))*freq)
        x=self.alphas(signals)
        return x
#Example
#arch=RSP_pytorch((2,2,16))
#x=torch.randint(0,2,(4,2)).type(torch.float32)
#print(arch(x))

#Fourier Signal Perceptron (FSP) implementation using pytorch. This version is defined in the signal perceptron paper
class FSP_pytorch(nn.Module):
    def __init__(self,inout, parameters):
        k,heads=inout
        n=parameters
        """
        Parameters:
        k=arity of number of input variables
        heads = is the number of outputs of the unit
        n= the number of signals that the FSP will have
        """
        super(FSP_pytorch, self).__init__()
        self.m = n
        self.k = k
        self.freq = nn.Linear(k, n,bias=False)
        self.alphas =nn.Linear(n, heads,bias=False)

    def forward(self, x):
        freq=self.freq(x)
        signals=torch.cos(freq)
        x=self.alphas(signals)
        return x
#Example
#x=torch.randn(2,2)
#arch=FSP_pytorch((2,1),(8))
#print(arch(x))
