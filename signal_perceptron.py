"""
    Description: This library contains all the functions used for defining the signal perceptron
    Please refer to each of the functions/classes for a full description of what they do.

    Functions (True are the implemented Functions):
        walshMatrix:
        SP_Matrix:
        SP_r_Matrix:
        frequencies_gen:
        GD_MSE_SP_step:
        MSELoss:

    Classes (True are the implemented classes):
        SP_numpy:True. Implementation of the signal perceptron using the "numpy" library (USED IN PAPER EXPERIMENTS)
        SP_r_numpy:True. Implementation of the real signal perceptron using the "numpy" library (USED IN PAPER EXPERIMENTS)
        SP_pytorch:False. Implementation of the signal perceptron using the "pytorch" library (Not functional as pytorch has problems to calculating complex gradients with respect of complex numbers)
        SP_r_pytorch:True. Implementation of the real signal perceptron using the "pytorch" library (USED IN PAPER EXPERIMENTS)
        SP_r_v2_pytorch:True. Second Implementation of the real signal perceptron using the "pytorch" library (EXPERIMENTAL NOT USED IN THE PAPER )
        SP_r_appx_pytorch:False. Implementation of the real signal perceptron with learnable frequencies and fixed ammount of signals (EXPERIMENTAL NOT USED IN THE PAPER)

           """
import numpy as np
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter, UninitializedParameter
import math

#Signal Perceptron Functions:
def walshMatrix(n):
    N=2**n
    H=np.zeros([N,N])
    for i in range(0,N):
        for j in range(0,N):
            l=0
            ij = i & j;
            while ij!=0: 
                ij=ij&(ij-1);
                l=l+1
            if(l%2==0):
                H[i,j] = 1;
            else:
                H[i,j] = -1;
    return H

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

def SP_r_Matrix(m,k):
    aix=np.zeros([k]); #Array of indexes (to order them)
    aiw=np.zeros([k]); #Array of indexes (to order them)
    ni=m**k   #Number of Iterations
    n=k  #No. of variables
    nn=m**n #|m^k| domain space
    nnn=m**nn #|Delta|=|m^m^k| function space
    # Matrix
    A=np.zeros([nn,nn]) 
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

def frequencies_gen(m,k):
    """
    Description: This function creates the frecuency array used by the signals of the signal perceptron
    Implemented:
        True
    Args:
        (m:int): The domain size , the amount of possible variables that each variable can take
        (k:int): The arity, the amount of variables that each signal can recieve

    Shape:
        - Input: integers that define the functional space
        - Output: an array of size :math:`m*k`

    Examples::
        frequencies = frequency_gen(2,2)
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
    arrw = np.asarray(wki)
    return arrw

def GD_MSE_SP_step(Y, X, model,lr=.01):
    N=len(X)
    #Calculate the gradient
    pred ,m_exp= model.forward(X)
    gradient= -2/N*np.dot((Y-pred),m_exp)
    #Update parameters
    model.alphas = model.alphas - lr * gradient

def MSE_Loss(y_label,y_pred):
	n=len(y_label)
	loss= (y_label-y_pred)**2
	#loss= 1/n*(np.sum(loss))
	return loss
    
#Signal Perceptron Classes:

class SP_numpy(object):
    def __init__(self,m,k):
        self.m=m
        self.freq=frequencies_gen(m,k)
        self.init_alphas=.5 * np.random.randn(1, m**k)
        self.alphas=self.init_alphas.copy()
    #print("frecuency matrix",arrw.shape)
    def forward(self,x):
        #print("x",x.shape)
        x = np.transpose(x)
        #print("x trans",x.shape)
        exp=np.dot(self.freq,x)
        #print("exponent",exp.shape)
        o_sp=np.exp((1j*np.pi/(self.m-1))*exp)
        #print("after exponential",o_sp)
        #print("theta vector",theta.shape)
        y_sp=np.dot(self.alphas,o_sp)
        #print("result",y_sp)
        return y_sp , o_sp
    def count(self):
        self.alphas=len(self.alphas)
    def reset_params(self):
        self.alphas=self.init_alphas

class SP_r_numpy(object):
    def __init__(self,m,k):
        self.m=m
        self.freq=frequencies_gen(m,k)
        self.init_alphas=.5 * np.random.randn(1, m**k)
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
        self.alphas=len(self.alphas)
    def reset_params(self):
        self.alphas=self.init_alphas

class SP_pytorch(nn.Module):
    def __init__(self, m, k, device=None, dtype=None):
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(SP_pytorch, self).__init__()
        self.m = m
        self.k = k
        self.freq=torch.from_numpy(frequencies_gen(m,k))
        self.freq=self.freq.type(torch.cuda.FloatTensor)
        self.freq=torch.transpose(self.freq,0,1)
        #print(self.freq)
        params=torch.empty((m**k),**factory_kwargs)
        params=torch.unsqueeze(params, 0)
        self.alphas = nn.Parameter(params)
        #print(self.alphas)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.alphas, a=math.sqrt(5))

    def forward(self, x):
        #exp=torch.dot(x,self.freq)
        y=[]
        exp=torch.mm(x,self.freq)
        #print("exponent",exp)
        o_sp=torch.exp((torch.tensor(math.pi)*j/(self.m-1))*exp)
        #print("after exponential",o_sp)
        y_sp=torch.mm(self.alphas,o_sp)
        y.append(y_sp)
        #print("result",y_sp)
        return y_sp

class SP_r_pytorch(nn.Module):
    def __init__(self, m, k, device=None, dtype=None):
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(SP_r_pytorch, self).__init__()
        self.m = m
        self.k = k
        self.freq=torch.from_numpy(frequencies_gen(m,k))
        self.freq=self.freq.type(torch.cuda.FloatTensor)#Here change if not using gpu
        self.freq=torch.transpose(self.freq,0,1)
        #print(self.freq)
        params=torch.empty((m**k),**factory_kwargs)
        params=torch.unsqueeze(params, 0)
        self.alphas = nn.Parameter(params)
        #print(self.alphas)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.alphas, a=math.sqrt(5))

    def forward(self, x):
        #exp=torch.dot(x,self.freq)
        y=[]
        exp=torch.mm(x,self.freq)
        #print("exponent",exp)
        o_sp=torch.cos((torch.tensor(math.pi)/(self.m-1))*exp)
        #print("after exponential",o_sp)
        y_sp=torch.mm(self.alphas,o_sp)
        y.append(y_sp)
        #print("result",y_sp)
        return y_sp

class SP_r_v2_pytorch(nn.Module):
    def __init__(self, m, k, device=None, dtype=None):
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(SP_r_v2_pytorch, self).__init__()
        self.m = m
        self.k = k
        params = torch.from_numpy(frequencies_gen(m,k))
        self.freq = nn.Linear(k, m**k,bias=False)
        self.freq.load_params(params)
        for param in self.freq.parameters():
            param.requires_grad = False
        self.alphas= nn.Linear(m**k, 1,bias=False)

    def forward(self, x):
        freq=self.freq(x)
        signals=torch.cos(freq)
        x=self.alphas(signals)
        return x

class SP_r_appx_pytorch(nn.Module):
    def __init__(self, n, k, device=None, dtype=None):
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(SP_r_appx_pytorch, self).__init__()
        self.m = n
        self.k = k
        self.freq = nn.Linear(k, n,bias=False)
        self.alphas =nn.Linear(n, 1,bias=False)

    def forward(self, x):
        freq=self.freq(x)
        signals=torch.cos(freq)
        x=self.alphas(signals)
        return x
