"""Some simple data generators"""
import numpy as np
import pickle

#Discrete funcitons#############################################################################################
def discrete_array(m,length,size):
    """Creates an array of a domain space, if the domain space is full then it repeats"""
    yki=[]
    aiy=np.zeros([length]);
    for i in range(0,size,1):
        ky=i;
        for j in range(0,length,1):
            aiy[j]= int ( ky % m );  
            ky=int(ky/m); 
        yt=[]
        for l in aiy:
            yt.append(l)
        yki.append(yt)
    y = np.asarray(yki)
    return y
#Example:
#Binary Boolean function
#print(discrete_array(2,2,4))
#Binary Boolean function overloaded
#print(discrete_array(2,2,6))

#Functional Space of Discrete functions
def discrete_fs(m,k,samples=None):
    if samples is not None:
        #Creating n random functions
        Y=np.random.randint(m, size=(samples,m**k))
    else:
        #Creating all possible functions 
        Y=discrete_array(m,length=m**k,size=m**(m**k))
    #Creating Dataset that is all possible combinations of the inputs
    X=discrete_array(m,length=k,size=(m**k))
    return X, Y

#Example:
#Generating one function
#print(discrete_fs(2,2,1))
#Generating whole funcitonal space
#print(discrete_fs(2,2))
