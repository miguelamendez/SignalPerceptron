import numpy as np
import pickle

def gen_finite_array(m, length, size):
    """Generates and returns a finite size array of functions based on the specified parameters.

    Args:
        m ([type]): [description]
        length ([type]): [description]
        size ([type]): [description]

    Returns:
        [type]: [description]
    """
    yki = []
    aiy = np.zeros([length]);

    for i in range(size):
        ky = i;
        for j in range(length):
            aiy[j] = int( ky % m );  
            ky = int( ky / m ); 
        yt = []
        for l in aiy:
            yt.append(l)
        yki.append(yt)
    
    y = np.asarray(yki)
    return y


def data_gen(m, k, func_samples=-1):
    """[summary]

    Args:
        m ([type]): [description]
        k ([type]): [description]
        func_samples (int, optional): [description]. Defaults to -1.

    Returns:
        [type]: [description]
    """
    if func_samples > 0:
        #Creating n random functions
        Y = np.random.randint(m, size=(func_samples, m**k))
    else:
        #Creating all possible functions 
        Y = gen_finite_array(m, length=m*k, size=m**(m**k))
    
    #Creating Dataset that is all possible combinations of the inputs
    X = gen_finite_array(m, length=k, size=(m**k))
    return X, Y


def partial_data_gen():
    """Stub function used for generation of train and test sets.

    Returns:
        [type]: [description]
    """
    return train_X, train_Y, test_X, test_Y


# TODO: Rafael: This seems to be unimplemented. 
def mnist_data_gent():
    """Generates data for experiments based on the MNIST database.

    Returns:
        [type]: [description]
    """
    file = open("Fruits.obj",'rb')
    return train_X, train_Y, test_X, test_Y
