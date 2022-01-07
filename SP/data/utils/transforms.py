"""This utilities are for data transformation"""
import numpy as np
import torch



#NUMPY FUNCTIONS
def int_to_bin(arr,num_bits=8):
    """This function transforms an array of integers into an equivalent array of binary numbers
    Inputs:
    arr [type: arr]. Description: arr is the array to be processed
    num_bits [type: int]. Description: numbits is the size of the bit array where each of the integers are going to be stored. Default:8
    """
    temp_arr=[]
    for i in range(0,len(arr)):
        f="0"+str(num_bits)+"b"
        a=format(arr[i],f)
        b=np.zeros(num_bits)
        for element in range(0, len(a)):
            b[element]=float(a[element])
        temp_arr.append(b)
    bin_arr=np.asarray(temp_arr)
    return bin_arr
#Example
#print(int_to_bin([12,13,23,24,255,23]))

def int_to_bin_dataset(dataset,num_bits=8):
    """This function transforms a complete dataset of integers into an equivalent array of binary numbers
    Inputs:
    arr [type: arr]. Description: arr is the array to be processed
    num_bits [type: int]. Description: numbits is the size of the bit array where each of the integers are going to be stored. Default:8
    """
    bin_train_list=[]
    for data in dataset:
        k=0
        bin_data_list=[]
        for element in data:
            bin_element=int_to_bin([element])
            k+=1
            bin_data_list.append(bin_element)
        bin_data_array=np.asarray(bin_data_list)
        bin_train_list.append(bin_data_array)
    bin_train_array=np.stack(bin_train_list)
    return bin_train_array
#Example
#print(int_to_bin_dataset([[2,12],[134,34],[23,51],[34,67]]))
#PYTORCH FUNCTIONS

#Image
