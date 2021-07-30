import numpy as np
import matplotlib.pyplot as plt
import time

"""This utilities are for data preprocessing"""
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
            bin_element=int_to_bin(element)
            k+=1
            bin_data_list.append(bin_element)
        bin_data_array=np.asarray(bin_data_list)
        bin_train_list.append(bin_data_array)
    bin_train_array=np.stack(bin_train_list)
    return bin_train_array

""" This utilities are for getting running times"""
class Timer(object):
    def __init__(self,function,inputs=[],iterations=1000):
        self.time_hist= np.zeros(iterations)
        for i in range(0,iterations):
            start = time.time()
            output=function(inputs)
            elapsed = time.time() - start
            self.time_hist[i]=elapsed
    def mean(self):
        return np.mean(self.time_hist)
    def std(self):
        return np.std(self.time_hist)
"""This utilities are for ploting loss"""
def functions_plot(data,title,path=[],labels=[]):
    print("dse",len(data))
    for i in range(0,len(data)):
        x_axis=[]
        y_axis=[]
        for j in data[i]:
            x_axis.append(j[0])
            y_axis.append(j[1])
        if not bool(labels):
            a= "Function:"+str(i)
        else:
            a=labels[i]
        plt.plot(x_axis,y_axis, label = a)
    titles=title
    plt.title(titles)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    if not bool(path):
        plt.show()
    else:
        plt.savefig(path)
        plt.show()
        

def single_function_plot(data,title):
    x_axis=[]
    y_axis=[]
    for j in data:
        x_axis.append(j[0])
        y_axis.append(j[1])
    a= "Function:"+str(i)
    plt.plot(x_axis,y_axis, label = a)
    titles=title
    plt.title(titles)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

"""Utilis for ploting models properties"""
def latex_table(description,label,columns_names,table_data):
    print("\begin{table}[h]")
    print("\centering")
    print("\captionsetup{justification=centering}")
    print("\caption{",description,"}")
    print("\vskip 0.15in")
    print("\scalebox{1}{")
    a=[]
    for i in range(0,len(colum_names))
        a=a+"c"
    print("\begin{tabular}{",a,"}")
    print("\toprule")
    a=[]
    for j in colum_names:
        a=a +j +"&"
    a[:-1]="\"
    a=a+"\"
    print(a)
    print("\midrule")
    for i in table_data:
        a=[]
        for j in i:
            a=a +j +"&"
        a[:-1]="\"
        a=a+"\"
        print(a)
    print("\bottomrule"
    print("\end{tabular} }"
    print("\label{table:",label,"}")
    print("\end{table}")

