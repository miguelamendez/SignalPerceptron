# example of loading the mnist dataset
#from keras.datasets import mnist
from matplotlib import pyplot
import numpy as np
# load dataset
#(trainX, trainy), (testX, testy) = mnist.load_data()
# summarize loaded dataset
#print('Train: X=%s, y=%s' % (trainX.shape, trainy.shape))
#print('Test: X=%s, y=%s' % (testX.shape, testy.shape))
# plot first few images
#for i in range(9):
#	# define subplot
#	pyplot.subplot(330 + 1 + i)
#	# plot raw pixel data
#	pyplot.imshow(trainX[i], cmap=pyplot.get_cmap('gray'))
# show the figure
#pyplot.show()
#print("here",trainy[0])

def int_to_bin(arr_,num_bits=8):
    temp_arr=[]
    for i in range(0,len(arr_)):
        f="0"+str(num_bits)+"b"
        a=format(arr_[i],f)
        b=np.zeros(num_bits)
        for element in range(0, len(a)):
            b[element]=float(a[element])
        temp_arr.append(b)
    bin_arr=np.asarray(temp_arr)
    return bin_arr




def dataset_to_bool(batch_dataset,num_bits=8):
    bin_train_list=[]
    for data in batch_dataset:
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

