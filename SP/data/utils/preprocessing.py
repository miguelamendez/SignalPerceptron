"""This utilities are for preprocessing datasets"""
#Internal libraries
import os
import sys
full_path = os.path.dirname(os.path.realpath(__file__))
print("[data]:preprocessing.py:",full_path)
print(os.path.dirname(os.path.dirname(full_path)))
sys.path.append(os.path.dirname(os.path.dirname(full_path)))


import numpy as np
import cv2 as cv
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import pandas as pd
import torchvision.transforms as transforms
#Tabular preprocessing ############################################################################################################################
#To do:
#Separate the functions from the class
#Create one class for dataset(when you have also the outputs) and one class for only inputs
class Tabular():
    def __init__(self,csv_path,cat_cols,num_cols,output_col,remove_cols=None):
        self.data = pd.read_csv(csv_path)
        if remove_cols is not None:
            self.remove_columns(remove_cols)
        self.cat_data=[]
        self.cat_emb=[]
        self.cat_proc(cat_cols)
        self.num_data=[]
        self.num_proc(num_cols)
        self.out_data=[]
        self.out_proc(output_col)
    def remove_columns(self,remove_columns):
        self.data =self.data.drop(remove_columns, axis=1)
    def transform_nan(self,index):
        return
    def cat_proc(self,categorical_columns):
        #Creating categorical data:
        for category in categorical_columns:
            self.data[category] = self.data[category].astype('category')
            #self.cat.append(self.data[category].cat.codes.values)
        cat=np.stack([self.data[col].cat.codes.values for col in categorical_columns], 1)
        self.cat_data = torch.tensor(cat, dtype=torch.int64)
        #Creating emmbedding sizes:
        categorical_column_sizes=[len(self.data[column].cat.categories) for column in categorical_columns]
        self.cat_emb= [(col_size, min(50, (col_size+1)//2)) for col_size in categorical_column_sizes]
    def num_proc(self,numerical_columns):
        num = np.stack([self.data[col].values for col in numerical_columns], 1)
        self.num_data = torch.tensor(num, dtype=torch.float)
    def out_proc(self,outputs):
        self.out_data=torch.tensor(self.data[outputs].values).flatten()
#Example
#Loading data
#data_set ="/home/misha/Projects/AI/data/datasets/tabular/Churn_Modelling.csv"
#List of unwanted columns
#remove_cols=['RowNumber','CustomerId','Surname']
#List of cat columns
#cat_cols = ['Geography', 'Gender', 'HasCrCard', 'IsActiveMember']
#List of num columns
#num_cols = ['CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'EstimatedSalary']
#List of output
#outputs = ['Exited']
#dataset=Tabular(data_set,cat_cols,num_cols,outputs,remove_cols)
#print(dataset.cat_data.shape,dataset.num_data.shape,dataset.out_data.shape,dataset.cat_emb)

#Image preprocessing ############################################################################################################################
#To do:
#Separate the functions from the class
#Create one class for dataset(when you have also the outputs) and one class for only inputs
class Image():
    def __init__(self,data,PIL=True,resize=None,interpolation=None,grayscale=False):
        if PIL:
            self.images,self.outs=self.pil2tensor(data)
        else:
            raise ValueError("Not Implemented")

        self.resize= resize 
        self.screen_w , self.screen_h = resize if resize is not None else (False , False)
        self.interpolation= interpolation if interpolation is not None else cv.INTER_NEAREST
        self.gray=grayscale
        print(self.resize,self.screen_w,self.interpolation,self.gray)
    def resize_fn(self,image,screen_w,screen_h,interpolation):
        return cv.resize(image, (screen_w,screen_h), interpolation=self.interpolation)
    def grayscale(self,image):
        return cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    def __call__(self,image):
        if self.resize is not None:
            image=self.resize_fn(image,self.screen_w , self.screen_h,self.interpolation) 
        if self.gray:
            image=self.grayscale(image)
        return image
    def pil2tensor(self,data):
        #numpy.asarray(PIL.Image.open('test.jpg')) pil2numpy
        image=[]
        outs=[]
        transform=transforms.ToTensor()
        for img , out in mnist_trainset:
            image.append(transform(img))
            outs.append(torch.tensor(out))
        print(len(image))
        return torch.stack(image) , torch.stack(outs)
    def process(self,data):
        pass
        
#Example
#import torchvision.datasets as datasets
#mnist_trainset = datasets.MNIST(root='home/misha/Projects/AI/data/datasets/image/MNIST', train=True, download=True, transform=None)
#dataset=Image(mnist_trainset)
#print(dataset.images[0].shape,dataset.outs.shape)
#plt.imshow(dataset.images[0][0], interpolation='nearest')
#plt.show()

class NGramLanguageModeler(nn.Module):

    def __init__(self, vocab_size, embedding_dim, context_size):
        super(NGramLanguageModeler, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.linear1 = nn.Linear(context_size * embedding_dim, 128)
        self.linear2 = nn.Linear(128, vocab_size)

    def forward(self, inputs):
        embeds = self.embeddings(inputs).view((1, -1))
        out = F.relu(self.linear1(embeds))
        out = self.linear2(out)
        log_probs = F.log_softmax(out, dim=1)
        return log_probs
#losses = []
#loss_function = nn.NLLLoss()
#model = NGramLanguageModeler(len(vocab), EMBEDDING_DIM, CONTEXT_SIZE)
