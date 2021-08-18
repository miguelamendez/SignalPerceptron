import torch
from torch import Tensor
import torch.nn as nn
import numpy as np
import signal_perceptron as sp
from utils import *
import time

#Train loops for first set of experiments (check exp1.py)
def train_pytorch(x_train,y_train,model,PATH,epochs,optimizer,loss_fn):
    total_hist=[]
    final_loss=[]
    learned_epochs=[]
    total_time=[]
    for i in y_train:
        model.load_state_dict(torch.load(PATH))
        i = i.unsqueeze(1)
        history_train=[]
        learned_epoch=[]
        time_backward=np.zeros(epochs)
        for j in range(0,epochs):
            start=time.time()
            pred=model(x_train)
            loss = loss_fn(pred,i)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            end = time.time()-start
            time_backward[j]=end
            history_train.append([j,loss])
            if not bool(learned_epoch):
                if loss<=.001:
                    learned_epoch.append(j)
        learned_epochs.append(learned_epoch)
        final_loss.append(loss.detach().numpy())
        total_hist.append(history_train)
        total_time.append(time_backward)
    l=0
    for i in final_loss:
        l=l+i
    avg_fl=l/len(final_loss)
    return total_hist,avg_fl,learned_epochs,total_time


def train_numpy(x_train,y_train,model,epochs,learning_rate,loss_fn):
    total_hist=[]
    final_loss=[]
    learned_epochs=[]
    total_time=[]
    for i in y_train:
        model.reset_params()
        #i = i.unsqueeze(1)
        history_train=[]
        learned_epoch=[]
        time_backward=np.zeros(epochs)
        for j in range(0,epochs):
            start=time.time()
            pred,signals=model.forward(x_train)
            loss = loss_fn(pred, i)
            loss = np.mean(loss)
            sp.GD_MSE_SP_step(i, x_train, model,learning_rate)
            end = time.time()-start
            time_backward[j]=end
            history_train.append([j,loss])
            if not bool(learned_epoch):
                if loss<=.001:
                    learned_epoch.append(j)
        learned_epochs.append(learned_epoch)
        final_loss.append(loss)
        total_hist.append(history_train)
        total_time.append(time_backward)
    #print(total_hist[1])
    l=0
    for i in final_loss:
        l=l+i
    avg_fl=l/len(final_loss)
    return total_hist,avg_fl,learned_epochs,total_time

#Train loops for second set of experiments (check exp2.py)
def train_mh_pytorch(x_train,y_train,model,PATH,epochs,optimizer,loss_fn):
    y_train=torch.transpose(y_train, 0, 1)
    total_hist=[]
    final_loss=[]
    learned_epoch=[]
    model.load_state_dict(torch.load(PATH))
    history_train=[]
    time_backward=np.zeros(epochs)
    for j in range(0,epochs):
        start=time.time()
        pred=model(x_train)
        loss = loss_fn(pred,y_train)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        end = time.time()-start
        time_backward[j]=end
        history_train.append([j,loss.detach().numpy()])
        if not bool(learned_epoch):
            if loss<=.001:
                learned_epoch.append(j)
    final_loss=loss.detach().numpy()
    total_hist.append(history_train)
    return total_hist,final_loss,learned_epoch,time_backward


def train_mh_numpy(x_train,y_train,model,epochs,learning_rate,loss_fn):
    total_hist=[]
    final_loss=[]
    learned_epoch=[]
    history_train=[]
    time_backward=np.zeros(epochs)
    for j in range(0,epochs):
        start=time.time()
        pred,signals=model.forward(x_train)
        loss = loss_fn(pred, y_train)
        loss = np.mean(loss)
        sp.GD_MSE_SP_step(y_train, x_train, model,learning_rate)
        end = time.time()-start
        time_backward[j]=end
        history_train.append([j,loss])
        if not bool(learned_epoch):
            if loss<=.001:
                learned_epoch.append(j)
    final_loss.append(loss)
    total_hist.append(history_train)
    return total_hist,final_loss,learned_epoch,time_backward
    
"""MNIST TRAINING LOOPS"""
def train_mnist(dataloader, model, loss_fn, optimizer,device):
    size = len(dataloader.dataset)
    time_backward=[]
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        # Compute prediction error
        start=time.time()
        pred = model(X)
        loss = loss_fn(pred, y)
        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        end = time.time()-start
        time_backward.append(end)
        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
        tb=np.asarray(time_backward)
    return loss ,tb

def test_mnist(dataloader, model, loss_fn,device):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    return (100*correct) ,test_loss

#Train loops for third set of experiments (check exp3.py)

def train_linear_numpy(y_train,sp_matrix):
    alphas=[]
    for i in y_train:
        alphas_i = np.linalg.inv(sp_matrix).dot(i)
        alphas.append(alphas_i)
    return alphas

def test_linear_numpy(x_test,y_test,model,alphas,loss_fn):
    total_loss=[]
    for i in range(0,len(y_test)):
        model.load_params(alphas[i])
        test_loss = 0
        y=y_test[i]
        for j in range(0,len(x_test)):
            pred=model.forward(j)
            test_loss += loss_fn(pred, y[j])
            #correct += ( np.sqrt(pred - y)<0.0001).type(np.float).sum().item()
        test_loss /= len(x_test)
        total_loss.append(test_loss)
    return total_loss
