import torch
from torch import Tensor
import torch.nn as nn
import numpy as np
import signal_perceptron as sp

def train_pytorch(x_train,y_train,model,PATH,epochs,optimizer,loss_fn):
    total_hist=[]
    final_loss=[]
    learned_epochs=[]
    for i in y_train:
        model.load_state_dict(torch.load(PATH))
        i = i.unsqueeze(1)
        history_train=[]
        learned_epoch=[]
        for j in range(0,epochs):
            pred=model(x_train)
            loss = loss_fn(pred,i)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            history_train.append([j,loss])
            if not bool(learned_epoch):
                if loss<=.001:
                    learned_epoch.append(j)
        learned_epochs.append(learned_epoch)
        final_loss.append(loss)
        total_hist.append(history_train)
    return total_hist,final_loss,learned_epochs


def train_numpy(x_train,y_train,model,epochs,learning_rate,loss_fn):
    total_hist=[]
    final_loss=[]
    learned_epochs=[]
    for i in y_train:
        model.reset_params()
        #i = i.unsqueeze(1)
        history_train=[]
        learned_epoch=[]
        for j in range(0,epochs):
            pred,signals=model.forward(x_train)
            loss = loss_fn(pred, i)
            loss = np.sum(loss)
            sp.GD_MSE_SP_step(i, x_train, model,learning_rate)
            history_train.append([j,loss])
            if not bool(learned_epoch):
                if loss<=.001:
                    learned_epoch.append(j)
        learned_epochs.append(learned_epoch)
        final_loss.append(loss)
        total_hist.append(history_train)
    #print(total_hist[1])
    return total_hist,final_loss,learned_epochs

def train_mh_pytorch(x_train,y_train,model,PATH,epochs,optimizer,loss_fn):
    y_train=torch.transpose(y_train, 0, 1)
    total_hist=[]
    final_loss=[]
    learned_epoch=[]
    model.load_state_dict(torch.load(PATH))
    history_train=[]
    for j in range(0,epochs):
        pred=model(x_train)
        loss = loss_fn(pred,y_train)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        history_train.append([j,loss])
        if not bool(learned_epoch):
            if loss<=.001:
                learned_epoch.append(j)
    final_loss=loss
    total_hist.append(history_train)
    return total_hist,final_loss,learned_epoch


def train_mh_numpy(x_train,y_train,model,epochs,learning_rate,loss_fn):
    total_hist=[]
    final_loss=[]
    learned_epoch=[]
    history_train=[]
    for j in range(0,epochs):
        pred,signals=model.forward(x_train)
        loss = loss_fn(pred, y_train)
        loss = np.sum(loss)
        sp.GD_MSE_SP_step(y_train, x_train, model,learning_rate)
        history_train.append([j,loss])
        if not bool(learned_epoch):
            if loss<=.001:
                learned_epoch.append(j)
    final_loss.append(loss)
    total_hist.append(history_train)
    return total_hist,final_loss,learned_epoch
    
"""MNIST TRAINING LOOPS"""
def train_mnist(dataloader, model, loss_fn, optimizer,device):
    size = len(dataloader.dataset)
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
            return loss 

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
