import torch
from torch import Tensor
import torch.nn as nn
import numpy as np
import signal_perceptron as sp

def train_loop_pytorch(x_train,y_train,model,PATH,epochs=1000,learning_rate=.01):
    loss_fn = nn.MSELoss()
    print(model)
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    #optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    total_hist=[]
    final_loss=[]
    for i in y_train:
        model.load_state_dict(torch.load(PATH))
        i = i.unsqueeze(1)
        history_train=[]
        for j in range(0,epochs):
            pred=model(x_train)
            loss = loss_fn(pred,i)
            #loss = torch.sum(loss)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            #print(j,loss)
            history_train.append([j,loss])
        final_loss.append(loss)
        total_hist.append(history_train)
    #print(total_hist[1])
    return total_hist,final_loss


def train_loop_numpy(x_train,y_train,model,epochs=1000,learning_rate=.01):

    total_hist=[]
    final_loss=[]
    for i in y_train:
        model.reset_params()
        #i = i.unsqueeze(1)
        history_train=[]
        for j in range(0,epochs):
            pred,signals=model.forward(x_train)
            #print(i,pred)
            loss = sp.MSE_Loss(pred, i)
            loss = np.sum(loss)
            sp.GD_MSE_SP_step(i, x_train, model)
            history_train.append([j,loss])
        final_loss.append(loss)
        total_hist.append(history_train)
    #print(total_hist[1])
    return total_hist,final_loss
