import torch
import torchvision.models as models
from torch.profiler import profile, record_function, ProfilerActivity
import time
from signal_perceptron import *
from sp_paper_models import *
from data_load import *
from train import *
import os
from utils import *
def full_analysis_train(run,epochs,lr,titlews,inputs,perceptron,plot):
    #Functional Space Hyperparameters
    m=2
    k=inputs
    sample=[]

    #Loading datasets:
    #######################################################################################################################
    #Generating Dataset
    x,y=data_gen(m,k,sample)
    print(y)
    #print(x,y)
    x_train=torch.tensor(x)
    x_train = x_train.type(torch.FloatTensor)
    y_train=torch.tensor(y)
    y_train = y_train.type(torch.FloatTensor)

    #Loading Model:
    ################################################################################################################################
    #Single functions:
    if perceptron=="rsp":
        model= RSP_pytorch(m,k,1)
        PATH="data/models/idm_rsp.pt"
        torch.save(model.state_dict(),PATH)
    elif perceptron=="rsp_np":
        model= RSP_numpy(m,k,1)
    elif perceptron=="sp":
        model= SP_numpy(m,k,1)
    elif perceptron=="mlp":
        model= MLP_pytorch(m**k,k,1)
        PATH="data/models/idm_mlp.pt"
        torch.save(model.state_dict(),PATH)
    elif perceptron=="fsp":
        model= FSP_pytorch(m**k,k,1)
        PATH="data/models/idm_fsp.pt"
        torch.save(model.state_dict(),PATH)
    elif perceptron=="gn":
        model= GN_pytorch(k)
        PATH="data/models/idm_gn.pt"
        torch.save(model.state_dict(),PATH)
    ##################################################################################################################################
    ##################################################################################################################################
    ##################################################################################################################################

    #Printing Learnable Parameters
    ################################################################################################################################
    if perceptron=="sp":
        l_params = model.count()
    elif perceptron=="rsp_np":
        l_params = model.count()
    else:
        params = filter(lambda p: p.requires_grad, model.parameters())
        l_params = sum([np.prod(p.size()) for p in params])

    print("Ammount of Learnable Parameters for model that learns a function from FS:")
    print(l_params)



#MODELS TRAINING
#################################################################################################################################
#################################################################################################################################
#################################################################################################################################
    print("Training models for space (m,k): (",m ,"," ,k,")")
    print("Single function learning")
    #Training Hyperparameters:
    #----------------------------------------------------------
    epochs = epochs
    lr = lr
    optimizer_name="sgd"
    loss = nn.MSELoss()
    if perceptron=="sp" or perceptron=="rsp_np":
        optimizer=[]
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    print("Parameters: ")
    print("Epochs: ",epochs," lr: ",lr," optimizer: ",optimizer_name)
    #----------------------------------------------------------
    print("Model: ",perceptron)
    if perceptron=="sp":
        total_hist,final_loss,learned_epochs,time_backward=train_numpy(x_train=x,y_train=y,model=model,epochs=epochs,learning_rate=lr,loss_fn=MSE_Loss)
    elif perceptron=="rsp_np":
        total_hist,final_loss,learned_epochs,time_backward=train_numpy(x_train=x,y_train=y,model=model,epochs=epochs,learning_rate=lr,loss_fn=MSE_Loss)
    else:
        total_hist,final_loss,learned_epochs,time_backward=train_pytorch(x_train=x_train,y_train=y_train,model=model,PATH=PATH,epochs=epochs,optimizer=optimizer,loss_fn=loss)
    #counting the amount of learned functions:
    lf_counter=0
    for i in learned_epochs:
        if len(i)!=0:
            lf_counter+=1
    print(np.round(final_loss,4),lf_counter)
    if plot=="y" or plot=="yes":
        #Ploting loss for trained networks.
        title='Training loss of '+str(len(total_hist))+' functions of the m:'+str(m)+',k:'+str(k)+' function space with '+perceptron
        dir_section="run"+str(run+1)+"/graphs"
        image_path="data/experiments/exp1_3/"+dir_section+"/single/"+titlews+str(k)+perceptron+".png"
        functions_plot(total_hist,title,image_path)



#Main loop
import sys
a=[.1]
b=[100,1000,2000]
print("This experiment is gona be run ",sys.argv[-2], " times:")
plot = str(sys.argv[-1])
n= int(sys.argv[-2])
inputs= int(sys.argv[-3])
perceptron = str(sys.argv[-4])


print(perceptron)
for i in range(n):
    for j in a:
         for k in b:
              orig_stdout = sys.stdout
              subfolder="run"+str(i+1)
              subname=str(j)+"_"+str(k)+"_"+perceptron+"_"+str(inputs)
              out="data/experiments/exp1_3/"+subfolder+"/data/"+subname+".txt"
              f = open(out, 'w+')
              sys.stdout = f
              full_analysis_train(i,k,j,subname,inputs,perceptron,plot)              
              sys.stdout = orig_stdout
              f.close()

