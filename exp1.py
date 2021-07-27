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

#Functional Space Hyperparameters
m=2
k=2
sample=[]

#Loading datasets:
#######################################################################################################################
#Generating Dataset
x,y=data_gen(m,k,sample)
#print(x,y)
x_train=torch.tensor(x)
x_train = x_train.type(torch.FloatTensor)
y_train=torch.tensor(y)
y_train = y_train.type(torch.FloatTensor)

#Loading Models:
################################################################################################################################
#Single functions:
sp_np= SP_numpy(m,k,1)
rsp_np= RSP_numpy(m,k,1)
fsp= FSP_pytorch(m**k,k,1)
rsp= RSP_pytorch(m,k,1)
mlp= MLP_pytorch(m**k,k,1)
PATH1="data/models/idm_RSP.pt"
torch.save(fsp.state_dict(),PATH1)
PATH2="data/models/idm_FSP.pt"
torch.save(rsp.state_dict(),PATH2)
PATH3="data/models/idm_MLP.pt"
torch.save(mlp.state_dict(),PATH3)
#Multiple functions:
heads=m**(m**k) #For learning all functions from the function_space
#heads=4
sp_np_mh= SP_numpy(m,k,heads)
rsp_np_mh= RSP_numpy(m,k,heads)
fsp_mh= FSP_pytorch(m**k,k,heads)
rsp_mh= RSP_pytorch(m,k,heads)
mlp_mh= MLP_pytorch(m**k,k,heads)
PATH4="data/models/idm_RSP_mh.pt"
torch.save(fsp_mh.state_dict(),PATH4)
PATH5="data/models/idm_FSP_mh.pt"
torch.save(rsp_mh.state_dict(),PATH5)
PATH6="data/models/idm_MLP_mh.pt"
torch.save(mlp_mh.state_dict(),PATH6)
#MODEL PROPERTIES:
##################################################################################################################################
##################################################################################################################################
##################################################################################################################################

#Printing Learnable Parameters
################################################################################################################################

sp_np_params = sp_np.count()
rsp_np_params = rsp_np.count()
fsp_parameters = filter(lambda p: p.requires_grad, fsp.parameters())
fsp_params = sum([np.prod(p.size()) for p in fsp_parameters])
rsp_parameters = filter(lambda p: p.requires_grad, rsp.parameters())
rsp_params = sum([np.prod(p.size()) for p in rsp_parameters])
mlp_parameters = filter(lambda p: p.requires_grad, mlp.parameters())
mlp_params = sum([np.prod(p.size()) for p in mlp_parameters])

print("Learnable Parameters for model that learns a function from FS:")
print("SP_np \t RSP_np \t RSP \t FSP \t MLP")
print(sp_np_params,"\t",rsp_np_params,"\t",rsp_params,"\t",fsp_params,"\t",mlp_params)

sp_np_params = sp_np_mh.count()
rsp_np_params = rsp_np_mh.count()
fsp_parameters = filter(lambda p: p.requires_grad, fsp_mh.parameters())
fsp_params = sum([np.prod(p.size()) for p in fsp_parameters])
rsp_parameters = filter(lambda p: p.requires_grad, rsp_mh.parameters())
rsp_params = sum([np.prod(p.size()) for p in rsp_parameters])
mlp_parameters = filter(lambda p: p.requires_grad, mlp_mh.parameters())
mlp_params = sum([np.prod(p.size()) for p in mlp_parameters])

print("Learnable Parameters for model that learns all FS:")
print("SP_np \t RSP_np \t RSP \t FSP \t MLP")
print(sp_np_params,"\t",rsp_np_params,"\t",rsp_params,"\t",fsp_params,"\t",mlp_params)

#################################################################################################################################
#Memory:
#Not Implemented Jet


#Forward PassTime
#################################################################################################################################
#Single functions
inputs_np = np.random.randint(m, size=(1,k))
inputs = torch.tensor(inputs_np)
inputs = inputs.type(torch.FloatTensor)
t1 = time.time()
pred1=sp_np.forward(inputs_np)
elapsed1 = time.time() - t1
t2 = time.time()
pred2=rsp_np.forward(inputs_np)
elapsed2 = time.time() - t2
t3 = time.time()
pred3=rsp(inputs) 
elapsed3 = time.time() - t3
t4 = time.time()
pred4=fsp(inputs)
elapsed4 = time.time() - t4
t5 = time.time()
pred5=mlp(inputs)
elapsed5 = time.time() - t5
print("Forward time for model that learns a function from FS:")
print("SP_np \t RSP_np \t RSP \t FSP \t MLP")
print(elapsed1,"\t",elapsed2,"\t",elapsed3,"\t",elapsed4,"\t",elapsed5)
#Profiler(Only Pytorch)------------------------------------
print("Forward time for model that learns a function from FS Profiler:")
with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof:
    with record_function("model_inference:rsp"):
        rsp(inputs)
print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))
with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof:
    with record_function("model_inference:fsp"):
        fsp(inputs)
print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))
with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof:
    with record_function("model_inference:mlp"):
        mlp(inputs)
print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))

#Multiple functions:
inputs_np =np.random.randint(m, size=(1,k))
inputs = torch.tensor(inputs_np)
inputs = inputs.type(torch.FloatTensor)
#Time----------------------------------------
t1 = time.time()
pred1=sp_np_mh.forward(inputs_np)
elapsed1 = time.time() - t1
t2 = time.time()
pred2=rsp_np_mh.forward(inputs_np)
elapsed2 = time.time() - t2
t3 = time.time()
pred3=rsp_mh(inputs) 
elapsed3 = time.time() - t3
t4 = time.time()
pred4=fsp_mh(inputs)
elapsed4 = time.time() - t4
t5 = time.time()
pred5=mlp_mh(inputs)
elapsed5 = time.time() - t5
print("Forward time for model that learns all FS:")
print("SP_np \t RSP_np \t RSP \t FSP \t MLP")
print(elapsed1,"\t",elapsed2,"\t",elapsed3,"\t",elapsed4,"\t",elapsed5)
#--------------------------------------------
#Profiler(Only Pytorch)------------------------------------
print("Forward time for model that learns all FS Profiler:")
with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof:
    with record_function("model_inference:rsp_mh"):
        rsp_mh(inputs)
print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))
with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof:
    with record_function("model_inference:fsp_mh"):
        fsp_mh(inputs)
print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))
with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof:
    with record_function("model_inference:mlp_mh"):
        mlp_mh(inputs)
print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))
#################################################################################################################################
#Backward PassTime
#################################################################################################################################


#MODELS TRAINING
#################################################################################################################################
#################################################################################################################################
#################################################################################################################################
def full_analysis_train():
    print("Training models for space (m,k): (",m ,"," ,k,")")
    print("Single function learning")
    #Training Hyperparameters:
    #----------------------------------------------------------
    epochs = 1000
    lr = .01
    optimizer_name="sgd"
    loss_fn_1 = nn.MSELoss()
    loss_fn_2 = nn.MSELoss()
    loss_fn_3 = nn.MSELoss()
    #optimizer1 = torch.optim.Adam(fsp.parameters(), lr=lr)
    #optimizer2 = torch.optim.Adam(rsp.parameters(), lr=lr)
    #optimizer3 = torch.optim.Adam(mlp.parameters(), lr=lr)
    optimizer1 = torch.optim.SGD(fsp.parameters(), lr=lr)
    optimizer2 = torch.optim.SGD(rsp.parameters(), lr=lr)
    optimizer3 = torch.optim.SGD(mlp.parameters(), lr=lr)
    print("Parameters: ")
    print("Epochs: ",epochs," lr: ",lr," optimizer: ",optimizer_name)
    #----------------------------------------------------------
    print("Signal Perceptron numpy")
    total_hist1,final_loss1,learned_epochs1=train_numpy(x_train=x,y_train=y,model=sp_np,epochs=epochs,learning_rate=lr,loss_fn=MSE_Loss)
    print(final_loss1,learned_epochs1)
    print("Real Signal Perceptron numpy")
    total_hist2,final_loss2,learned_epochs2=train_numpy(x_train=x,y_train=y,model=rsp_np,epochs=epochs,learning_rate=lr,loss_fn=MSE_Loss)
    print(final_loss2,learned_epochs2)
    print("Real Signal Perceptron pytorch")
    total_hist3,final_loss3,learned_epochs3=train_pytorch(x_train=x_train,y_train=y_train,model=rsp,PATH=PATH2,epochs=epochs,optimizer=optimizer2,loss_fn=loss_fn_1)
    print(final_loss3,learned_epochs3)
    print("Fourier Signal Perceptron pytorch")
    total_hist4,final_loss4,learned_epochs4=train_pytorch(x_train=x_train,y_train=y_train,model=fsp,PATH=PATH1,epochs=epochs,optimizer=optimizer1,loss_fn=loss_fn_2)
    print(final_loss4,learned_epochs4)
    print(" Multilayer Perceptron pytorch")
    total_hist5,final_loss5,learned_epochs5=train_pytorch(x_train=x_train,y_train=y_train,model=mlp,PATH=PATH3,epochs=epochs,optimizer=optimizer3,loss_fn=loss_fn_3)
    print(final_loss5,learned_epochs5)
    #Ploting loss for trained networks.
    title1='Training loss of '+str(len(total_hist1))+' functions of the m:'+str(m)+',k:'+str(k)+' function space with SP_np'
    title2='Training loss of '+str(len(total_hist2))+' functions of the m:'+str(m)+',k:'+str(k)+' function space with RSP_np'
    title3='Training loss of '+str(len(total_hist3))+' functions of the m:'+str(m)+',k:'+str(k)+' function space with SP_pt'
    title4='Training loss of '+str(len(total_hist4))+' functions of the m:'+str(m)+',k:'+str(k)+' function space with FSP_pt'
    title5='Training loss of '+str(len(total_hist5))+' functions of the m:'+str(m)+',k:'+str(k)+' function space with MLP_pt'
    dir_section="run1/sgd/"
    image_path1="data/experiments/exp1/"+dir_section+"sp_np.png"
    image_path2="data/experiments/exp1/"+dir_section+"rsp_np.png"
    image_path3="data/experiments/exp1/"+dir_section+"rsp_pt.png"
    image_path4="data/experiments/exp1/"+dir_section+"fsp_pt.png"
    image_path5="data/experiments/exp1/"+dir_section+"mlp_pt.png"
    #image_path3="data/experiments/exp1/run1/adam/rsp_pt"
    #image_path4="data/experiments/exp1/run1/adam/fsp_pt"
    #image_path5="data/experiments/exp1/run1/adam/mlp_pt"
    functions_plot(total_hist1,title1,image_path1)
    functions_plot(total_hist2,title2,image_path2)
    functions_plot(total_hist3,title3,image_path3)
    functions_plot(total_hist4,title4,image_path4)
    functions_plot(total_hist5,title5,image_path5)
    print("Training models for space (m,k): (",m ,"," ,k,")")
    print("Multiple function learning")
    #Training Hyperparameters:
    #----------------------------------------------------------
    epochs = 1000
    lr = .01
    optimizer_name="SGD"
    loss_fn_pt1 = nn.MSELoss()
    loss_fn_pt2 = nn.MSELoss()
    loss_fn_pt3 = nn.MSELoss()
    #optimizer1 = torch.optim.Adam(fsp_mh.parameters(), lr=lr)
    #optimizer2 = torch.optim.Adam(rsp_mh.parameters(), lr=lr)
    #optimizer3 = torch.optim.Adam(mlp_mh.parameters(), lr=lr)
    optimizer1 = torch.optim.SGD(fsp_mh.parameters(), lr=lr)
    optimizer2 = torch.optim.SGD(rsp_mh.parameters(), lr=lr)
    optimizer3 = torch.optim.SGD(mlp_mh.parameters(), lr=lr)
    print("Parameters: ")
    print("Epochs: ",epochs," lr: ",lr," optimizer: ",optimizer_name)
    #----------------------------------------------------------
    print("Signal Perceptron numpy")
    total_hist1,final_loss1,learned_epochs1=train_mh_numpy(x_train=x,y_train=y,model=sp_np_mh,epochs=epochs,learning_rate=lr,loss_fn=MSE_Loss)
    print(final_loss1,learned_epochs1)
    print("Real Signal Perceptron numpy")
    total_hist2,final_loss2,learned_epochs2=train_mh_numpy(x_train=x,y_train=y,model=rsp_np_mh,epochs=epochs,learning_rate=lr,loss_fn=MSE_Loss)
    print(final_loss2,learned_epochs2)
    print("Real Signal Perceptron pytorch")
    total_hist3,final_loss3,learned_epochs3=train_mh_pytorch(x_train=x_train,y_train=y_train,model=rsp_mh,PATH=PATH5,epochs=epochs,optimizer=optimizer2,loss_fn=loss_fn_pt1)
    print(final_loss3,learned_epochs3)
    print("Fourier Signal Perceptron pytorch")
    total_hist4,final_loss4,learned_epochs4=train_mh_pytorch(x_train=x_train,y_train=y_train,model=fsp_mh,PATH=PATH4,epochs=epochs,optimizer=optimizer1,loss_fn=loss_fn_pt2)
    print(final_loss4,learned_epochs4)
    print(" Multilayer Perceptron pytorch")
    total_hist5,final_loss5,learned_epochs5=train_mh_pytorch(x_train=x_train,y_train=y_train,model=mlp_mh,PATH=PATH6,epochs=epochs,optimizer=optimizer3,loss_fn=loss_fn_pt3)
    print(final_loss5,learned_epochs5)
    #Ploting loss for trained networks. 
    title1='Training loss of '+str(len(total_hist1))+' functions of the m:'+str(m)+',k:'+str(k)+' function space with SP_np'
    title2='Training loss of '+str(len(total_hist2))+' functions of the m:'+str(m)+',k:'+str(k)+' function space with RSP_np'
    title3='Training loss of '+str(len(total_hist3))+' functions of the m:'+str(m)+',k:'+str(k)+' function space with SP_pt'
    title4='Training loss of '+str(len(total_hist4))+' functions of the m:'+str(m)+',k:'+str(k)+' function space with FSP_pt'
    title5='Training loss of '+str(len(total_hist5))+' functions of the m:'+str(m)+',k:'+str(k)+' function space with MLP_pt'
    dir_section="run1/sgd/"
    image_path1="data/experiments/exp1/"+dir_section+"sp_np_mh.png"
    image_path2="data/experiments/exp1/"+dir_section+"rsp_np_mh.png"
    image_path3="data/experiments/exp1/"+dir_section+"rsp_pt_mh.png"
    image_path4="data/experiments/exp1/"+dir_section+"fsp_pt_mh.png"
    image_path5="data/experiments/exp1/"+dir_section+"mlp_pt_mh.png"
    #image_path3="data/experiments/exp1/run1/adam/rsp_pt_mh"
    #image_path4="data/experiments/exp1/run1/adam/fsp_pt_mh"
    #image_path5="data/experiments/exp1/run1/adam/mlp_pt_mh"
    functions_plot(total_hist1,title1,image_path1)
    functions_plot(total_hist2,title2,image_path2)
    functions_plot(total_hist3,title3,image_path3)
    functions_plot(total_hist4,title4,image_path4)
    functions_plot(total_hist5,title5,image_path5)
    #################################################################################################################################
    #################################################################################################################################
    #################################################################################################################################
full_analysis_train()