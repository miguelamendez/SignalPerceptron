import torch
import torchvision.models as models
from torch.profiler import profile, record_function, ProfilerActivity
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda, Compose
import matplotlib.pyplot as plt
from signal_perceptron import *
from sp_paper_models import *
import time
from train import *
#Loading datasets:
#######################################################################################################################

training_data_mnist = datasets.MNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor(),
)

# Download test data from open datasets.
test_data_mnist = datasets.MNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor(),
)

training_data_f_mnist = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor(),
)

# Download test data from open datasets.
test_data_f_mnist = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor(),
)
batch_size=64
# Create data loaders.
train_mnist_dataloader = DataLoader(training_data_mnist, batch_size=batch_size)
test_mnist_dataloader = DataLoader(test_data_mnist, batch_size=batch_size)
train_f_mnist_dataloader = DataLoader(training_data_f_mnist, batch_size=batch_size)
test_f_mnist_dataloader = DataLoader(test_data_f_mnist, batch_size=batch_size)



#Loading Models:
##################################################################################################################################
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using {} device".format(device))

fsp128 = FSP_mnist(128).to(device)
fsp = FSP_mnist().to(device)
mlp1 = MLP1_mnist().to(device)
mlp2 = MLP2_mnist().to(device)
#Saving Initial values of parameters
PATH="data/models/idm_FSP128_mnist.pt"
torch.save(fsp.state_dict(),PATH)
PATH1="data/models/idm_FSP512_mnist.pt"
torch.save(fsp.state_dict(),PATH1)
PATH2="data/models/idm_MLP1_mnist.pt"
torch.save(mlp1.state_dict(),PATH2)
PATH3="data/models/idm_MLP2_mnist.pt"
torch.save(mlp2.state_dict(),PATH3)

#MODEL PROPERTIES:
##################################################################################################################################
##################################################################################################################################
##################################################################################################################################

#Printing Learnable Parameters
################################################################################################################################
fsp128_parameters = filter(lambda p: p.requires_grad, fsp128.parameters())
fsp128_params = sum([np.prod(p.size()) for p in fsp128_parameters])
fsp_parameters = filter(lambda p: p.requires_grad, fsp.parameters())
fsp_params = sum([np.prod(p.size()) for p in fsp_parameters])
mlp1_parameters = filter(lambda p: p.requires_grad, mlp1.parameters())
mlp1_params = sum([np.prod(p.size()) for p in mlp1_parameters])
mlp2_parameters = filter(lambda p: p.requires_grad, mlp2.parameters())
mlp2_params = sum([np.prod(p.size()) for p in mlp2_parameters])

print("Learnable Parameters for MNIST models:")
print("FSP128 \t FSP512 \t MLP 1 hidden  \t MLP 2 hidden")
print(fsp128_params,"\t",fsp_params,"\t",mlp1_params,"\t",mlp2_params)

#################################################################################################################################
#Memory:


#Forward PassTime
#################################################################################################################################

train_features, train_labels = next(iter(train_mnist_dataloader))
inputs=train_features[0]
inputs=inputs.to(device)
print(train_features.size())
print(inputs.size())
t2 = time.time()
pred2=mlp1(inputs)
elapsed2 = time.time() - t2
t22 = time.time()
pred22=mlp1(inputs)
elapsed22 = time.time() - t22
t3 = time.time()
pred3=mlp2(inputs) 
elapsed3 = time.time() - t3
t11 = time.time()
pred1=fsp128(inputs)
elapsed11 = time.time() - t11
t1 = time.time()
pred1=fsp(inputs)
elapsed1 = time.time() - t1
print("Forward time for MNIST models:")
print("FSP128 \t FSP512 \t MLP 1 hidden  \t MLP 2 hidden")
print(elapsed11,"\t",elapsed1,"\t",elapsed22,"\t",elapsed3)
#Profiler(Only Pytorch)------------------------------------


with profile(activities=[
        ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
    with record_function("model_inference mlp1"):
        mlp1(inputs)
#print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
with profile(activities=[
        ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
    with record_function("model_inference mlp1"):
        mlp1(inputs)
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
with profile(activities=[
        ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
    with record_function("model_inference mlp2"):
        mlp2(inputs)
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
with profile(activities=[
        ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
    with record_function("model_inference fsp128"):
        fsp128(inputs)
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
with profile(activities=[
        ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
    with record_function("model_inference fsp512"):
        fsp(inputs)
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
#################################################################################################################################
#Backward PassTime
#################################################################################################################################

#################################################################################################################################
#################################################################################################################################
#################################################################################################################################



#MODELS TRAINING
#################################################################################################################################
#################################################################################################################################
#################################################################################################################################
#Training Hyperparameters:
#----------------------------------------------------------
epochs = 9
loss_fn = nn.CrossEntropyLoss()
#optimizer = torch.optim.Adam(fsp128.parameters(), lr=.001)
#optimizer1 = torch.optim.Adam(fsp.parameters(), lr=.001)
#optimizer2 = torch.optim.Adam(mlp1.parameters(), lr=.001)
#optimizer3 = torch.optim.Adam(mlp2.parameters(), lr=.001)
optimizer = torch.optim.SGD(fsp128.parameters(), lr=.001)
optimizer1 = torch.optim.SGD(fsp.parameters(), lr=.001)
optimizer2 = torch.optim.SGD(mlp1.parameters(), lr=.001)
optimizer3 = torch.optim.SGD(mlp2.parameters(), lr=.001)
#----------------------------------------------------------
print("Training models with MNIST DATASET :")
print("Fourier Signal Perceptron 128")
optimal_epoch=[]
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    loss1=train_mnist(train_mnist_dataloader, fsp128, loss_fn, optimizer,device)
    accuracy,loss2=test_mnist(test_mnist_dataloader, fsp128, loss_fn,device)
    if not bool(optimal_epoch):
        optimal_epoch=[t,accuracy, loss2,loss1]
    if bool(optimal_epoch):
        if optimal_epoch[2]>loss2:
            optimal_epoch=[t,accuracy, loss2,loss1]
print("Final  epoch:")
print(epochs,accuracy,loss2,loss1)
print("Optimal  epoch:")
print(optimal_epoch)

print("Fourier Signal Perceptron 512")
optimal_epoch=[]
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    loss1=train_mnist(train_mnist_dataloader, fsp, loss_fn, optimizer1,device)
    accuracy,loss2=test_mnist(test_mnist_dataloader, fsp, loss_fn,device)
    if not bool(optimal_epoch):
        optimal_epoch=[t,accuracy, loss2]
    if bool(optimal_epoch):
        if optimal_epoch[2]>loss2:
            optimal_epoch=[t,accuracy, loss2]
print("Final  epoch:")
print(epochs,accuracy,loss2,loss1)
print("Optimal  epoch:")
print(optimal_epoch)

print("MLP 1 hidden layer Signal Perceptron")
optimal_epoch=[]
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    loss1=train_mnist(train_mnist_dataloader, mlp1, loss_fn, optimizer2,device)
    accuracy,loss2=test_mnist(test_mnist_dataloader, mlp1, loss_fn,device)
    if not bool(optimal_epoch):
        optimal_epoch=[t,accuracy, loss2,loss1]
    if bool(optimal_epoch):
        if optimal_epoch[2]>loss2:
            optimal_epoch=[t,accuracy, loss2,loss1]
print("Final  epoch:")
print(epochs,accuracy,loss2,loss1)
print("Optimal  epoch:")
print(optimal_epoch)

print("MLP 2 hidden layer Signal Perceptron")
optimal_epoch=[]
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    loss1=train_mnist(train_mnist_dataloader, mlp2, loss_fn, optimizer3,device)
    accuracy,loss2=test_mnist(test_mnist_dataloader, mlp2, loss_fn,device)
    if not bool(optimal_epoch):
        optimal_epoch=[t,accuracy, loss2,loss1]
    if bool(optimal_epoch):
        if optimal_epoch[2]>loss2:
            optimal_epoch=[t,accuracy, loss2,loss1]
print("Final  epoch:")
print(epochs,accuracy,loss2,loss1)
print("Optimal  epoch:")
print(optimal_epoch)

print("Training models with FashionMNIST DATASET :")
fsp.load_state_dict(torch.load(PATH1))
mlp1.load_state_dict(torch.load(PATH2))
mlp2.load_state_dict(torch.load(PATH3))
epochs = 9
print("Fourier Signal Perceptron 128")
optimal_epoch=[]
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    loss1=train_mnist(train_f_mnist_dataloader, fsp128, loss_fn, optimizer,device)
    accuracy,loss2=test_mnist(test_f_mnist_dataloader, fsp128, loss_fn,device)
    if not bool(optimal_epoch):
        optimal_epoch=[t,accuracy, loss2,loss1]
    if bool(optimal_epoch):
        if optimal_epoch[2]>loss2:
            optimal_epoch=[t,accuracy, loss2,loss1]
print("Final  epoch:")
print(epochs,accuracy,loss2,loss1)
print("Optimal  epoch:")
print(optimal_epoch)

print("Fourier Signal Perceptron 512")
optimal_epoch=[]
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    loss1=train_mnist(train_f_mnist_dataloader, fsp, loss_fn, optimizer1,device)
    accuracy,loss2=test_mnist(test_f_mnist_dataloader, fsp, loss_fn,device)
    if not bool(optimal_epoch):
        optimal_epoch=[t,accuracy, loss2,loss1]
    if bool(optimal_epoch):
        if optimal_epoch[2]>loss2:
            optimal_epoch=[t,accuracy, loss2,loss1]
print("Final  epoch:")
print(epochs,accuracy,loss2,loss1)
print("Optimal  epoch:")
print(optimal_epoch)

print("MLP 1 hidden layer Signal Perceptron")
optimal_epoch=[]
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    loss1=train_mnist(train_f_mnist_dataloader, mlp1, loss_fn, optimizer2,device)
    accuracy,loss2=test_mnist(test_f_mnist_dataloader, mlp1, loss_fn,device)
    if not bool(optimal_epoch):
        optimal_epoch=[t,accuracy, loss2,loss1]
    if bool(optimal_epoch):
        if optimal_epoch[2]>loss2:
            optimal_epoch=[t,accuracy, loss2,loss1]
print("Final  epoch:")
print(epochs,accuracy,loss2,loss1)
print("Optimal  epoch:")
print(optimal_epoch)

print("MLP 2 hidden layer Signal Perceptron")
optimal_epoch=[]
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    loss1=train_mnist(train_f_mnist_dataloader, mlp2, loss_fn, optimizer3,device)
    accuracy,loss2=test_mnist(test_f_mnist_dataloader, mlp2, loss_fn,device)
    if not bool(optimal_epoch):
        optimal_epoch=[t,accuracy, loss2,loss1]
    if bool(optimal_epoch):
        if optimal_epoch[2]>loss2:
            optimal_epoch=[t,accuracy, loss2,loss1]
print("Final  epoch:")
print(epochs,accuracy,loss2,loss1)
print("Optimal  epoch:")
print(optimal_epoch)

#################################################################################################################################
#################################################################################################################################
#################################################################################################################################
