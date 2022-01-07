import sys
import os
print(os.path.dirname(os.path.dirname(os.getcwd())))
sys.path.append(os.path.dirname(os.path.dirname(os.getcwd())))
import torch
from ml.model import *
from ml.archs.SP.sp_paper_models import *
from ml.train.static import *

fsp1=FSP_mnist(784)
fsp=nn.Linear(4, 1)
model1=Model(fsp,nn.CrossEntropyLoss(),torch.optim.Adam(fsp.parameters(), lr=.001),dummy_train,"linear.pth")
model2=Model(fsp1,nn.CrossEntropyLoss(),torch.optim.Adam(fsp1.parameters(), lr=.001),dummy_train,"fsp_784.pth")
model1.print_specs()
model2.print_specs()
model1.save_checkpoint()
x=torch.tensor([[0.,0.,0.,0.],[0.,0.,0.,0.],[0.,0.,0.,0.],[0.,0.,0.,0.],[0.,0.,0.,0.]],device=model1.device)
#x=x.to(model1.device)
y=torch.tensor([1.,0.,1.,1.],device=model1.device)
print("device",x.device,y.device)
#y=y.to(model1.device)
print(model1.device)
z,y=model1.pred(x)
print(z,y)
y=y.sum()
y.backward()


model1.learn([x,y])


def train_mnist(dataloader, model1):
    size = len(dataloader.dataset)
    time_backward=[]
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(model1.device), y.to(model1.device)
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
