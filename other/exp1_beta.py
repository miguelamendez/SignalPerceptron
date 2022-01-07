from signal_perceptron import *
from sp_paper_models import *
from train import *
from utils import *
from data_load import *

def main_pytorch(m,k,epochs,lr,sample=[]):
    #Generating Dataset
    x,y=data_gen(m,k,sample)
    print(x,y)
    #device = 'cuda' if torch.cuda.is_available() else 'cpu'
    x_train=torch.tensor(x)
    x_train = x_train.type(torch.FloatTensor)
    y_train=torch.tensor(y)
    y_train = y_train.type(torch.FloatTensor)
    #Defining Models
    model=RSP_pytorch(m,k,16)
    model1=FSP_pytorch(m**k,k,16)
    model2=MLP_pytorch(m**k,k,16)
    PATH="data/models/idm_RSP.pt"
    torch.save(model.state_dict(),PATH)
    PATH1="data/models/idm_FSP.pt"
    torch.save(model1.state_dict(),PATH1)
    PATH2="data/models/idm_MLP.pt"
    torch.save(model2.state_dict(),PATH2)
    #Counting number of parameters
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    model_parameters = filter(lambda p: p.requires_grad, model1.parameters())
    params1 = sum([np.prod(p.size()) for p in model_parameters])
    model_parameters = filter(lambda p: p.requires_grad, model2.parameters())
    params2 = sum([np.prod(p.size()) for p in model_parameters])
    print("RSP number of params:",params)
    print("FSP number of params:",params1)
    print("MLP number of params:",params2)
    loss_fn1 = nn.MSELoss()
    loss_fn2 = nn.MSELoss()
    loss_fn3 = nn.MSELoss()
    #optimizer1 = torch.optim.Adam(model.parameters(), lr=lr)
    #optimizer2 = torch.optim.Adam(model1.parameters(), lr=lr)
    #optimizer3 = torch.optim.Adam(model2.parameters(), lr=lr)
    optimizer1 = torch.optim.SGD(model.parameters(), lr=lr)
    optimizer2 = torch.optim.SGD(model1.parameters(), lr=lr)
    optimizer3 = torch.optim.SGD(model2.parameters(), lr=lr)
    #Training loop
    total_hist ,final_loss,learned_epoch=train_mh_pytorch(x_train,y_train,model,PATH,epochs=epochs,optimizer=optimizer1,loss_fn=loss_fn1)
    total_hist1 ,final_loss1,learned_epoch1=train_mh_pytorch(x_train,y_train,model1,PATH1,epochs=epochs,optimizer=optimizer2,loss_fn=loss_fn2)
    total_hist2 ,final_loss2,learned_epoch2=train_mh_pytorch(x_train,y_train,model2,PATH2,epochs=epochs,optimizer=optimizer3,loss_fn=loss_fn3)
    print("r:",final_loss,learned_epoch)
    print("r1:",final_loss1,learned_epoch1)
    print("r2:",final_loss2,learned_epoch2)
    #Ploting loss for trained networks.
    #[print(i.item()) for i in final_loss]
    title='Training loss of '+str(len(total_hist))+' functions of the'+str(m)+'-valued '+str(k)+'-ary  function space with RSP_pytorch'
    title1='Training loss of '+str(len(total_hist1))+' functions of the'+str(m)+'-valued '+str(k)+'-ary  function space with FSP_pytorch'
    title2='Training loss of '+str(len(total_hist2))+' functions of the'+str(m)+'-valued '+str(k)+'-ary  function space with MLP_pytorch'
    functions_plot(total_hist,title)
    functions_plot(total_hist1,title1)
    functions_plot(total_hist2,title2)
    return final_loss,final_loss1,final_loss2

def main_numpy(m,k,epochs,lr,sample=[]):
    #Generating Dataset
    x_train,y_train=data_gen(m,k,sample)
    print(x_train,y_train)
    #Defining Models
    model1=SP_numpy(m,k,16)
    model2=RSP_numpy(m,k,16)
    #Counting number of parameters
    params1 = model1.count()
    params2 = model2.count()
    print("SP number of params:",params1)
    print("SP_r number of params:",params2)
    #Training loop
    total_hist1 ,final_loss1,learned_epoch1=train_numpy(x_train,y_train,model1,epochs=epochs,learning_rate=lr,loss_fn=MSE_Loss)
    total_hist2 ,final_loss2,learned_epoch2=train_numpy(x_train,y_train,model2,epochs=epochs,learning_rate=lr,loss_fn=MSE_Loss)
    print("r1:",final_loss1,learned_epoch1)
    print("r2:",final_loss2,learned_epoch2)
    #Ploting loss for trained networks.
    title1='Training loss of '+str(len(total_hist1))+' functions of the'+str(m)+'-valued '+str(k)+'-ary  function space with SP_numpy'
    title2='Training loss of '+str(len(total_hist2))+' functions of the'+str(m)+'-valued '+str(k)+'-ary  function space with SP_r_numpy'
    functions_plot(total_hist1,title1)
    functions_plot(total_hist2,title2)
    return final_loss1,final_loss2

#Experiment number 1
#main_numpy(2,2,1000,.01)
main_pytorch(2,2,15000,.01)
