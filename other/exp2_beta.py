from signal_perceptron import *
from mlp import *
from train import *
from utils import *
from data_load import *

def main(m,k,epochs,lr,sample=[]):
    #Generating Dataset
    x,y=data_gen(m,k,sample)
    print(x,y)
    x_train=torch.tensor(x)
    x_train = x_train.type(torch.cuda.FloatTensor)
    y_train=torch.tensor(y)
    y_train = y_train.type(torch.cuda.FloatTensor)
    #Defining Models
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    #model1=SP_r_v2_pytorch(m,k).to(device)
    model2=SP_r_appx_pytorch(5,k,m**(m**k)).to(device)
    #PATH1="data/models/init_dict_model_SP_r_v2.pt"
    #torch.save(model1.state_dict(),PATH1)
    PATH2="data/models/init_dict_model_SP_r_appx.pt"
    torch.save(model2.state_dict(),PATH2)
    #Counting number of parameters
    #model_parameters = filter(lambda p: p.requires_grad, model1.parameters())
    #params1 = sum([np.prod(p.size()) for p in model_parameters])
    model_parameters = filter(lambda p: p.requires_grad, model2.parameters())
    params2 = sum([np.prod(p.size()) for p in model_parameters])
    #print("SP_r_v2 number of params:",params1)
    print("SP_r_appx number of params:",params2)
    #Training loop
    #total_hist1 ,final_loss1=train_loop_pytorch(x_train,y_train,model1,PATH1,epochs=epochs,learning_rate=lr)
    total_hist2 ,final_loss2=train_loop_pytorch(x_train,y_train,model2,PATH2,epochs=epochs,learning_rate=lr)
    #Ploting loss for trained networks.
    #[print(i.item()) for i in final_loss1]
    [print(i.item()) for i in final_loss2]
    #title1='Training loss of '+str(len(total_hist1))+' functions of the'+str(m)+'-valued '+str(k)+'-ary  function space with SP_r_v2_pytorch'
    title2='Training loss of '+str(len(total_hist2))+' functions of the'+str(m)+'-valued '+str(k)+'-ary  function space with SP_r_appx_pytorch'
    #functions_plot(total_hist1,title1)
    functions_plot(total_hist2,title2)
    #return final_loss1,final_loss2
    return final_loss2
#main(2,3,1000,.01,4)
import gzip
import pickle
with gzip.open('data/mnist/mnist.pkl.gz', 'rb') as f:
    u = pickle._Unpickler( f )
    u.encoding = 'latin1'
    train, val, test = u.load()
train_x, train_y = train
x=np.array(train_x)
y=np.array(train_y)
