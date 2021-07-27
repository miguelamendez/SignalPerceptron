from signal_perceptron import *
from mlp import *
from train import *
from utils import *
from data_load import *


def main_pytorch(m, k, epochs, lr, sample=-1):
    """Main function for running PyTorch based experiments.

    Args:
        m ([type]): [description]
        k ([type]): [description]
        epochs ([type]): [description]
        lr ([type]): [description]
        sample (int, optional): [description]. Defaults to -1.

    Returns:
        [type]: [description]
    """
    # Dataset generation
    x, y = data_gen(m, k, sample)
    print(x,y)

    # Convert data to tensors
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    x_train=torch.tensor(x)
    x_train = x_train.type(torch.cuda.FloatTensor)
    y_train=torch.tensor(y)
    y_train = y_train.type(torch.cuda.FloatTensor)
    
    # Definition of models
    model1 = SP_r_pytorch(m,k).to(device)
    model2 = SP_r_appx_pytorch(m, k).to(device)
    model2 = MLP_pytorch(m**k, k).to(device)
    PATH1 = "data/models/init_dict_model_SP_r.pt"
    torch.save(model1.state_dict(), PATH1)
    PATH2 = "data/models/init_dict_model_MLP.pt"
    torch.save(model2.state_dict(), PATH2)
    
    # Calculate the number of parameters used by each model 
    model_parameters = filter(lambda p: p.requires_grad, model1.parameters())
    params1 = sum([np.prod(p.size()) for p in model_parameters])
    model_parameters = filter(lambda p: p.requires_grad, model2.parameters())
    params2 = sum([np.prod(p.size()) for p in model_parameters])
    print("SP_r number of params:", params1)
    print("MLP number of params:", params2)

    # Training loop
    total_hist1, final_loss1 = train_loop_pytorch(x_train, y_train,model1, PATH1, epochs=epochs, learning_rate=lr)
    total_hist2, final_loss2 = train_loop_pytorch(x_train, y_train,model2, PATH2, epochs=epochs, learning_rate=lr)
    
    # Loss plotting for trained networks.
    [print(i.item()) for i in final_loss1] # Raf: There was a bug in this line. Changed it to final_loss1.
    title1 = 'Training loss of ' + str(len(total_hist1)) + ' functions of the' + \
        str(m) + '-valued ' + str(k) + '-ary  function space with SP_r_pytorch'
    title2 = 'Training loss of ' + str(len(total_hist2)) + ' functions of the' + \
        str(m) + '-valued ' + str(k) + '-ary  function space with MLP_pytorch'
    functions_plot(total_hist1, title1)
    functions_plot(total_hist2, title2)
    
    return final_loss1, final_loss2


def main_numpy(m, k, epochs, lr, sample=-1):
    """Main function used to run numpy experiments.

    Args:
        m ([type]): [description]
        k ([type]): [description]
        epochs ([type]): [description]
        lr ([type]): [description]
        sample (int, optional): [description]. Defaults to -1.

    Returns:
        [type]: [description]
    """
    # Dataset generation
    x_train, y_train = data_gen(m, k, sample)
    print(x_train, y_train)

    # Definition of Models
    model1 = SP_numpy(m, k)
    model2 = SP_r_numpy(m, k)

    # Calculate number of parameters
    params1 = model1.count()
    params2 = model2.count()
    print("SP number of params:", params1)
    print("SP_r number of params:",  params2)
    
    # Training loop
    total_hist1, final_loss1 = train_loop_numpy(x_train, y_train, model1, epochs=epochs, learning_rate=lr)
    total_hist2, final_loss2 = train_loop_numpy(x_train, y_train, model2, epochs=epochs, learning_rate=lr)
    
    # Loss plotting for trained networks.
    [print(i.item()) for i in final_loss1]
    title1 = 'Training loss of ' + str(len(total_hist1)) + ' functions of the' + \
        str(m) + '-valued ' + str(k) + '-ary  function space with SP_numpy'
    title2 = 'Training loss of ' + str(len(total_hist2)) + ' functions of the' + \
        str(m) + '-valued ' + str(k) + '-ary  function space with SP_r_numpy'
    functions_plot(total_hist1, title1)
    functions_plot(total_hist2, title2)

    return final_loss1, final_loss2


# Run experiment number 1
main_numpy(2, 8, 200, .5, 4)
main_pytorch(2, 8, 200, .5, 4)
