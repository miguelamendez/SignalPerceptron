"""
    Description: This library contains all the class model for building ml models.
    Summary: Machione learning modules are of two type  learnable or not learnable.
    If not learnable they can only be used for inference , if learnable the follow a learning function that it can be called.
    The inference function model (aka forward)computes the value of the current architecture)
    The learning function 
    Please refer to each of the functions/classes for a full description of what they do.

    Functions ([first,second] "first bool" notes the implemented functions "second bool" notes the implemented documentation):
        get_obj_fn:[True,False] Returns the objective function using the dictionary of the available obj_fn
        get_optim:[True,False] Returns the optimizer function using the dictionary of the available optimizers
        get_train_fn:[True,False] Returns the train function using the dictionary of the available train functions
        get_specs:[True,False] Returns current information of the structure of the model
        build_model:[True,False] Returns a model build with the specifications given by the model dictionary


    Classes ([first,second] "first bool" notes the implemented classes "second bool" notes the implemented documentation):
        name:[True,False] summary
        name:[True,False] summary
        name:[True,False] summary
        name:[True,False] summary
        """
#Internal libraries
import os
import sys
full_path = os.path.dirname(os.path.realpath(__file__))
print("build.py:",full_path)
sys.path.append(os.path.dirname(os.path.dirname(full_path)))# one directory above

#from ml.models.config import data as models_dict #File that contains information of developed models
#from ml.archs.build import arch as arch_build #File for creating/building an architecture (only use here build)

#Libraries for learnable models 
#from ml.functional.optimizers import *
#from ml.functional.obj_fn import *
#from ml.functional.train_fn import* 
from ml.archs.build import *
#from dir.to.file import * #files for imports
#from dir.to.file import *

import torch
import numpy as np
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
#External libraries

class Model():
    #Built-in methods constructor, destructor and call methods
    def __init__(self,model_id,arch,value_size,value_ids=None,learn_mode=True,batch_size=4,seq_size=1,obj_fn=None,optim_fn=None,path=None):
        """The model class is the main function for ML functions. The attributes of such function can be devided in to two main categories: Inference and Learning
        Inference : self.arch
        Learning : self.arch , self.learn_fn
        Args:
            model_id([Str]):
            arch([Arch object]):
            value_size([Int]):
            mem_size([Int]): Is the seq total lenght ,
            value_ids([List[Str]]):
            learn_mode([Bool]):
            obj_fn=None:
            optim_fn=None:
            path=None:
        """
        #Essential structures#############################
        self.id=id
        self.arch=arch
        self.device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if isinstance(self.arch,nn.Module):
            self.arch.to(self.device)
        else:
            print("Warning: Arch is not a pytorch or tensorflow module")
        if path is not None:
            self.path=path+self.id+".pt"
        #Flags############################################
        self.start=True
        #For learning#####################################
        self.pred_memory=MemoryBuffer(value_size=[batch_size,],value_ids=None,mem_size=32)
        self.loss_memory=MemoryBuffer(value_size=[batch_size,],value_ids=None,mem_size=32)
        self.obj_fn=obj_fn
        self.optim_fn=optim_fn
        self.learn_fn= learn_fn
        self.learn_mode=self.check_learn_mode(learn_mode)
        if learn_mode and not self.learn_mode:
            print("Warning: Model expected to be learnable but doesnt have minimum requirements")
        if self.learn_fn is None and self.learn_mode:
            print("Warining: Missing learning function for local learning")
        #Data Storing and Processing#######################
        #We have a memory buffer to store the last n predictions
        if value_ids is not None:
            self.memory=MemoryBuffer(value_size,value_ids,mem_size)
        elif value_size>1: 
            print("Warning: memory is storing values as a list instead of dictionary may cause problems")

    def check_learn_mode(self, learn_flag):
        if learn_flag:
            #Check if structure contains learnable parameters:
            if isinstance(self.arch,nn.Module):
                params = filter(lambda p: p.requires_grad, self.arch.parameters())
                total_params = sum([np.prod(p.size()) for p in params])
                if total_params==0:
                    return False
                else:
                    if self.obj_fn is not None and self.optim_fn is not None:
                        return True
                    else:
                        return False
            else:
                return False
        else:
            return False

    def __call__(self,x):
        
        if self.learn_mode:
            if self.learn_fn is not None:
                if self.data_mngr.learn():
                    x,y=self.data_mngr()
                    learn_data_dict=self.learn_fn(x,y,self.obj_fn)
                    self.data_mngr.process(learn_data_dict)
        #Compute the forward value after learning
        return self.forward(x)

    def forward(self,x,params=None):
        if self.start:
            print("Start parameters:",params)
            self.start=False
        self.values = self.arch(x.to(self.device))
        if self.values_ids is not None:
            return dict(zip(self.values_ids,self.values))
        else:
            return self.values
    
    def loss(self,y,y_pred):
        return self.obj_fn(last_pred,y_pred)

    def learn(self,X,y):
        loss=process(self.loss_memory)
        optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

#Debug Information for Model's Architecture
    def get_specs(self):
        """[Funciton that prints the ammount of learnable parameters in the arch and the device used]
        Args:
            arg1 ([type]): [description] . Defaults to"""
        info={}
        info["model_id"]=self.id
        info["device"]=self.device
        if isinstance(self.arch,nn.Module):
            total_params = sum(p.numel() for p in self.arch.parameters())
            params = filter(lambda p: p.requires_grad, self.arch.parameters())
            l_params = sum([np.prod(p.size()) for p in params])
            info["learn_params"]=l_params
            info["total_params"]=total_params
        else:
            info["learn_params"]=None
            info["total_params"]=None
        return info

    def reset(self):
        self.start=True
        self.data_mngr.reset()
#Save and Loading arch models.
    def save_checkpoint(self):
        """[Functions that stores the learned model]
        Args:
            arg1 ([type]): [description] . Defaults to"""

        torch.save(self.arch.state_dict(),self.path)

    def load_checkpoint(self):
        """[summary]
        Args:
            arg1 ([type]): [description] . Defaults to
        """
        self.arch.load_state_dict(torch.load(self.checkpoint_file))

#Example
#from ml.archs.config import data as arch_dict
#arch1=arch(arch_dict["rl"],"simple_ac",{"actor":[4,3],"critic":[4,1],"enc":[8,4]})
#model=Model("ac",arch1,learn_mode=True)
#model1=Model("ac_with_vals",arch1,values_ids=["sample","sample_grad","vals"],learn_mode=True)
#print(model.get_specs())
#x=torch.randn(2,8)
#print(x,"\n",model(x),"\n",model1(x))

class Modelv2():
    #Built-in methods constructor, destructor and call methods
    def __init__(self,model_id,archs,seq_size=1,obj_fn=None,optim_fn=None,path=None,learn_mode=True):
        """The model class is the main function for ML functions. The attributes of such function can be devided in to two main categories: Inference and Learning
        Inference : self.arch
        Learning : self.arch , self.learn_fn
        Args:
            model_id([Str]):
            archs([list of Arch object]):
            seq_size([Int]):Defines the size of a sequence when processing information used by loss_seq and pred_seq variables.
            learn_mode([Bool])=TRUE:Enables learning
            obj_fn=None:the loss functions used by the model
            optim_fn=None:The optimizer for the model parameters
            path=None:path for saving loading the model arch parameters should be under model_id/version_id.ptc/
        """
        #Essential structures#############################
        self.id=model_id
        self.archs=archs
        self.device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.seq_len=seq_size
        self.pred=[] #Stores last prediction tensor contains gradient
        self.pred_seq=[]#Stores last self.seq_len detached predictions
        for arch in self.archs:
            if isinstance(arch,nn.Module):
                arch.to(self.device)
            else:
                print("Warning: Arch is not a pytorch or tensorflow module")
        if path is not None:
            self.path=path+self.id+"/v1.pt"
        #For learning#########################################################
        self.loss_seq=[]
        self.obj_fn=obj_fn
        self.optim_fn=optim_fn

    def __call__(self,x,y=None):
        return self.forward(x,y)

    def forward(self,x,y):
        if y is not None:
        #First calculate loss of last prediction
            loss=self.obj_fn(self.input_seq[:-1],data_y)
            #Try to learn:
            if self.can_learn:
                self.learn(loss)
            loss=loss.detach()
        else:
            loss=None
        #Next prediction assuming only one arch needs to be implemented for multiple
        self.pred=self.archs[0](x)
        #Add prediction to the pred buffer
        self.pred_seq.append(self.pred.detach())
        pred=self.pred.detach()
        return pred,loss
    
    def learn(self,loss):
            self.loss_seq.append(loss)
            #The model learns only for completed sequences
            if len(model_loss_seq)==self.seq_len:
                #We average all losses
                loss=pad_sequence(model_loss_seq).avg()
                #Backprop##########################
                self.optimizer.zero_grad() 
                loss.backward() 
                self.optimizer.step()
                ##################################
                #Reset model_loss_seq
                self.loss_seq=[]
#Debug Information for Model's Architecture
    def get_specs(self):
        """[Funciton that prints the ammount of learnable parameters in the arch and the device used]
        Args:
            arg1 ([type]): [description] . Defaults to"""
        info={}
        info["model_id"]=self.id
        info["device"]=self.device
        info["learn_params"]=0
        info["total_params"]=0
        for arch in self.archs:
            if isinstance(arch,nn.Module):
                total_params = sum(p.numel() for p in arch.parameters())
                params = filter(lambda p: p.requires_grad, arch.parameters())
                l_params = sum([np.prod(p.size()) for p in params])
                info["learn_params"]+=l_params
                info["total_params"]+=total_params
        return info

    def reset(self,x):
        pred= self.archs[0](x)
        self.pred_seq.append(pred)
        return pred.detach()
        
#Save and Loading arch models.
    def save_checkpoint(self):
        """[Functions that stores the learned model]
        Args:
            arg1 ([type]): [description] . Defaults to"""

        torch.save(self.arch.state_dict(),self.path)

    def load_checkpoint(self):
        """[summary]
        Args:
            arg1 ([type]): [description] . Defaults to
        """
        self.arch.load_state_dict(torch.load(self.checkpoint_file))

#Example
import ml.archs.sp.baselines as sp
fsp_arch=Arch("FSP",sp.FSP_pytorch((16,4),256))
fsp_model=Modelv2(fsp_arch.id,[fsp_arch],learn_mode=True)
print(fsp_model.get_specs())
x=torch.randn(2,16)
pred=fsp_model.reset(x)
print(x)
print(pred)
print(fsp_model.pred)
x1=torch.rand(2,16)
pred,loss=fsp_model(x1)
print(x1)
print(pred)
print(fsp_model.pred)

#Start of the functions used to build models###########################################################


def model(model_type,model_id,inout=None,values_ids=None,learn_mode=True,path=None):
    models=models_dict[model_type]
    #Getting model data
    model_data=models[model_id]
    #print("Model data:",model_data)
    #Getting arch information :
    arch_data  = model_data["arch"]
    print("Arch data:",arch_data)
    #Building arch
    if inout is not None:
        arch=arch_build(arch_data["type"],arch_data["id"],inout)
    else:
        arch=arch_build(arch_data["type"],arch_data["id"],arch_data["inout"])
    if learn_mode:
        obj_fn_params=model_data["obj_fn"]["params"]
        obj_fn=model_data["obj_fn"]["id"]()
        optim_params=model_data["optim"]["params"]
        optimizer=model_data["optim"]["id"](arch.parameters(),lr=optim_params["lr"])
        train_params=model_data["train_fn"]["params"]
        train_fn=model_data["train_fn"]["id"]
        return Model(model_id,arch,obj_fn=obj_fn,optim_fn=optimizer,path=None,learn_fn=train_fn)
    else:
        return Model(model_id,arch,values_ids,learn_mode=learn_mode)

#Example
#new_model=model("agent_nodes","simple_ac",learn_mode=True)
#X , y =torch.randn(4,128),torch.randn(4,1)
#print("Specs:",new_model.get_specs())
#new_model.update(X,y)
