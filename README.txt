This Repo contains the implementation of the algorithms defined in the Signal Perceptron Paper.

Pre-Requirements:
You need to install the following libraries in order to run the code:
numpy
pytorch
torchvision
matplotlib

Running Procedure:
To replicate the experiments in the paper run exp1.py , exp2.py with the commands:
##############################################################################################
For the experiments of learning boolean function spaces run:
python3 exp1.py 5

Warning:This is gona be run a total of five times just like in the paper so it my take a while to finish (around 2 hours to generate all data and graphs)
If you just want to run all the experiments for one time please write the command:

python exp1.py 1
Warning: This will take  around 20 min to generate all data and graphs

The results of each run are stored in the folder data/experiments/exp1/data

################################################################################################
For the experiments of MNIST datasets run:
python3 exp2.py 5

Warning:This is gona be run a total of five times just like in the paper so it my take a while to finish (around 30 min to train all models)
If you just want to train all models for one time please write the command:
python exp2.py 1

The results of each run are stored in the folder data/experiments/exp2/ 
You can find the learned models under the folder data/models/
Warning: The models stored in this folder are the ones trained in the last run.
##################################################################################################
For the experiments results provided in the Appendix A and Appendix B:
python exp1_3.py n k r p
n: is the name of the mechanism to run which can be:
 sp for signalperceptron numpy version
 rsp_np for realsignalperceptron numpy version
 rsp for realsignalperceptron pytorch version
 fsp for furiersignal perceptron 
 mlp for the single layer MLP
 gn for generalized neuron
k: defines the arity of the boolean space to be learned.
r: defines the number of runs that the experiment will be conducted we allowed only up to 5 times as in the article.
p: is if you want the plots of the graphs of the loss functions, if you want them write yes , other whise no.
Example.
For conducting 4 times the experiments of learn all ternary boolean functions and using the FurierSgnalPerceptron (with plots):
python exp1_3.py fsp 3 4 yes

The results of each run are stored in the folder data/experiments/exp1_3/ 
You can find the learned models under the folder data/models/
Warning: The models stored in this folder are the ones trained in the last run.
##################################################################################################
All the functions are documented inside the .py files here we only explain in general what each file do. For more information check the documentation inside the files.
 
The following  files are used by the experiments.py:
train.py : Train file contains the training loops for the gradient descent algorithm using numpy and pytorch libraries for the different datasets used in the experiments.

data_load.py : This functions generate/load the datasets used by the training of the Signal Perceptron and MLP. The datasets are build by generating the whole domain space using the function data_gen().

signal_perceptron.py :This library contains the functions used by the SignalPerceptron and its different implementations using the numpy and pytorch libraries.

sp_paper_models.py : This library contains the functions used by the MultilayerPerceptron of 1 hidden layer as well as the implementation of the MLP and FSP for learning mnist datasets using the pytorch library.

utils.py : Ploting functions.
exp1.py:File used for runing all experiments part 1 from the paper (binary boolean functions)
exp2.py:File used for runing all experiments part 2 from the paper (MNIST datasets)

The  files mentioned avobe and the following files are used in the supplementary material:
exp3.py:File used for learning the parameters using the system of linear equations for general $k$-ary function spaces.
 This function obtains the parameters using the system of linear equations algorithm rather than gradient descent.
##################################################################################################
Other files for future work (not part of the paper):
exp1_beta.py :File used for training all variations of SP and MLP to learn a functional space.[Deprecated check exp1] 
exp2_beta.py :File used for training a Fourier Signal Perceptron and MLP to learn the mnist dataset.[Deprecated check exp2]
