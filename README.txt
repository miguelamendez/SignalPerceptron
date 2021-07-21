This Repo contains the implementation of the algorithms defined in the Signal Perceptron Paper.
To replicate the experiments in the paper run the experiments.py

All the functions are documented inside the .py files here we only explain in general what each file do. For more information check the documentation inside the files.
 
The following  files are used by the experiments.py:
train.py : Train file contains the training loops for the gradient descent algorithm using numpy and pytorch libraries.

data_load.py : This functions generate/load the datasets used by the training of the Signal Perceptron and MLP. The datasets are build by generating the whole domain space using the function data_gen().

signal_perceptron.py :This library contains the functions used by the SignalPerceptron and its different implementations using the numpy and pytorch libraries.

mlp.py : This library contains the functions used by the MultilayerPerceptron of 1 hidden layer as well as the implementation of the MLP using the pytorch library.

utils.py :

The  files mentioned avobe and the following files are used in the supplementary material:
train_linear.py: This function obtains the parameters using the system of linear equations algorithm rather than gradient descent.

Other files for future work (not part of the paper):
mnist.py : File used for training a Neural Network of Signal Perceptrons in order to learn the mnist dataset.[Currently not Implemented]

debbug.py: File used for testing some functions.

