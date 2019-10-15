"""
---------------------------------------------------------------------
-- Author: Jhosimar George Arias Figueroa
---------------------------------------------------------------------

Main file to execute the model with the MNIST dataset

"""

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from keras.datasets import mnist
from utils.partition import *
from model.GMVAE import *
import os
from scipy.io import loadmat

def del_all_flags(FLAGS):
    flags_dict = FLAGS._flags()
    keys_list = [keys for keys in flags_dict]
    for keys in keys_list:
        FLAGS.__delattr__(keys)
del_all_flags(tf.flags.FLAGS)

flags = tf.flags
PARAMETERS = flags.FLAGS

#########################################################
## Input Parameters
#########################################################

## Dataset
flags.DEFINE_string('dataset', 'mnist', 'Specify the desired dataset (mnist)')
flags.DEFINE_integer('seed', -1, 'Random Seed')

## GPU
flags.DEFINE_integer('gpu', 1, 'Using Cuda, 1 to enable')
flags.DEFINE_integer('gpuID', 0, 'Set GPU Id to use')

## Training
flags.DEFINE_integer('batch_size', 64, 'Batch size of training data')
flags.DEFINE_integer('num_epochs', 100, 'Number of epochs in training phase')
flags.DEFINE_float('learning_rate', 1e-3, 'Learning rate for training')
flags.DEFINE_float('decay_epoch', -1, 'Reduces the learning rate every decay_epoch')
flags.DEFINE_float('lr_decay', 0.5, 'Learning rate decay for training')

## Architecture
flags.DEFINE_integer('num_classes', 10, 'Number of clusters')
flags.DEFINE_integer('gaussian_size', 64, 'Size of the gaussian learnt by the network')

## Partition parameters
flags.DEFINE_float('train_proportion', 1.0, 'Proportion of examples to consider for training only  (0.0-1.0)')
flags.DEFINE_integer('batch_size_val', 200, 'Batch size of validation data')
flags.DEFINE_integer('batch_size_test', 200, 'Batch size of test data')

## Gumbel parameters
flags.DEFINE_float('temperature', 1.0, 'Initial temperature used in gumbel-softmax (recommended 0.5-1.0)')
flags.DEFINE_integer('decay_temperature', 1, 'Set 1 to decay gumbel temperature at every epoch')
flags.DEFINE_integer('hard_gumbel', 0, 'Hard version of gumbel-softmax')
flags.DEFINE_float('min_temperature', 0.5, 'Minimum temperature of gumbel-softmax after annealing' )
flags.DEFINE_float('decay_temp_rate', 0.00693, 'Temperature decay rate at every epoch')

## Loss function parameters
flags.DEFINE_string('loss_type', 'bce', 'Desired loss function to train (mse, bce)')
flags.DEFINE_float('w_gaussian', 1.0, 'Weight of Gaussian regularization')
flags.DEFINE_float('w_categorical', 1.0, 'Weight of Categorical regularization')
flags.DEFINE_float('w_reconstruction', 1.0, 'Weight of Reconstruction loss')

## Others
flags.DEFINE_integer('verbose', 0, "Print extra information at every epoch.")
flags.DEFINE_integer('random_search_it', 20, 'Iterations of random search')

if PARAMETERS.gpu == 1:
   os.environ["CUDA_VISIBLE_DEVICES"] = str(PARAMETERS.gpuID)

if PARAMETERS.seed < 0:
   np.random.seed(None)
else:
   np.random.seed(PARAMETERS.seed)


#########################################################
## Read Data
#########################################################
if PARAMETERS.dataset == "mnist":
   print("Loading mnist dataset...")
   # load mnist data
   (x_train, y_train), (x_test, y_test) = mnist.load_data()

   x_train = x_train / 255.0
   x_test = x_test / 255.0
else:
   raise "invalid dataset, valid datasets are: mnist"


## Set datatypes
x_train = x_train.astype(np.float32)
x_test = x_test.astype(np.float32)
y_train = y_train.astype(np.int64)
y_test = y_test.astype(np.int64)

## Flatten data
x_train = flatten_array(x_train)
x_test = flatten_array(x_test)

#########################################################
## Data Partition
#########################################################
test_data, test_labels = x_test, y_test

if PARAMETERS.train_proportion == 1.0:
   train_data, train_labels, val_data, val_labels = x_train, y_train, x_test, y_test   
else:
   train_data, train_labels, val_data, val_labels = partition_train_val(x_train, y_train, PARAMETERS.train_proportion, PARAMETERS.num_classes) 

if PARAMETERS.verbose == 1:
   print("Train size: %dx%d" % (train_data.shape[0], train_data.shape[1]))
   if PARAMETERS.train_proportion < 1.0:
      print("Validation size: %dx%d" % (val_data.shape[0], val_data.shape[1]))
   print("Test size: %dx%d" % (test_data.shape[0], test_data.shape[1]))

#########################################################
## Train and Test Model
#########################################################
tf.reset_default_graph()
if PARAMETERS.seed > -1:
   tf.set_random_seed(PARAMETERS.seed)

## Model Initialization
gmvae = GMVAE(PARAMETERS)

## Training Phase
history_loss = gmvae.train(train_data, train_labels, val_data, val_labels)

## Testing Phase
accuracy, nmi = gmvae.test(test_data, test_labels, PARAMETERS.batch_size_test)

print("Testing phase...")
print("Accuracy: %.5lf,  NMI: %.5lf" % (accuracy, nmi) )

