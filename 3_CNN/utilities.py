# IMPORTS
import tensorflow as tf
import tf_keras as keras

from tf_keras.models import Sequential, Model
from tf_keras.layers import Input, Dense, BatchNormalization, Dropout,
Conv2D, MaxPooling2D, Flatten
from tf_keras.optimizers import SGD, Adam

# Set seed from random number generator, for better comparisons
import numpy as np
from numpy.random import seed
seed(123)

import matplotlib.pyplot as plt

# define function that builds a CNN model
def build_CNN(input_shape, loss,
              n_conv_layers:int=2,
              n_filters:int=16,
              n_dense_layers:int=0,
              n_nodes:int=50,
              use_dropout:bool=False,
              learning_rate:float=0.01,
              act_fun='relu',
              optimizer:str='adam',
              print_summary:bool=False):
    """
    Builds a Convolutional Neural Network (CNN) model based on the
provided parameters.

    Parameters:
    input_shape (tuple): Shape of the input data (excluding batch size).
    loss (tf_keras.losses): Loss function to use in the model.
    n_conv_layers (int, optional): Number of convolutional layers in
the model. Default is 2.
    n_filters (int, optional): Number of filters in each convolutional
layer. Default is 16.
    n_dense_layers (int, optional): Number of dense layers in the
model. Default is 0.
    n_nodes (int, optional): Number of nodes in each dense layer. Default is 50.
    use_dropout (bool, optional): Whether to use Dropout after each
layer. Default is False.
    learning_rate (float, optional): Learning rate for the optimizer.
Default is 0.01.
    act_fun (str, optional): Activation function to use in each layer.
Default is 'relu'.
    optimizer (str, optional): Optimizer to use in the model. Default is 'adam'.
    print_summary (bool, optional): Whether to print a summary of the
model. Default is False.

    Returns:
    model (Sequential): Compiled Keras Sequential model.
    """

    # Setup optimizer, depending on input parameter string
    if optimizer == 'sgd':
        optimizer = SGD(learning_rate=learning_rate)
    elif optimizer == 'adam':
        optimizer = Adam(learning_rate=learning_rate)
    else:
        raise ValueError("Optimizer not recognized. Use 'sgd' or 'adam'.")

    # Setup a sequential model
    model = Sequential()
     
    # Add convolutional layers
    for i in range(n_conv_layers):
        if i == 0:
            # First layer: specify input_shape
            model.add(Conv2D(filters=n_filters * (2**i),
kernel_size=(3, 3), padding='same', activation=act_fun,
input_shape=input_shape))
        else:
            # Subsequent layers: do not specify input_shape
            model.add(Conv2D(filters=n_filters * (2**i),
kernel_size=(3, 3), padding='same', activation=act_fun))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(2, 2)))
        if use_dropout:
            model.add(Dropout(0.25))

    # Flatten the output of the convolutional layers
    model.add(Flatten())

    # Add dense layers
    for i in range(n_dense_layers):
        model.add(Dense(n_nodes, activation=act_fun))
        model.add(BatchNormalization())
        if use_dropout:
            model.add(Dropout(0.5))

    # Add output layer
    model.add(Dense(10, activation='softmax'))  # 10 classes for CIFAR10

    # Compile the model
    model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])

    # Print model summary if requested
    if print_summary:
        model.summary()

    return model


# =======================================
# PLOTTING FUNCTIONS
# =======================================

# TRAINING CURVES PLOT FUNCTION
def plot_results(history):
    """
    Plots the training and validation loss and accuracy from a Keras history object.
    Parameters:
    history (keras.callbacks.History): A History object returned by the fit method of a Keras model. 
                                       It contains the training and validation loss and accuracy for each epoch.
    Returns:
    None
    """
    
    val_loss = history.history['val_loss']
    acc = history.history['accuracy']
    loss = history.history['loss']
    val_acc = history.history['val_accuracy']
    
    plt.figure(figsize=(10,4))
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.plot(loss)
    plt.plot(val_loss)
    plt.legend(['Training','Validation'])

    plt.figure(figsize=(10,4))
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.plot(acc)
    plt.plot(val_acc)
    plt.legend(['Training','Validation'])

    plt.show()


# =======================================
# AUGMENTATIONS FUNCTIONS
# =======================================

# ROTATE IMAGES BY () DEGREES
def myrotate(images):

    images_rot = np.rot90(images, axes=(1,2))
    
    return images_rot