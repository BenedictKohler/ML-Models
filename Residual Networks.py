# Implementation of a Residual Network

import numpy as np
from tensorflow.keras import layers
from tensorflow.keras.layers import Input, Add, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, AveragePooling2D, MaxPooling2D, GlobalMaxPooling2D
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.initializers import glorot_uniform

def identity_block(X, f, filters, stage, block) :
    """
    X: input tensor of shape (batch_size, prev_height, prev_width, prev_conv_depth)
    f: integer, specifying the shape of the middle CONV's window for the main path
    filters: list of integers, defining the number of filters in the CONV layers of the main path
    stage: integer used to name the layers depending on their position in the network
    block: string/character used to name the layers depending on their position in the network
    """
    
    # defining name basis
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    # Retrieve filters
    F1, F2, F3 = filters

    # Save the input value which gets used after skipping some layers.
    # In residual networks we skip over some layers in order to reduce the chance of exploading and
    # vanishing gradients which occur from backpropagating in deep neural networks. This is because
    # when backpropagating we continuously multiply due to chain rule which drives values either 
    # extremely large or small
    X_shortcut = X
    
    # First component of main path
    X = Conv2D(filters=F1, kernel_size=(1, 1), strides=(1, 1), padding='valid', name=conv_name_base + '2a', kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name=bn_name_base + '2a')(X)
    X = Activation('relu')(X)
    
    # Second component of main path
    X = Conv2D(filters=F2, kernel_size=(f, f), strides=(1, 1), padding='same', name=conv_name_base + '2b', kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name=bn_name_base + '2b')(X)
    X = Activation('relu')(X)
    
    # Third component of main path
    X = Conv2D(filters=F3, kernel_size=(1, 1), strides=(1, 1), padding='valid', name=conv_name_base + '2c', kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name=bn_name_base + '2c')(X)
    
    # Merge the main path with the alternative path
    X = Add()([X, X_shortcut])
    X = Activation('relu')(X)
    
    return X


def convolutional_block(X, f, filters, stage, block, stride=2) :
    
    # defining name basis
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'
    
    # Retrieve Filters
    F1, F2, F3 = filters

    # Save the input value
    X_shortcut = X
    
    # First component of main path 
    X = Conv2D(filters=F1, kernel_size=(1, 1), strides=(stride, stride), padding='valid', name=conv_name_base + '2a', kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name=bn_name_base + '2a')(X)
    X = Activation('relu')(X)
    
    # Second component of main path
    X = Conv2D(filters=F2, kernel_size=(f, f), strides=(1, 1), padding='same', name=conv_name_base + '2b', kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name=bn_name_base + '2b')(X)
    X = Activation('relu')(X)
    
    # Third component of main path 
    X = Conv2D(filters=F3, kernel_size=(1, 1), strides=(1, 1), padding='valid', name=conv_name_base + '2c', kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name=bn_name_base + '2c')(X)
    
    # Shortcut Path. Skipping over layers to reduce chance of vanishing/exploding gradients
    X_shortcut = Conv2D(filters=F3, kernel_size=(1, 1), strides=(stride, stride), padding='valid', name=conv_name_base + '1', kernel_initializer=glorot_uniform(seed=0))(X_shortcut)
    X_shortcut = BatchNormalization(axis=3, name=bn_name_base + '1')(X_shortcut)
    
    # Final step: Add shortcut value back to main path
    X = Add()([X, X_shortcut])
    X = Activation('relu')(X)
    
    return X

def ResNet50Layers(input_shape=(64, 64, 3), classes=6) : # An example of classification with 6 possible outputs
    
    X_input = Input(input_shape)
    
    X = ZeroPadding2D((3, 3))(X_input)
    
    # Stage 1
    X = Conv2D(64, (7, 7), strides=(2, 2), name='conv1', kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name='bn_conv1')(X)
    X = Activation('relu')(X)
    X = MaxPooling2D((3, 3), strides=(2, 2))(X)

    # Stage 2
    X = convolutional_block(X, f=3, filters=[64, 64, 256], stage=2, block='a', stride=1)
    X = identity_block(X, 3, [64, 64, 256], stage=2, block='b')
    X = identity_block(X, 3, [64, 64, 256], stage=2, block='c')
    
    # Stage 3
    X = convolutional_block(X, f=3, filters=[128, 128, 512], stage=3, block='a', stride=2)
    X = identity_block(X, 3, [128, 128, 512], stage=3, block='b')
    X = identity_block(X, 3, [128, 128, 512], stage=3, block='c')
    X = identity_block(X, 3, [128, 128, 512], stage=3, block='d')
    
    # Stage 4
    X = convolutional_block(X, f=3, filters=[256, 256, 1024], stage=4, block='a', stride=2)
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='b')
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='c')
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='d')
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='e')
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='f')
    
    # Stage 5
    X = X = convolutional_block(X, f=3, filters=[512, 512, 2048], stage=5, block='a', stride=2)
    X = identity_block(X, 3, [512, 512, 2048], stage=5, block='b')
    X = identity_block(X, 3, [512, 512, 2048], stage=5, block='c')
    
    # AVGPOOL
    X = AveragePooling2D(pool_size=(2, 2), padding='same')(X)
    
    # output layer
    X = Flatten()(X)
    X = Dense(classes, activation='softmax', name='fc' + str(classes), kernel_initializer=glorot_uniform(seed=0))(X)
    
    # Create the model
    model = Model(inputs=X_input, outputs=X, name='ResNet50')
    
    return model


# Build models graph
model = ResNet50Layers(input_shape=(64, 64, 3), classes=6)

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Code below shows an implementation of the adam optimizer used above
# Adam optimization algorithm is used to speed up learning by taking into account the direction that the weights have
# have previously been moving in (momentum) as well as the number of iterations. Resulting in convergence to a more optimal 
# solution faster than regular gradient descent.  
def adam_initialization(parameters) : # parameters contains weights "W.." and biases "b.."
    num_layers = len(parameters) // 2 # We divide by two as parameters separates bias from other weights
    exp_weighted_avg = {}
    exp_weighted_avg_squared = {}
    
    for l in range(num_layers) :
        exp_weighted_avg["dW" + str(l)] = np.zeros(parameters["W" + str(l+1)].shape)
        exp_weighted_avg["db" + str(l)] = np.zeros(parameters["b" + str(l+1)].shape)
        exp_weighted_avg_squared["dW" + str(l)] = np.zeros(parameters["W" + str(l+1)].shape)
        exp_weighted_avg_squared["db" + str(l)] = np.zeros(parameters["b" + str(l+1)].shape)
    
    return exp_weighted_avg, exp_weighted_avg_squared

def adam_parameter_update(parameters, gradients, exp_weighted_avg, exp_weighted_avg_squared,
                          t, beta1, beta2, epsilon=1e-8, learning_rate=0.01) :
    
    """ 
    parameters: Dictionary containing weights and biases
    gradients: Dictionary of gradients/rates of changes for each parameter
    exp_weighted_avg: Exp Moving avg of gradient
    exp_weighted_avg_squared: Exp Moving avg of squared gradient
    t: number of timesteps
    learning_rate: hyperparamter that decides how much we can change values in direction of gradient
    beta1: Exponential decay hyperparameter for exp_weighted_avg
    beta2: Exponential decay hyperparameter for exp_weighted_avg_squared
    epsilon: prevents division by 0 during updates
    """
    
    num_layers = len(parameters) // 2        # number of layers in the neural networks
    v_corrected = {}                         # Initializing first moment estimate, python dictionary
    s_corrected = {}                         # Initializing second moment estimate, python dictionary
    
    for l in range(num_layers) :
        exp_weighted_avg["dW" + str(l+1)] = beta1 * exp_weighted_avg["dW" + str(l+1)] + (1 - beta1) * gradients["dW" + str(l+1)]  
        exp_weighted_avg["db" + str(l+1)] = beta1 * exp_weighted_avg["db" + str(l+1)] + (1 - beta1) * gradients["db" + str(l+1)]
        
        v_corrected["dW" + str(l+1)] = exp_weighted_avg["dW" + str(l+1)] / (1 - beta1**t)
        v_corrected["db" + str(l+1)] = exp_weighted_avg["db" + str(l+1)] / (1 - beta1**t)
        
        exp_weighted_avg_squared["dW" + str(l+1)] = beta2 * exp_weighted_avg_squared["dW" + str(l+1)] + (1 - beta2) * (gradients["dW" + str(l+1)] ** 2)
        exp_weighted_avg_squared["db" + str(l+1)] = beta2 * exp_weighted_avg_squared["db" + str(l+1)] + (1 - beta2) * (gradients["db" + str(l+1)] ** 2)

        s_corrected["dW" + str(l+1)] = exp_weighted_avg_squared["dW" + str(l+1)] / (1 - beta2 ** t)
        s_corrected["db" + str(l+1)] = exp_weighted_avg_squared["db" + str(l+1)] / (1 - beta2 ** t)
        
        parameters["W" + str(l+1)] = parameters["W" + str(l+1)] - learning_rate * v_corrected["dW" + str(l+1)] / np.sqrt(s_corrected["dW" + str(l+1)] + epsilon)
        parameters["b" + str(l+1)] = parameters["b" + str(l+1)] - learning_rate * v_corrected["db" + str(l+1)] / np.sqrt(s_corrected["db" + str(l+1)] + epsilon)
    
    return parameters, exp_weighted_avg, exp_weighted_avg_squared