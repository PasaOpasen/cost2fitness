# -*- coding: utf-8 -*-
"""
Created on Wed Jan  6 15:31:49 2021

@author: qtckp
"""

import sys
sys.path.append('..')

import numpy as np 

from cost2fitness import arr_to_weigths, Bias, MatrixDot, NNStep, Pl, Softmax, Relu, Tanh


from keras.layers import Dense
from keras import Sequential


input_size = 30


pipe = Pl([
            NNStep(from_size = input_size, to_size = 10),
            Relu(),
            NNStep(from_size = 10, to_size = 10),
            Relu(),
            NNStep(from_size = 10, to_size = 100),
            Tanh(),
            NNStep(from_size = 100, to_size = 1000),
            Relu(),
            NNStep(from_size = 1000, to_size = 10),
            Tanh(),
            NNStep(from_size = 10, to_size = 10),
            Relu(),
            NNStep(from_size = 10, to_size = 10),
            Tanh(),
            NNStep(from_size = 10, to_size = 10),
            Relu(),
            NNStep(from_size = 10, to_size = 10),
            Tanh(),
            NNStep(from_size = 10, to_size = 10),
            Relu(),
            NNStep(from_size = 10, to_size = 10),
            #Tanh(),
            #NNStep(from_size = 10, to_size = 6),
            Softmax()
        ])

# that pipe is equal to this Keras model
keras_model = Sequential()
keras_model.add(Dense(10, input_shape = (input_size,), activation = 'relu'))
keras_model.add(Dense(10, activation = 'relu'))
keras_model.add(Dense(100, activation = 'tanh'))
keras_model.add(Dense(1000, activation = 'relu'))
keras_model.add(Dense(10, activation = 'tanh'))
keras_model.add(Dense(10, activation = 'relu'))
keras_model.add(Dense(10, activation = 'tanh'))
keras_model.add(Dense(10, activation = 'relu'))
keras_model.add(Dense(10, activation = 'tanh'))
keras_model.add(Dense(10, activation = 'relu'))
#keras_model.add(Dense(10, activation = 'tanh'))
keras_model.add(Dense(6, activation = 'softmax'))

keras_model.summary()


shapes = pipe.get_shapes() # get array of shapes for pipe model
shapes = [arr.shape for arr in keras_model.get_weights()] # equal way for keras model

# count total weights
total_weights = pipe.total_weights()

# just random weights for test
random_weights = np.random.random(total_weights)

# convert random_weights (1D-vector) to list of arrays with needed shapes  
list_of_weights = arr_to_weigths(random_weights, shapes)

# set weights
pipe.set_weights(list_of_weights)
keras_model.set_weights(list_of_weights)


#
# Running
#

# just random input
arr = np.random.random(input_size)


%timeit pipe.transform(arr)
# arr2 = arr[np.newaxis, :]  no influence for speed
%timeit keras_model.predict(arr[np.newaxis, :])



