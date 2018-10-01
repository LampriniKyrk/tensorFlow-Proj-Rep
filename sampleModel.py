import cPickle, sys
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers.convolutional import Convolution2D
from keras.layers.core import Flatten, Dense, Activation
from keras.utils import np_utils


#------------UNPICKLE-------------#
f = open('fridge/fridge1_PAA_64_GADF 0.pkl', 'rb')
pickledFile = cPickle.load(f)
#pickledFile = f.readlines()
print pickledFile

traintuple = pickledFile[0] #eikona
testtuple = pickledFile[1] #labels eikonas

temp_X = traintuple[0]
temp_Y = traintuple[1]

print temp_X.shape , temp_Y.shape
#print temp_X

#test_X = np.asarray(testtuple[0])
#test_Y = np.asarray(testtuple[1])



#----------CONVOLUTIONAL NN--------------#
#MUST CHANGE THE INPUT SHAPE TO (HEIGHT, WIDTH, 3) ACCORDING TO DOCUMENTATION
#that should be a tensor

#use conv2d
n_images = 1
train_X = np.reshape(temp_X, (n_images, 64, 64, 1))
#train_Y = np.reshape(temp_Y,(7200))

print train_X.shape
#print train_Y

#try conv2d
#Create model
model = Sequential()
model.add(Convolution2D(32, kernel_size=(3, 3), activation='relu', input_shape=(64,64,1)))
model.add(Flatten())
model.add(Dense(64))
model.add(Activation('softmax'))

model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

model.fit(train_X,temp_Y, batch_size= 1)




