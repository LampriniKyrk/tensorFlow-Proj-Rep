import cPickle, sys
import numpy as np
import matplotlib.pyplot as plt

#------------UNPICKLE-------------#
f = open('microwave1_PAA_64_GADF 0.pkl', 'rb')
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

train_X = np.reshape(temp_X, (7200, 7200, 1))
train_Y = np.reshape(temp_Y,(7200))

print train_Y.shape
print train_Y

#try conv2d 