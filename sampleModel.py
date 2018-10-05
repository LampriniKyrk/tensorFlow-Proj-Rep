import cPickle, sys
import numpy as np
import matplotlib.pyplot as plt
from keras.layers import MaxPooling2D, Conv2D, Dropout
from keras.models import Sequential
from keras.layers.convolutional import Convolution2D
from keras.layers.core import Flatten, Dense, Activation
from keras.utils import np_utils
from sklearn import preprocessing

#put all images to one big array- dynamic list? - read in batches?
n_images = 13726
trainSize = 10000
filename = 'fridge/fridge1_PAA_64_GADF'
imageList = []
labelList = []

for image in range(0,n_images):
    #------------UNPICKLE-------------#
    f = open(filename+' '+str(image)+'.pkl', 'rb')
    pickledFile = cPickle.load(f)

    traintuple = pickledFile[0] #train eikonas
    testtuple = pickledFile[1] #test eikonas - empty

    # print pickledFile

    temp_X = traintuple[0]
    temp_Y = traintuple[1]

    #print temp_X.shape

    imageList.append(temp_X)#.flatten())
    labelList.append(np.average(temp_Y))#(np.reshape(temp_Y,temp_Y.size))
    #print labelList

#prepare data for conv input

#imageList = preprocessing.minmax_scale(imageList)


#----------CONVOLUTIONAL NN--------------#

#use conv2d
train_X = np.reshape(imageList[:trainSize], (trainSize, 64, 64, 1))
test_X = np.reshape(imageList[trainSize:], (n_images-trainSize, 64, 64, 1))
train_Y = np.asarray(labelList[:trainSize])
test_Y = np.asarray(labelList[trainSize:])
test_Y = np.reshape(test_Y,(n_images-trainSize, 1))
#train_Y = np.reshape(train_Y, (1, train_Y.size))

print train_Y
print train_X.shape

#try conv2d
#Create model
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), input_shape=(64, 64, 1)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))


model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation('linear'))

model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

model.fit(train_X, train_Y, batch_size = 10, epochs= 1)

pred = model.predict(test_X,1)

f = file('results.txt','wb')
for i,j in zip(pred, test_Y):
    f.write(str(i)+' - '+str(j)+'\n')
f.close()

#print pred, test_Y




