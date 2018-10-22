import cPickle, sys
import numpy as np
from keras.preprocessing import image
from keras import utils, optimizers, activations
import matplotlib.pyplot as plt
from keras.layers import MaxPooling2D, Conv2D, Dropout
from keras.models import Sequential
from keras.layers.convolutional import Convolution2D
from keras.layers.core import Flatten, Dense, Activation
from sklearn import preprocessing

#NOTES
#32 pix images for 4 months data are 54900
#32 pix images for 8 months data are 100000+
#62 pix images for 2 months data are 13700

def parseImgs(numOfImages, path):
    imageList = []
    for i in range(0, numOfImages):
        img_path = path + str(i)+'.png'
        temp = image.load_img(img_path, target_size=(100, 100))
        temp = image.img_to_array(temp)
        imageList.append(temp)
    return imageList


def cnnModel(in_shape, out_size):
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3), input_shape=in_shape))
    model.add(Activation('relu'))
    model.add(Conv2D(32, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    #
    # model.add(MaxPooling2D(pool_size=(2, 2)))
    #
    # model.add(Conv2D(64, (3, 3)))
    # model.add(Activation('relu'))
    # model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(100, activation = 'relu'))
    model.add(Dropout(0.5))
    model.add(Dense(out_size))
    model.add(Activation('relu'))
#adamax for binary
    #sgd for regretion
    model.compile(loss='mse',
                      optimizer= optimizers.Adam(),
                      metrics=['mae'])
    return model

def cnnBinaryModel(in_shape, out_size):
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3), input_shape=in_shape))
    model.add(Activation('relu'))
    model.add(Conv2D(32, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    #
    # model.add(MaxPooling2D(pool_size=(2, 2)))
    #
    # model.add(Conv2D(64, (3, 3)))
    # model.add(Activation('relu'))
    # model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(100, activation = 'relu'))
    model.add(Dropout(0.5))
    model.add(Dense(out_size))
    model.add(Activation('sigmoid'))
#adamax for binary
    #sgd for regretion
    model.compile(loss='binary_crossentropy',
                      optimizer= optimizers.Adamax(),
                      metrics=['accuracy'])
    return model

def cnnCategoricalModel(in_shape, out_size):
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3), input_shape=in_shape))
    model.add(Activation('relu'))

    model.add(Conv2D(32, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(100, activation = 'relu'))
    model.add(Dropout(0.5))
    model.add(Dense(out_size))
    model.add(Activation('sigmoid'))
#adamax for binary
    #sgd for regretion
    model.compile(loss='categorical_crossentropy',
                      optimizer= optimizers.Adamax(),
                      metrics=['accuracy'])
    return model

def categorical_step_func(predicted, threshold):
    npred = []
    for i in predicted:
        first, second = 0.0 , 0.0
        if i[0] > threshold:
            first = 1.0
        if i[1] > threshold:
            second = 1.0
        npred.append([first,second])
    return np.asarray(npred)

def binary_step_func(predicted, threshold):
    npred = []
    for i in predicted:
        first = 0.0
        if i[0] > threshold:
            first = 1.0
        npred.append([first])
    return np.asarray(npred)



#put all images to one big array- dynamic list? - read in batches?
n_images = 109818#13726
trainSize = 100000
filename = 'dishwasher32/dish washer1_PAA_32_GADF'
imageList = []
labelList = []

for img in range(0,n_images):
    #------------UNPICKLE-------------#
    f = open(filename+' '+str(img)+'.pkl', 'rb')
    pickledFile = cPickle.load(f)

    traintuple = pickledFile[0] #train eikonas
    testtuple = pickledFile[1] #test eikonas - empty

    temp_X = traintuple[0]
    temp_Y = traintuple[1]

    imageList.append(temp_X.flatten())

    # labelList.append(np.reshape(temp_Y,temp_Y.size))
    averageLabel = np.average(temp_Y)
    # if averageLabel>0.015:
    #     averageLabel = 1
    # else: averageLabel = 0
    labelList.append(averageLabel)

#prepare data for conv input

imageList = preprocessing.minmax_scale(imageList)
labelList = preprocessing.minmax_scale(labelList,(0,1))

#----------CONVOLUTIONAL NN--------------#
# imgList = parseImgs(n_images,'img/fig-')
data_shape = (32,32,1)

train_X = np.reshape(imageList[:trainSize], (trainSize, 32,32,1))
test_X = np.reshape(imageList[trainSize:], (n_images-trainSize, 32,32,1))

train_Y = np.asarray(labelList[:trainSize])
test_Y = np.asarray(labelList[trainSize:])

#for categorical
y_train_binary = utils.to_categorical(train_Y)
y_test_binary = utils.to_categorical(test_Y)

model = cnnModel(data_shape,1)
model.fit(train_X, train_Y, batch_size = 10, epochs= 1,
          verbose=1, validation_data=(test_X, test_Y), shuffle=True)

#model = cnnBinaryModel(data_shape,1)
# model = cnnCategoricalModel(data_shape,2)
# model.fit(train_X,y_train_binary,batch_size=10,epochs=1,validation_data=(test_X,y_test_binary))

pred = model.predict(test_X,1)
#pred = binary_step_func(pred, 0.5)
#pred = categorical_step_func(pred, 0.5)

# import matplotlib.pyplot as plt
# plt.plot(pred.flatten(), label = 'pred')
# plt.plot(y_test_binary.flatten(), label= 'Y')
# #plt.plot(test_Y.flatten(), label= 'Y')
# plt.show()


# f = file('results.txt','wb')
# for i,j in zip(pred, y_test_binary):
#     f.write(str(i)+' - '+str(j)+'\n')
# f.close()

import matplotlib.pyplot as plt
plt.plot(pred.flatten(), label = 'pred')
plt.plot(test_Y.flatten(), label= 'Y')
plt.show()
#
# #print pred, test_Y




