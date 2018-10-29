from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
import numpy as np
import sklearn
from sklearn import tree
import PIL

#READ LABELS
# from sklearn.cross_validation import cross_val_score
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier, AdaBoostRegressor, RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import f1_score, accuracy_score, recall_score, precision_score, mean_absolute_error


def getFeatures(num_of_imgs):
    model = VGG16(include_top=False, weights='imagenet', input_tensor=None, input_shape=(100, 100, 3), pooling='avg',
                  classes=1000)
    # model.compile()

    vgg16_feature_list = []
    img_path = 'img-dishwasher4m/fig-'

    for i in range(0, num_of_imgs):
        path = img_path + str(i) + '.png'
        img = image.load_img(path, target_size=(100, 100))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)

        vgg16_feature = model.predict(x)
        vgg16_feature_np = np.array(vgg16_feature)
        vgg16_feature_list.append(vgg16_feature_np.flatten())

    feature_array = np.array(vgg16_feature_list)
    return feature_array


def saveFeatures(filename, farray):
    np.save(filename, farray)


def readFeatures(filename):
    return np.load(filename)


f = open('dish washer4m-labels').readlines()
# thresshold for fridge = 0.0062
labelList = []
for line in f:
    label = line.split(' ')
    labelarr = np.asarray(label).astype(np.float)
    labelavg = np.average(labelarr)
    if (labelavg > 0.00125):
        labelavg = 1
    else: labelavg = 0
    labelList.append(labelavg)
    # labelList.append(np.average(labelarr))

labelList = np.asarray(labelList)
train_Y = labelList[:13000]
test_Y = labelList[13000:15000]
print(test_Y.shape)

print('completed reading labels')
num_of_imgs = 15000

# vgg16_feature_array = getFeatures(num_of_imgs)
# saveFeatures('dishwasher32-vgg16-15000-features', vgg16_feature_array)
# print('save completed')

vgg16_feature_array = readFeatures('dishwasher32-vgg16-15000-features.npy')

train_X = vgg16_feature_array[:13000,:]
test_X = vgg16_feature_array[13000:,:]

# clf = AdaBoostRegressor(DecisionTreeRegressor(), n_estimators=5, learning_rate=0.5)

# clf = RandomForestRegressor(n_estimators=5)

# clf = AdaBoostClassifier(DecisionTreeClassifier(max_depth=2),n_estimators=40, learning_rate=0.25)

# clf = AdaBoostClassifier(RandomForestClassifier(random_state=0.7), n_estimators=50, learning_rate=0.5)

# clf = DecisionTreeClassifier(max_depth=3)

clf = RandomForestClassifier(n_estimators=5)

# cv = cross_val_score(model_tree, train_X, train_Y, cv=10)
# print("Accuracy: %0.2f (+/- %0.2f)" % (cv.mean(), cv.std() * 2))

clf.fit(train_X,train_Y)
pred = clf.predict(test_X)

#metrics
##CLASSIFICATION METRICS

f1 = f1_score(test_Y, pred, average='macro')
acc = accuracy_score(test_Y,pred)
rec = recall_score(test_Y,pred)
prec = precision_score(test_Y,pred)
print('f1:',f1)
print('acc: ',acc)
print('recall: ',rec)
print('precision: ',prec)

# ##REGRESSION METRICS
# mae = mean_absolute_error(test_Y,pred)
# print('mae: ',mae)
# E_pred = sum(pred)
# E_ground = sum(test_Y)
# rete = abs(E_pred-E_ground)/float(max(E_ground,E_pred))
# print('relative error total energy: ',rete)

import matplotlib.pyplot as plt
plt.plot(pred.flatten(), label = 'pred')
plt.plot(test_Y.flatten(), label= 'Y')
plt.show()