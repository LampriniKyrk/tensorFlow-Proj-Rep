from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
import numpy as np
from sklearn.externals import joblib
import sklearn
from sklearn import tree
import PIL

#READ LABELS
# from sklearn.cross_validation import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier, AdaBoostRegressor, RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import f1_score, accuracy_score, recall_score, precision_score, mean_absolute_error, \
    confusion_matrix


def getFeatures(num_of_imgs):
    model = VGG16(include_top=False, weights='imagenet', input_tensor=None, input_shape=(100, 100, 3), pooling='avg',
                  classes=1000)
    # model.compile()

    vgg16_feature_list = []
    img_path = 'b5/fig-'

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

def create_multilable_y(filenameA, filenameB, thressholdA, thressholdB):
    fA = open(filenameA)
    fB = open(filenameB)
    new_Y= []
    for l1, l2 in zip(fA, fB):
        l1 = l1.split(' ')
        l2 = l2.split(' ')
        l1 = np.asarray(l1).astype(np.float)
        l2 = np.asarray(l2).astype(np.float)
        avg1 = np.average(l1)
        avg2 = np.average(l2)
        if (avg1 > thressholdA):
            avg1 = 1
        else: avg1 = 0
        if (avg2 > thressholdB):
            avg2 = 1
        else: avg2 = 0
        new_Y.append([avg1, avg2])
    return np.asarray(new_Y)


f = open('data/washing machineb2-labels').readlines()
# thresshold for fridge = 0.0062
labelList = []
for line in f:
    label = line.split(' ')
    labelarr = np.asarray(label).astype(np.float)
    labelavg = np.average(labelarr)
    if (labelavg > 0.0025):
        labelavg = 1
    else: labelavg = 0
    labelList.append(labelavg)
    # labelList.append(np.average(labelarr))

labelList = np.asarray(labelList)
# train_Y = labelList[:78000]
# test_Y = labelList[78000:82000]
# print(test_Y.shape)

# labelList = create_multilable_y('fridge1y-labels','microwave1y-labels', 0.0062, 0.025)
print('completed reading labels')
# print(labelList.shape)
# labelList = labelList[:82000]
num_of_imgs = labelList.__len__()

# vgg16_feature_array = getFeatures(num_of_imgs)
# saveFeatures('numpy-files/fridge-vgg16-b5', vgg16_feature_array)
# print('save completed')

vgg16_feature_array = readFeatures('numpy-files/vgg16-b2.npy')
vgg16_feature_array = vgg16_feature_array[:labelList.__len__()]
train_X, test_X, train_Y, test_Y = train_test_split(vgg16_feature_array, labelList, test_size=0.90, random_state=42)
# print (test_X.shape)
# train_X = vgg16_feature_array[:78000,:]
# test_X = vgg16_feature_array[78000:,:]

# clf = AdaBoostRegressor(DecisionTreeRegressor(), n_estimators=5, learning_rate=0.5)

# clf = RandomForestRegressor(n_estimators=10)

# clf = MLPRegressor(hidden_layer_sizes=20, activation='tanh')

# clf = AdaBoostClassifier(DecisionTreeClassifier(max_depth=4),n_estimators=1000, learning_rate=0.25)

# clf = AdaBoostClassifier(n_estimators=1000, learning_rate=0.25)

# clf = AdaBoostClassifier(RandomForestClassifier(random_state=0.7), n_estimators=500, learning_rate=0.5)

# clf = DecisionTreeClassifier(max_depth=15)

# clf = RandomForestClassifier(n_estimators=1000)

clf = MLPClassifier(hidden_layer_sizes=50, batch_size=20)

# cv = cross_val_score(model_tree, train_X, train_Y, cv=10)
# print("Accuracy: %0.2f (+/- %0.2f)" % (cv.mean(), cv.std() * 2))
#
# clf.fit(train_X,train_Y)
# joblib.dump(clf, 'MLP50-whashingmachine-13-14.joblib')
clf = joblib.load('MLP50-whashingmachine-13-14.joblib')
pred = clf.predict(test_X)

confMatrix = confusion_matrix(test_Y, pred)
print("confusion matrix: ", confMatrix)

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
#
import matplotlib.pyplot as plt
plt.plot(pred.flatten(), label = 'pred')
plt.plot(test_Y.flatten(), label= 'Y')
plt.show()