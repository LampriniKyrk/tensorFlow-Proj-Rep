from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
import numpy as np
import sklearn
from sklearn import tree
import PIL

#READ LABELS
# from sklearn.cross_validation import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.svm import NuSVC, LinearSVC
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier, AdaBoostRegressor, RandomForestClassifier, RandomForestRegressor, \
    GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.metrics import f1_score, accuracy_score, recall_score, precision_score, mean_absolute_error, \
    label_ranking_average_precision_score, label_ranking_loss, confusion_matrix


def getFeatures(num_of_imgs):
    model = VGG16(include_top=False, weights='imagenet', input_tensor=None, input_shape=(100, 100, 3), pooling='avg',
                  classes=1000)
    # model.compile()

    vgg16_feature_list = []
    img_path = 'img-fridge1y/fig-'

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

def get_multilable_y(filenameList, thressholdList):
    array_y =[]
    idx = 0
    for fn, thr in zip(filenameList, thressholdList):
        temp_arr = []
        f = open(fn)
        for line in f:
            line = line.strip().split(" ")
            line = np.asarray(line).astype(np.float)
            avg = np.average(line)

            if (avg > thr):
                avg = 1
            else:
                avg = 0

            temp_arr.append(avg)

        f.close()
        temp_arr = np.asarray(temp_arr[:30000])
        # print temp_arr.shape
        array_y.append(temp_arr.flatten())
        idx += 1

    array_y = np.asarray(array_y)

    print(array_y.shape)
    return array_y


#~~~~~~~~~~~MAIN~~~~~~~~~~#
fnList =['data/kettleb2-labels',
         'data/fridgeb2-labels',
         'data/dish washerb2-labels',
         'data/microwaveb2-labels',
         'data/washing machineb2-labels']
thList = [0.0013, 0.0062, 0.00125, 0.025, 0.0025]


labelList = get_multilable_y(fnList,thList).T
print(labelList.shape)
print('completed reading labels')


num_of_imgs = labelList.__len__()

# vgg16_feature_array = getFeatures(num_of_imgs)
# saveFeatures('numpy-files/fridge64-vgg16-82000-features', vgg16_feature_array)
# print('save completed')

vgg16_feature_array = readFeatures('numpy-files/vgg16-b2.npy')
print vgg16_feature_array.shape
train_X = vgg16_feature_array[:21000,:]
train_Y = labelList[:21000,:]
test_X = vgg16_feature_array[21000:30000,]
test_Y = labelList[21000:,]
# train_X, test_X, train_Y, test_Y = train_test_split(vgg16_feature_array[:30000,], labelList, test_size=0.30, random_state=42)
print (test_X.shape)

# clf = OneVsRestClassifier(DecisionTreeClassifier())
clf = MLPClassifier()
# clf = RandomForestClassifier(n_estimators=1000)
# clf = ExtraTreesClassifier(n_estimators=500)

clf.fit(train_X,train_Y)
pred = clf.predict(test_X)

# confMatrix = confusion_matrix(test_Y, pred)
# print("confusion matrix: ", confMatrix)

#metrics
##MULTILABLE CLASSIFICATION METRICS
f1 = f1_score(test_Y, pred, average='micro')
prec = label_ranking_average_precision_score(test_Y, pred)
rank_loss = label_ranking_loss(test_Y, pred)
for i in range(0,5):
    print('label ',i,' f1: ', f1_score(test_Y[:,i], pred[:,i], average='micro'))

print('f1:',f1)
print('prec: ',prec)
print('ranking loss: ',rank_loss)

