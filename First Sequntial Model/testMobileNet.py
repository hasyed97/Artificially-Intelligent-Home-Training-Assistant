import cv2
import keras
from keras import backend as K
from keras.callbacks import ModelCheckpoint
from keras.engine.saving import load_model
from keras.layers.core import Dense, Activation
from keras.optimizers import Adam
from keras.metrics import categorical_crossentropy
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
from keras.models import Model
from keras.applications import imagenet_utils
from keras.layers import Dense,GlobalAveragePooling2D
from keras.applications import MobileNet
from keras.applications.mobilenet import preprocess_input
import numpy as np
from IPython.display import Image
from keras.optimizers import Adam
import pandas as pd
import seaborn as sn
from matplotlib.pyplot import figure
import matplotlib
from networkx.drawing.tests.test_pylab import plt

matplotlib.use("TkAgg")
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score


def load_image(img_path, show=False):

    # img = image.load_img(img_path, target_size=(128, 228
    #                                             ))
    img_tensor = image.img_to_array(img_path)                    # (height, width, channels)
    img_tensor = np.expand_dims(img_tensor, axis=0)         # (1, height, width, channels), add a dimension because the model expects this shape: (batch_size, height, width, channels)
    img_tensor /= 255.                                      # imshow expects values in the range [0, 1]

    if show:
        plt.imshow(img_tensor[0])
        plt.axis('off')
        plt.show()

    return img_tensor
def Confusion_Matrix_Built_IN( ConfusMat):
    CM = ConfusMat
    # print(CM)
    df_cm = pd.DataFrame(CM, range(5), range(5))

    sn.set(font_scale=1.5)  # for label size
    sn.heatmap(df_cm, vmin=1, vmax=max(CM.diagonal()), linewidth=.5, annot=True, fmt="d",cmap=plt.cm.Blues, xticklabels=False,
               yticklabels=False)  # font size
    plt.ylabel('CS5   CS4   CS3   CS2   CS1')
    plt.xlabel('CS1   CS2   CS3   CS4   CS5')
    plt.show()


train_datagen = ImageDataGenerator() #included in our dependencies
train_data_dir ='/home/ahmedhasan/Desktop/FYP/NewDataSet/testingData'

train_generator = train_datagen.flow_from_directory(train_data_dir,
                                                 target_size=(128,228),
                                                 color_mode='rgb',
                                                 batch_size=1,
                                                 # class_mode='categorical',
                                                 shuffle=True)

# train_generator = train_datagen.flow_from_directory(
#     train_data_dir,
#     target_size=(128,228),
#     batch_size=32,
#     shuffle=True,
#     subset='training') # set as training data
#
# validation_generator = train_datagen.flow_from_directory(
#     train_data_dir, # same directory as training data
#     target_size=(128,228),
#     batch_size=32,
#     shuffle=True,
#     subset='validation') # set as validation data
testrest = []
predicrest = []


model = load_model('/home/ahmedhasan/Desktop/FYP/NewDataSet/trainingData/BestWeightSequential.hdf5')
model.summary()
count = 0
i=0
for j in range(407):
    x,y = train_generator.next()
    # cv2.imshow('qwerty', x[1])
    # cv2.waitKey()
    # print(y)
    lable = np.where(y[i] == np.max(y[i]))
    new_image = load_image(x[i])
    pred = model.predict(new_image)
    predic = np.where(pred[0]==np.max(pred[0]))
    if(predic[0][0]==lable[0][0]):
        # failimg = cv2.convertScaleAbs(x[i], alpha=(255.0))
        # cv2.imwrite('testexercisefail/img'+str(j)+'.png',failimg)
        # cv2.imshow('qwerty', x[i])
        # cv2.waitKey()
        count=count+1
    # else:
    #     failimg = cv2.convertScaleAbs(x[i], alpha=(255.0))
    #     cv2.imwrite('testexercisepass/img' + str(j) + '.png', failimg)
    print(predic[0][0])
    print(lable[0][0])
    testrest.append(lable[0][0])
    predicrest.append(predic[0][0])
    print(i,j)
print(count/1035*100)


def precision(label, confusion_matrix):
    col = confusion_matrix[:, label]
    return confusion_matrix[label, label] / col.sum()


def recall(label, confusion_matrix):
    row = confusion_matrix[label, :]
    return confusion_matrix[label, label] / row.sum()


def precision_macro_average(confusion_matrix):
    rows, columns = confusion_matrix.shape
    sum_of_precisions = 0
    for label in range(rows):
        sum_of_precisions += precision(label, confusion_matrix)
    return sum_of_precisions / rows


def recall_macro_average(confusion_matrix):
    rows, columns = confusion_matrix.shape
    sum_of_recalls = 0
    for label in range(columns):
        sum_of_recalls += recall(label, confusion_matrix)
    return sum_of_recalls / columns

ConfuMat = confusion_matrix(testrest, predicrest)
# accuracy: (tp + tn) / (p + n)
accuracy = accuracy_score(testrest, predicrest)
print('Accuracy: %f' % accuracy)
# precision tp / (tp + fp)
prec = precision_macro_average(ConfuMat)
print('Precision: %f' % prec)
# recall: tp / (tp + fn)
rec = recall_macro_average(ConfuMat)
print('Recall: %f' % rec)
# f1: 2 tp / (2 tp + fp + fn)
f1 = 2*((prec*rec)/(prec+rec))
print('F1 score: %f' % f1)

Confusion_Matrix_Built_IN(ConfuMat)


# testX = np.load('/home/ahmedhasan/PycharmProjects/FYP/savedata/TestX.npy')
# testY = np.load('/home/ahmedhasan/PycharmProjects/FYP/savedata/TestY.npy')
# print(testX.shape)
# name=''
# for i in range(np.size(testY)):
#     if testY[i] == 0:
#         name = "Squats"
#     elif testY[i] == 1:
#         name = "SidePlank"
#     elif testY[i] == 2:
#         name = "Push ups"
#     elif testY[i] == 3:
#         name = "Bench Press"
#     elif testY[i] == 4:
#         name = "Abs"
#     cv2.imwrite("TestingDataSet/" + name + "/image" + str(i) + ".png", testX[i])