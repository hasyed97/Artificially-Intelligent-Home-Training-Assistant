# first neural network with keras tutorial
import sys


import xlsxwriter
from cv2 import imshow
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from matplotlib import pyplot
import numpy as np
import tensorflow as tf
from numpy import loadtxt
from keras.models import Sequential
from keras.layers import Dense, Conv1D, Dropout, MaxPooling1D, Flatten, Reshape, GlobalAveragePooling1D
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt
import seaborn as sns
def summarize_diagnostics(history):
    # plot loss
    pyplot.subplot(211)
    pyplot.title("Cross Entropy Loss")
    pyplot.plot(history.history["loss"], color="blue", label="train")
    pyplot.plot(history.history["val_loss"], color="orange", label="test")
    pyplot.legend()
    # plot accuracy
    pyplot.subplot(212)
    pyplot.title("Classification Accuracy")
    pyplot.plot(history.history["acc"], color="blue", label="train")
    pyplot.plot(history.history["val_acc"], color="orange", label="test")
    pyplot.legend()
    # save plot to file
    pyplot.savefig("NormalizedDatasetWithRealData LeftSho-LeftHipDropNoseFinal.png")
    pyplot.close()


def define_model():
    model_m = Sequential()
    model_m.add(Reshape((17, 1), input_shape=(17,)))
    model_m.add(Conv1D(32, 2, activation='relu', input_shape=(17,1)))
    model_m.add(Conv1D(64, 2, activation='relu'))
    model_m.add(MaxPooling1D(1))
    model_m.add(Conv1D(128, 2, activation='relu'))
    model_m.add(Conv1D(256, 1, activation='relu'))
    model_m.add(GlobalAveragePooling1D())
    model_m.add(Dropout(0.5))
    model_m.add(Dense(8, activation='softmax'))
    # print(model_m.summary())

    return model_m

# Importing dataset
dataset = pd.read_excel("NormalizedDatasetWithRealData LeftSho-LeftHip.xlsx")
# dataset = dataset.loc[dataset['Lable'] != 'Abs']
# dataset = dataset.drop(['Nose'], axis=1)
dataset = dataset.sample(frac=1).reset_index(drop=True)
print(dataset)
# dataset = dataset.drop(['NOSE'], axis=1)
# X_train,X_test,y_train,y_test = train_test_split(dataset,dataset.pop('Lable'),test_size=0)
# print(X_train.reset_index(drop=True))
TrainY = dataset['Lable']
TrainX = dataset.drop(['Lable'], axis=1)
# TestX = X_test.reset_index(drop=True)
# TestY = y_test.reset_index(drop=True)

lables = ['Abs','Background', 'Bench Press', 'Down', 'Push ups',  'SidePlank', 'Squats','up']

i =0
for lab in lables:
    TrainY = TrainY.replace([lab], i)
    # TestY = TestY.replace([lab], i)
    i = i+1

train_x, test_x, train_y, test_y = train_test_split(TrainX, TrainY, test_size=0.20)

model = define_model()
print(model.summary())
# print(TrainX.shape)
# compile the keras model
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['acc'])

filepath = "CNN1D_Posenet_NormalizedDatasetRealDataDropNoseFinal.hdf5"

earlyStopping = EarlyStopping(monitor='val_acc', patience=30, verbose=0, mode='max')
mcp_save = ModelCheckpoint(filepath, save_best_only=True, monitor='val_acc', mode='max')
reduce_lr_loss = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=7, verbose=1, epsilon=1e-4, mode='min')

callbacks_list = [earlyStopping,mcp_save, reduce_lr_loss]
# fit the keras model on the datasetx
history = model.fit(train_x, train_y, batch_size=5, epochs=400, verbose=2, callbacks=callbacks_list, validation_split=0.2)

summarize_diagnostics(history)
model.save('CNN1D_Posenet_NormalizedDatasetRealDataDropNoseFinal.h5')




# model= tf.keras.models.load_model(filepath='CNN1D_Posenet_NormalizedDatasetRealData.hdf5')
# model = tf.keras.models.load_model(filepath='/home/ahmedhasan/PycharmProjects/FYP/Algorithmic-trading-master/ANN400.h5')
# evaluate the keras model
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
def Confusion_Matrix_Built_IN( ConfusMat):
    CM = ConfusMat
    # print(CM)
    df_cm = pd.DataFrame(CM, range(8), range(8))

    sns.set()  # for label size
    sns.heatmap(df_cm, vmin=1, vmax=max(CM.diagonal()), linewidth=.5, annot=True, fmt="d",cmap=plt.cm.Blues, xticklabels=False,
               yticklabels=False)  # font size
    plt.xlabel('ABS  BGD  BPS  DNB  PUP  SPK   SQT  UPB')
    plt.ylabel('UPB  SQT  SPK  PUP  DNB  BPS   BGD  ABS')
    plt.show()


y_pred = model.predict(test_x)
y_pred = np.argmax(y_pred, axis=1)
# print(y_pred)

ConfuMat = confusion_matrix(test_y,y_pred )
# accuracy: (tp + tn) / (p + n)
_, accuracy = model.evaluate(TrainX, TrainY)
print('Accuracy: %f' % (accuracy*100))
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




# _, accuracy = model.evaluate(TrainX, TrainY)
# print('Accuracy: %.2f' % (accuracy*100))
# y_pred = model.predict(TrainX)
# y_pred = np.argmax(y_pred, axis=1)
# # print(y_pred)
#
# cm = confusion_matrix(TrainY,y_pred )
# print(cm)
# ax= plt.subplot()
# sns.heatmap(cm, annot=True, ax = ax) #annot=True to annotate cells
#
# # labels, title and ticks
# ax.set_xlabel('Predicted labels')
# ax.set_ylabel('True labels')
# ax.set_title('Confusion Matrix')
# ax.xaxis.set_ticklabels(lables)
# ax.yaxis.set_ticklabels(lables)
# plt.show()





# import tensorflow as tf
#
# # Specify that all features have real-value data
# feature_columns = [tf.contrib.layers.real_valued_column("", dimension=8)]
#
# # Build 3 layer DNN with 512, 256, 128 units respectively.
# classifier = tf.contrib.learn.DNNClassifier(feature_columns=feature_columns,
#                                             hidden_units=[512, 256, 128],
#                                             n_classes=8,
#                                             optimizer=tf.train.ProximalAdagradOptimizer(
#                                                 learning_rate=0.15,
#                                                 l1_regularization_strength=0.001
#                                             ))
#
# # Define the training inputs
# def get_train_inputs():
#     x = tf.constant(TrainX)
#     y = tf.constant(TrainY)
#     return x, y
#
# # Fit model.
# his = classifier.fit(input_fn=get_train_inputs, steps=1200)
#
# # Define the test inputs
# def get_test_inputs():
#     x = tf.constant(TrainX)
#     y = tf.constant(TrainY)
#
#     return x, y
#
#
# # Evaluate accuracy.
# # print(classifier.evaluate(input_fn=get_test_inputs, steps=1))
# accuracy_score = classifier.evaluate(input_fn=get_test_inputs, steps=1)["accuracy"]*100
#
# graph_location = '/home/ahmedhasan/PycharmProjects/FYP/Algorithmic-trading-master/car-evaluation'
# print('Saving graph to: %s' % graph_location)
# train_writer = tf.summary.FileWriter(graph_location)
# train_writer.add_graph(tf.get_default_graph())
# print("Test Accuracy: {0:f}".format(accuracy_score))



