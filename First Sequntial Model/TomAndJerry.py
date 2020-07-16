# import sys
#
# import cv2
# import keras
# from keras import backend as K, Sequential
# from keras.callbacks import ModelCheckpoint
# from keras.engine.saving import load_model
# from keras.layers.core import Dense, Activation, Dropout, Flatten
# from keras.optimizers import Adam, SGD
# from keras.metrics import categorical_crossentropy
# from keras.preprocessing.image import ImageDataGenerator
# from keras.preprocessing import image
# from keras.models import Model
# from keras.applications import imagenet_utils
# from keras.layers import Dense, GlobalAveragePooling2D, Conv2D, MaxPooling2D
# from keras.applications import MobileNet
# from keras.applications.mobilenet import preprocess_input
# from matplotlib import pyplot
# import numpy as np
# from IPython.display import Image
# from keras.optimizers import Adam
#
#
# # plot diagnostic learning curves
#
#
# def summarize_diagnostics(history):
#     # plot loss
#     pyplot.subplot(211)
#     pyplot.title("Cross Entropy Loss")
#     pyplot.plot(history.history["loss"], color="blue", label="train")
#     pyplot.plot(history.history["val_loss"], color="orange", label="test")
#     pyplot.legend()
#     # plot accuracy
#     pyplot.subplot(212)
#     pyplot.title("Classification Accuracy")
#     pyplot.plot(history.history["acc"], color="blue", label="train")
#     pyplot.plot(history.history["val_acc"], color="orange", label="test")
#     pyplot.legend()
#     # save plot to file
#     filename = sys.argv[0].split("/")[-1]
#     pyplot.savefig(filename + "NewData8classes_plot1.png")
#     pyplot.close()
#
#
# def define_model():
#     model = Sequential()
#     model.add(Conv2D(32, (3, 3), padding='same', input_shape=(128, 228, 3)))
#     model.add(Activation('relu'))
#     model.add(Conv2D(32, (3, 3)))
#     model.add(Activation('relu'))
#     model.add(MaxPooling2D(pool_size=(2, 2)))
#     model.add(Dropout(0.25))
#
#     model.add(Conv2D(64, (3, 3), padding='same'))
#     model.add(Activation('relu'))
#     model.add(Conv2D(64, (3, 3)))
#     model.add(Activation('relu'))
#     model.add(MaxPooling2D(pool_size=(2, 2)))
#     model.add(Dropout(0.25))
#
#     model.add(Flatten())
#
#     model.add(Dense(512))
#     model.add(Activation('relu'))
#     model.add(Dropout(0.5))
#     model.add(Dense(8))
#     model.add(Activation('softmax'))
#
#     return model
#
#
# train_data_dir = '/home/ahmedhasan/Desktop/FYP/NewDataSet/trainingData'
# train_datagen = ImageDataGenerator(preprocessing_function=preprocess_input,
#                                    validation_split=0.2)  # included in our dependencies
# train_generator = train_datagen.flow_from_directory(train_data_dir,
#                                                     target_size=(128, 228),
#                                                     color_mode='rgb',
#                                                     batch_size=32,
#                                                     class_mode='categorical',
#                                                     shuffle=True,
#                                                     subset='training')
#
# validation_generator = train_datagen.flow_from_directory(train_data_dir,
#                                                          target_size=(128, 228),
#                                                          color_mode='rgb',
#                                                          batch_size=32,
#                                                          class_mode='categorical',
#                                                          shuffle=True,
#                                                          subset='validation')  # set as validation data
#
# model = define_model()
#
# model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['acc'])
# # Adam optimizer
# # loss function will be categorical cross entropy
# # evaluation metric will be accuracy
# filepath = "/home/ahmedhasan/Desktop/FYP/NewDataSet/trainingData/BestWeightSequential8Classes(background).hdf5"
# checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
#
# callbacks_list = [checkpoint]
#
# history = model.fit_generator(
#     generator=train_generator,
#     steps_per_epoch=train_generator.samples // train_generator.batch_size,
#     validation_data=validation_generator,
#     validation_steps=validation_generator.samples // validation_generator.batch_size,
#     epochs=25, callbacks=callbacks_list)
# summarize_diagnostics(history)
# model.save('/home/ahmedhasan/Desktop/FYP/NewDataSet/trainingData/BestWeightSequential8Classes(background).h5')
# import tensorflow as tf
#
# print(tf.__version__)
#
#
# new_model= tf.keras.models.load_model(filepath= '/home/ahmedhasan/Desktop/FYP/NewDataSet/trainingData/BestWeightSequential8Classes(background).hdf5')
# converter = tf.lite.TFLiteConverter.from_keras_model( new_model ) # Your model's name
# model = converter.convert()
# file = open( '/home/ahmedhasan/Desktop/FYP/NewDataSet/trainingData/BestWeightSequential8Classes(background)TnsorfowLitemodel.tflite' , 'wb' )
# file.write( model )
import numpy as np
import tensorflow as tf

# Load TFLite model and allocate tensors.
interpreter = tf.lite.Interpreter(model_path="/home/ahmedhasan/Desktop/FYP/NewDataSet/trainingData/BestWeightSequential8Classes(background)TnsorfowLitemodel.tflite")
interpreter.allocate_tensors()

# Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Test model on random input data.
input_shape = input_details[0]['shape']
input_data = np.array(np.random.random_sample(input_shape), dtype=np.float32)
interpreter.set_tensor(input_details[0]['index'], input_data)

interpreter.invoke()

# The function `get_tensor()` returns a copy of the tensor data.
# Use `tensor()` in order to get a pointer to the tensor.
output_data = interpreter.get_tensor(output_details[0]['index'])
print(output_data)




# i = 0
# name = ''
# for j in range(4915):
#     x, y = train_generator.next()
#     # cv2.imshow('qwerty', x[1])
#     # cv2.waitKey()
#     # print(y)
#     lable = np.where(y[i] == np.max(y[i]))
#
#     if lable[0][0] == 0:
#         name = 'Abs'
#     elif lable[0][0] == 1:
#         name = 'Bench Press'
#     elif lable[0][0] == 2:
#         name = 'Push ups'
#     elif lable[0][0] == 3:
#         name = 'SlidrPlank'
#     elif lable[0][0] == 4:
#         name = 'Squats'
#     # image = cv2.convertScaleAbs(x[i], alpha=(255.0))
#     cv2.imwrite('TrainingDataSet/' + name + "/image" + str(j) + ".png", x[i])
#
# for j in range(1227):
#     x, y = validation_generator.next()
#     # cv2.imshow('qwerty', x[1])
#     # cv2.waitKey()
#     # print(y)
#     lable = np.where(y[i] == np.max(y[i]))
#
#     if lable[0][0] == 0:
#         name = "Squats"
#     elif lable[0][0] == 1:
#         name = "SidePlank"
#     elif lable[0][0] == 2:
#         name = "Push ups"
#     elif lable[0][0] == 3:
#         name = "Bench Press"
#     elif lable[0][0] == 4:
#         name = "Abs"
#     image = cv2.convertScaleAbs(x[i], alpha=(255.0))
#     cv2.imwrite('TrainingDataSet/' + name + "/image" + str(j) + ".png", image)
