import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import math
import enum
import xlsxwriter
import os
from collections import deque


class BodyPart(enum.Enum):
  NOSE = "NOSE"
  LEFT_EYE = "LEFT_EYE"
  RIGHT_EYE = "RIGHT_EYE"
  LEFT_EAR = "LEFT_EAR"
  RIGHT_EAR = "RIGHT_EAR"
  LEFT_SHOULDER = "LEFT_SHOULDER"
  RIGHT_SHOULDER = "RIGHT_SHOULDER"
  LEFT_ELBOW = "LEFT_ELBOW"
  RIGHT_ELBOW = "RIGHT_ELBOW"
  LEFT_WRIST = "LEFT_WRIST"
  RIGHT_WRIST = "RIGHT_WRIST"
  LEFT_HIP = "LEFT_HIP"
  RIGHT_HIP = "RIGHT_HIP"
  LEFT_KNEE = "LEFT_KNEE"
  RIGHT_KNEE = "RIGHT_KNEE"
  LEFT_ANKLE = "LEFT_ANKLE"
  RIGHT_ANKLE = "RIGHT_ANKLE"
class Position:
    # constructor of class
    def __init__(self):
        self.x = 0
        self.y = 0
class Person:
    # constructor of class
    def __init__(self):
        self.keyPoints = []
        self.score = 0.0
class KeyPoint:
    # constructor of class
    def __init__(self):
        self.bodyPart = BodyPart.NOSE
        self.position =Position()
        self.score = 0.0
def sigmoid(x):
    return 1.0 / (1.0 + math.exp(-x))
def get_Euclideandistance(x1, y1,x2, y2):
        distance = math.sqrt(math.pow((x2 - x1), 2.0) + math.pow((y2 - y1), 2.0))
        return distance
def EvaluatingModel(tflite_interpreter,img):
    input_details = tflite_interpreter.get_input_details()
    output_details = tflite_interpreter.get_output_details()

    img_height = input_details[0]['shape'][1]
    img_width = input_details[0]['shape'][2]
    dim = (img_width, img_height)
    resized = cv2.resize(img, dim)
    reimage = []
    reimage.append(resized)
    # ii = np.array(reimage, dtype=np.float32)
    input_data = (np.float32(reimage) - 127.5) / 127.5
    # print('Resized Dimensions : ', ii.dtype)

    tflite_interpreter.set_tensor(input_details[0]['index'], input_data)
    tflite_interpreter.invoke()
    outputResults = {0: np.squeeze(tflite_interpreter.get_tensor(output_details[0]['index'])),
                     1: np.squeeze(tflite_interpreter.get_tensor(output_details[1]['index'])),
                     2: np.squeeze(tflite_interpreter.get_tensor(output_details[2]['index'])),
                     3: np.squeeze(tflite_interpreter.get_tensor(output_details[3]['index']))}

    # print(outputResults[0].shape)
    # print(len(outputResults[0][0][0]))
    # cv2.imshow("fghjk",resized)
    heatmaps = outputResults[0]
    offsets = outputResults[1]

    height = len(heatmaps)
    width = len(heatmaps[0])
    numKeypoints = len(heatmaps[0][0])

    # print(height,width,numKeypoints)
    #  Finds the (row, col) locations of where the keypoints are most likely to be.
    keypointPositions = []
    # print(keypointPositions)
    for keypoint in range(0, numKeypoints):
        maxVal = heatmaps[0][0][keypoint]
        maxRow = 0
        maxCol = 0
        for row in range(0, height):
            for col in range(0, width):
                if heatmaps[row][col][keypoint] > maxVal:
                    maxVal = heatmaps[row][col][keypoint]
                    maxRow = row
                    maxCol = col
                # print(maxVal)
        keypointPositions.append((maxRow, maxCol))
    # print("keypointPositions")
    # Calculating the x and y coordinates of the keypoints with offset adjustment.
    xCoords = []
    yCoords = []
    confidenceScores = []
    for idx, position in enumerate(keypointPositions):
        positionY = keypointPositions[idx][0]
        positionX = keypointPositions[idx][1]
        yCoords.append(int(
            position[0] / (height - 1) * img_height +
            offsets[positionY][positionX][idx]
        ))
        xCoords.append(int(
            position[1] / (width - 1) * img_width +
            offsets[positionY][positionX][idx + numKeypoints]
        ))
        confidenceScores.append(sigmoid(heatmaps[positionY][positionX][idx]))
    # print(xCoords,yCoords)
    person = Person()
    totalScore = 0.0
    idx = 0
    for it in BodyPart:
        temp = KeyPoint()
        temp.bodyPart = it.value
        temp.position.x = xCoords[idx]
        temp.position.y = yCoords[idx]
        temp.score = confidenceScores[idx]
        person.keyPoints.append(temp)
        totalScore += confidenceScores[idx]
        idx += 1

    person.score = totalScore / numKeypoints
    return person


# interpreter = tf.lite.Interpreter(model_path="/home/ahmedhasan/PycharmProjects/FYP/Algorithmic-trading-master/posenet_model.tflite")
# interpreter.allocate_tensors()
#
# image = cv2.imread('/home/ahmedhasan/PycharmProjects/FYP/OpenPose/frame13.jpg')
#
# personclass = EvaluatingModel(interpreter,image)
# # print(personclass.keyPoints[0].position.x)
#
# inputArray = []
# for idx , keyPoint in enumerate(personclass.keyPoints):
#     posi = keyPoint.position
#     distance = float("{:.3f}".format(get_Euclideandistance(posi.x,posi.y,
#         personclass.keyPoints[0].position.x,personclass.keyPoints[0].position.y)))
#     inputArray.append(distance)
#     # print(posi.x,posi.y,)
# # print(inputArray)

interpreter = tf.lite.Interpreter(model_path="/home/ahmedhasan/PycharmProjects/FYP/Algorithmic-trading-master/posenet_model.tflite")
interpreter.allocate_tensors()
# load the trained model and label binarizer from disk
print("[INFO] loading model and label binarizer...")
from keras.models import load_model
model= tf.keras.models.load_model(filepath='CNN1D_Posenet_NormalizedDatasetRealData.hdf5')

model.summary()
# initialize the image mean for mean subtraction along with the
Q = deque(maxlen=10)

# initialize the video stream, pointer to output video file, and
# frame dimensions
# vs = cv2.VideoCapture('/home/ahmedhasan/Desktop/FYP/real data.mp4')

vs = cv2.VideoCapture('/home/ahmedhasan/Desktop/FYP/29 January 2020 at 11_38 am 2020-01-29 11-41-28.mp4')
# vs = cv2.VideoCapture(0)
(W, H) = (None, None)
labl = ['Abs', 'Background', 'Bench Press', 'Down', 'Push ups',  'SidePlank', 'Squats','Up']
keypointsMapping = ['Nose', 'Neck', 'R-Sho', 'R-Elb', 'R-Wr', 'L-Sho', 'L-Elb', 'L-Wr', 'R-Hip', 'R-Knee', 'R-Ank',
                    'L-Hip', 'L-Knee', 'L-Ank', 'R-Eye', 'L-Eye', 'R-Ear']
newDF = pd.DataFrame( columns=keypointsMapping)
rep = 0
lableinit ='start'
count =0
# loop over frames from the video file stream

while True:
    # read the next frame from the file
    (grabbed, frame) = vs.read()

    # if the frame was not grabbed, then we have reached the end
    # of the stream
    if not grabbed:
        break
    count = count + 1
    if count%3!=0:
        continue
    # if the frame dimensions are empty, grab them
    if W is None or H is None:
        (H, W) = frame.shape[:2]

    # clone the output frame, then convert it from BGR to RGB
    # ordering, resize the frame to a fixed 224x224, and then
    # perform mean subtraction
    output = frame.copy()
    personclass = EvaluatingModel(interpreter,frame)
    inputArray = []
    Lshol = 0
    Lhip = 0
    for idx, keyPoint in enumerate(personclass.keyPoints):
        keypoint_name = keyPoint.bodyPart
        posi = keyPoint.position
        if (keypoint_name == "LEFT_SHOULDER"):
            Lshol = float("{:.3f}".format(get_Euclideandistance(posi.x, posi.y,
                                                                personclass.keyPoints[0].position.x,
                                                                personclass.keyPoints[0].position.y)))
        if (keypoint_name == "LEFT_HIP"):
            Lhip = float("{:.3f}".format(get_Euclideandistance(posi.x, posi.y,
                                                               personclass.keyPoints[0].position.x,
                                                               personclass.keyPoints[0].position.y)))
    normalization_scale = (Lhip - Lshol) / 2 + 1

    for idx, keyPoint in enumerate(personclass.keyPoints):
        posi = keyPoint.position
        newDF.at[1, keypointsMapping[idx]] = (float("{:.3f}".format(get_Euclideandistance(posi.x, posi.y,
                                                               personclass.keyPoints[0].position.x,
                                                               personclass.keyPoints[0].position.y))))/normalization_scale
        # inputArray.append(distance)
    # newDF = pd.DataFrame(inputArray)
    # make predictions on the frame and then update the predictions
    # newDF = newDF.drop(['Nose'], axis=1)
    preds = model.predict(newDF)[0]
    Q.append(preds)

    # perform prediction averaging over the current history of
    # previous predictions
    results = np.array(Q).mean(axis=0)
    i = np.argmax(results)
    print(labl[i])
    label = labl[i]
    if lableinit == 'start':
        if (label=='Up' or label=='Down'):
            lableinit= label
            rep = 0
    if (label == 'Up' or label == 'Down'):
        if(lableinit!=label):
            lableinit = label
            rep = rep+0.5
    if (label == 'Up' or label == 'Down'):
        label = 'Bench Press'
    # draw the activity on the output frame
    text = "activity: {}".format(label)
    cv2.putText(output, text, (35, 50), cv2.FONT_HERSHEY_SIMPLEX,
        1.25, (0, 255, 0), 5)
    cv2.putText(output, str(rep), (1000, 50), cv2.FONT_HERSHEY_SIMPLEX,
                1.25, (0, 255, 0), 5)

    # show the output image
    cv2.namedWindow('Frame', cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty('Frame', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.imshow('Frame', output)
    key = cv2.waitKey(1) & 0xFF
    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break

# release the file pointers
print("[INFO] cleaning up...")

vs.release()