# cnn model
from turtle import backward

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import math
import enum
import xlsxwriter
import os


# keypointsMapping = ['Nose', 'Neck', 'R-Sho', 'R-Elb', 'R-Wr', 'L-Sho', 'L-Elb', 'L-Wr', 'R-Hip', 'R-Knee', 'R-Ank', 'L-Hip', 'L-Knee', 'L-Ank', 'R-Eye', 'L-Eye', 'R-Ear', 'L-Ear']
# dataset = pd.read_excel('/home/ahmedhasan/PycharmProjects/FYP/OpenPose/BodyJointDataSet2confidencecheck.xlsx')
# # dataset.set_index("Lable", inplace=True)
# labl = ['Abs', 'Background', 'Bench Press', 'Down', 'Push ups',  'SidePlank', 'Squats','up']
# data = []
# for label in labl:
#     data.append(dataset.loc[dataset['Lable'] == label].mean(axis= 0))
#
# print(data)
#
# df = pd.DataFrame({'Abs': data[0],
#                    'Push ups': data[4],
#                    'Background': data[1],
#                    'Bench Press': data[2],
#                    'Down': data[3],
#                    'Up': data[7],
#                    'Squats': data[6],
#                    'SidePlank': data[5]
#                    }, index=keypointsMapping)
# ax = df.plot.bar(rot=0)
#
# plt.show()
#
#  Returns value within[0, 1].
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

# Workbook() takes one, non-optional, argument
# which is the filename that we want to create.
workbook = xlsxwriter.Workbook('NormalizedDatasetWithRealData LeftSho-LeftHip.xlsx')

# The workbook object is then used to add new
# worksheet via the add_worksheet() method.
worksheet = workbook.add_worksheet()
col = 0
for b_part in BodyPart:
    worksheet.write(0, col,b_part.value)
    col += 1
worksheet.write(0, 17, "Lable")

interpreter = tf.lite.Interpreter(model_path="/home/ahmedhasan/PycharmProjects/FYP/Algorithmic-trading-master/posenet_model.tflite")
interpreter.allocate_tensors()

labl = ['Abs', 'Background', 'Bench Press', 'Down', 'Push ups',  'SidePlank', 'Squats','up']
folder ="/home/ahmedhasan/Desktop/FYP/NewDataSet/trainingData/"
count = 1
lablcount1 = 6
framenumpy =[]

# print(folder+labl[lablcount1])
for lablecount in range(8):
    for filename in os.listdir(folder+labl[lablecount]):
        image = cv2.imread(os.path.join(folder+labl[lablecount],filename))
        if image is not None:
            person_cods = EvaluatingModel(interpreter,image)
            Lshol = 0
            Lhip = 0
            for idx, keyPoint in enumerate(person_cods.keyPoints):
                keypoint_name = keyPoint.bodyPart
                posi = keyPoint.position
                if (keypoint_name=="LEFT_SHOULDER" ):
                    Lshol = float("{:.3f}".format(get_Euclideandistance(posi.x, posi.y,
                                                                           person_cods.keyPoints[0].position.x,
                                                                           person_cods.keyPoints[0].position.y)))
                if (keypoint_name=="LEFT_HIP" ):
                    Lhip = float("{:.3f}".format(get_Euclideandistance(posi.x, posi.y,
                                                                           person_cods.keyPoints[0].position.x,
                                                                           person_cods.keyPoints[0].position.y)))
            normalization_scale = (Lhip-Lshol)/2 +1
            for idx, keyPoint in enumerate(person_cods.keyPoints):
                posi = keyPoint.position

                if (keyPoint.score > 0):

                    distance = float("{:.3f}".format(get_Euclideandistance(posi.x, posi.y,
                                                                       person_cods.keyPoints[0].position.x,
                                                                       person_cods.keyPoints[0].position.y)))
                    worksheet.write(count, idx, distance/normalization_scale)
            worksheet.write(count, 17, labl[lablecount])
            print (count)
            count = count+1
        # if count%4 == 0:
        #     break

workbook.close()


# interpreter = tf.lite.Interpreter(model_path="/home/ahmedhasan/PycharmProjects/FYP/Algorithmic-trading-master/posenet_model.tflite")
# interpreter.allocate_tensors()
#
# image = cv2.imread('/home/ahmedhasan/Desktop/FYP/NewDataSet/AugmentedTrainData/Abs/frame4.jpg52.jpg')
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
# print(inputArray)
# img_height, img_width, img_channels = image.shape
# widthRatio = img_width/257
# heightRatio = img_height/257
#  # Draw key points over the image.
# for keyPoint in personclass.keyPoints :
#   if (keyPoint.score > 0):
#     position = keyPoint.position
#     adjustedX = position.x * widthRatio
#     adjustedY = position.y * heightRatio
#     # Center coordinates
#     center_coordinates = (120, 50)
#     # Radius of circle
#     radius = 5
#     # Blue color in BGR
#     color = (255, 0, 0)
#     # Line thickness of 2 px
#     thickness = 2
#     # Using cv2.circle() method
#     # Draw a circle with blue line borders of thickness of 2 px
#     image = cv2.circle(image, (int(adjustedX), int(adjustedY)), radius, color, thickness)
#     # canvas.drawCircle(adjustedX, adjustedY, circleRadius, paint)
# cv2.imshow("output",image)
# cv2.waitKey()








