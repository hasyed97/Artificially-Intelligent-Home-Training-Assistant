import cv2
vidcap = cv2.VideoCapture('/home/ahmedhasan/Desktop/FYP/NewDataSet/Real Data M7/real data.mp4')
success,image = vidcap.read()
count = 0
frame =0
success = True
while success:
  success,image = vidcap.read()
  if(frame==0):
    try:
        cv2.imwrite("/home/ahmedhasan/Desktop/FYP/NewDataSet/Real Data M7/RealM7DataSidePlank/realM75frame%d.jpg" % (count/2), image)     # save frame as JPEG file
    except:
        continue
  if(frame==2):
    frame=-1
  frame += 1
  # print(frame)
  count += 1

# importing libraries
# import cv2
# import numpy as np
#
# # Create a VideoCapture object and read from input file
# cap = cv2.VideoCapture('/home/ahmedhasan/Desktop/FYP/29 January 2020 at 11_38 am 2020-01-29 11-41-28.mp4')
# # Check if camera opened successfully
# if (cap.isOpened() == False):
#     print("Error opening video  file")
#
# # Read until video is completed
# while (cap.isOpened()):
#
#     # Capture frame-by-frame
#     ret, frame = cap.read()
#     if ret == True:
#         cv2.namedWindow('Frame', cv2.WND_PROP_FULLSCREEN)
#         cv2.setWindowProperty('Frame', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
#
#         # Display the resulting frame
#         cv2.imshow('Frame', frame)
#
#         # Press Q on keyboard to  exit
#         if cv2.waitKey(500) & 0xFF == ord('q'):
#             break
#
#     # Break the loop
#     else:
#         break
#
# # When everything done, release
# # the video capture object
# cap.release()
#
# # Closes all the frames
# cv2.destroyAllWindows()

