#---------------------<Library Imports>-------------------
import os
from cv2 import dnn
import numpy as np

#-----------------------<Video Loading>----------------------
video=cv.VideoCapture('/ubc/ece/home/ts/grads/alexandre/Documents/YOLOv7/yolov7/alexandre_thesis_code/P18.avi')
#Checking Paths
if(video.isOpened()==False):
    print("Video Cannot be Opened")

#----------------------<Model Loading>-------------------

#------------------<Function Definitions>----------------

#Subfunction to pre-process one left/right frames from raw video
def process_oneframe(cropped_frame):
    gray=cv.cvtColor(cropped_frame,cv.COLOR_BGR2GRAY)
    restructured=np.zeros_like(cropped_frame)

    #Passing the gray image to each channel of the image
    restructured[:,:,0]=gray
    restructured[:,:,1]=gray
    restructured[:,:,2]=gray

    return restructured

#Function that pre-processes the video frame and splits it into two frames
def process_frame(frame):

    #Crops Frames
    left_frame=frame[:480,0:640]
    right_frame=frame[:480,640:]
    
    left_restructured=process_oneframe(left_frame)
    right_restructured=process_oneframe(right_frame)

    return left_restructured,right_restructured


#------------------------------<Main Loop>--------------------------------
while(video.isOpened()):    #Loops for each frame in the video
    ret,frame=video.read() #Reads frame
    
    if ret==True:
    #--------------------<Frame pre-processing>---------------------------
        left_restructured,right_restructured=process_frame(frame)




        cv.imshow('Frame',left_restructured)
        cv.waitKey(0)

    else:
        print("Frame Not Read Correctly")
        break
