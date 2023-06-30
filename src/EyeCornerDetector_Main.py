'''
Author: Alexandre Banks
Date: June 30, 2023
Institution: UBC Robotics and Control Laboratory
Description: Script that uses yolov7 to create eye corner bounding boxes, then finds eye corner
landmarks using conventional computer vision techniques.
'''


#---------------------<Library Imports>-------------------
import os
import cv2
from cv2 import dnn
import numpy as np
from collections import namedtuple
import dataclasses
from dataclasses import dataclass
from dataclasses import field


import torch
import torch.backends.cudnn as cudnn
from numpy import random

from models.experimental import attempt_load
from utils.general import check_img_size, non_max_suppression
from utils.torch_utils import select_device
#----------------------------------------------<Setting Variables>----------------------------------
img_path="resources/P18.avi"
model_path="resources/yolov7_tiny_custom.pt"
INPUT_WIDTH=640
INPUT_HEIGHT=640
CONF_THRESH=0.3
IOU_THRESH=0.4

#Class to hold the prediction results and cropped frame
@dataclass
class pred_res:
    ids: list=field(default_factory=list)
    xyxy: list=field(default_factory=list)
    cropped_eye: list=field(default_factory=list)

class_names=['Inner Corner', 'Outer Corner']

#Canny Edge Detector Parameters
low_threshold=20
ratio_val=3
upper_threshold=low_threshold*ratio_val
canny_kernel_size=3
filter_size=9

#Morph op parameters
erosion_size=4
dilation_size=4

#Curvature Calculation Parameters
step_size=4




#----------------------------------------------<Function Definitions>-------------------------------
#Subfunction to pre-process one left/right frames from raw video
def process_oneframe(cropped_frame):
    gray=cv2.cvtColor(cropped_frame,cv2.COLOR_BGR2GRAY)
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

def reshape_frame(cropped_frame,imgsz):
        #Finds the ratio between the cropped image and the network input size
        dw=INPUT_WIDTH/cropped_frame.shape[1]
        dh=INPUT_HEIGHT/cropped_frame.shape[0]

        left_restructured=cv2.resize(cropped_frame,(imgsz,imgsz),cv2.INTER_LINEAR)
              
        #Convert the image from BGR to RGB
        left_restructured=left_restructured[:,:,::-1].transpose(2,0,1)
        #right_restructured=right_restructured[:,:,::-1].transpose(2,0,1)

        left_restructured=np.ascontiguousarray(left_restructured)
        #right_restructured=np.ascontiguousarray(right_restructured)
        return left_restructured,dw,dh

def detect(frame,model,device,old_img_b,old_img_h,old_img_w):
    #Make sure to return old_img_b, old_img_h, and old_img_w
    names={'Inner Corner','Outer Corner'}
    img=torch.from_numpy(frame).to(device) #Loads in the image and casts to device
    img=img.float()     #Converts it to fp32
    img/=255.0 #Normalizes the image to 0.0-1.0
    if img.ndimension()==3:
        img=img.unsqueeze(0)
    #Warmup (maybe get rid of this)

    if device.type != 'cpu' and (old_img_b != img.shape[0] or old_img_h != img.shape[2] or old_img_w != img.shape[3]):
        old_img_b = img.shape[0]
        old_img_h = img.shape[2]
        old_img_w = img.shape[3]
        for i in range(3):
            model(img,augment=False)[0]
    
    #Run Inference
    with torch.no_grad():   # Calculating gradients would cause a GPU memory leak
        pred = model(img, augment=False)[0]
    
    #Apply nms
    pred=non_max_suppression(pred,CONF_THRESH,IOU_THRESH)

    return pred,old_img_b,old_img_h,old_img_w


def process_detections(pred,img0,dw,dh):
    pred_results=pred_res() #Creates class object
    print(pred_results)
    for i,det in enumerate(pred): #Loops through all detections in the prediction results
       
        if len(det):    #Enters if the detection exists
            for *xyxy, conf, cls in reversed(det): #Loops for the bounding boxes

                class_type=int(cls)
                xyxy_list=torch.tensor(xyxy).view(1,4).tolist()[0] #Converts tensor detection to list\

                #Scaling the detection                
                xyxy_list[0]=xyxy_list[0]*(1/dw)
                xyxy_list[1]=xyxy_list[1]*(1/dh)
                xyxy_list[2]=xyxy_list[2]*(1/dw)
                xyxy_list[3]=xyxy_list[3]*(1/dh)   

                #Converting the list to integer             
                xyxy_list=[int(item) for item in xyxy_list]
                               
                #Cropping the eyecorner ROI                
                cropped_eyecorner=img0[xyxy_list[1]:xyxy_list[3],xyxy_list[0]:xyxy_list[2]]
                #Adding Results to prediction results class
                pred_results.ids.append(class_type)
                pred_results.xyxy.append(xyxy_list)
                pred_results.cropped_eye.append(cropped_eyecorner)
    return (pred_results)



#Functions for eye corner detection



#--------------------------------------------------<Video Loading>---------------------------------------------
video=cv2.VideoCapture(img_path)
#Checking Paths
if(video.isOpened()==False):
    print("Video Cannot be Opened")

#---------------------------------------------------<Model Loading>-----------------------------------

device=select_device('0') #Sets up the GPU Cuda device
model=attempt_load(model_path,map_location=device)
stride_size=int(model.stride.max())
imgsz=check_img_size(INPUT_HEIGHT,s=stride_size) #Checks that stride requirements are met

#Setup model
if device.type != 'cpu':
    model(torch.zeros(1,3,imgsz,imgsz).to(device).type_as(next(model.parameters()))) #runs it once
left_old_img_w=left_old_img_h=imgsz
left_old_img_b=1

right_old_img_w=right_old_img_h=imgsz
right_old_img_b=1


#-----------------------------------------------------<Main Loop>--------------------------------------------
while(video.isOpened()):    #Loops for each frame in the video
    ret,frame=video.read() #Reads frame
    
    if ret==True:
    #--------------------------------------------<Frame pre-processing>------------------------------------------
        left_cropped,right_cropped=process_frame(frame) #Crops the frame to right and left images

        left_restructured,dw_l,dh_l=reshape_frame(left_cropped,imgsz) #Reshapes the images to 1x3x640x640
        right_restructured,dw_r,dh_r=reshape_frame(right_cropped,imgsz)

    #----------------------------------------------<Running Inference>---------------------------------------------
        pred_left,left_old_img_b,left_old_img_h,left_old_img_w=detect(left_restructured,model,device,left_old_img_b,left_old_img_h,left_old_img_w)
        pred_right,right_old_img_b,right_old_img_h,right_old_img_w=detect(right_restructured,model,device,right_old_img_b,right_old_img_h,right_old_img_w)
    #----------------------------------------------<Process Detections>--------------------------------------
        left_results=process_detections(pred_left,left_cropped,dw_l,dh_l)
        right_results=process_detections(pred_right,right_cropped,dw_r,dh_r)

    #----------------------------------------------<Eye Corner Detection>------------------------------
        cv2.imshow('Original',left_cropped)
        cv2.imshow('Cropped Eyecorner',left_results.cropped_eye[0])
        cv2.waitKey(0)
        if len(left_results.ids) and len(right_results.ids): #Got resuls for both eyes

        elif len(left_results.ids): #Got results for the left eye

        elif len(right_results.ids): #Got results for the right eye

        else: #No corners detected for this frame
            continue
    else:
        print("Frame Not Read Correctly")
        break
