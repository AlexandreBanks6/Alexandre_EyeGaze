#---------------------<Library Imports>-------------------
import os
import cv2
from cv2 import dnn
import numpy as np

import torch
import torch.backends.cudnn as cudnn
from numpy import random

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
scale_coords,xyxy2xywh, strip_optimizer,set_logging,increment_path
from utils.torch_utils import select_device, load_classifier, time_synchronized, TracedModel

#------------------------<Setting Variables>---------------------
img_path="../resources/P18.avi"
model_path="../resources/yolov7_tiny.pt"
INPUT_WIDTH,INPUT_HEIGHT=640


#------------------<Function Definitions>----------------
#Letterbox
#It scales the image to INPUT_WIDTHxINPUT_HEIGHT and pads with zeros
def letterbox(img, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
    # Resize and pad image while meeting stride-multiple constraints
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better test mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return img, ratio, (dw, dh)


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



def detect(frame,model,device,imgsz,old_img_b,old_img_h,old_img_w):
    img=torch.from_numpy(frame).to(device) #Loads in the image and casts to device
    img=img.float()     #Converts it to fp32
    img/=255.0 #Normalizes the image to 0.0-1.0
    if img.ndimensions==3:
        img=img.unsqueeze(0)
    #Warmup (maybe get rid of this)
    if device.type!='cpu' and (old_img_b!=img.shape[0] or old_img_h!=img.shape[2] or old_img_w!=img.shape[3]):
        old_img_b=img.shape[0]
        old_img_h=img.shape[2]
        old_img_w=img.shape[3]
        for i in range(3):
            model(img,augment=False)[0]
    

    




#-----------------------<Video Loading>----------------------
video=cv.VideoCapture(img_path)
#Checking Paths
if(video.isOpened()==False):
    print("Video Cannot be Opened")

#----------------------<Model Loading>-------------------
device=select_device('0')
model=attempt_load(model_path,map_location=device)
stride=int(model.stride.max())
imgsz=check_img_size((INPUT_HEIGHT,INPUT_WIDTH),s=stride)
#Setup model
if device.type != 'cpu':
    model(torch.zeros(1,3,imgsz,imgsz).to(device).type_as(next(model.parameters()))) #runs it once
old_img_w=old_img_h=INPUT_HEIGHT
old_img_b=1




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
