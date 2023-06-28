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
model_path="../resources/yolov7_tiny_custom.pt"
INPUT_WIDTH=640
INPUT_HEIGHT=640
CONF_THRESH=0.3
IOU_THRESH=0.4
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



def detect(frame,model,device,imgsz,old_img_b,old_img_h,old_img_w):
    #Make sure to return old_img_b, old_img_h, and old_img_w
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
    
    #Run Inference
    with torch.no_grad():
        pred=model(img,augment=False)
    
    #Apply nms
    pred=non_max_suppression(pred,CONF_THRESH,IOU_THRESH)
    
    return pred,old_img_b,old_img_h,old_img_w

#-----------------------<Video Loading>----------------------
video=cv2.VideoCapture(img_path)
#Checking Paths
if(video.isOpened()==False):
    print("Video Cannot be Opened")

#----------------------<Model Loading>-------------------

device=select_device('0')
model=attempt_load(model_path,map_location=device)
stride_size=int(model.stride.max())
imgsz=check_img_size(INPUT_HEIGHT,s=stride_size)

#Setup model
if device.type != 'cpu':
    model(torch.zeros(1,3,imgsz,imgsz).to(device).type_as(next(model.parameters()))) #runs it once
old_img_w=old_img_h=imgsz
old_img_b=1




#------------------------------<Main Loop>--------------------------------
while(video.isOpened()):    #Loops for each frame in the video
    ret,frame=video.read() #Reads frame
    
    if ret==True:
    #--------------------<Frame pre-processing>---------------------------
        left_restructured,right_restructured=process_frame(frame)
        left_restructured=letterbox(left_restructured,(INPUT_HEIGHT,INPUT_WIDTH),stride=stride_size)
        right_restructured=letterbox(right_restructured,(INPUT_HEIGHT,INPUT_WIDTH),stride=stride_size)

        #Convert the image
        left_restructured=left_restructured[:,:,::-1].transpose(2,0,1)
        right_restructured=right_restructured[:,:,::-1].transpose(2,0,1)

        left_restructured=np.ascontiguousarray(left_restructured)
        right_restructured=np.ascontiguousarray(right_restructured)


    #-------------------------<Running Inference>-----------------------
        pred,old_img_b,old_img_h,old_img_w=detect(left_restructured,model,device,imgsz,old_img_b,old_img_h,old_img_w)

        #cv2.imshow('Frame',left_restructured)
        #cv2.waitKey(0)

    else:
        print("Frame Not Read Correctly")
        break
