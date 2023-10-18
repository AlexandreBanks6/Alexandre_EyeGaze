
'''
Date:August 20th, 2023
Author: Alexandre Banks

venv: 'eyecorner_main_venv'
'''

#----------------------<Imports>----------------------------
import cv2
from cv2 import dnn
from collections import namedtuple
from dataclasses import dataclass
from dataclasses import field,fields
import math
import time
from itertools import count
from multiprocessing import Process
import multiprocessing



import os
import numpy as np
import re

import torch
import torch.backends.cudnn as cudnn
from numpy import random

from models.experimental import attempt_load
from utils.general import check_img_size, non_max_suppression
from utils.torch_utils import select_device



#---------------<Initializing Variables>--------------------

data_root="E:/Alexandre_EyeGazeProject_Extra/eyecorner_userstudy2_converted"
YOLOV7_MODEL_PATH='resources/eyegaze_model_new_aug18.pt'
PI=3.1415926535

INPUT_IMAGE_SIZE=640
CONF_THRESH=0.5
IOU_THRESH=0.5

@dataclass
class bbxs:
    right_eye_inner: list=field(default_factory=list)
    right_eye_outer: list=field(default_factory=list)
    left_eye_outer: list=field(default_factory=list)
    left_eye_inner: list=field(default_factory=list)
    right_eye: list=field(default_factory=list)
    left_eye: list=field(default_factory=list)
    
@dataclass
class eye_imgs:
    right_eye: list=field(default_factory=list)
    left_eye: list=field(default_factory=list)
    
@dataclass
class eye_corners:
    
    #Current corner point 
    right_eye_inner: list=field(default_factory=list)
    right_eye_outer: list=field(default_factory=list)
    left_eye_outer: list=field(default_factory=list)
    left_eye_inner: list=field(default_factory=list)

@dataclass
class eye_corner_list:
    #List of contour points for filtering
    right_eye_inner: list=field(default_factory=list)
    right_eye_outer: list=field(default_factory=list)
    left_eye_outer: list=field(default_factory=list)
    left_eye_inner: list=field(default_factory=list)
    #bbox_centers: list=field(default_factory=list)




#----------------Global Variables------------------------------
#List with the eye corners
LIST_EYE_CORNERS=eye_corner_list()
LIST_EYE_CORNERS.right_eye_inner=[]
LIST_EYE_CORNERS.right_eye_outer=[]
LIST_EYE_CORNERS.right_eye_outer=[]
LIST_EYE_CORNERS.right_eye_inner=[]
#LIST_EYE_CORNERS.bbox_centers=[]

#List that contains the results from the previous eye corner
LIST_LAST_EYE_CORNER=[]


NON_DETECT_RIGHT_INNER=0
NON_DETECT_RIGHT_OUTER=0
NON_DETECT_LEFT_INNER=0
NON_DETECT_LEFT_OUTER=0




#ADd this many pixels to height and width of bounding boxes of eyes
ADD_HEIGHT_EYE=20
ADD_WIDTH_EYE=8 

ADD_WIDTH_CORNER=8

FILTER_SIZE=45 #Filter for gaussian blur

THRESHOLD_FILTER_SIZE=15
THRESHOLD_FILTER_CONSTANT=1

#Contour Filtering Params:
DILATION_SIZE=5
EROSION_SIZE=4 
EROSION_ELEMENT=cv2.getStructuringElement(cv2.MORPH_RECT,(EROSION_SIZE,EROSION_SIZE))
DILATION_ELEMENT=cv2.getStructuringElement(cv2.MORPH_RECT,(DILATION_SIZE,DILATION_SIZE))

CONTOUR_THRESH=9 #Only keeping top 9 largest contours

MAX_NUM_PERCENTAGE=0.1 #This sets how many maximum points we will evaluate as candidate eye corners
STEP_SIZE=2 #Step size of curvature detector


MAV_LENGTH=15
OUTLIER_MULT=1.01
#OUTLIER_MULT_BBOX=1.05
#----------------<Function Definitions>---------------------


#Frame Pre-processing Function
def preprocessFrame(frame):
    #Finds the ratio between the cropped image and the network input size
    dw=INPUT_IMAGE_SIZE/frame.shape[1]
    dh=INPUT_IMAGE_SIZE/frame.shape[0]
    
    
    resized_img=cv2.resize(frame,[INPUT_IMAGE_SIZE,INPUT_IMAGE_SIZE],interpolation=cv2.INTER_LINEAR)
              
    #Convert the image from BGR to RGB
    resized_img=resized_img[:,:,::-1].transpose(2,0,1)

    resized_img=np.ascontiguousarray(resized_img)

    return resized_img,dw,dh





#YOLOv7 detection
def detect(frame,model,device,old_img_b,old_img_h,old_img_w):
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
    
    with torch.no_grad():   # Calculating gradients would cause a GPU memory leak
        pred = model(img, augment=False)[0]
        
    #Apply nms
    pred=non_max_suppression(pred,CONF_THRESH,IOU_THRESH)
    #return_dict[pred]=pred
    #return_dict[old_img_b]=old_img_b
    #return_dict[old_img_h]=old_img_h
    #return_dict[old_img_w]=old_img_w
    return pred,old_img_b,old_img_h,old_img_w





#Process the yolov7 detection
def process_detections(pred,dw,dh):
    bounding_boxes=bbxs() #Creates class object
    found=False #Boolean to see if we found a detection
    for i,det in enumerate(pred): #Loops through all detections in the prediction results
        if len(det):    #Enters if the detection exists
            for *xyxy, conf, cls in reversed(det): #Loops for the bounding boxes
                xyxy_list=torch.tensor(xyxy).view(1,4).tolist()[0] #Converts tensor detection to list\
                
                #Scaling the detection                
                xyxy_list[0]=xyxy_list[0]*(1/dw)
                xyxy_list[1]=xyxy_list[1]*(1/dh)
                xyxy_list[2]=xyxy_list[2]*(1/dw)
                xyxy_list[3]=xyxy_list[3]*(1/dh)  
                
                #Converting the list to integer             
                xyxy_list=[int(round(item)) for item in xyxy_list]
                
                #adding results to bounding box class
                class_type=int(cls)
                if class_type==0:
                    #xyxy_list[0]= xyxy_list[0]-ADD_WIDTH_CORNER
                    #xyxy_list[2]= xyxy_list[2]+ADD_WIDTH_CORNER
                    #print('Right Eye Inner:',xyxy_list)
                    if ((xyxy_list[0]+xyxy_list[2])/2)>=640:
                        if len(bounding_boxes.right_eye_inner):#We have a previous bounding box detection
                            if ((xyxy_list[0]+xyxy_list[2])/2)>((bounding_boxes.right_eye_inner[0][0]+bounding_boxes.right_eye_inner[0][2])/2):
                                del bounding_boxes.right_eye_inner[0]
                                bounding_boxes.right_eye_inner.append(xyxy_list)
                                found=True
                            else:
                                continue                                
                        else: #First detection
                            bounding_boxes.right_eye_inner.append(xyxy_list)
                            found=True
                    else:
                        continue
                elif class_type==1:
                    
                    #print('Right Eye Outer:',xyxy_list)
                    #xyxy_list[2]= xyxy_list[2]-ADD_WIDTH_CORNER
                    if ((xyxy_list[0]+xyxy_list[2])/2)>=640:
                        if len(bounding_boxes.right_eye_outer):#We have a previous bounding box detection
                            if ((xyxy_list[0]+xyxy_list[2])/2)<((bounding_boxes.right_eye_outer[0][0]+bounding_boxes.right_eye_outer[0][2])/2):
                                del bounding_boxes.right_eye_outer[0]
                                bounding_boxes.right_eye_outer.append(xyxy_list)
                                found=True
                            else:
                                continue                                
                        else: #First detection
                            bounding_boxes.right_eye_outer.append(xyxy_list)
                            found=True
                    else:
                        continue
                elif class_type==2:
                    #xyxy_list[0]= xyxy_list[0]+ADD_WIDTH_CORNER
                    
                    #print('Left Eye Outer:',xyxy_list)
                    if ((xyxy_list[0]+xyxy_list[2])/2)<=640:
                        if len(bounding_boxes.left_eye_outer):#We have a previous bounding box detection
                            if ((xyxy_list[0]+xyxy_list[2])/2)>((bounding_boxes.left_eye_outer[0][0]+bounding_boxes.left_eye_outer[0][2])/2):
                                del bounding_boxes.left_eye_outer[0]
                                bounding_boxes.left_eye_outer.append(xyxy_list)
                                found=True
                            else:
                                continue                                
                        else: #First detection
                            bounding_boxes.left_eye_outer.append(xyxy_list)
                            found=True
                    else:
                        continue
                elif class_type==3:
                    #xyxy_list[0]= xyxy_list[0]-ADD_WIDTH_CORNER
                    #xyxy_list[2]= xyxy_list[2]+ADD_WIDTH_CORNER
                    #print('Left Eye Inner:',xyxy_list)
                    if ((xyxy_list[0]+xyxy_list[2])/2)<=640: 
                        if len(bounding_boxes.left_eye_inner):#We have a previous bounding box detection
                            if ((xyxy_list[0]+xyxy_list[2])/2)<((bounding_boxes.left_eye_inner[0][0]+bounding_boxes.left_eye_inner[0][2])/2):
                                del bounding_boxes.left_eye_inner[0]
                                bounding_boxes.left_eye_inner.append(xyxy_list)
                                found=True
                            else:
                                continue                                
                        else: #First detection
                            bounding_boxes.left_eye_inner.append(xyxy_list)
                            found=True
                    else:
                        continue
                elif class_type==4:
                    if ((xyxy_list[0]+xyxy_list[2])/2)>=640:   
                        xyxy_list[0]= xyxy_list[0]-ADD_WIDTH_EYE
                        xyxy_list[2]= xyxy_list[2]+ADD_WIDTH_EYE
                        xyxy_list[1]= xyxy_list[1]-ADD_HEIGHT_EYE
                        xyxy_list[3]= xyxy_list[3]+ADD_HEIGHT_EYE
                        bounding_boxes.right_eye.append(xyxy_list)
                        found=True
                    else:
                        continue
                elif class_type==5:
                    if ((xyxy_list[0]+xyxy_list[2])/2)<=640:   
                        xyxy_list[0]= xyxy_list[0]-ADD_WIDTH_EYE
                        xyxy_list[2]= xyxy_list[2]+ADD_WIDTH_EYE
                        xyxy_list[1]= xyxy_list[1]-ADD_HEIGHT_EYE
                        xyxy_list[3]= xyxy_list[3]+ADD_HEIGHT_EYE
                        bounding_boxes.left_eye.append(xyxy_list)
                        found=True
                    else:
                        continue
                else:
                    continue
    return bounding_boxes,found                        
    
    
    
    

#Function to display bounding boxes
def showBoxes(frame,bboxes=bbxs()):
    new_frame=np.copy(frame)
    for field in fields(bboxes):
        field_name=field.name
        if field_name=='right_eye_outer': #Red
            color=(0,0,255)
        elif field_name=='right_eye_inner': #Green
            color=(0,255,0)
        elif field_name=='left_eye_inner': #Blue
            color=(255,0,0)
        elif field_name=='left_eye_outer': #White
            color=(255,255,255)
        else:   #Purple
            color=(255,0,255)
            
        bound_box=getattr(bboxes,field_name) #Returns list of bounding boxes
        if len(bound_box):
            top_left=(bound_box[0][0],bound_box[0][1])
            bottom_right=(bound_box[0][2],bound_box[0][3])
            cv2.rectangle(new_frame,top_left,bottom_right,color=color,thickness=2)
            cv2.imshow('Bounding Boxes',new_frame)
            cv2.waitKey(2)
      
      
      
        

#Used to crop out regions defined by a bounding box
def cropImage(frame,bbox):
    cropped=frame[bbox[0][1]:bbox[0][3],bbox[0][0]:bbox[0][2]]
    return cropped




#Used to crop out the eyes from rest of frame
def cropBothEyes(frame,bboxes):
    eye_found=False
    eyes=eye_imgs() #Creates empty class of eye images
    for field in fields(bboxes):
        field_name=field.name
        
        if (field_name=='right_eye') or (field_name=='left_eye'):
            bound_box=getattr(bboxes,field_name) #Returns list of bounding boxes
            if len(bound_box):
                eye_found=True #Sets bool to true
                cropped_eye=cropImage(frame,bound_box)
                #cropped_eye.convertTo(cropped_eye,cv2.CV_8UC1)
                if field_name=='right_eye':
                    eyes.right_eye.append(cropped_eye)
                else:
                    eyes.left_eye.append(cropped_eye)
            #cv2.imshow('cropped eye',cropped_eye)
            #cv2.waitKey(0)
    
    return eyes,eye_found
         
         
#------------------~Corner Finding Functions~------------------
#Sorts the contours
def findLargestContours(contours):
    num_contours=len(contours)
    cont_lengths=[]

    if num_contours>CONTOUR_THRESH: #Filter out small contours if we have more than 5
        for cnt in contours:
            cont_lengths.append(cv2.arcLength(cnt,True)) #Finds the length of each contour
        sorted_lengths=sorted(cont_lengths,reverse=True) #Sorts list from highes to lowest
        sorted_lengths=sorted_lengths[0:CONTOUR_THRESH] #Takes 5 top values of the list
        sorted_lengths=set(sorted_lengths)
        indx_list=[i for i, val in enumerate(cont_lengths) if val in sorted_lengths]
        contours=[val for i,val in enumerate(contours) if i in indx_list]
    return contours




#Finds the contours for a cropped eye image
def findEyeContours(frame):
    frame_gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    frame_blur=cv2.GaussianBlur(frame_gray,(FILTER_SIZE,FILTER_SIZE), cv2.BORDER_DEFAULT) #FIlters image
    #cv2.imshow('blurred',frame_blur)
    #cv2.waitKey(0)

    frame_thresholded=cv2.adaptiveThreshold(frame_blur,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,THRESHOLD_FILTER_SIZE,THRESHOLD_FILTER_CONSTANT)
    #cv2.imshow('frame thresholded',frame_thresholded)
    #cv2.waitKey(0)

    #!!!!Maybe Switch This Order!!!!
    frame_eroded=cv2.erode(frame_thresholded,EROSION_ELEMENT)
    frame_dilated=cv2.dilate(frame_eroded,DILATION_ELEMENT)
    
    #cv2.imshow('frame eroded',frame_dilated)
    #cv2.waitKey(0)
    
    contours, _ = cv2.findContours(frame_dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE) 
    contours=findLargestContours(contours) #Finds top 5 largest contours
    
    #frame_colour=np.copy(frame)
    
    #Fit contours using PolyDP
    #for cnt in contours:
        #color_rand=list(np.random.random(size=3)*256)
        #cv2.drawContours(frame_colour,[cnt],-1,color=(0,0,255),thickness=1)
        #cv2.imshow('contours',frame_colour)
        #cv2.waitKey(0)
    return contours


#Finds contours inside the bounding box, if none are in bounding box, then we set the flag to true
def filterCountours(contours,bbox,bbox_eye,frame):
    #Takes in countours and bbox
    is_empty=True
    bbx_x0=bbox[0][0]
    bbx_x1=bbox[0][2]
    bbx_y0=bbox[0][1]
    bbx_y1=bbox[0][3]
    
    bbx_eye_x0=bbox_eye[0][0]
    bbx_eye_y0=bbox_eye[0][1]
    
    bbx_center=[(bbx_x0+bbx_x1)/2,(bbx_y0+bbx_y1)/2]
    contours_new=[]
    for cnt in contours:
        temp_list=[]
        add_contour=False
        for val in cnt:
            if (val[0][0]+bbx_eye_x0>=bbx_x0) and (val[0][0]+bbx_eye_x0<=bbx_x1) and (val[0][1]+bbx_eye_y0>=bbx_y0) and (val[0][1]+bbx_eye_y0<=bbx_y1):
                temp_list.append(val)
                is_empty=False
                add_contour=True
        if add_contour==True:
            contours_new.append(temp_list)
    
    #past_img=cv2.imread('past_image.jpg')
    #if past_img is None:
        frame_colour=np.copy(frame)
    #else:
     #   frame_colour=past_img

    #for cnt in contours_new:
     #   for val in cnt:
      #      frame_colour=cv2.circle(frame_colour,val[0],1,(0,0,255),1)
    #cv2.imwrite('past_image.jpg',frame_colour)
    #cv2.imshow('contours',frame_colour)
    #cv2.waitKey(0)
    return contours_new,bbx_center,is_empty





#Rejects outlier corners in the retained list        
def outlierReject(corner_list):
    #Finds mean and standard deviation of list in both the x- and y- directions
    check=True
    x_list=[coord[0] for coord in corner_list]
    y_list=[coord[1] for coord in corner_list]

    std_x=np.std(x_list)
    std_y=np.std(y_list)
    avg_x=np.average(x_list)
    avg_y=np.average(y_list)

    new_list_x=[]
    new_list_y=[]
    smallest_diff=10000
    #smallest_coord=[]
    for coord in corner_list:
        dif_x=abs(coord[0]-avg_x)
        dif_y=abs(coord[1]-avg_y)
        if ((dif_x+dif_y)/2)<smallest_diff:
            smallest_diff=(dif_x+dif_y)/2
            smallest_coord=coord
        if (dif_x<=OUTLIER_MULT*std_x) and (dif_y<=OUTLIER_MULT*std_y):
            new_list_x.append(coord[0])
            new_list_y.append(coord[1])
    if not len(new_list_x):
        new_list_x.append(smallest_coord[0])
        new_list_y.append(smallest_coord[1])
        check=False
    return new_list_x,new_list_y,check



#Filters eye corner points
def mavFilter(corner_list):
    #Performs filtering and outlier rejection
    list_len=len(corner_list)
    
    if MAV_LENGTH<=list_len: #Uses a MAV window of size MAV_LENGTH
        sliced_list=corner_list[-MAV_LENGTH:]
        new_list_x,new_list_y,check=outlierReject(sliced_list)
        avg_x=sum(new_list_x)/len(new_list_x) #Finds average eye corner
        avg_y=sum(new_list_y)/len(new_list_y)

    else: #We haven't filled list yet to size MAV_LENGTH
        sliced_list=corner_list[-list_len:]
        new_list_x,new_list_y,check=outlierReject(sliced_list)
        avg_x=sum(new_list_x)/len(new_list_x) #Finds average eye corner
        avg_y=sum(new_list_y)/len(new_list_y)
    avg_corner=[avg_x,avg_y]
    return avg_corner,sliced_list,check

#Gets the curvature values of the contour points
def getCurvature(contour_points,step):
    contour_points=list(contour_points)
    #This Function takes a single contour

    num_contour_points=len(contour_points)
    vec_curvature=[0]*num_contour_points

    #Checks that the number of contour points is greater than the step size
    if(num_contour_points<step):
        return vec_curvature
    
    fronToBack=contour_points[1]-contour_points[1]
    isClosed=int(max(abs(fronToBack[0][0]),abs(fronToBack[0][1])))<=1

    #Init Variables
    fplus=[0,0]
    fminus=[0,0]
    f2plus=[0,0]
    f2minus=[0,0]
    f1stDerivative=[0,0]
    f2ndDerivative=[0,0]

    for i in range(num_contour_points): #Loops for the number of contour points
        fpos=list(contour_points[i])

        if isClosed: #Closed Curve
            iminus=i-step
            iplus=i+step
            fminus=list(contour_points[iminus+num_contour_points if (iminus<0) else iminus])
            fplus=list(contour_points[iplus-num_contour_points if (iplus>=num_contour_points) else iplus])

            #Derivative Approximations
            f1stDerivative[0]=(fplus[0][0]-fminus[0][0])/(2*step) #0=x direction
            f1stDerivative[1]=(fplus[0][1]-fminus[0][1])/(2*step) #1=y direction
            f2ndDerivative[0]=(fplus[0][0]-2*fpos[0][0]+fminus[0][0])/(step**2) #0=x direction
            f2ndDerivative[1]=(fplus[0][1]-2*fpos[0][1]+fminus[0][1])/(step**2) #1=y direction
        else: #Open Curve
            if ((i-step)<0) and ((i+2*step)<num_contour_points): #We are at start of curve
                iplus=i+step
                i2plus=i+2*step
                fplus=list(contour_points[iplus])
                f2plus=list(contour_points[i2plus])
                
                
                #One Sided Derivative Approximations (forward)
                f1stDerivative[0]=(-f2plus[0][0]+4*fplus[0][0]-3*fpos[0][0])/(2*step)
                f1stDerivative[1]=(-f2plus[0][1]+4*fplus[0][1]-3*fpos[0][1])/(2*step)

                f2ndDerivative[0]=(f2plus[0][0]-2*fplus[0][0]+fpos[0][0])/(step**2)
                f2ndDerivative[1]=(f2plus[0][1]-2*fplus[0][1]+fpos[0][1])/(step**2)

            elif ((i+step)>=num_contour_points) and ((i-2*step)>=0): #End of curve
                iminus=i-step
                i2minus=i-2*step
                fminus=list(contour_points[iminus])
                f2minus=list(contour_points[i2minus])

                #One Sided Derivative Approximations (backward)
                f1stDerivative[0]=(3*fpos[0][0]-4*fminus[0][0]+f2minus[0][0])/(2*step)
                f1stDerivative[1]=(3*fpos[0][1]-4*fminus[0][1]+f2minus[0][1])/(2*step)

                f2ndDerivative[0]=(fpos[0][0]-2*fminus[0][0]+f2minus[0][0])/(step**2)
                f2ndDerivative[1]=(fpos[0][1]-2*fminus[0][1]+f2minus[0][1])/(step**2)
            elif ((i+step)<num_contour_points) and ((i-step)>=0):  #Middle of curve
                iminus=i-step
                iplus=i+step
                fminus=list(contour_points[iminus+num_contour_points if (iminus<0) else iminus])
                fplus=list(contour_points[iplus-num_contour_points if (iplus>=num_contour_points) else iplus])

                #Derivative Approximations
                f1stDerivative[0]=(fplus[0][0]-fminus[0][0])/(2*step) #0=x direction
                f1stDerivative[1]=(fplus[0][1]-fminus[0][1])/(2*step) #1=y direction
                f2ndDerivative[0]=(fplus[0][0]-2*fpos[0][0]+fminus[0][0])/(step**2) #0=x direction
                f2ndDerivative[1]=(fplus[0][1]-2*fpos[0][1]+fminus[0][1])/(step**2) #1=y direction
            else:
                return vec_curvature
        #Calculating Curvature
        divisor=f1stDerivative[0]**2+f1stDerivative[1]**2
        if abs(divisor)>10e-6:
            curvature2D=abs(f2ndDerivative[1]*f1stDerivative[0]-f2ndDerivative[0]*f1stDerivative[1])/(math.sqrt(divisor)**3)
        else:
            curvature2D=float('inf')
        vec_curvature[i]=curvature2D
    return vec_curvature

def maxPoints(contours): 
    #This function returns the max threshold 'MAX_NUM_PERCENTAGE' amount of points from the list of contours for the eye corner
        
    number_of_max_points=0
    for cnt in contours:
        number_of_max_points=number_of_max_points+len(cnt)
    number_of_max_points=int(math.ceil(number_of_max_points*MAX_NUM_PERCENTAGE))+1
    max_mags=[] #Values of curvature point
    max_points=[] #Coresponding indices of curvature points
    
    for cnt in contours: #Loops for all the contours for a given eye corner
        contour_mag=getCurvature(cnt,STEP_SIZE) #Gets the curvature value for a given contour
        for i in range(len(contour_mag)): #Loops for all the curvature values
            if contour_mag[i]=='inf':
                continue
            max_mags.append(contour_mag[i])
            max_points.append(cnt[i])
    #Sort max points from highest to lowest
    sorted_max_mags=sorted(max_mags,reverse=True)
    sorted_max_mags=sorted_max_mags[0:number_of_max_points] #Takes top threshold number of maximum points
    sorted_max_mags_set=set(sorted_max_mags)
    indx_list=[i for i,val in enumerate(max_mags) if val in sorted_max_mags_set]
    corner_candidates=[val for i,val in enumerate(max_points) if i in indx_list]
    return corner_candidates

def filterCornerCandidates(corner_candidates,bbx_center,bbox_eye,contours):
    #Function that finds corner candidate closest to the bounding box center
    
    diff_vals=[]
    bbx_eye_x0=bbox_eye[0][0]
    bbx_eye_y0=bbox_eye[0][1]
    for candidate in corner_candidates:
        diff_vals.append(math.sqrt((candidate[0][0]+bbx_eye_x0-bbx_center[0])**2+(candidate[0][1]+bbx_eye_y0-bbx_center[1])**2))
    #sorted_diff_vals=sorted(diff_vals) #Sorts the dif vals from smalles to largest
        smallest_val=min(diff_vals)
    smallest_inx=diff_vals.index(smallest_val)
    corner_point=corner_candidates[smallest_inx]
    #frame_colour=np.copy(frame)
    #frame_colour=cv2.circle(frame_colour,corner_point[0],radius=2,color=(0,255,0),thickness=2)
    #cv2.imshow('corner point',frame_colour)
    #cv2.waitKey(0)

    corner_point[0][0]=corner_point[0][0]+bbx_eye_x0
    corner_point[0][1]=corner_point[0][1]+bbx_eye_y0
    corner_point=list(corner_point[0])
    
    return corner_point
    
                        
def singleCornerDetector(eye_image,bbox_corner,bbox_eye,frame,corner_list,left_right):
    #print(eye_image[0].size)
    if len(eye_image): #Extra check that the eye image array is not empty
        if len(bbox_corner) and (not(eye_image[0].size==0)): #Yolov7 detected an eye corner and a bounding box
            contours=findEyeContours(eye_image[0])
            corner_contours,bbx_center,is_empty=filterCountours(contours,bbox_corner,bbox_eye,eye_image[0])
            #print("is empty: ",is_empty)
            if is_empty==True:
                if left_right=='left' and bbx_center[0]<=640:
                    corner_point=bbx_center
                elif left_right=='right' and bbx_center[0]>=640:
                    corner_point=bbx_center
                else:
                    corner_point=[-1,-1] #What we define NaN as

            else:
                if left_right=='left' and bbx_center[0]<=640:
                    corner_candidates=maxPoints(corner_contours)
                    corner_point=filterCornerCandidates(corner_candidates,bbx_center,bbox_eye,corner_contours)
                elif left_right=='right' and bbx_center[0]>=640:
                    corner_candidates=maxPoints(corner_contours)
                    corner_point=filterCornerCandidates(corner_candidates,bbx_center,bbox_eye,corner_contours)
                else:
                    corner_point=[-1,-1]
                    
            
            #corner_point_new=[int(elem) for elem in corner_point]
            #frame_colour=np.copy(frame)
            #frame_colour=cv2.cvtColor(frame_colour,cv2.COLOR_GRAY2BGR)
            #frame_colour=cv2.circle(frame_colour,corner_point_new,radius=3,color=(0,0,255),thickness=1)
            #cv2.imshow('corner',frame_colour)
            #cv2.waitKey(0)
        elif len(bbox_corner): #We only have a bounding box around the corner
            contours=findEyeContours(frame)
            corner_contours,bbx_center,is_empty=filterCountours(contours,bbox_corner,[[0,0]],frame)
            if is_empty==True:
                if left_right=='left' and bbx_center[0]<=640:
                    corner_point=bbx_center
                elif left_right=='right' and bbx_center[0]>=640:
                    corner_point=bbx_center
                else:
                    corner_point=[-1,-1] #What we define NaN as

            else:
                if left_right=='left' and bbx_center[0]<=640:
                    corner_candidates=maxPoints(corner_contours)
                    corner_point=filterCornerCandidates(corner_candidates,bbx_center,[[0,0]],corner_contours)
                elif left_right=='right' and bbx_center[0]>=640:
                    corner_candidates=maxPoints(corner_contours)
                    corner_point=filterCornerCandidates(corner_candidates,bbx_center,[[0,0]],corner_contours)
                else:
                    corner_point=[-1,-1]
        else:
            corner_point=[-1,-1] #What we define NaN as
    elif len(bbox_corner): #We only have a bounding box around the corner:
        contours=findEyeContours(frame)
        corner_contours,bbx_center,is_empty=filterCountours(contours,bbox_corner,[[0,0]],frame)
        if is_empty==True:
            if left_right=='left' and bbx_center[0]<=640:
                corner_point=bbx_center
            elif left_right=='right' and bbx_center[0]>=640:
                corner_point=bbx_center
            else:
                corner_point=[-1,-1] #What we define NaN as

        else:
            if left_right=='left' and bbx_center[0]<=640:
                corner_candidates=maxPoints(corner_contours)
                corner_point=filterCornerCandidates(corner_candidates,bbx_center,[[0,0]],corner_contours)
            elif left_right=='right' and bbx_center[0]>=640:
                corner_candidates=maxPoints(corner_contours)
                corner_point=filterCornerCandidates(corner_candidates,bbx_center,[[0,0]],corner_contours)
            else:
                corner_point=[-1,-1]
            
    else:
        corner_point=[-1,-1]
        
    return corner_point 
    
    '''
    
    
    global LIST_EYE_CORNERS
    if len(bbox_corner) and len(eye_image): #We have both a bounding box and an eye image
        contours=findEyeContours(eye_image[0])
        
        if len(corner_list)>=1: #We have more than one detection
        #Check that bounding box passes rejection criteria
            bbx_x0=bbox_corner[0][0]
            bbx_x1=bbox_corner[0][2]
            bbx_y0=bbox_corner[0][1]
            bbx_y1=bbox_corner[0][3]       
            bbx_center=[(bbx_x0+bbx_x1)/2,(bbx_y0+bbx_y1)/2]
            
            x_list=[coord[0] for coord in corner_list]
            y_list=[coord[1] for coord in corner_list]

            std_x=np.std(x_list)
            std_y=np.std(y_list)
            avg_x=np.average(x_list)
            avg_y=np.average(y_list)
            
            dif_x=abs(bbx_center[0]-avg_x)
            dif_y=abs(bbx_center[1]-avg_y)
            
            if (dif_x<=OUTLIER_MULT_BBOX*std_x) and (dif_y<=OUTLIER_MULT_BBOX*std_y): #Bounding box is valid
                corner_contours,_,is_empty=filterCountours(contours,bbox_corner,bbox_eye,eye_image[0])
                LIST_EYE_CORNERS.bbox_centers.append(bbx_center)
                if is_empty==True:
                    corner_point=bbx_center
                else:
                    corner_candidates=maxPoints(corner_contours)
                    corner_point=filterCornerCandidates(corner_candidates,bbx_center,bbox_eye,corner_contours)
                    
            else: #Bounding box is not valid, and we do corner detection on pure contours
                #bbx_eye_x0=bbox_eye[0][0]
                #bbx_eye_y0=bbox_eye[0][1]
                corner_candidates=maxPoints(contours)
                past_corn,LIST_EYE_CORNERS.bbox_centers,_,=mavFilter(LIST_EYE_CORNERS.bbox_centers)
                corner_point=filterCornerCandidates(corner_candidates,past_corn,bbox_eye,contours)
                #corner_point=corner_candidates[0] #Take point with highest magnitude
                #corner_point[0][0]=corner_point[0][0]+bbx_eye_x0
                #corner_point[0][1]=corner_point[0][1]+bbx_eye_y0
                #corner_point=list(corner_point[0])
        else:
            corner_contours,bbx_center,is_empty=filterCountours(contours,bbox_corner,bbox_eye,eye_image[0])
            LIST_EYE_CORNERS.bbox_centers.append(bbx_center)
            if is_empty==True:
                corner_point=bbx_center
            else:
                corner_candidates=maxPoints(corner_contours)
                corner_point=filterCornerCandidates(corner_candidates,bbx_center,bbox_eye,corner_contours)
    elif len(bbox_corner): #We have a bounding box around the corner, but not around the eye
        bbx_x0=bbox_corner[0][0]
        bbx_x1=bbox_corner[0][2]
        bbx_y0=bbox_corner[0][1]
        bbx_y1=bbox_corner[0][3]       
        bbx_center=[(bbx_x0+bbx_x1)/2,(bbx_y0+bbx_y1)/2]
        
        x_list=[coord[0] for coord in corner_list]
        y_list=[coord[1] for coord in corner_list]

        std_x=np.std(x_list)
        std_y=np.std(y_list)
        avg_x=np.average(x_list)
        avg_y=np.average(y_list)
        
        dif_x=abs(bbx_center[0]-avg_x)
        dif_y=abs(bbx_center[1]-avg_y)
        
        if (dif_x<=OUTLIER_MULT_BBOX*std_x) and (dif_y<=OUTLIER_MULT_BBOX*std_y): #Bounding box is valid
            corner_point=bbx_center
            LIST_EYE_CORNERS.bbox_centers.append(bbx_center)
        else:
            corner_point=[-1,-1]
    else:
        corner_point=[-1,-1]
    '''
            
            
        
            

    
    '''

    #print('contours: ',contours)
    if len(bbox_corner) and len(eye_image): #Continues only if yolov7 detected an eye corner
        contours=findEyeContours(eye_image[0])
        corner_contours,bbx_center,is_empty=filterCountours(contours,bbox_corner,bbox_eye,eye_image[0])
        #print("is empty: ",is_empty)
        if is_empty==True:
            corner_point=bbx_center
        else:
            corner_candidates=maxPoints(corner_contours)
            corner_point=filterCornerCandidates(corner_candidates,bbx_center,bbox_eye,corner_contours)
        
        #corner_point_new=[int(elem) for elem in corner_point]
        #frame_colour=np.copy(frame)
        #frame_colour=cv2.cvtColor(frame_colour,cv2.COLOR_GRAY2BGR)
        #frame_colour=cv2.circle(frame_colour,corner_point_new,radius=3,color=(0,0,255),thickness=1)
        #cv2.imshow('corner',frame_colour)
        #cv2.waitKey(0)
          
    else:
        corner_point=[-1,-1] #What we define NaN as
        
        
    '''
    

    


def findCorners(frame,eye_images=eye_imgs(),bounding_boxes=bbxs()):
    #Returns the four corner coordinates (if found) of each eye
    eye_corner_results=eye_corners()
    global LIST_EYE_CORNERS
    global NON_DETECT_RIGHT_OUTER
    global NON_DETECT_RIGHT_INNER
    global NON_DETECT_LEFT_OUTER
    global NON_DETECT_LEFT_INNER
    global LIST_LAST_EYE_CORNER
    
    for field in fields(eye_images):
        field_name=field.name
        #print(field_name)        
        if field_name=='right_eye': #We have a right eye
            eye_corner_results.right_eye_inner.append(singleCornerDetector(eye_images.right_eye,bounding_boxes.right_eye_inner,bounding_boxes.right_eye,frame,LIST_EYE_CORNERS.right_eye_inner,'right'))
            eye_corner_results.right_eye_outer.append(singleCornerDetector(eye_images.right_eye,bounding_boxes.right_eye_outer,bounding_boxes.right_eye,frame,LIST_EYE_CORNERS.right_eye_outer,'right'))
       
        elif field_name=='left_eye':
            eye_corner_results.left_eye_inner.append(singleCornerDetector(eye_images.left_eye,bounding_boxes.left_eye_inner,bounding_boxes.left_eye,frame,LIST_EYE_CORNERS.left_eye_inner,'left'))
            eye_corner_results.left_eye_outer.append(singleCornerDetector(eye_images.left_eye,bounding_boxes.left_eye_outer,bounding_boxes.left_eye,frame,LIST_EYE_CORNERS.left_eye_outer,'left'))
            #eye_corner_results.left_eye_outer.append([-1,-1])

    #We put the constraint that right_outer_x is > right_inner_x and left_inner_x>left_outer_x with this function:
    eye_corner_results=constrainCorners(eye_corner_results)
    LIST_LAST_EYE_CORNER=[eye_corner_results.left_eye_outer[0],eye_corner_results.left_eye_inner[0],eye_corner_results.right_eye_inner[0],eye_corner_results.right_eye_outer[0]]

    #Now that we have found all current eye corners, we filter them, and check for lost eye corners, then we return the eye_corner_results dataclass
    corners_new=eye_corners()

    for corner in fields(eye_corner_results):
        corner_name=corner.name
        corner_point=getattr(eye_corner_results,corner_name)[0]
        #print('cornerpoint length',len(corner_point))
        if (not len(corner_point)) or (corner_point==[-1,-1]): #We don't have a current detection and must determine whether we filter or return NaN
            if corner_name=='right_eye_inner':
                NON_DETECT_RIGHT_INNER+=1
                if NON_DETECT_RIGHT_INNER>=MAV_LENGTH:
                    corners_new.right_eye_inner.append(math.nan)
                elif (not len(LIST_EYE_CORNERS.right_eye_inner)):
                    corners_new.right_eye_inner.append(math.nan)
                else:
                    corner_point,LIST_EYE_CORNERS.right_eye_inner,_=mavFilter(LIST_EYE_CORNERS.right_eye_inner)
                    corners_new.right_eye_inner.append(corner_point)
                    
            if corner_name=='right_eye_outer':
                NON_DETECT_RIGHT_OUTER+=1
                if NON_DETECT_RIGHT_OUTER>=MAV_LENGTH:
                    corners_new.right_eye_outer.append(math.nan)
                elif (not len(LIST_EYE_CORNERS.right_eye_outer)):
                    corners_new.right_eye_outer.append(math.nan)
                else:
                    corner_point,LIST_EYE_CORNERS.right_eye_outer,_=mavFilter(LIST_EYE_CORNERS.right_eye_outer)
                    corners_new.right_eye_outer.append(corner_point)
                    
            if corner_name=='left_eye_inner':
                NON_DETECT_LEFT_INNER+=1
                if NON_DETECT_LEFT_INNER>=MAV_LENGTH:
                    corners_new.left_eye_inner.append(math.nan)
                elif (not len(LIST_EYE_CORNERS.left_eye_inner)):
                    corners_new.left_eye_inner.append(math.nan)
                else:
                    corner_point,LIST_EYE_CORNERS.left_eye_inner,_=mavFilter(LIST_EYE_CORNERS.left_eye_inner)
                    corners_new.left_eye_inner.append(corner_point)
                    
            if corner_name=='left_eye_outer':
                NON_DETECT_LEFT_OUTER+=1
                if NON_DETECT_LEFT_OUTER>=MAV_LENGTH:
                    corners_new.left_eye_outer.append(math.nan)
                elif (not len(LIST_EYE_CORNERS.left_eye_outer)):
                    corners_new.left_eye_outer.append(math.nan)
                else:
                    corner_point,LIST_EYE_CORNERS.left_eye_outer,_=mavFilter(LIST_EYE_CORNERS.left_eye_outer)
                    corners_new.left_eye_outer.append(corner_point)
                
        else: #We have a current detection
            if corner_name=='right_eye_inner':
                LIST_EYE_CORNERS.right_eye_inner.append(corner_point)
                corner_point,LIST_EYE_CORNERS.right_eye_inner,check=mavFilter(LIST_EYE_CORNERS.right_eye_inner)
                
                if check==False:
                    NON_DETECT_RIGHT_INNER+=1
                else:
                    NON_DETECT_RIGHT_INNER=0
                    
                if NON_DETECT_RIGHT_INNER>=MAV_LENGTH:
                    corners_new.right_eye_inner.append(math.nan)
                else:
                    corners_new.right_eye_inner.append(corner_point)
                
            if corner_name=='right_eye_outer':
                LIST_EYE_CORNERS.right_eye_outer.append(corner_point)
                corner_point,LIST_EYE_CORNERS.right_eye_outer,check=mavFilter(LIST_EYE_CORNERS.right_eye_outer)
                if check==False:
                    NON_DETECT_RIGHT_OUTER+=1
                else:
                    NON_DETECT_RIGHT_OUTER=0
                    
                if NON_DETECT_RIGHT_OUTER>=MAV_LENGTH:
                    corners_new.right_eye_outer.append(math.nan)
                else:
                    corners_new.right_eye_outer.append(corner_point)
                    
                    
            if corner_name=='left_eye_inner':
                LIST_EYE_CORNERS.left_eye_inner.append(corner_point)
                corner_point,LIST_EYE_CORNERS.left_eye_inner,check=mavFilter(LIST_EYE_CORNERS.left_eye_inner)
                if check==False:
                    NON_DETECT_LEFT_INNER+=1
                else:
                    NON_DETECT_LEFT_INNER=0
                    
                if NON_DETECT_LEFT_INNER>=MAV_LENGTH:
                    corners_new.left_eye_inner.append(math.nan)
                else:
                    corners_new.left_eye_inner.append(corner_point)
                    
            if corner_name=='left_eye_outer':
                LIST_EYE_CORNERS.left_eye_outer.append(corner_point)
                corner_point,LIST_EYE_CORNERS.left_eye_outer,check=mavFilter(LIST_EYE_CORNERS.left_eye_outer)
                if check==False:
                    NON_DETECT_LEFT_OUTER+=1
                else:
                    NON_DETECT_LEFT_OUTER=0
                    
                if NON_DETECT_LEFT_OUTER>=MAV_LENGTH:
                    corners_new.left_eye_outer.append(math.nan)
                else:
                    corners_new.left_eye_outer.append(corner_point)
    return corners_new

def constrainCorners(eye_corner_results):
    global LIST_LAST_EYE_CORNER
    for corner in fields(eye_corner_results):
        corner_name=corner.name
        corner_point=getattr(eye_corner_results,corner_name)[0]
        if (not len(corner_point)) or (corner_point==[-1,-1]): #We don't have a corner point (it's nan) and continue
            continue
        else: #We have a corner detection

            #Right Outer
            if corner_name=='right_eye_outer': #We do right outer
                if not len(LIST_LAST_EYE_CORNER): #This is the first corner point
                    right_inner=getattr(eye_corner_results,'right_eye_inner')[0]
                    if right_inner[0]<corner_point[0]: #We swap the values
                        old_right_outer=corner_point
                        eye_corner_results.right_eye_outer[0]=right_inner
                        eye_corner_results.right_eye_inner[0]=old_right_outer
                else:   #We had a previous corner point
                 
                    if (not len(getattr(eye_corner_results,'right_eye_inner')[0])) or (getattr(eye_corner_results,'right_eye_inner')[0]==[-1,-1]):
                        #The other eye corner is nan                                                 
                        if (len(LIST_LAST_EYE_CORNER[2]) or (not LIST_LAST_EYE_CORNER[2]==[-1,-1])) or (len(LIST_LAST_EYE_CORNER[3]) or (not LIST_LAST_EYE_CORNER[3]==[-1,-1])):
                             #We had two detections last time
                            dist_to_inner=abs(corner_point[0]-LIST_LAST_EYE_CORNER[2][0]) #Distance in x-direction 
                            dist_to_outer=abs(corner_point[0]-LIST_LAST_EYE_CORNER[3][0])
                            if dist_to_inner<dist_to_outer: #We swap the values
                                old_right_outer=corner_point
                                eye_corner_results.right_eye_outer[0]=getattr(eye_corner_results,'right_eye_inner')[0]
                                eye_corner_results.right_eye_inner[0]=old_right_outer
                    else:
                        #the other eye corner is a number
                        right_inner=getattr(eye_corner_results,'right_eye_inner')[0]
                        if right_inner[0]<corner_point[0]: #We swap the values
                            old_right_outer=corner_point
                            eye_corner_results.right_eye_outer[0]=right_inner
                            eye_corner_results.right_eye_inner[0]=old_right_outer
            
            #Right Inner
            elif corner_name=='right_eye_inner': #We do right inner
                if not len(LIST_LAST_EYE_CORNER): #This is the first corner point
                    right_outer=getattr(eye_corner_results,'right_eye_outer')[0]
                    if corner_point[0]<right_outer[0]: #We swap the values
                        old_right_inner=corner_point
                        eye_corner_results.right_eye_inner[0]=right_outer
                        eye_corner_results.right_eye_outer[0]=old_right_inner
                else:   #We had a previous corner point
                 
                    if (not len(getattr(eye_corner_results,'right_eye_outer')[0])) or (getattr(eye_corner_results,'right_eye_outer')[0]==[-1,-1]):
                        #The other eye corner is nan                                                 
                        if (len(LIST_LAST_EYE_CORNER[2]) or (not LIST_LAST_EYE_CORNER[2]==[-1,-1])) or (len(LIST_LAST_EYE_CORNER[3]) or (not LIST_LAST_EYE_CORNER[3]==[-1,-1])):
                             #We had two detections last time
                            dist_to_inner=abs(corner_point[0]-LIST_LAST_EYE_CORNER[2][0]) #Distance in x-direction 
                            dist_to_outer=abs(corner_point[0]-LIST_LAST_EYE_CORNER[3][0])
                            if dist_to_inner>dist_to_outer: #We swap the values
                                old_right_inner=corner_point
                                eye_corner_results.right_eye_inner[0]=getattr(eye_corner_results,'right_eye_outer')[0]
                                eye_corner_results.right_eye_outer[0]=old_right_inner
                    else:
                        #the other eye corner is a number
                        right_outer=getattr(eye_corner_results,'right_eye_outer')[0]
                        if corner_point[0]<right_outer[0]: #We swap the values
                            old_right_inner=corner_point
                            eye_corner_results.right_eye_inner[0]=right_outer
                            eye_corner_results.right_eye_outer[0]=old_right_inner


            #Left Inner
            if corner_name=='left_eye_inner': #We do left inner
                if not len(LIST_LAST_EYE_CORNER): #This is the first corner point
                    left_outer=getattr(eye_corner_results,'left_eye_outer')[0]
                    if left_outer[0]<corner_point[0]: #We swap the values
                        old_left_inner=corner_point
                        eye_corner_results.left_eye_inner[0]=left_outer
                        eye_corner_results.right_eye_outer[0]=old_left_inner
                else:   #We had a previous corner point
                 
                    if (not len(getattr(eye_corner_results,'left_eye_outer')[0])) or (getattr(eye_corner_results,'left_eye_outer')[0]==[-1,-1]):
                        #The other eye corner is nan                                                 
                        if (len(LIST_LAST_EYE_CORNER[0]) or (not LIST_LAST_EYE_CORNER[0]==[-1,-1])) or (len(LIST_LAST_EYE_CORNER[1]) or (not LIST_LAST_EYE_CORNER[1]==[-1,-1])):
                             #We had two detections last time
                            dist_to_inner=abs(corner_point[0]-LIST_LAST_EYE_CORNER[1][0]) #Distance in x-direction 
                            dist_to_outer=abs(corner_point[0]-LIST_LAST_EYE_CORNER[0][0])
                            if dist_to_inner>dist_to_outer: #We swap the values
                                old_left_inner=corner_point
                                eye_corner_results.left_eye_inner[0]=getattr(eye_corner_results,'left_eye_outer')[0]
                                eye_corner_results.right_eye_outer[0]=old_left_inner
                    else:
                        #the other eye corner is a number
                        left_outer=getattr(eye_corner_results,'left_eye_outer')[0]
                        if left_outer[0]<corner_point[0]: #We swap the values
                            old_left_inner=corner_point
                            eye_corner_results.left_eye_inner[0]=left_outer
                            eye_corner_results.right_eye_outer[0]=old_left_inner
            
            #Left Outer
            elif corner_name=='left_eye_outer': #We do left outer
                if not len(LIST_LAST_EYE_CORNER): #This is the first corner point
                    left_inner=getattr(eye_corner_results,'left_eye_inner')[0]
                    if corner_point[0]<left_inner[0]: #We swap the values
                        old_left_outer=corner_point
                        eye_corner_results.left_eye_outer[0]=left_inner
                        eye_corner_results.left_eye_inner[0]=old_left_outer
                else:   #We had a previous corner point
                 
                    if (not len(getattr(eye_corner_results,'left_eye_inner')[0])) or (getattr(eye_corner_results,'left_eye_inner')[0]==[-1,-1]):
                        #The other eye corner is nan                                                 
                        if (len(LIST_LAST_EYE_CORNER[0]) or (not LIST_LAST_EYE_CORNER[0]==[-1,-1])) or (len(LIST_LAST_EYE_CORNER[1]) or (not LIST_LAST_EYE_CORNER[1]==[-1,-1])):
                             #We had two detections last time
                            dist_to_inner=abs(corner_point[0]-LIST_LAST_EYE_CORNER[1][0]) #Distance in x-direction 
                            dist_to_outer=abs(corner_point[0]-LIST_LAST_EYE_CORNER[0][0])
                            if dist_to_inner<dist_to_outer: #We swap the values
                                old_left_outer=corner_point
                                eye_corner_results.left_eye_outer[0]=getattr(eye_corner_results,'left_eye_inner')[0]
                                eye_corner_results.left_eye_inner[0]=old_left_outer
                    else:
                        #the other eye corner is a number
                        left_inner=getattr(eye_corner_results,'left_eye_inner')[0]
                        if corner_point[0]<left_inner[0]: #We swap the values
                            old_left_outer=corner_point
                            eye_corner_results.left_eye_outer[0]=left_inner
                            eye_corner_results.left_eye_inner[0]=old_left_outer
    return eye_corner_results
                    














    
def showCorners(corners,frame):
    frame_colour=np.copy(frame)
    for corner in fields(corners):
        corner_name=corner.name
        corner_point=getattr(corners,corner_name)[0]
        if type(corner_point) is float:
            if math.isnan(corner_point):
                continue
        if corner_name=='left_eye_inner': #blue
            pintcolor=(255,0,0)
        elif corner_name=='left_eye_outer': #green
            pintcolor=(0,255,0)
        elif corner_name=='right_eye_outer':    #red
            pintcolor=(0,0,255)
        elif corner_name=='right_eye_inner':    #white
            pintcolor=(255,255,255)
        corner_point=[int(elem) for elem in corner_point]
        frame_colour=cv2.circle(frame_colour,corner_point,radius=3,color=pintcolor,thickness=2)
    cv2.imshow('corners',frame_colour)
    cv2.waitKey(2)


#-----------------------------<Main>-------------------------------
#if __name__=="__main__":
device=select_device('0') #Sets up the GPU Cuda device
model=attempt_load(YOLOV7_MODEL_PATH,map_location=device)
stride_size=int(model.stride.max())
imgsz=check_img_size(INPUT_IMAGE_SIZE,s=stride_size) #Checks that stride requirements are met

#Setup model
if device.type != 'cpu':
    model(torch.zeros(1,3,imgsz,imgsz).to(device).type_as(next(model.parameters()))) #runs it once
left_old_img_w=left_old_img_h=imgsz
left_old_img_b=1

right_old_img_w=right_old_img_h=imgsz
right_old_img_b=1




#Looping through all videos in directory
dest_path='E:/Alexandre_EyeGazeProject_Extra/EvaluatingCornerDetection_Accuracy'
csv_eyecorner=open(dest_path+'/CornerResults.csv',mode='w')
record_num=1
for entry in os.scandir(data_root):
    if entry.is_dir():
        part_name=entry.name
        video_dir=data_root+'/'+part_name+'/EyeGaze_Data'
        entry_num=re.sub("[P]","",part_name)
        entry_num=int(entry_num)
        if entry_num<=10:
            for file in os.listdir(video_dir): #Loops for all the eye videos in the directory
                if file.endswith('.avi'):
                    root,ext=os.path.splitext(file)
                    video=cv2.VideoCapture(video_dir+'/'+file)
                    if root=='eyeVideo_Calib_Init':
                        if(video.isOpened()==False):
                            print("video "+file+" cannot be opened")
                            continue
                        '''
                        #Creates a csv file to store the eyecorners
                        
                        csv_name=data_root+'/'+part_name+'/'+'eyecorners_'+root[9:]+'.csv'
                        #print('Current File:',csv_name)
                        if os.path.isfile(csv_name):    #Enters if the csv alread exists and opens for appending
                            csv_eyecorner=open(csv_name,mode='r')
                            lines=csv_eyecorner.readlines()
                            csv_eyecorner.close()
                            if len(lines):
                                last_line=lines[-1]
                                last_line=last_line.split(',')
                                if last_line[0].isnumeric():
                                    video.set(cv2.CAP_PROP_POS_FRAMES,int(last_line[0]))
                                    csv_eyecorner=open(csv_name,'a')
                                else:
                                    csv_eyecorner=open(csv_name,mode='w')
                                    csv_eyecorner.write('Frame_No,Right_Inner_x,Right_Inner_y,Right_Outer_x,Right_Outer_y,Left_Outer_x,Left_Outer_y,Left_Inner_x,Left_Inner_y\n')

                            else:
                                csv_eyecorner=open(csv_name,mode='w')
                                csv_eyecorner.write('Frame_No,Right_Inner_x,Right_Inner_y,Right_Outer_x,Right_Outer_y,Left_Outer_x,Left_Outer_y,Left_Inner_x,Left_Inner_y\n')
                                #video.set(cv2.CAP_PROP_POS_FRAMES,0)
                        else:
                            csv_eyecorner=open(csv_name,mode='w')
                            #Writing Header:
                            csv_eyecorner.write('Frame_No,Right_Inner_x,Right_Inner_y,Right_Outer_x,Right_Outer_y,Left_Outer_x,Left_Outer_y,Left_Inner_x,Left_Inner_y\n')
                        '''
                        sucess,frame=video.read()
                        #frame_count=0
                        #print('Current File is: ',csv_name)
                        while(sucess):
                            frame_no=video.get(cv2.CAP_PROP_POS_FRAMES)
                        
                            #frame_count+=1
                            #print(frame_no)
                            frame_processed,dw,dh=preprocessFrame(frame)
                            #t0=time.time()
                            #manager=multiprocessing.Manager()
                            #return_dict=manager.dict()
                            #process=Process(target=detect,args=(frame_processed,model,device,left_old_img_b,left_old_img_h,left_old_img_w,return_dict))
                            #process.start()
                            #process.join(timeout=20)
                            
                            #if process.is_alive():
                                #process.terminate()
                                #process.join()
                                #continue

                            #return_vals=return_dict.values()
                            #predictions=return_dict.pred
                            #left_old_img_b=return_dict.left_old_img_b
                            #left_old_img_h=return_dict.left_old_img_h
                            #left_old_img_w=return_dict.left_old_img_w
                            predictions,left_old_img_b,left_old_img_h,left_old_img_w=detect(frame_processed,model,device,left_old_img_b,left_old_img_h,left_old_img_w)
                            #t1=time.time()
                            #print('Prediction time is: ',t1-t0)
                            bounding_boxes,is_detect=process_detections(predictions,dw,dh)
                            if is_detect==False: #Checks that we have a detection
                                sucess,frame=video.read()
                                continue

                            eye_images,eye_found=cropBothEyes(frame,bounding_boxes)
                            if eye_found==False: #No eyes are found
                                sucess,frame=video.read()
                                continue
                            #t3=time.time()
                            eye_corner_results=findCorners(frame,eye_images,bounding_boxes)
                            #t4=time.time()
                            #print('Corner Detection Time: ',t4-t3)
                            
                            #if frame_count==50:
                            #showBoxes(frame,bounding_boxes)

                            #print(frame_count)
                            #frame_count=0
                            
                            #Saving the eye corner results
                            #del eye_corner_results.left_eye_inner[0]
                            #eye_corner_results.left_eye_inner.append(math.nan)
                            #print(eye_corner_results)

                            
                            results_list=[math.nan]*9 #Initializes our results list
                            #results_list[0]=int(frame_no)
                            for corner in fields(eye_corner_results):
                                corner_name=corner.name
                                corner_point=getattr(eye_corner_results,corner_name)[0]
                                if type(corner_point) is float:
                                    if math.isnan(corner_point): #The results is a nan
                                        if corner_name=='right_eye_inner':
                                            results_list[1]='nan'
                                            results_list[2]='nan'
                                        elif corner_name=='right_eye_outer':
                                            results_list[3]='nan'
                                            results_list[4]='nan'
                                        elif corner_name=='left_eye_outer':
                                            results_list[5]='nan'
                                            results_list[6]='nan'
                                        elif corner_name=='left_eye_inner':
                                            results_list[7]='nan'
                                            results_list[8]='nan'
                                        else:
                                            continue
                                    else:
                                        continue
                                else:
                                    if corner_name=='right_eye_inner':
                                        results_list[1]=corner_point[0]
                                        results_list[2]=corner_point[1]
                                    elif corner_name=='right_eye_outer':
                                        results_list[3]=corner_point[0]
                                        results_list[4]=corner_point[1]
                                    elif corner_name=='left_eye_outer':
                                        results_list[5]=corner_point[0]
                                        results_list[6]=corner_point[1]
                                    elif corner_name=='left_eye_inner':
                                        results_list[7]=corner_point[0]
                                        results_list[8]=corner_point[1]
                                    else:
                                        continue
                            print('Frame_no: ',int(frame_no))
                            if frame_no>=40:
                                showCorners(eye_corner_results,frame)
                                results_list[0]=record_num
                                record_num+=1
                                csv_eyecorner.write('{},{},{},{},{},{},{},{},{}\n'.format(results_list[0],results_list[1],results_list[2],results_list[3],results_list[4],results_list[5],results_list[6],results_list[7],results_list[8]))
                                gray_frame=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
                                frame_path=dest_path+'/frames_unlabeled/'+str(record_num)+'.png'
                                cv2.imwrite(frame_path,gray_frame)
                            #print(results_list)
                            #csv_eyecorner.write('{},{},{},{},{},{},{},{},{}\n'.format(results_list[0],results_list[1],results_list[2],results_list[3],results_list[4],results_list[5],results_list[6],results_list[7],results_list[8]))
                            
                            #print('Frame_no: ',int(frame_no))
                            sucess,frame=video.read()
                            if frame_no>=50:
                                sucess=False
                            
                        #csv_eyecorner.close()
csv_eyecorner.close()
                                    
                                
                            

                        
                        
                        
                        
                        
                        
                        