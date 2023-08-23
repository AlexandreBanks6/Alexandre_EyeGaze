'''
Author: Alexandre Banks
Date: July 26, 2023
Institution: UBC Robotics and Control Laboratory
Description: Script that uses yolov7 to create eye corner bounding boxes, then finds eye corner
landmarks using conventional computer vision techniques. This version only finds inner eye corners.
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
import math
import time


import torch
import torch.backends.cudnn as cudnn
from numpy import random

from models.experimental import attempt_load
from utils.general import check_img_size, non_max_suppression
from utils.torch_utils import select_device
#----------------------------------------------<Setting Variables>----------------------------------
img_path="/media/alexandre/My Passport/Alexandre_EyeGazeProject/RefinedData/SelectedVideos/dataset/training_videos/P20.avi"
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
    cropped_eyecorner: list=field(default_factory=list)
    contours_eyecorner: list=field(default_factory=list)
    original: list=field(default_factory=list)
    eyecorner_point: list=field(default_factory=list)

class_names=['Inner Corner', 'Outer Corner']

#~~~~~~~~~~~~Params for Eye Corner Filtering~~~~~~~~~~~~~~~~~
#------Inner Eye Corner Params
#Canny Edge Detector Parameters
FILTER_SIZE_INNER=9
LOW_THRESHOLD_INNER=17
RATIO_VAL_INNER=3
UPPER_THRESHOLD_INNER=LOW_THRESHOLD_INNER*RATIO_VAL_INNER
CANNY_KERNAL_SIZE_INNER=3


#Morph op parameters
dilation_size_inner=4
erosion_size_inner=4
EROSION_ELEMENT_INNER=cv2.getStructuringElement(cv2.MORPH_RECT,(erosion_size_inner,erosion_size_inner))
DILATION_ELEMENT_INNER=cv2.getStructuringElement(cv2.MORPH_RECT,(dilation_size_inner,dilation_size_inner))

#Thresholding Params:
BLOCK_SIZE_INNER=31
THRESH_CONSTANT_INNER=2
erosion_size_outer=3
THRESHOLD_DILATION_ELEMENT_INNER=cv2.getStructuringElement(cv2.MORPH_RECT,(erosion_size_outer,erosion_size_outer))

#Curvature Calculation Parameters
STEP_SIZE_INNER=20 

#Moving Average Filter Parameters
STD_MULT=1.2 #Multiplier of standard deviation for outlier rejection
MAV_LENGTH=18 #Length of the moving average filter for the eye corners



#Trackbar stuff
def nothing(x):
    pass


'''
window_name='Processed Outer Edges'
cv2.namedWindow(window_name)
cv2.createTrackbar('filter_size',window_name,11,30,nothing)
cv2.createTrackbar('block_size',window_name,19,30,nothing)
cv2.createTrackbar('threshold_constant',window_name,1,30,nothing)
cv2.createTrackbar('erosion_size',window_name,3,11,nothing)
cv2.createTrackbar('lower_canny',window_name,17,50,nothing)
cv2.createTrackbar('canny_kernel',window_name,3,50,nothing)
cv2.createTrackbar('ratio_val',window_name,10,20,nothing)
cv2.waitKey(0)
'''
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


def process_detections(pred,img0,dw,dh,left_right):
    pred_results=pred_res() #Creates class object
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
                pred_results.cropped_eyecorner.append(cropped_eyecorner)
    return (pred_results)

#Funcions that performs pre-processing on the cropped eye-corner image
def process_innercorner(cropped_corner):
        eye_grey=cv2.cvtColor(cropped_corner,cv2.COLOR_BGR2GRAY) #Converts to grayscale (already grayscale but for check)
        eye_grey=cv2.GaussianBlur(eye_grey,(FILTER_SIZE_INNER,FILTER_SIZE_INNER), cv2.BORDER_DEFAULT) #FIlters image
        detected_edges=cv2.Canny(eye_grey,LOW_THRESHOLD_INNER,UPPER_THRESHOLD_INNER,apertureSize=CANNY_KERNAL_SIZE_INNER,L2gradient=True) #Runs edge detector
        dilated_edges=cv2.dilate(detected_edges,DILATION_ELEMENT_INNER)
        eroded_edges=cv2.erode(dilated_edges,EROSION_ELEMENT_INNER)
        #Clustering contour points
        contours,hierarchy=cv2.findContours(eroded_edges,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
        cnt=sorted(contours,key=cv2.contourArea)[-1] #Gets contour with the largest area
        return cnt
def process_eyecorner(pred_results=pred_res()):
    for i in range(len(pred_results.ids)):
        if pred_results.ids[i]==0: #Inner Corner 
            
            contours=process_innercorner(pred_results.cropped_eyecorner[i])
            pred_results.contours_eyecorner.append(contours)
        else: #If another ID, appends blank contours
            pred_results.contours_eyecorner.append([])
        
    return pred_results

def process_contour(pred_results=pred_res()):
    for k in range(len(pred_results.ids)): #Loops for inner (0) and outer (1) eye corners
        if pred_results.ids[k]==0: #Inner Corner
            contours=pred_results.contours_eyecorner[k]
            middle_list=math.floor(len(contours)/2)
            if middle_list<1:
                continue
            contours_new=list(contours[:middle_list])
            pred_results.contours_eyecorner[k]=contours_new
  
    return pred_results
    
def getCurvature(contour_points,step):
    contour_points=list(contour_points)
    #This Function takes a single contour

    num_contour_points=len(contour_points)
    vec_curvature=[0]*num_contour_points

    #Checks that the number of contour points is greater than the step size
    if(num_contour_points<step):
        return vec_curvature
    
    fronToBack=contour_points[0]-contour_points[-1]
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
        if abs(divisor)>10e-10:
            curvature2D=abs(f2ndDerivative[1]*f1stDerivative[0]-f2ndDerivative[0]*f1stDerivative[1])/(math.sqrt(divisor)**3)
        else:
            curvature2D=float('inf')
        vec_curvature[i]=curvature2D
    return vec_curvature

def maxCurve(pred_results=pred_res()):
    for k in range(len(pred_results.ids)): #Loops for the inner/outer eye corners
        if pred_results.ids[k]==0: #Inner Corner
            contours=list(pred_results.contours_eyecorner[k]) #List of list containing all the contours
            max_curvature=0 #Init max curvature value  
            contour_mag=getCurvature(contours,STEP_SIZE_INNER)
            max_point=contours[0] #Inits the corresponding maximum contour point
            for j in range(len(contour_mag)): #looping for all curvature values
                if contour_mag[j]>max_curvature and not math.isinf(contour_mag[j]): #Takes largest value, as long as it isn't infinity
                    max_curvature=contour_mag[j]
                    max_point=contours[j]
            pred_results.eyecorner_point.append(max_point) #Adds the detected eye corner
        else:
            pred_results.eyecorner_point.append([])
           
    return pred_results

def outlier_detection(corner_list):
    #Returns new corner list with outliers removed
    #corner_list=list(corner_list)
    list_x=[]
    list_y=[]
    for i in range(len(corner_list)):
        list_x.append(corner_list[i][0][0])
        list_y.append(corner_list[i][0][1])
    std_x=np.std(list_x)
    std_y=np.std(list_y)
    mean_x=np.mean(list_x)
    mean_y=np.mean(list_y)
    corner_list_new=[]
    smallest_dif=[300,300]
    smallest_avg=corner_list[0]
    for i in range(len(corner_list)):
        #if abs(corner_list[i][0][0]-mean_x)<smallest_dif[0] or abs(corner_list[i][0][1]-mean_y)<smallest_dif[1]: #Update the closest point to the mean
         #   smallest_avg=corner_list[i]
          #  smallest_dif=[abs(corner_list[i][0][0]-mean_x),abs(corner_list[i][0][1]-mean_y)]
        if (abs(corner_list[i][0][0]-mean_x)>STD_MULT*std_x) or (abs(corner_list[i][0][1]-mean_y)>STD_MULT*std_y): #If the point lies outside the mean +- STD_MULT*standard deviation then we don't add to the new list
            continue
        corner_list_new.append(corner_list[i])
    #if len(corner_list_new):
    return corner_list_new
    #else:
     #   return smallest_avg


def mavFilter(corner_list):
    #Input: List of all eye corner points, with the most recent eye corner point at end
    #Output: Filtered eye corner point
    #Description: First, takes all points within a "MAV_LENGTH" window, rejects outlier points, and computes average of this window
    #Returns the filtered corner point
    corner_list=list(corner_list) #List of list where each sublist is (x,y) set of eye corner points
    list_len=len(corner_list)
    if MAV_LENGTH<=list_len: #Uses a MAV window of size mav_length
        sliced_list=corner_list[-MAV_LENGTH:]
        new_list=outlier_detection(sliced_list)
        avg_corner=sum(list(new_list))
        avg_corner=avg_corner/len(new_list)
        
    else: #Initializing the filter (just taking list_len number as average)
        sliced_list=corner_list[-list_len:]
        new_list=outlier_detection(sliced_list)
        avg_corner=sum(list(new_list))
        avg_corner=avg_corner/len(new_list)
    
    avg_corner=list(avg_corner)
    avg_corner=list(np.around(np.array(avg_corner)))
    avg_corner=[int(elem) for elem in avg_corner[0]]
    return avg_corner

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



#Init List of List for eye corners (used in a moving avverage filter)
left_corners=[]
right_corners=[]

time_list=[]

#Setup windows
cv2.namedWindow('Left Eye Corner')
cv2.namedWindow('Right Eye Corner')
cv2.waitKey(0)
#-----------------------------------------------------<Main Loop>--------------------------------------------
frame_num=1
while(video.isOpened()):    #Loops for each frame in the video
    ret,frame=video.read() #Reads frame
    
    if ret==True:
    #--------------------------------------------<Frame pre-processing>------------------------------------------
        t0=time.time()
        left_cropped,right_cropped=process_frame(frame) #Crops the frame to right and left images
        left_restructured,dw_l,dh_l=reshape_frame(left_cropped,imgsz) #Reshapes the images to 1x3x640x640
        right_restructured,dw_r,dh_r=reshape_frame(right_cropped,imgsz)

    #----------------------------------------------<Running Inference>---------------------------------------------
        pred_left,left_old_img_b,left_old_img_h,left_old_img_w=detect(left_restructured,model,device,left_old_img_b,left_old_img_h,left_old_img_w)
        pred_right,right_old_img_b,right_old_img_h,right_old_img_w=detect(right_restructured,model,device,right_old_img_b,right_old_img_h,right_old_img_w)
    #----------------------------------------------<Process Detections>--------------------------------------
        left_results=process_detections(pred_left,left_cropped,dw_l,dh_l,'left')
        right_results=process_detections(pred_right,right_cropped,dw_r,dh_r,'right')
        t1=time.time()
        process_time=t1-t0

        if len(left_results.ids) and len(right_results.ids): #New Corner For Both Eyes
            #Updating class objects with original 
            t2=time.time()
            left_results.original=left_cropped
            right_results.original=right_cropped
            #Gets the contours
            left_results=process_eyecorner(left_results)
            right_results=process_eyecorner(right_results)
            #Processes the extracted contour points
            left_results=process_contour(left_results)
            right_results=process_contour(right_results)
            #Extract the eye corner point (unfiltered)
            left_results=maxCurve(left_results)
            right_results=maxCurve(right_results)

            for i in range(len(left_results.eyecorner_point)):
                if left_results.ids[i]==1:
                    continue
                left_corners.append(left_results.eyecorner_point[i])

            for i in range(len(right_results.eyecorner_point)):
                if right_results.ids[i]==1:
                    continue
                right_corners.append(right_results.eyecorner_point[i])
            
            t3=time.time()
            curve_time=t3-t2
                


        elif len(left_results.ids): #New Corner for Left Eye
            left_results.original=left_cropped
            left_results=process_eyecorner(left_results)
            left_results=process_contour(left_results)
            left_results=maxCurve(left_results)
            for i in range(len(left_results.eyecorner_point)):
                if left_results.ids[i]==1:
                    continue
                left_corners.append(left_results.eyecorner_point[i])

        elif len(right_results.ids): #New Corner For Right Eye
            right_results.original=right_cropped
            right_results=process_eyecorner(right_results)
            right_results=process_contour(right_results)
            right_results=maxCurve(right_results)
            for i in range(len(right_results.eyecorner_point)):
                if right_results.ids[i]==1:
                    continue
                right_corners.append(right_results.eyecorner_point[i])

        else: #No corner ROIs detected for this frame
            continue

        #Computing filtered eye corner and displaying results for left/right

        t4=time.time()
        left_corner=mavFilter(left_corners)
        right_corner=mavFilter(right_corners)
        t5=time.time()
        filter_time=t5-t4
        time_list.append(process_time+curve_time+filter_time)
         #Displaying the left eye corner result
        if len(left_results.ids):
            for i in range(len(left_results.ids)):
                if left_results.ids[i]==1:
                    continue
                #eye_img=left_results.cropped_eyecorner[i]
                #eye_grey=cv2.cvtColor(eye_img,cv2.COLOR_BGR2GRAY)
                left_corner_new=[0,0]
                left_corner_new[0]=left_corner[0]+int(left_results.xyxy[i][0])
                left_corner_new[1]=left_corner[1]+int(left_results.xyxy[i][1])
                #width_ratio=left_results.cropped_eyecorner[i].shape[0]/left_cropped.shape[0]
                #height_ratio=left_results.cropped_eyecorner[i].shape[1]/left_cropped.shape[1]
                #left_corner_new[0]=int(left_corner[0]*(width_ratio))
                #left_corner_new[1]=int(left_corner[1]*(height_ratio))
                superimposed=cv2.circle(left_cropped,left_corner_new,1,(0,0,255),3)
                cv2.imshow('Left Eye Corner',superimposed)
                cv2.waitKey(16)
                

        #Displaying the right eye corner result
        if len(right_results.ids):
            for i in range(len(right_results.ids)):
                if right_results.ids[i]==1:
                    continue
                right_corner_new=[0,0]
                right_corner_new[0]=right_corner[0]+int(right_results.xyxy[i][0])
                right_corner_new[1]=right_corner[1]+int(right_results.xyxy[i][1])
                #eye_img=right_results.cropped_eyecorner[i]
                #eye_grey=cv2.cvtColor(eye_img,cv2.COLOR_BGR2GRAY)
                superimposed=cv2.circle(right_cropped,right_corner_new,1,(0,0,255),3)
                cv2.imshow('Right Eye Corner',superimposed)
                cv2.waitKey(16)

    else:
        print("Frame Not Read Correctly")
        break
    
    '''
    if frame_num==200:
        print('Computation Time:')
        av_time=sum(time_list)/(len(time_list))
        print(av_time)
    frame_num=frame_num+1
    '''
    


