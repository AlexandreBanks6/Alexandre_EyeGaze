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
import math
import time


import torch
import torch.backends.cudnn as cudnn
from numpy import random

from models.experimental import attempt_load
from utils.general import check_img_size, non_max_suppression
from utils.torch_utils import select_device
#----------------------------------------------<Setting Variables>----------------------------------
img_path="resources/P19.avi"
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
filter_size_inner=9
low_threshold_inner=17
ratio_val_inner=3
upper_threshold_inner=low_threshold_inner*ratio_val_inner
canny_kernel_size_inner=3


#Morph op parameters
dilation_size_inner=4
erosion_size_inner=4
erosion_element_inner=cv2.getStructuringElement(cv2.MORPH_RECT,(erosion_size_inner,erosion_size_inner))
dilation_element_inner=cv2.getStructuringElement(cv2.MORPH_RECT,(dilation_size_inner,dilation_size_inner))

#Curvature Calculation Parameters
#step_size_inner=18 

#--------Outer Eye Corner Params
#Thresholding Params
FILTER_SIZE_OUTER=11
BLOCK_SIZE_OUTER=19
THRESH_CONSTANT_OUTER=1
erosion_size_outer=3
THRESHOLD_DILATION_ELEMENT_OUTER=cv2.getStructuringElement(cv2.MORPH_RECT,(erosion_size_outer,erosion_size_outer))
#Canny Edge Detection and Post-Processing Parameters
CANNY_THRESH_LOW_OUTER=17
ratio_val_outer=3
CANNY_THRESH_UP_OUTER=CANNY_THRESH_LOW_OUTER*ratio_val_outer
CANNY_SIZE_OUTER=3
dilation_size_outer=4
erosion_size_outer=4
DILATION_ELEMENT_OUTER=cv2.getStructuringElement(cv2.MORPH_RECT,(erosion_size_outer,erosion_size_outer))
DILATION_ELEMENT_INNER=cv2.getStructuringElement(cv2.MORPH_RECT,(dilation_size_outer,dilation_size_outer))

#Pre-process For Corner Detector Param
#filter_size_forcorner_outer=19
#filter_size_forcorner_outer=9
#corner_thresh_outer=0.1
#blocksize_outer=2
#ksize_outer=3
#harris_param_outer=0.09

#Curvature Calculation Parameters
step_size=18



mav_length=10 #Length of the moving average filter for the eye corners



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



#Functions for eye corner detection

def refine_contours(corners,contours):
    #1. Create list of all x,y coordinates of corner points
    #2. Search each contour, and loop through each contour point, if any point is within a threshold of the corner points, keep that contour
    index=corners==255
    corner_points=np.where(index)
    print(corner_points)


#Function that stretching the image to span all values between 0->255
def contrast_stretching(old_img):
    '''
    MaxVal=old_img.max()
    MinVal=old_img.min()
    factor=(255/(MaxVal-MinVal))
    img_stretch=(old_img-MinVal)*factor
    img_stretch=(img_stretch/img_stretch.max())*255
    '''
    #Normalizing Input Image
    norm_img=cv2.normalize(old_img,None,alpha=0,beta=1,norm_type=cv2.NORM_MINMAX,dtype=cv2.CV_32F)
    #Scaling back to 0->255 as uint8
    norm_img=(255*norm_img).astype(np.uint8)

    cv2.imshow('Stretched',norm_img)
    cv2.waitKey(0)

    return norm_img


#Funcions that performs pre-processing on the cropped eye-corner image
def process_innercorner(cropped_corner):
        eye_grey=cv2.cvtColor(cropped_corner,cv2.COLOR_BGR2GRAY) #Converts to grayscale (already grayscale but for check)
        #eye_grey=cv2.equalizeHist(eye_grey) --> Maybe add later

        eye_grey=cv2.GaussianBlur(eye_grey,(filter_size_inner,filter_size_inner), cv2.BORDER_DEFAULT) #FIlters image
        
        
        #cv2.imshow('Original',cropped_corner)
        #cv2.waitKey(0)
        #cv2.imshow('Equalized',eye_grey)
        #cv2.waitKey(0)

        #cv2.imshow('Gray Blurred',eye_grey)
        #cv2.waitKey(0)
        


        detected_edges=cv2.Canny(eye_grey,low_threshold_inner,upper_threshold_inner,apertureSize=canny_kernel_size_inner,L2gradient=True) #Runs edge detector
        #cv2.imshow('Edges',detected_edges)
        #cv2.waitKey(0)
        #Morphological operations on edges
        dilated_edges=cv2.dilate(detected_edges,dilation_element_inner)
        eroded_edges=cv2.erode(dilated_edges,erosion_element_inner)
        #cv2.imshow('Morph Edges',eroded_edges)
        #cv2.waitKey(0)
        #Clustering contour points
        contours,hierarchy=cv2.findContours(eroded_edges,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
        


        #cv2.imshow('Edges',detected_edges)
        #cv2.waitKey(0)

        #cv2.imshow('Morph Edges',eroded_edges)
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()
        return contours






#def CannyThresholds(eye_grey,window_name):
#    filtersize=cv2.getTrackbarPos('filter_size',window_name)
#    block_size=cv2.getTrackbarPos('block_size',window_name)
#    thresh_constant=cv2.getTrackbarPos('threshold_constant',window_name)
#    erosion_size=cv2.getTrackbarPos('erosion_size',window_name)
#    thresh=cv2.getTrackbarPos('lower_canny',window_name)
#    canny_kernel=cv2.getTrackbarPos('canny_kernel',window_name)
#    ratio=cv2.getTrackbarPos('ratio_val',window_name)
#    ratio=2+ratio/10.0
#    thresh_up=ratio*thresh
    

'''
    thresh=cv2.getTrackbarPos('lower_canny',window_name)
    canny_kernel=cv2.getTrackbarPos('canny_kernel',window_name)
    ratio=cv2.getTrackbarPos('ratio_val',window_name)
    ratio=2+ratio/10.0
    thresh_up=ratio*thresh
    '''
    
    #First find glints and extract from image (turn those regions black)
    #erosion_element_glintmask=cv2.getStructuringElement(cv2.MORPH_RECT,(5,5))
    #_,glint_mask=cv2.threshold(eye_grey,190,255,cv2.THRESH_BINARY_INV)
    #glint_mask=cv2.erode(glint_mask,erosion_element_glintmask)
    ##cv2.imshow('glint mask',glint_mask)
    #cv2.waitKey(0)

    #eye_masked=cv2.bitwise_and(eye_grey,glint_mask)

    #cv2.imshow('masked eye',eye_masked)
    #cv2.waitKey(0)

    #Blur the image first
    #eye_grey=cv2.equalizeHist(eye_grey) 
    #cv2.imshow('Equalized',eye_grey)
    #cv2.waitKey(0)

#    eye_grey=cv2.GaussianBlur(eye_grey,(filtersize,filtersize), cv2.BORDER_DEFAULT) #FIlters image
    #cv2.imshow('Blurred',eye_grey)
    #cv2.waitKey(0)

    #Adaptive Threshold to increase contrast between sclera and edge
#    thresholded=cv2.adaptiveThreshold(eye_grey,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,block_size,thresh_constant)
    #cv2.imshow('thresholded',thresholded)
    #cv2.waitKey(0)

    #Erosion First (de-noising)
    #erosion_element_glintmask=cv2.getStructuringElement(cv2.MORPH_RECT,(erosion_size,erosion_size))
    #eroded=cv2.erode(thresholded,erosion_element_glintmask)
    #cv2.imshow('eroded',eroded)
    #cv2.waitKey(0)
    

    #Dilation Next
#    dilation_element_glintmask=cv2.getStructuringElement(cv2.MORPH_RECT,(erosion_size,erosion_size))
#    dilated=cv2.dilate(thresholded,dilation_element_glintmask)
    #cv2.imshow('dilated',dilated)
    #cv2.waitKey(0)


#    detected_edges=cv2.Canny(dilated,thresh,thresh_up,apertureSize=canny_kernel,L2gradient=True) #Runs edge detector
#    dilated_edges=cv2.dilate(detected_edges,dilation_element_outer)
#    eroded_edges=cv2.erode(dilated_edges,erosion_element_outer)
#    cv2.imshow(window_name,detected_edges)
#    cv2.waitKey(0)

#    cv2.imshow('eroded edges',eroded_edges)

    #Detects Edges
    #detected_edges=cv2.Canny(eye_grey,low_threshold_inner,upper_threshold_inner,apertureSize=canny_kernel_size_inner,L2gradient=True) #Runs edge detector
    #Dilation of edges followed by erosion
    #dilated_edges=cv2.dilate(detected_edges,dilation_element_inner)
    #eroded_edges=cv2.erode(dilated_edges,erosion_element_inner)

    #eye_grey=cv2.equalizeHist(eye_grey) 
    #cv2.imshow('Equalized',eye_grey)
    #cv2.waitKey(0)
    #kernel=np.array([[0,-1,0],[-1,5,-1],[0,-1,0]])
    #eye_grey=cv2.filter2D(src=eye_grey,ddepth=-1,kernel=kernel)
    #cv2.imshow('Sharpened',eye_grey)
    #cv2.waitKey(0)


    #eye_sharp=cv2.addWeighted(src1=eye_blur,alpha=1.5,src2=eye_grey,beta=-0.5,gamma=0)
    #cv2.imshow('Sharpened',eye_sharp)
    #cv2.waitKey(0)

'''
    detected_edges=cv2.Canny(eye_grey,thresh,thresh_up,apertureSize=canny_kernel,L2gradient=True) #Runs edge detector
    dilated_edges=cv2.dilate(detected_edges,dilation_element_outer)
    eroded_edges=cv2.erode(dilated_edges,erosion_element_outer)
    cv2.imshow(window_name,detected_edges)
    cv2.waitKey(0)
    '''


def process_outercorner(cropped_corner):

    cv2.imshow('Original',cropped_corner)
    cv2.waitKey(0)
    
    eye_grey=cv2.cvtColor(cropped_corner,cv2.COLOR_BGR2GRAY) #Converts to grayscale (already grayscale but for check)

    eye_grey=cv2.GaussianBlur(eye_grey,(FILTER_SIZE_OUTER,FILTER_SIZE_OUTER), cv2.BORDER_DEFAULT)

    thresholded=cv2.adaptiveThreshold(eye_grey,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,BLOCK_SIZE_OUTER,THRESH_CONSTANT_OUTER)

    dilated=cv2.dilate(thresholded,THRESHOLD_DILATION_ELEMENT_OUTER)

    detected_edges=cv2.Canny(dilated,CANNY_THRESH_LOW_OUTER,CANNY_THRESH_UP_OUTER,apertureSize=CANNY_SIZE_OUTER,L2gradient=True) #Runs edge detector
    dilated_edges=cv2.dilate(detected_edges,DILATION_ELEMENT_OUTER)
    eroded_edges=cv2.erode(dilated_edges,DILATION_ELEMENT_INNER)
    contours,hierarchy=cv2.findContours(eroded_edges,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
    return contours

    #CannyThresholds(eye_grey,window_name)


    #Perform histogram equalization (normalizes brightness and increases image contrast)
    #eye_grey=cv2.equalizeHist(eye_grey)
    #cv2.imshow('Adjusted Brightness/Contrast',eye_grey)
    #cv2.waitKey(0)
    
    #eye_grey_corner=eye_grey.copy()
    ''' Maybe Do Refinement with corners later
    #---------------<Corner Detector to refine Contours>--------------
    eye_grey_corner=cv2.GaussianBlur(eye_grey_corner,(filter_size_forcorner_outer,filter_size_forcorner_outer), cv2.BORDER_DEFAULT) #FIlters image
    #eye_grey=cv2.blur(eye_grey,(filter_size_outer,filter_size_outer))
    
    
    #cv2.imshow('Gray Blurred Corner',eye_grey_corner)
    #cv2.waitKey(0)
    
    #Use the Harris Corner Detector (maybe with subpixel accuracy)
    #corner_mask=eye_grey_corner.copy()
    corner_mask=np.zeros_like(eye_grey,dtype=np.uint8)

    corners=cv2.cornerHarris(eye_grey_corner,blockSize=blocksize_outer,ksize=ksize_outer,k=harris_param_outer)
    corners=cv2.dilate(corners,None) #Dilates the corners
    
    corner_mask[corners>corner_thresh_outer*corners.max()]=255
    cv2.imshow('Corners',corner_mask)
    cv2.waitKey(0)
    '''
    #------------------<Extracting Contours>----------------
    '''
    filtersize=int(cv2.getTrackbarPos('filter_size',window_name))
    thresh=int(cv2.getTrackbarPos('lower_canny',window_name))
    thresh_up=ratio_val_outer*thresh
    eye_grey=cv2.GaussianBlur(eye_grey,(filtersize,filtersize), cv2.BORDER_DEFAULT) #FIlters image
    cv2.imshow('Gray Blurred Canny',eye_grey)
    cv2.waitKey(0)
    detected_edges=cv2.Canny(eye_grey,thresh,thresh_up,apertureSize=canny_kernel_size_outer,L2gradient=True) #Runs edge detector

    dilated_edges=cv2.dilate(detected_edges,dilation_element_outer)
    eroded_edges=cv2.erode(dilated_edges,erosion_element_outer)
    cv2.imshow('Edges',detected_edges)
    cv2.waitKey(0)

    #cv2.imshow('Morph Edges',eroded_edges)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
    cv2.destroyWindow('Gray Blurred Canny')
    cv2.destroyWindow('Edges')
    cv2.destroyWindow('Morph Edges')
    '''
    
    # Uncomment this after trackbar
    
    #eye_grey=cv2.GaussianBlur(eye_grey,(filter_size_outer,filter_size_outer), cv2.BORDER_DEFAULT) #FIlters image

    #cv2.imshow('Gray Blurred Canny',eye_grey)
    #cv2.waitKey(0)
    #detected_edges=cv2.Canny(eye_grey,low_threshold_outer,upper_threshold_outer,apertureSize=canny_kernel_size_outer,L2gradient=True) #Runs edge detector

    #Perform Morphological Closing before refining contours
    #dilated_edges=cv2.dilate(detected_edges,dilation_element_outer)
    #eroded_edges=cv2.erode(dilated_edges,erosion_element_outer)
    #cv2.imshow('Morph Edges',eroded_edges)
    #cv2.waitKey(0)

    

    
    #refine_contours(corner_mask,contours)
    
    #contour_mask=cv2.drawContours(eye_grey,contours,-1,(0,255,0),1)
    
    #cv2.imshow('Contours',contour_mask)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
    

    #Morphological operations on edges
    '''
    erosion_size=int(cv2.getTrackbarPos('erosion_size',window_name))
    dilation_size=int(cv2.getTrackbarPos('dilation_size',window_name))
    erosion_element_outer=cv2.getStructuringElement(cv2.MORPH_RECT,(erosion_size,erosion_size))
    dilation_element_outer=cv2.getStructuringElement(cv2.MORPH_RECT,(dilation_size,dilation_size))

    dilated_edges=cv2.dilate(detected_edges,dilation_element_outer)
    eroded_edges=cv2.erode(dilated_edges,erosion_element_outer)

    
    #Clustering contour points
    contours,hierarchy=cv2.findContours(eroded_edges,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
    


    cv2.imshow('Edges',detected_edges)
    cv2.waitKey(0)

    cv2.imshow('Morph Edges',eroded_edges)
    cv2.waitKey(0)

    #cv2.destroyAllWindows()
    cv2.destroyWindow('Edges')
    cv2.destroyWindow('Morph Edge')
    '''
    


    







    #Change brightness and contrast
    #eye_grey=cv2.convertScaleAbs(eye_grey,alpha=alpha_inner,beta=beta_inner)

    #Perform histogram stretching
    #eye_grey=contrast_stretching(eye_grey)



    #ret,thresholded=cv2.threshold(eye_grey,70,255,cv2.THRESH_BINARY)
    #cv2.imshow('Thresholded',thresholded)
    #cv2.waitKey(0)

    '''
    detected_edges=cv2.Canny(eye_grey,low_threshold_outer,upper_threshold_outer,apertureSize=canny_kernel_size_outer,L2gradient=True) #Runs edge detector

    #Morphological operations on edges
    dilated_edges=cv2.dilate(detected_edges,dilation_element_outer)
    eroded_edges=cv2.erode(dilated_edges,erosion_element_outer)
    #Clustering contour points
    contours,hierarchy=cv2.findContours(eroded_edges,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
    


    cv2.imshow('Edges',detected_edges)
    cv2.waitKey(0)

    cv2.imshow('Morph Edges',eroded_edges)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    '''




    #ret,corners=cv2.threshold(corners,corner_thresh*corners.max(),255,0)
    #corners=np.uint8(corners)

    #Find centroids
    #ret,labels,stats,centroids=cv2.connectedComponentsWithStats(corners)
    #Define the criteria to stop and refine the corners
    #criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 1000, 0.001)
    #corner_subpix = cv2.cornerSubPix(eye_grey,np.float32(centroids),(5,5),(-1,-1),criteria)

    #Draw them
    # Now draw them
    #res = np.hstack((centroids,corner_subpix))
    #res = np.int0(res)
    #eye_grey[res[:,3],res[:,2]] = 255


    #corner_mask=np.zeros_like(corners,dtype=np.uint8)
    
    

    #contours,_=cv2.findContours(corner_mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    #cv2.drawContours(eye_grey,contours,-1,(0,255,0),3)
    #corner_points=[cv2.minEnclosingCircle(cnt)[0] for cnt in contours]
    #corner=corner_points[1]
    #cv2.circle(eye_grey,tuple(map(int,corner)),3,(0,255,0),2)

def process_eyecorner(pred_results=pred_res()):
    for i in range(len(pred_results.ids)):
        if pred_results.ids[i]==0: #Inner Corner
            
            contours=process_innercorner(pred_results.cropped_eyecorner[i])
            pred_results.contours_eyecorner.append(contours)
        elif pred_results.ids[i]==1: #Outer corner
            #continue #-----------------------------<Added tHIS>
            #process_outercorner(pred_results.cropped_eyecorner[i])
            contours=process_outercorner(pred_results.cropped_eyecorner[i])
            pred_results.contours_eyecorner.append(contours)
        else:
            pred_results.contours_eyecorner.append([])
        
    return pred_results

def process_contour(pred_results=pred_res()):
    for k in range(len(pred_results.ids)): #Loops for inner (0) and outer (1) eye corners
   #     if pred_results.ids[k]==1: #Outer Corner
     #       continue #-----------------------------<Added tHIS>
        contours=pred_results.contours_eyecorner[k]
        cont_len=len(contours)
        #contours_new=[[] for i in range(cont_len)]
        contours_new=[]

        for i in range(cont_len): #Loops for the number of contours
            #for j in range(math.floor(len(contours[i])/2)):
            middle_list=math.floor(len(contours[i])/2)
            if middle_list<1:
                continue
            contours_new.append(list(contours[i][:middle_list]))
        pred_results.contours_eyecorner[k]=contours_new
    
    return pred_results
    
def getCurvature(contour_points,step):
    contour_points=list(contour_points)
    #This Function takes a single contour

    num_contour_points=len(contour_points)
    vec_curvature=[0]*num_contour_points

    #Checks that the number of contour points is less than the step size
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
        fpos=contour_points[i]
        pos=fpos[0][0]

        if isClosed: #Closed Curve
            iminus=i-step
            iplus=i+step
            fminus=contour_points[iminus+num_contour_points if (iminus<0) else iminus]
            fplus=contour_points[iplus-num_contour_points if (iplus>=num_contour_points) else iplus]

            #Derivative Approximations
            f1stDerivative[0]=(fplus[0][0]-fminus[0][0])/(2*step) #0=x direction
            f1stDerivative[1]=(fplus[0][1]-fminus[0][1])/(2*step) #1=y direction
            f2ndDerivative[0]=(fplus[0][0]-2*fpos[0][0]+fminus[0][0])/(step**2) #0=x direction
            f2ndDerivative[1]=(fplus[0][1]-2*fpos[0][1]+fminus[0][1])/(step**2) #1=y direction
        else: #Open Curve
            if ((i-step)<0) and ((i+2*step)<num_contour_points): #We are at start of curve
                iplus=i+step
                i2plus=i+2*step
                fplus=contour_points[iplus]
                f2plus=contour_points[i2plus]
                
                
                #One Sided Derivative Approximations (forward)
                f1stDerivative[0]=(-f2plus[0][0]+4*fplus[0][0]-3*fpos[0][0])/(2*step)
                f1stDerivative[1]=(-f2plus[0][1]+4*fplus[0][1]-3*fpos[0][1])/(2*step)

                f2ndDerivative[0]=(f2plus[0][0]-2*fplus[0][0]+fpos[0][0])/(step**2)
                f2ndDerivative[1]=(f2plus[0][1]-2*fplus[0][1]+fpos[0][1])/(step**2)

            elif ((i+step)>=num_contour_points) and ((i-2*step)>=0): #End of curve
                iminus=i-step
                i2minus=i-2*step
                fminus=contour_points[iminus]
                f2minus=contour_points[i2minus]

                #One Sided Derivative Approximations (backward)
                f1stDerivative[0]=(3*fpos[0][0]-4*fminus[0][0]+f2minus[0][0])/(2*step)
                f1stDerivative[1]=(3*fpos[0][1]-4*fminus[0][1]+f2minus[0][1])/(2*step)

                f2ndDerivative[0]=(fpos[0][0]-2*fminus[0][0]+f2minus[0][0])/(step**2)
                f2ndDerivative[1]=(fpos[0][1]-2*fminus[0][1]+f2minus[0][1])/(step**2)
            elif ((i+step)<num_contour_points) and ((i-step)>=0):  #Middle of curve
                iminus=i-step
                iplus=i+step
                fminus=contour_points[iminus+num_contour_points if (iminus<0) else iminus]
                fplus=contour_points[iplus-num_contour_points if (iplus>=num_contour_points) else iplus]

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
            curvature2D=abs(f2ndDerivative[1]*f1stDerivative[0]-f2ndDerivative[0]*f1stDerivative[1])/(divisor**(3/2))
        else:
            curvature2D=float('inf')
        vec_curvature[i]=curvature2D
    return vec_curvature

def maxCurve(step,pred_results=pred_res()):
    for k in range(len(pred_results.ids)): #Loops for the inner/outer eye corners
        #if pred_results.ids[k]==1: #Outer Corner
          #  pred_results.eyecorner_point.append([])
            #continue #-----------------------------<Added tHIS>
        contours=list(pred_results.contours_eyecorner[k]) #List of list containing all the contours
        num_contours=len(contours)
        if num_contours>0: #Checks that we have contours
            max_curvature=0 #Init max curvature value
            #contour_mag=[[] for i in range(num_contours)]
            contour_mag=[]
            max_point=contours[0][0] #Inits the corresponding maximum contour point

            for i in range(num_contours):   #Looping for all contours in cropped eye corner
                #contour_mag[i]=getCurvature(contours[i],step) #Getting value of curvature at every point in contour
                contour_mag.append(getCurvature(contours[i],step)) #Getting value of curvature at every point in contour
                for j in range(len(contour_mag[i])): #looping for all curvature values in contour "i"
                    if contour_mag[i][j]>max_curvature:
                        max_curvature=contour_mag[i][j]
                        max_point=contours[i][j]
            pred_results.eyecorner_point.append(max_point[0]) #Adds the detected eye corner
        else:
            pred_results.eyecorner_point.append([])

            
    return pred_results


def mavFilter(corner_list):
    #Returns the filtered corner point
    corner_list=list(corner_list)
    list_len=len(corner_list)
    #x_list,y_list=[]
    '''
    for i in range(len(corner_list)): #Loops for each list item
        x_list.append(corner_list[i][0])
        y_list.append(corner_list[i][1])
        '''
    if mav_length<=list_len: #Uses a MAV window of size mav_length
        sliced_list=corner_list[-mav_length:]
        avg_corner=sum(list(sliced_list))
        avg_corner=avg_corner/mav_length
        
    else: #Initializing the filter (just taking list_len number as average)
        sliced_list=corner_list[-list_len:]
        avg_corner=sum(list(sliced_list))
        avg_corner=avg_corner/list_len
    avg_corner=list(avg_corner)
    avg_corner=list(np.around(np.array(avg_corner)))
    avg_corner=[int(elem) for elem in avg_corner]
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

    #----------------------------------------------<Eye Corner Detection>------------------------------
        #cv2.imshow('Original',left_cropped)
        #cv2.imshow('Cropped Eyecorner',left_results.cropped_eye[0])
        #cv2.waitKey(0)

        if len(left_results.ids) and len(right_results.ids): #Got resuls for both eyes
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

            #Extract the eye corner
            left_results=maxCurve(step_size,left_results)
            right_results=maxCurve(step_size,right_results)

            '''

            for i in range(len(left_results.eyecorner_point)):
                if left_results.ids[i]==1:
                    continue
                left_corners.append(left_results.eyecorner_point[i])

            for i in range(len(right_results.eyecorner_point)):
                if right_results.ids[i]==1:
                    continue
                right_corners.append(right_results.eyecorner_point[i])
            '''
            t3=time.time()
            curve_time=t3-t2

            #Displaying the left eye corner result
            '''
            for i in range(len(left_results.eyecorner_point)):
                if left_results.ids[i]==1:
                    continue
                eye_img=left_results.cropped_eyecorner[i]
                #eye_grey=cv2.cvtColor(eye_img,cv2.COLOR_BGR2GRAY)
                superimposed=cv2.circle(eye_img,left_results.eyecorner_point[i],1,(0,0,255),3)
                cv2.imshow('Eye Corner',superimposed)
                cv2.waitKey(0)
            '''
                


        elif len(left_results.ids): #Got results for the left eye
            left_results.original=left_cropped
            left_results=process_eyecorner(left_results)
            left_results=process_contour(left_results)
            left_results=maxCurve(step_size,left_results)
            '''
            for i in range(len(left_results.eyecorner_point)):
                if left_results.ids[i]==1:
                    continue
                left_corners.append(left_results.eyecorner_point[i])
            '''

        elif len(right_results.ids): #Got results for the right eye
            right_results.original=right_cropped
            right_results=process_eyecorner(right_results)
            right_results=process_contour(right_results)
            right_results=maxCurve(step_size,right_results)
            '''
            for i in range(len(right_results.eyecorner_point)):
                if right_results.ids[i]==1:
                    continue
                right_corners.append(right_results.eyecorner_point[i])
            '''

        else: #No corner ROIs detected for this frame
            continue

        #Computing filtered eye corner and displaying results for left/right
        '''
        t4=time.time()
        left_corner=mavFilter(left_corners)
        right_corner=mavFilter(right_corners)
        t5=time.time()
        filter_time=t5-t4
        time_list.append(process_time+curve_time+filter_time)
        '''

        '''
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
        '''
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
    


