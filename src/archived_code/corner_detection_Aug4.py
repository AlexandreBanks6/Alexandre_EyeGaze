'''

Descrition: Runs eye corner detection using custom model. Applies eye corner filtering.
Does it for all eye videos for a given subject
'''
#----------------------<Imports>----------------------------
import cv2
import os
import numpy as np
import math


#---------------<Initializing Variables>--------------------
EYE_DIR='C:/Users/playf/OneDrive/Documents/UBC/Thesis/Custom_Model_Code/data/images_cropped'
PI=3.1415926535

FILTER_SIZE=45 #Filter for gaussian blur
LOW_THRESHOLD=8
ratio_val_inner=2.5
UPPER_THRESHOLD=LOW_THRESHOLD*ratio_val_inner
CANNY_KERNEL_SIZE=3
DILATION_SIZE=5
EROSION_SIZE=4
EROSION_ELEMENT=cv2.getStructuringElement(cv2.MORPH_RECT,(EROSION_SIZE,EROSION_SIZE))
DILATION_ELEMENT=cv2.getStructuringElement(cv2.MORPH_RECT,(DILATION_SIZE,DILATION_SIZE))

'''
DILATION_SIZE_EDGES=4
EROSION_SIZE_EDGES=4
EROSION_ELEMENT_EDGES=cv2.getStructuringElement(cv2.MORPH_RECT,(EROSION_SIZE_EDGES,EROSION_SIZE_EDGES))
DILATION_ELEMENT_EDGES=cv2.getStructuringElement(cv2.MORPH_RECT,(DILATION_SIZE_EDGES,DILATION_SIZE_EDGES))
'''

CONTOUR_THRESH=5 #Only keeping top 5 largest contours
EPSILON_VALUE=0.01 #allowable percentage error for fitting contours with polynomial
MAV_LENGTH=8
OUTLIER_MULT=2
#----------------<Function Definitions>---------------------

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
        '''
        max_length=sorted_lengths[0]
        max_ind=cont_lengths.index(max_length)
        largest_contour=contours()
        '''
    return contours

def bestEllipse(ellipses,img_shape):
    #Sorts through list of ellipses and scores each
    #ellipse depending on how close it is to the center
    #and its size, rejects ellipses with center outside of image
    
    #Scores list for closeness to center
    c_x=img_shape[1]/2
    c_y=img_shape[0]/2
    dst_vec=[]
    sz_vec=[]
    new_ellipse=[]

    ellipse_count=0
    for ellipse in ellipses:
        ellipse_x=ellipse[0][0]
        ellipse_y=ellipse[0][1]
        if (ellipse_x<0 or ellipse_x>img_shape[1]) or (ellipse_y<0 or ellipse_y>img_shape[0]):
            continue
        new_ellipse.append(ellipse)
        dst_vec.append(math.sqrt((ellipse_x-c_x)**2+(ellipse_y-c_y)**2))
        sz_vec.append(PI*ellipse[1][0]*ellipse[1][1])
        ellipse_count+=1
    score_list=[]
    for i in range(len(dst_vec)):
        score_list.append((sz_vec[i]/max(sz_vec))+1.5*(min(dst_vec)/dst_vec[i]))
    max_val=max(score_list)
    max_ind=score_list.index(max_val)
    best_ellipse=new_ellipse[max_ind]
    return best_ellipse

def findCorners(frame):
    frame_blur=cv2.GaussianBlur(frame,(FILTER_SIZE,FILTER_SIZE), cv2.BORDER_DEFAULT) #FIlters image
    cv2.imshow('blurred',frame_blur)
    cv2.waitKey(0)
    #This is to remove effect of eye lashes:
    #_,eye_lashes=cv2.threshold(frame_blur,50,255,cv2.THRESH_BINARY)

    #cv2.imshow("lashes",eye_lashes)
    #cv2.waitKey(0)
    #_,frame_thresholded=cv2.threshold(frame,80,255,cv2.THRESH_BINARY_INV)
    frame_thresholded=cv2.adaptiveThreshold(frame_blur,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,11,1.5)
    cv2.imshow('frame thresholded',frame_thresholded)
    cv2.waitKey(0)

    #detected_edges=cv2.Canny( ,LOW_THRESHOLD,UPPER_THRESHOLD,apertureSize=CANNY_KERNEL_SIZE,L2gradient=True)
    #dilated_edges=cv2.dilate(detected_edges,DILATION_ELEMENT_EDGES)
    #eroded_edges=cv2.erode(dilated_edges,EROSION_ELEMENT_EDGES)
    #cv2.imshow('eroded edges',eroded_edges)
    #cv2.waitKey(0)

    frame_dilated=cv2.dilate(frame_thresholded,DILATION_ELEMENT)
    frame_eroded=cv2.erode(frame_dilated,EROSION_ELEMENT)
    cv2.imshow('frame eroded',frame_eroded)
    cv2.waitKey(0)
    contours, _ = cv2.findContours(frame_eroded, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE) 
    contours=findLargestContours(contours) #Finds top 5 largest contours
    
    #cnt_count=0
    frame_colour=cv2.cvtColor(frame,cv2.COLOR_GRAY2BGR)
    ellipses=[] #Will store the ellipses
    
    #Fit contours using PolyDP and then fit with ellipse
    #Fits an ellipse to all contours 
    for cnt in contours:
        #epsilon=EPSILON_VALUE*cv2.arcLength(cnt,True)
        #approx=cv2.approxPolyDP(cnt,epsilon,True)

        approx_size=cnt.size
        if approx_size/2<=5:
            continue
        ellipse=cv2.fitEllipse(cnt)
        ellipses.append(ellipse)
        #hull=cv2.convexHull(cnt,returnPoints=True)
        #cv2.drawContours(frame_colour,[hull],-1,color=(0,255,0),thickness=3)
        #cv2.drawContours(frame_colour,[approx],-1,color=(0,0,255),thickness=1)
        #cv2.drawContours(frame_colour,[cnt],-1,color=(255,0,0),thickness=1)

    img_shape=frame.shape
    best_ellipse=bestEllipse(ellipses,img_shape)
    cv2.ellipse(frame_colour,best_ellipse,(255,255,0),1)
    cv2.imshow('contours',frame_colour)
    cv2.waitKey(0)
        #cnt_count+=1
    #cv2.imwrite(TEST_IMAGES_DIR+'edge_imaeg'+'_'+str(frame_count)+'.jpg')
#--------------------<Main Loop>---------------------------

#Looping through all videos in directory
for file in os.listdir(EYE_DIR): #Loops for all the eye videos in the directory
    frame=cv2.imread(EYE_DIR+'/'+file)
    img=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    findCorners(img)
