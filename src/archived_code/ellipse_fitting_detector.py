'''

Descrition: Runs eye corner detection using custom model. Applies eye corner filtering.
Does it for all eye videos for a given subject
'''
#----------------------<Imports>----------------------------
import cv2
import os
import numpy as np


#---------------<Initializing Variables>--------------------
EYE_VIDEO_DIR='E:/Alexandre_EyeGazeProject/eyecorner_userstudy/test_01/EyeGaze_Data'
TEST_IMAGES_DIR='C:/Users/playf/OneDrive/Documents/UBC/Thesis/Custom_Model_Code/data/ellipse_test_results/'
#Loading model
#interpreter=tf.lite.Interpreter(model_path=TFLITE_MODEL_PATH)
#interpreter.allocate_tensors()

FILTER_SIZE=21 #Filter for gaussian blur
LOW_THRESHOLD=10
ratio_val_inner=2.5
UPPER_THRESHOLD=LOW_THRESHOLD*ratio_val_inner
CANNY_KERNEL_SIZE=3
DILATION_SIZE=3
EROSION_SIZE=2
EROSION_ELEMENT=cv2.getStructuringElement(cv2.MORPH_RECT,(EROSION_SIZE,EROSION_SIZE))
DILATION_ELEMENT=cv2.getStructuringElement(cv2.MORPH_RECT,(DILATION_SIZE,DILATION_SIZE))
CONT_THRESH=120 #Threshold on size of contours considered

MAV_LENGTH=8
OUTLIER_MULT=2
#----------------<Function Definitions>---------------------
def outlierReject(corner_list):
    #Finds mean and standard deviation of list in both the x- and y- directions
    x_list=[coord[0] for coord in corner_list]
    y_list=[coord[1] for coord in corner_list]

    std_x=np.std(x_list)
    std_y=np.std(y_list)
    avg_x=np.average(x_list)
    avg_y=np.average(y_list)

    new_list_x=[]
    new_list_y=[]
    smallest_diff=10000
    for coord in corner_list:
        dif_x=abs(coord[0]-avg_x)
        dif_y=abs(coord[1]-avg_y)
        if (dif_x<=OUTLIER_MULT*std_x) and (dif_y<=OUTLIER_MULT*std_y):
            if ((dif_x+dif_y)/2)<smallest_diff:
                smallest_diff=(dif_x+dif_y)/2
                smallest_coord=coord
            new_list_x.append(coord[0])
            new_list_y.append(coord[1])
    if not len(new_list_x):
        new_list_x=smallest_coord[0]
        new_list_y=smallest_coord[1]
    return new_list_x,new_list_y

def mavFilter(corner_list):
    #Performs filtering and outlier rejection
    list_len=len(corner_list)
    
    if MAV_LENGTH<=list_len: #Uses a MAV window of size MAV_LENGTH
        sliced_list=corner_list[-MAV_LENGTH:]
        new_list_x,new_list_y=outlierReject(sliced_list)
        avg_x=sum(new_list_x)/MAV_LENGTH #Finds average eye corner
        avg_y=sum(new_list_y)/MAV_LENGTH

    else: #We haven't filled list yet to size MAV_LENGTH
        sliced_list=corner_list[-list_len:]
        new_list_x,new_list_y=outlierReject(sliced_list)
        avg_x=sum(new_list_x)/list_len #Finds average eye corner
        avg_y=sum(new_list_y)/list_len
    avg_corner=[avg_x,avg_y]
    return avg_corner

def processFrame(frame):
    #Crops Frames
    img=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY) #Converts into gray
    left_frame=img[:480,0:640]
    right_frame=img[:480,640:]

    return left_frame, right_frame

def findCorners(frame,frame_count):
    frame_blur=cv2.GaussianBlur(frame,(FILTER_SIZE,FILTER_SIZE), cv2.BORDER_DEFAULT) #FIlters image
    frame_thresholded=cv2.adaptiveThreshold(frame_blur,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,11,1.05)

    #detected_edges=cv2.Canny(frame_blur,LOW_THRESHOLD,UPPER_THRESHOLD,apertureSize=CANNY_KERNEL_SIZE,L2gradient=True)
    frame_dilated=cv2.dilate(frame_thresholded,DILATION_ELEMENT)
    frame_eroded=cv2.erode(frame_dilated,EROSION_ELEMENT)
    contours, _ = cv2.findContours(frame_eroded, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnt_count=0
    for cnt in contours:
        if cnt.size<CONT_THRESH:
            cnt_count+=1
            continue
        ellipse=cv2.fitEllipse(cnt)
        frame_colour=cv2.cvtColor(frame,cv2.COLOR_GRAY2BGR)
        cv2.drawContours(frame_colour,contours,contourIdx=cnt_count,color=(0,255,0),thickness=1)
        cv2.ellipse(frame_colour,ellipse,(0,0,255),1)
        cv2.imshow('eroded',frame_colour)
        cv2.waitKey(0)
        cnt_count+=1
    #cv2.imwrite(TEST_IMAGES_DIR+'edge_imaeg'+'_'+str(frame_count)+'.jpg')
#--------------------<Main Loop>---------------------------

#Looping through all videos in directory
for file in os.listdir(EYE_VIDEO_DIR): #Loops for all the eye videos in the directory
    if file.endswith('.avi'):
        #We have a video file
        root,ext=os.path.splitext(file)
        #output_filename=EYE_VIDEO_DIR+'/'+root+'.csv'
        #csv_file=open(output_filename,'w')
        #writing headers:
        #csv_file.write('frame_num,corner_x0,corner_y0,corner_x1,corner_y1,corner_x2,corner_y2,corner_x3,corner_y3\n')
        video=cv2.VideoCapture(EYE_VIDEO_DIR+'/'+file)
        if(video.isOpened()==False):
            print("video "+file+" cannot be opened")
            continue
        frame_count=0
        right_inner_list=[]
        right_outer_list=[]
        left_outer_list=[]
        left_inner_list=[]

        while(video.isOpened): #Loops for each frame in the video
            ret,frame=video.read()
            if ret==True:
                #Pre-process the frame
                left_frame,right_frame=processFrame(frame)
                findCorners(left_frame,frame_count)
                findCorners(right_frame,frame_count)

                
                

                '''
                #Updating the list of eye corners
                #right_inner_list.append([output_data[0][0],output_data[0][1]])
                #right_outer_list.append([output_data[0][2],output_data[0][3]])
                #left_outer_list.append([output_data[0][4],output_data[0][5]])
                #left_inner_list.append([output_data[0][6],output_data[0][7]])

                #Applying filtering and outlier rejection to lists
                right_inner_corner=mavFilter(right_inner_list)
                right_outer_corner=mavFilter(right_outer_list)
                left_outer_corner=mavFilter(left_outer_list)
                left_inner_corner=mavFilter(left_inner_list)
                
                #Displaying results
                #circle_coords=[right_inner_corner,right_outer_corner,left_outer_corner,left_inner_corner]
                #circle_tuple=tuple(tuple([int(round(sub[0])),int(round(sub[1]))]) for sub in circle_coords)
                superimposed=cv2.circle(frame,center=[int(round(right_inner_corner[0])),int(round(right_inner_corner[1]))],radius=3,color=(0,0,255),thickness=1)
                superimposed=cv2.circle(superimposed,center=[int(round(right_outer_corner[0])),int(round(right_outer_corner[1]))],radius=3,color=(0,0,255),thickness=1)
                superimposed=cv2.circle(superimposed,center=[int(round(left_outer_corner[0])),int(round(left_outer_corner[1]))],radius=3,color=(0,0,255),thickness=1)
                superimposed=cv2.circle(superimposed,center=[int(round(left_inner_corner[0])),int(round(left_inner_corner[1]))],radius=3,color=(0,0,255),thickness=1)
                cv2.imwrite(TEST_IMAGES_DIR+root+'_'+str(frame_count)+'.jpg',superimposed)
                cv2.waitKey(0)

                #Saving Results
                #csv_file.write('{},{},{},{},{},{},{},{},{}\n'.format(frame_count,right_inner_corner[0],right_inner_corner[1],right_outer_corner[0],right_outer_corner[1],left_outer_corner[0],left_outer_corner[1],left_inner_corner[0],left_inner_corner[1]))
                '''
                frame_count+=1
            else:
                print("frame %d could not be read",frame_count)
                frame_count+=1
        #csv_file.close()
        video.release()
                

