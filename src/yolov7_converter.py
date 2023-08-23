import json
import random
import cv2
import os
import string
import math
#Converts bounding boxes for the two eyes that are in yolo format, and 
#Keypoints of the eye corners that are in a .json, into 6 bounding boxes
#With the following id's:
'''
id:         label:

0           right_eye_inner
1           right_eye_outer
2           left_eye_outer
3           left_eye_inner
4           right_eye         
5           left_eye

'''


IMG_SIZE_NEW=640
JSON_KEYPOINTS='C:/Users/playf/OneDrive/Documents/UBC/Thesis/Custom_Model_Code/data/yolov7_cvat_files/added_data_keypoints.json'
BOXES_DIR='C:/Users/playf/OneDrive/Documents/UBC/Thesis/Custom_Model_Code/data/yolov7_cvat_files/bbox_files_added_data/'
OUTPUT_DIR='C:/Users/playf/OneDrive/Documents/UBC/Thesis/Custom_Model_Code/data/yolov7_new_data'
OUTPUT_TEST_IMAGES_DIR=OUTPUT_DIR+'/test/images/'
OUTPUT_TEST_LABELS_DIR=OUTPUT_DIR+'/test/labels/'

OUTPUT_TRAIN_IMAGES_DIR=OUTPUT_DIR+'/train/images/'
OUTPUT_TRAIN_LABELS_DIR=OUTPUT_DIR+'/train/labels/'

OUTPUT_VALID_IMAGES_DIR=OUTPUT_DIR+'/valid/images/'
OUTPUT_VALID_LABELS_DIR=OUTPUT_DIR+'/valid/labels/'

IMG_PATH='C:/Users/playf/OneDrive/Documents/UBC/Thesis/Custom_Model_Code/data/images_added_data/'

#------------Function Defs

def showBoxes(annotations,image,img_shape):
    indxs=[0,4,8,12,16,20]
    for i in indxs:
        top_left=(int(round((annotations[i]-annotations[i+2]/2)*img_shape[1])),int(round((annotations[i+1]-annotations[i+3]/2)*img_shape[0])))
        bottom_right=(int(round((annotations[i]+annotations[i+2]/2)*img_shape[1])),int(round((annotations[i+1]+annotations[i+3]/2)*img_shape[0])))
        cv2.rectangle(image,top_left,bottom_right,color=(0,255,0),thickness=1)
        cv2.imshow('Bounding Boxes',image)
        cv2.waitKey(0)
def writeTxt(annotations,file_path,file_root):
    indxs=[0,4,8,12,16,20]
    ids=[0,1,2,3,4,5]
    output_txt=open(file_path+file_root+'.txt','w')
    count=0
    
    for i in indxs:
        out_val=[]
        out_val.append(ids[count])
        for elemen in annotations[i:i+4]:
            out_val.append(round(elemen,6))
        output_txt.write('{} {} {} {} {}\n'.format(out_val[0],out_val[1],out_val[2],out_val[3],out_val[4]))
        count+=1
    output_txt.close()





s=json.load(open(JSON_KEYPOINTS))


#Making random list to determine testing/training data
num_labels=len(s['annotations'])
test_train=['train']*(int(num_labels*0.8))+['test']*(int(num_labels*0.1))+['valid']*(int(num_labels*0.1)) #Split labels for training/testing dataset
if not (len(test_train)==num_labels):
    if len(test_train)<num_labels:
        test_train=test_train+['test']*(num_labels-len(test_train))
    else:
        test_train=test_train[:-(len(test_train)-num_labels)]

#Randomly shuffles test_train to create two datasets (one for training, the other for testing)
random.shuffle(test_train)

#Loops for all keypoints in the json file
for ann in s['annotations']:
    image_id=ann['image_id'] #Finds the image ID for the annotation
    check=False
    for im in s['images']: #Looks for the ID and corresponding image name
        id=im['id']
        if id==image_id: #Found corresponding image
            check=True
            image_name=im['file_name']
            #Updates the image name if it contains a path
            image_name=image_name.split('/')
            image_name=image_name[-1]
            break
    if check==True: #Saving our data

        update_name=IMG_PATH+image_name #New Images name
        #----Scales the image to 256x256

        #Getting image size
        img=cv2.imread(update_name) #Reads in the image
        img_size=img.shape #Original Image Size
        resized_img=cv2.resize(img,[IMG_SIZE_NEW,IMG_SIZE_NEW],interpolation=cv2.INTER_LINEAR) #Make sure to replicate this for prediction

        #Keypoint labels normalized by image size, and we convert them into bounding boxes by making the 10x10 px
        annotations=[] #List that will hold all the bounding boxes
        annotations.append(ann['keypoints'][0]*(1/img_size[1]))
        annotations.append(ann['keypoints'][1]*(1/img_size[0]))
        annotations.append(60*(1/img_size[1]))
        annotations.append(40*(1/img_size[0]))

        annotations.append(ann['keypoints'][3]*(1/img_size[1]))
        annotations.append(ann['keypoints'][4]*(1/img_size[0]))
        annotations.append(40*(1/img_size[1]))
        annotations.append(40*(1/img_size[0]))

        annotations.append(ann['keypoints'][6]*(1/img_size[1]))
        annotations.append(ann['keypoints'][7]*(1/img_size[0]))
        annotations.append(40*(1/img_size[1]))
        annotations.append(40*(1/img_size[0]))

        annotations.append(ann['keypoints'][9]*(1/img_size[1]))
        annotations.append(ann['keypoints'][10]*(1/img_size[0]))
        annotations.append(60*(1/img_size[1]))
        annotations.append(40*(1/img_size[0]))

        #Now read in the yolo .txt file with the two bounding boxes of the eyes
        root,_=os.path.splitext(image_name)
        #Open the yolo text file to read from
        yolo_src=open(BOXES_DIR+root+'.txt')
        lines=yolo_src.readlines()
        yolo_src.close()
        for line in lines: #Loop through the lines in the .txt (in these files, 5=right eye and 6=left eye)
            line_list=line.split(' ')
            if line[0]=='5':    #right_eye
                right_eye_list=[float(x) for x in line_list[1:5]]
            else:               #left_eye
                left_eye_list=[float(x) for x in line_list[1:5]]
        for x in right_eye_list:
            annotations.append(x) 
        for x in left_eye_list:
            annotations.append(x)  
        


        #showBoxes(annotations,img,img.shape)

        #Writing files
        if test_train[image_id-1]=='train':
            writeTxt(annotations,OUTPUT_TRAIN_LABELS_DIR,root)
            cv2.imwrite(OUTPUT_TRAIN_IMAGES_DIR+image_name,resized_img)
        elif test_train[image_id-1]=='test':
            writeTxt(annotations,OUTPUT_TEST_LABELS_DIR,root)
            cv2.imwrite(OUTPUT_TEST_IMAGES_DIR+image_name,resized_img)
        else:
            writeTxt(annotations,OUTPUT_VALID_LABELS_DIR,root)
            cv2.imwrite(OUTPUT_VALID_IMAGES_DIR+image_name,resized_img)

