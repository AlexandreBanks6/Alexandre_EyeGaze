import json
import random
import cv2

IMG_SIZE_NEW=512
JSON_FILE='C:/Users/playf/OneDrive/Documents/UBC/Thesis/Custom_Model_Code/data/labels/keypoints.json'
OUTPUT_FILE_TRAIN=JSON_FILE[:-5]+'_train'+'.csv'
OUTPUT_FILE_TEST=JSON_FILE[:-5]+'_test'+'.csv'
OUTPUT_FILE_VALID=JSON_FILE[:-5]+'_valid'+'.csv'

OUTPUT_TEXT_SPACES_TRAIN=JSON_FILE[:-5]+'_spaces_train.txt'
OUTPUT_TEXT_COMMAS_TRAIN=JSON_FILE[:-5]+'_commas_train.txt'
OUTPUT_TEXT_CUSTOM_TRAIN=JSON_FILE[:-5]+'_custom_train.txt'
OUTPUT_TEXT_COMMAS_TEST=JSON_FILE[:-5]+'_commas_test.txt'
OUTPUT_TEXT_CUSTOM_TEST=JSON_FILE[:-5]+'_custom_test.txt'
OUTPUT_TEXT_SPACES_VALID=JSON_FILE[:-5]+'_spaces_valid.txt'
OUTPUT_TEXT_COMMAS_VALID=JSON_FILE[:-5]+'_commas_valid.txt'
OUTPUT_TEXT_CUSTOM_VALID=JSON_FILE[:-5]+'_custom_valid.txt'
IMG_PATH='C:/Users/playf/OneDrive/Documents/UBC/Thesis/Custom_Model_Code/data/images/'
NEW_IMG_PATH='C:/Users/playf/OneDrive/Documents/UBC/Thesis/Custom_Model_Code/data/resized_images/'
s=json.load(open(JSON_FILE))

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

csv_train=open(OUTPUT_FILE_TRAIN,'w')
csv_test=open(OUTPUT_FILE_TEST,'w')
csv_valid=open(OUTPUT_FILE_VALID,'w')
csv_train.write('img_path,x0,y0,x1,y1,x2,y2,x3,y3\n')
csv_test.write('img_path,x0,y0,x1,y1,x2,y2,x3,y3\n')
csv_valid.write('img_path,x0,y0,x1,y1,x2,y2,x3,y3\n')

txt_spaces_train=open(OUTPUT_TEXT_SPACES_TRAIN,'w')
txt_commas_train=open(OUTPUT_TEXT_COMMAS_TRAIN,'w')
txt_custom_train=open(OUTPUT_TEXT_CUSTOM_TRAIN,'w')

txt_spaces_test=open(OUTPUT_TEXT_SPACES_TEST,'w')
txt_commas_test=open(OUTPUT_TEXT_COMMAS_TEST,'w')
txt_custom_test=open(OUTPUT_TEXT_CUSTOM_TEST,'w')

txt_spaces_valid=open(OUTPUT_TEXT_SPACES_VALID,'w')
txt_commas_valid=open(OUTPUT_TEXT_COMMAS_VALID,'w')
txt_custom_valid=open(OUTPUT_TEXT_CUSTOM_VALID,'w')

for ann in s['annotations']:
    image_id=ann['image_id'] #Finds the image ID for the annotation
    check=False
    for im in s['images']: #Looks for the ID and corresponding image name
        id=im['id']
        if id==image_id: #Found corresponding image
            check=True
            image_name=im['file_name']
            break
    if check==True:
        update_name=IMG_PATH+image_name
        #----Scales the image to 256x256
        #Getting image size
        img=cv2.imread(update_name)
        resized_img=cv2.resize(img,[IMG_SIZE_NEW,IMG_SIZE_NEW],interpolation=cv2.INTER_AREA) #Make sure to replicate this for prediction
        cv2.imwrite(NEW_IMG_PATH+image_name,resized_img)
        img_size=img.shape

        #Normalizing labels by image size
        x0=ann['keypoints'][0]*(1/img_size[1])
        y0=ann['keypoints'][1]*(1/img_size[0])
        x1=ann['keypoints'][3]*(1/img_size[1])
        y1=ann['keypoints'][4]*(1/img_size[0])
        x2=ann['keypoints'][6]*(1/img_size[1])
        y2=ann['keypoints'][7]*(1/img_size[0])
        x3=ann['keypoints'][9]*(1/img_size[1])
        y3=ann['keypoints'][10]*(1/img_size[0])
        if test_train[image_id-1]=='train':
            csv_train.write('{},{},{},{},{},{},{},{},{}\n'.format(update_name,x0,y0,x1,y1,x2,y2,x3,y3))
            txt_spaces_train.write('{} {} {} {} {} {} {} {} {}\n'.format(NEW_IMG_PATH+image_name,x0,y0,x1,y1,x2,y2,x3,y3))
            txt_commas_train.write('{},{},{},{},{},{},{},{},{}\n'.format(NEW_IMG_PATH+image_name,x0,y0,x1,y1,x2,y2,x3,y3))
            txt_custom_train.write('{} {} {} {} {} {} {} {},{}\n'.format(NEW_IMG_PATH+image_name,x0,y0,x1,y1,x2,y2,x3,y3))
        elif test_train[image_id-1]=='test':
            csv_test.write('{},{},{},{},{},{},{},{},{}\n'.format(NEW_IMG_PATH+image_name,x0,y0,x1,y1,x2,y2,x3,y3))
            txt_spaces_test.write('{} {} {} {} {} {} {} {} {}\n'.format(NEW_IMG_PATH+image_name,x0,y0,x1,y1,x2,y2,x3,y3))
            txt_commas_test.write('{},{},{},{},{},{},{},{},{}\n'.format(NEW_IMG_PATH+image_name,x0,y0,x1,y1,x2,y2,x3,y3))
            txt_custom_test.write('{} {} {} {} {} {} {} {},{}\n'.format(NEW_IMG_PATH+image_name,x0,y0,x1,y1,x2,y2,x3,y3))
        else:
            csv_valid.write('{},{},{},{},{},{},{},{},{}\n'.format(NEW_IMG_PATH+image_name,x0,y0,x1,y1,x2,y2,x3,y3))
            txt_spaces_valid.write('{} {} {} {} {} {} {} {} {}\n'.format(NEW_IMG_PATH+image_name,x0,y0,x1,y1,x2,y2,x3,y3))
            txt_commas_valid.write('{},{},{},{},{},{},{},{},{}\n'.format(NEW_IMG_PATH+image_name,x0,y0,x1,y1,x2,y2,x3,y3))
            txt_custom_valid.write('{} {} {} {} {} {} {} {},{}\n'.format(NEW_IMG_PATH+image_name,x0,y0,x1,y1,x2,y2,x3,y3))
csv_train.close()
csv_test.close()
csv_valid.close()

txt_spaces_train.close()
txt_commas_train.close()
txt_custom_train.close()

txt_spaces_test.close()
txt_commas_test.close()
txt_custom_test.close()

txt_spaces_valid.close()
txt_commas_valid.close()
txt_custom_valid.close()



'''
#Creating two .txt files: with spaces, and comma separated
csv_train=open(OUTPUT_FILE_TRAIN,'r')
csv_test=open(OUTPUT_FILE_TEST,'r')

lines=csv_file.readlines()
txt_spaces=open(OUTPUT_TEXT_SPACES,'w')
txt_commas=open(OUTPUT_TEXT_COMMAS,'w')
txt_custom=open(OUTPUT_TEXT_CUSTOM,'w')
count=0
for line in lines:
    if count==0: #Skipping the header
        count+=1
        continue

    #Writing txt with comma separation
    txt_commas.write(line)

    #Writing txt with space separation
    spaced_line=line.replace(","," ")
    txt_spaces.write(spaced_line)

    #Writing txt with last space as comma
    split_line=line.split(',')
    custom_line=""
    length_entries=len(split_line)
    i=1
    for elem in split_line:
        if i==length_entries:
            custom_line=custom_line[:-2]+','+elem
        else:
            custom_line=custom_line+elem+' '
        i+=1
    txt_custom.write(custom_line)

txt_commas.close()
txt_spaces.close()
txt_custom.close()
csv_file.close()
'''
