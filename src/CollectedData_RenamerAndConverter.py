import os
import re
from datetime import datetime

DATE_FORMAT_VIDEO='%m-%d-%Y_%H-%M-%S'
DATE_FORMAT='%d-%m-%Y_%H-%M-%S'
#------------------Function Definitions-----------------
def save_time(filename):
    root,ext=os.path.splitext(filename)
    if ext=='.avi': #We have a video which has a different naming convention
        date_str=root[-19:]
        date_obj=datetime.strptime(date_str,DATE_FORMAT_VIDEO)
    else:
        date_str=root[-19:]
        date_obj=datetime.strptime(date_str,DATE_FORMAT)
        
    return date_obj
    




data_root='/media/alexandre/My Passport/Alexandre_EyeGazeProject/eyecorner_userstudy_converted'
guide_csv_root='/media/alexandre/My Passport/Alexandre_EyeGazeProject/eyecorner_userstudy_converted/DataCollection_ParticipantList.csv'

for entry in os.scandir(data_root):
    if entry.is_dir():
        entry_name=entry.name
        csv_file=open(guide_csv_root,mode='r',encoding='utf-8-sig')
        lines=csv_file.readlines()
        for line in lines:
            line=line.strip()
            line_list=line.split(",")
            if not line_list[0]=='':
                line_num=int(line_list[0])
                entry_num=re.sub("[P]","",entry_name)
                entry_num=int(entry_num)
                
                if line_num==entry_num and entry_num<9 and entry_num>7: #The number in the csv matches that in the folder and we can now loop through each element in the folder and rename according to the order in the csv
                    #Looping for video data
                    video_root=data_root+'/'+entry_name+'/EyeGaze_Data'
                    vid_files=os.listdir(video_root)
                    vid_files.sort(key=save_time)
                    avi_count=1
                    caliblog_count=1
                    gazelog_count=1
                    for filename in vid_files: #Renaming the files
                        root,ext=os.path.splitext(filename)
                        if filename.endswith('.avi'): #Have our avi we are going to change the name of
                            old_name=video_root+'/'+filename
                            new_name=video_root+'/'+root[0:8]+'_'+line_list[avi_count]+'.avi'
                            os.rename(old_name,new_name)
                            avi_count+=1
                        if filename.endswith('.txt') and root[0]=='c': #Have our caliblog that we are going to rename
                            old_name=video_root+'/'+filename
                            new_name=video_root+'/'+root[0:8]+'_'+line_list[caliblog_count]+'.txt'
                            os.rename(old_name,new_name)
                            caliblog_count+=1
                        if filename.endswith('.txt') and root[0]=='g': #Have our gazelog that we are going to rename
                            old_name=video_root+'/'+filename
                            new_name=video_root+'/'+root[0:7]+'_'+line_list[gazelog_count]+'.csv'
                            os.rename(old_name,new_name)
                            gazelog_count+=1
                    #Looping for NDI data
                    ndi_root=data_root+'/'+entry_name+'/NDI_Data'
                    ndi_files=os.listdir(ndi_root)
                    ndi_files.sort(key=save_time)
                    ndi_count=1
                    for filename in ndi_files: #Renaming the files
                        if filename.endswith('.csv'): #Have our avi we are going to change the name of
                            old_name=ndi_root+'/'+filename
                            new_name=ndi_root+'/'+line_list[ndi_count]+'.csv'
                            os.rename(old_name,new_name)
                            ndi_count+=1

        csv_file.close()
                        
            
        

