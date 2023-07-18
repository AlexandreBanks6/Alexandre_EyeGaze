%% Startup
clear
clc
close all

%% 
%{
Description: Takes in a .csv file from an eye gaze calibration, as well as a .csv file from the corresponding NDI tool measurements. Synchronizes the two and creates a merged file called eye_NDI_merged.csv 
saved in a specified directory.
Also creates a merged csv with only the part with calibration information. Also trims the corresponding video accordingly.

Author: Alexandre Banks
Date: July 12, 2023
Affiliation: Robotics and Control Laboratory

%}

%-----------------------------<Setting Parameters>-------------------
GAZE_DATAPATH="/ubc/ece/home/ts/grads/alexandre/Documents/eye_gaze_data/test2_july17/Quaternions/gazelog_17-07-2023_08-51-00.csv";
VIDEO_DATAPATH="/ubc/ece/home/ts/grads/alexandre/Documents/eye_gaze_data/test2_july17/Quaternions/eyeVideo_07-17-2023_08-51-00.avi";
NDI_DATAPATH="/ubc/ece/home/ts/grads/alexandre/Documents/eye_gaze_data/test2_july17/Quaternions/ndi_headpose_17-07-2023_08-50-51.csv";
FULL_DATAPATH="/ubc/ece/home/ts/grads/alexandre/Documents/eye_gaze_data/test2_july17/Quaternions/full_data.csv";
CALIB_ONLY_DATAPATH="/ubc/ece/home/ts/grads/alexandre/Documents/eye_gaze_data/test2_july17/Quaternions/calib_only.csv";
OUTPUT_VIDEO_DATAPATH="/ubc/ece/home/ts/grads/alexandre/Documents/eye_gaze_data/test2_july17/Quaternions/eyeVideo_calib_only.avi";


FREQUENCY=60; 
PERIOD=1/FREQUENCY;


%% Loading Data
gaze_data=readmatrix(GAZE_DATAPATH);
ndi_data=readmatrix(NDI_DATAPATH);
video_data=VideoReader(VIDEO_DATAPATH);

%% Separating Tools in NDI Data

%Fixing gaze_data time to have decimals after 10th digit (places after
%decimal are fractions of a second)
gaze_time=gaze_data(:,1);
gaze_time=gaze_time/1e6;

%Create a cell array where each cell has the matrix of data corresponding
%to one tool, cell indices correspond to tool ID
unique_ids=unique(ndi_data(:,1)); %Returns vector of ID values with no repetitions
num_tools=length(unique_ids);

ndi_data_cell=cell(1,num_tools);

for i=[1:num_tools] %Loop for number of tools, and insert corresponding rows
    tool_ind=find(ndi_data(:,1)==unique_ids(i));
    ndi_data_cell{i}=ndi_data(tool_ind,:);

end



%% Time Synching data
%Create a single file with: eye gaze data, tool 1 data, tool 2 data, tool 3 data, ... etc.

%Converting the ndi_time to start with the initial time + the frame number
%and frequency, in unix time
ndi_time=cell(1,num_tools);
for i=[1:num_tools]
    ndi_time{i}=ndi_data_cell{i}(:,2); %Extracts the time column
    ndi_frames=ndi_data_cell{i}(:,3);
    for(j=2:length(ndi_time)) %starts at second index
        ndi_time{i}(j)=ndi_time{i}(j-1)+(ndi_frames(j)-ndi_frames(j-1))*PERIOD;
    end
end

%Assume first tool to use to find start, then we align other tools

%ndi_time is in unix time with fractions of seconds shown
%Find the first closest timestamp between both the ndi_time and the
%gaze_time (this is to syncrhonize)
[min_gazetime,i_gazetime]=min(abs(ndi_time{1}(1)-gaze_time));
[min_nditime,i_nditime]=min(abs(gaze_time(1)-ndi_time{1})); 

%Refromats both datasets to start at the same point
%Only taking the timestamp,framecount,calib_x,calib_y,calib_valid
%in gaze_data
%Taking all of ndi_data
ndi_data_new=cell(1,num_tools);
if(min_gazetime<min_nditime) %We started gaze data before ndi data
    gaze_data_new=gaze_data(i_gazetime:end,[1,3,end-2,end-1,end]);
    gaze_time=gaze_time(i_gazetime:end);
    gaze_data_new(:,1)=gaze_time;
    for i=[1:num_tools] %Loops for the number of tools
        if i==1
            ndi_data_new{i}=ndi_data_cell{i};
            ndi_data_new{i}(:,2)=ndi_time{i};
        else %Another tool and not the first one
            [min_time,i_closest]=min(abs(ndi_time{1}(1)-ndi_time{i}));
            ndi_data_new{i}=ndi_data_cell{i}(i_closest:end,:);
            ndi_time{i}=ndi_time{i}(i_closest:end);
            ndi_data_new{i}(:,2)=ndi_time{i};
        end
    end
else %We started ndi data before gaze data
    gaze_data_new=gaze_data(:,[1,3,end-2,end-1,end]);
    gaze_data_new(:,1)=gaze_time;
    for i=[1:num_tools] %Loops for the number of tools
        if i==1
            ndi_data_new{i}=ndi_data_cell{i}(i_nditime:end,:);
            ndi_time{i}=ndi_time{i}(i_nditime:end);
            ndi_data_new{i}(:,2)=ndi_time{i};
        else
            [min_time,i_closest]=min(abs(ndi_time{1}(1)-ndi_time{i}));
            ndi_data_new{i}=ndi_data_cell{i}(i_closest:end,:);
            ndi_time{i}=ndi_time{i}(i_closest:end);
            ndi_data_new{i}(:,2)=ndi_time{i};
        end
    end
end


%Formatting both time columns to start at zero
gaze_data_new(:,1)=gaze_data_new(:,1)-gaze_data_new(1,1);
first_ndi_time=ndi_data_new{1}(1,2); %Takes first tool as the zero
for i=[1:num_tools] %Loops for the number of tools
    ndi_data_new{i}(:,2)=ndi_data_new{i}(:,2)-first_ndi_time;
end

%% Trimming NDI data to get rid of NaN (tool not in view) at the start and
%end

%Trimming NaN from start
lowest_val_ind=1;
lowest_tool=1;
for i=[1:num_tools]
    [num_data,~]=size(ndi_data_new{i});
    last_val=NaN(1);
    for j=[1:num_data]
        cur_val=ndi_data_new{i}(j,4);
        if ((~isnan(cur_val))&&isnan(last_val))
            if(lowest_val_ind<j)
                lowest_val_ind=j;
                lowest_tool=i;
            end
        
            break;
        end
        last_val=cur_val;
    end

end

%Trims the data
ndi_data_new{lowest_tool}=ndi_data_new{lowest_tool}(lowest_val_ind:end,:);
%Looops to update the gaze data and other tools

[min_val,i_gaze]=min(abs(ndi_data_new{lowest_tool}(1,2)-gaze_data_new(:,1)));
gaze_data_new=gaze_data_new(i_gaze:end,:);
gaze_data_new(:,1)=gaze_data_new(:,1)-gaze_data_new(1,1); %Zero starts

for i=[1:num_tools]
    if i==lowest_tool
        continue;
    end
    [min_val,i_gaze]=min(abs(ndi_data_new{lowest_tool}(1,2)-ndi_data_new{i}(:,2)));
    ndi_data_new{i}=ndi_data_new{i}(i_gaze:end,:);
    ndi_data_new{i}(:,2)=ndi_data_new{i}(:,2)-ndi_data_new{i}(1,2); %Zero starts

end

ndi_data_new{lowest_tool}(:,2)=ndi_data_new{lowest_tool}(:,2)-ndi_data_new{lowest_tool}(1,2);


%Trimming NaN from end
[highest_val_ind,~]=size(ndi_data_new{1});
highest_tool=1;
for i=[1:num_tools]
    [num_data,~]=size(ndi_data_new{i});
    last_val=NaN(1);
    for j=[num_data:-1:1]
        cur_val=ndi_data_new{i}(j,4);
        if ((~isnan(cur_val))&&isnan(last_val))
            if(highest_val_ind>j)
                highest_val_ind=j;
                highest_tool=i;
            end
        
            break;
        end
        last_val=cur_val;
    end
end

%Trims the data
[min_val,i_gaze]=min(abs(ndi_data_new{highest_tool}(highest_val_ind,2)-gaze_data_new(:,1)));
gaze_data_new=gaze_data_new(1:i_gaze,:);

for i=[1:num_tools]
    if i==highest_tool
        continue;
    end
    [min_val,i_gaze]=min(abs(ndi_data_new{highest_tool}(highest_val_ind,2)-ndi_data_new{i}(:,2)));
    ndi_data_new{i}=ndi_data_new{i}(1:i_gaze,:);
end

ndi_data_new{highest_tool}=ndi_data_new{highest_tool}(1:highest_val_ind,:);



%% Normalizing Quaternions
for i=[1:num_tools]
    [num_data,~]=size(ndi_data_new{i});
    for j=[1:num_data]
        quat=quaternion(ndi_data_new{i}(j,7),ndi_data_new{i}(j,8),ndi_data_new{i}(j,9),ndi_data_new{i}(j,10));
        norm_quat=normalize(quat);
        quat_array=compact(norm_quat); %Array of quaternion magnitudes
        ndi_data_new{i}(j,7:10)=quat_array;
    end

end
%% Displaying Pose before interpolation
figname='Position Before Interpolation';
figure('Name',figname);
[num_data,~]=size(ndi_data_new{1});
for i=[1:num_data-1]
    
    pos=ndi_data_new{1}(1,4:6);
    quat=quaternion(ndi_data_new{1}(i,7),ndi_data_new{1}(i,8),ndi_data_new{1}(i,9),ndi_data_new{1}(i,10));
    
    pos2=ndi_data_new{2}(1,4:6);
    quat2=quaternion(ndi_data_new{2}(i,7),ndi_data_new{2}(i,8),ndi_data_new{2}(i,9),ndi_data_new{2}(i,10));
    
    poseplot(quat,pos);
    hold on;
    poseplot(quat2,pos2);
    hold off;
    pause(ndi_data_new{1}(i+1,2)-ndi_data_new{1}(i,2));
end



%% Deleting Gaze Data longer than NDI data
%First, Deleting Gaze data that is longer (in time) than the ndi data
smaller_ind=zeros(1,num_tools);
for i=[1:num_tools]
    ind=find(gaze_data_new(:,1)>ndi_data_new{i}(end,2));
    smaller_ind(i)=ind(1); %Takes first index where gaze_data is longer (in time) than ndi_data
end
trunc_ind=min(smaller_ind);
gaze_data_new(trunc_ind:end,:)=[];

%% Fixing rows with NaNs in ndi_data_new
%Any rows with NaNs are deleted
for i=[1:num_tools]
    nan_vals=find(isnan(ndi_data_new{i}(:,4))); %Finds rows with NaN values
    ndi_data_new{i}(nan_vals,:)=[];
end

%% Running Interpolation (to make equal data lengths)
%(Every Eye Gaze Entry has a corresponding head rotation/translation)

%Running interpolation on translation part (using 1D spline inmterpolation)
ndi_data_interp=cell(1,num_tools);
for i=[1:num_tools]
    sample_time=ndi_data_new{i}(:,2);
    query_time=gaze_data_new(:,1); %Time that we want to interpolate to
    interp_trans=interp1(sample_time,ndi_data_new{i}(:,4:6),query_time,'spline');
    ndi_data_interp{i}(:,4:6)=interp_trans;
end
%Running interpolation on quaternion part
for i=[1:num_tools] %Does interpolation for each tool
    [num_gaze,~]=size(gaze_data_new);
    [num_ndi,~]=size(ndi_data_new{i}); 
    ndi_count=2;
    for j=[1:num_gaze]
        if ndi_count>num_ndi
            break;
        end

        if gaze_data_new(j,1)>ndi_data_new{i}(ndi_count,2)
            while(1)
                ndi_count=ndi_count+1; %Loops until we find time in ndi_data_new greater than gaze data
                if gaze_data_new(j,1)<=ndi_data_new{i}(ndi_count,2)
                    break;
                end

            end
        end

        quat_start=quaternion(ndi_data_new{i}(ndi_count-1,7),ndi_data_new{i}(ndi_count-1,8),ndi_data_new{i}(ndi_count-1,9),ndi_data_new{i}(ndi_count-1,10));
        quat_end=quaternion(ndi_data_new{i}(ndi_count,7),ndi_data_new{i}(ndi_count,8),ndi_data_new{i}(ndi_count,9),ndi_data_new{i}(ndi_count,10));
        interval_frac=(gaze_data_new(j,1)-ndi_data_new{i}(ndi_count-1,2))/(ndi_data_new{i}(ndi_count,2)-ndi_data_new{i}(ndi_count-1,2));
        
        qi=quatinterp(quat_start,quat_end,interval_frac,'slerp');
        quat_array=compact(qi);
        ndi_data_interp{i}(j,7:10)=quat_array; %Updates quaternion
        ndi_data_interp{i}(j,1)=ndi_data_new{i}(ndi_count-1,1); %Updates tool ID
        ndi_data_interp{i}(j,2)=gaze_data_new(j,1); %Updates time
        ndi_data_interp{i}(j,3)=ndi_data_new{i}(ndi_count-1,3); %Updates frame number
        ndi_data_interp{i}(j,11)=ndi_data_new{i}(ndi_count-1,end); %Updates quality measure
     %   ndi_data_interp{i}=
    end
end


%% Displaying Results to Check Interpolation
figname='Position After Interpolation';
figure('Name',figname);
[num_data,~]=size(ndi_data_interp{1});
for i=[1:num_data-1]
    
    pos=ndi_data_interp{1}(1,4:6);
    quat=quaternion(ndi_data_interp{1}(i,7),ndi_data_interp{1}(i,8),ndi_data_interp{1}(i,9),ndi_data_interp{1}(i,10));
    
    pos2=ndi_data_interp{2}(1,4:6);
    quat2=quaternion(ndi_data_interp{2}(i,7),ndi_data_interp{2}(i,8),ndi_data_interp{2}(i,9),ndi_data_interp{2}(i,10));
    
    poseplot(quat,pos);
    hold on;
    poseplot(quat2,pos2);
    hold off;
    pause(ndi_data_interp{1}(i+1,2)-ndi_data_interp{1}(i,2));
end
%% Formating All Interpolated data
%Data saved as a .csv with gaze_data,tool_1_data,tool_2_data...
%so: timestamp,video_frame,calib_x,calib_y,calib_valid,tool ID 1, NDI
%frame, Tx,Ty,Tz,Q0,Qx,Qy,Qz,tracking quality, Tool ID 2, NDI Frame, Tx,...
[data_rows,~]=size(gaze_data_new);
full_data=gaze_data_new;
for i=[1:num_tools]
    full_data=[full_data,ndi_data_interp{i}(:,1),ndi_data_interp{i}(:,3:end)];
end


%% Calibration Only Data
%Extracting rows of full_data that correspond to calibration points only
calib_ind=find(full_data(:,5)==1);
calib_only_data=full_data(calib_ind,:);
[num_rows,~]=size(calib_only_data);

%Extracting frames of the video that correspond only to calibration points
counter=1;
output_video=VideoWriter(OUTPUT_VIDEO_DATAPATH); %Creates Video Writer Object
open(output_video);
num_frames=video_data.NumFrames;
for i=[1:num_frames]
    frame=readFrame(video_data);
    if(counter==num_rows)
        break
    end
    if(i==calib_only_data(counter,2))
        counter=counter+1;
        writeVideo(output_video,frame);
    end
end
close(output_video);

%% Saving Synchronized/Interpolated Data
%Creating data header
data_header={'timestamp','video_frame','calib_x','calib_y','calib_valid'};
for i=[1:num_tools]
    data_header=[data_header,{['Tool ID',num2str(i)],['NDI Frame',num2str(i)],['Tx',num2str(i)],['Ty',num2str(i)],['Tz',num2str(i)],['Q0',num2str(i)],['Qx',num2str(i)],['Qy',num2str(i)],['Qz',num2str(i)],['Track Quality',num2str(i)]}];

end
%Saving Full Data
full_data_table=array2table(full_data,'VariableNames',data_header);
%full_data_table.Properties.VariableNames(1:length(data_header))=data_header;
writetable(full_data_table,FULL_DATAPATH);

%Saving Calibration Only .csv
calib_only_table=array2table(calib_only_data,'VariableNames',data_header);
%calib_only_table.VariableNames=data_header;
writetable(calib_only_table,CALIB_ONLY_DATAPATH);

