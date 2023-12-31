%%Startup
clear
clc
close all

%{
Description: Takes in a .csv file from an eye gaze calibration, as well as a .csv file from the corresponding NDI tool measurements. Synchronizes the two and creates a merged file called eye_NDI_merged.csv 
saved in a specified directory.
Also creates a merged csv with only the part with calibration information. Also trims the corresponding video accordingly.

Author: Alexandre Banks
Date: August 16, 2023
Affiliation: Robotics and Control Laboratory

%}
            
FREQUENCY=60; 
PERIOD=1/FREQUENCY;

DATA_DIR='E:/Alexandre_EyeGazeProject_Extra/eyecorner_userstudy2_converted';
folder_list=dir(DATA_DIR);
dirnames={folder_list([folder_list.isdir]).name};
num_dir=length(dirnames);
for m=[1:num_dir]
    if dirnames{m}(1)=='P'
        p_num=str2double(dirnames{m}(2:end));
        if p_num>22
        NDI_ROOT=[DATA_DIR,'/',dirnames{m},'/','NDI_Data'];
        NDI_list=dir(NDI_ROOT);
        NDI_names={NDI_list(~[NDI_list.isdir]).name};
        num_ndi_names=length(NDI_names);
        for p=[1:num_ndi_names]
            NDI_DATAPATH=[NDI_ROOT,'/',NDI_names{p}];
            gaze_name=NDI_names{p};
            gaze_name=erase(gaze_name,'.csv');
            GAZE_DATAPATH=[DATA_DIR,'/',dirnames{m},'/','EyeGaze_Data','/','gazelog_',gaze_name,'.csv'];
            FULL_DATAPATH=[DATA_DIR,'/',dirnames{m},'/','full_data_merged_',gaze_name,'.csv'];
            CALIB_ONLY_DATAPATH=[DATA_DIR,'/',dirnames{m},'/','calib_only_merged_',gaze_name,'.csv'];
            %-----------------------------<Setting Parameters>-------------------
            %GAZE_DATAPATH="/ubc/ece/home/ts/grads/alexandre/Documents/eye_gaze_data/test2_july17/Quaternions/gazelog_17-07-2023_08-51-00.csv";
            %NDI_DATAPATH="/ubc/ece/home/ts/grads/alexandre/Documents/eye_gaze_data/test2_july17/Quaternions/ndi_headpose_17-07-2023_08-50-51.csv";
            
            %FULL_DATAPATH="/ubc/ece/home/ts/grads/alexandre/Documents/eye_gaze_data/test2_july17/Quaternions/full_data.csv";
            %CALIB_ONLY_DATAPATH="/ubc/ece/home/ts/grads/alexandre/Documents/eye_gaze_data/test2_july17/Quaternions/calib_only.csv";
            
            
            %VIDEO_DATAPATH="/ubc/ece/home/ts/grads/alexandre/Documents/eye_gaze_data/test2_july17/Quaternions/eyeVideo_07-17-2023_08-51-00.avi";
            %OUTPUT_VIDEO_DATAPATH="/ubc/ece/home/ts/grads/alexandre/Documents/eye_gaze_data/test2_july17/Quaternions/eyeVideo_calib_only.avi";

            
            
            %%Loading Data
            gaze_data=readmatrix(GAZE_DATAPATH);
            if p_num==25 && p==11
                gaze_data=gaze_data(2:end,:);
            end
            ndi_data=readmatrix(NDI_DATAPATH);
            %video_data=VideoReader(VIDEO_DATAPATH);
            
            %%Separating Tools in NDI Data
            
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
            
            
            
            %%Time Synching data
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
                gaze_data_new=gaze_data(i_gazetime:end,[1,3,13:24,34:48]);
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
                gaze_data_new=gaze_data(:,[1,3,13:24,34:48]);
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
            
            %%Trimming NDI data to get rid of NaN (tool not in view) at the start and
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
            
            
            
            %%Normalizing Quaternions
            for i=[1:num_tools]
                [num_data,~]=size(ndi_data_new{i});
                for j=[1:num_data]
                    quat=quaternion(ndi_data_new{i}(j,7),ndi_data_new{i}(j,8),ndi_data_new{i}(j,9),ndi_data_new{i}(j,10));
                    norm_quat=normalize(quat);
                    quat_array=compact(norm_quat); %Array of quaternion magnitudes
                    ndi_data_new{i}(j,7:10)=quat_array;
                end
            
            end
            %%Displaying Pose before interpolation
            %{
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
            %}
            
            
            %%Deleting Gaze Data longer than NDI data
            %First, Deleting Gaze data that is longer (in time) than the ndi data
            smaller_ind=zeros(1,num_tools);
            check=false;
            for i=[1:num_tools]
                ind=find(gaze_data_new(:,1)>ndi_data_new{i}(end,2));
                if ~isempty(ind)
                    smaller_ind(i)=ind(1); %Takes first index where gaze_data is longer (in time) than ndi_data
                    check=true;
                end
            end
            if check==true
                trunc_ind=min(smaller_ind);
                gaze_data_new(trunc_ind:end,:)=[];
            end
            
            %%Fixing rows with NaNs in ndi_data_new
            %Any rows with NaNs are deleted
            for i=[1:num_tools]
                nan_vals=find(isnan(ndi_data_new{i}(:,4))); %Finds rows with NaN values
                ndi_data_new{i}(nan_vals,:)=[];
            end
            
            %%Deleting NDI Data Longer than gaze data
            %max_gaze_time=gaze_data_new(end,1);
            %max_ndi_time=ndi_data_new{1}(end,2);
            %if max_ndi_time>max_gaze_time
             %   smaller_val_inds=find(ndi_data_new{1}(:,2)<=max_gaze_time+0.1);
              %  trim_ind=smaller_val_inds(end);
               % ndi_data_new{1}=ndi_data_new{1}(1:trim_ind,:);
                %ndi_data_new{2}=ndi_data_new{2}(1:trim_ind,:);
            %end

            %%Running Interpolation (to make equal data lengths)
            %(Every Eye Gaze Entry has a corresponding head rotation/translation)
            
            %Running interpolation on translation part (using 1D spline inmterpolation)
            ndi_data_interp=cell(1,num_tools);
            for i=[1:num_tools]
                sample_time=ndi_data_new{i}(:,2);
                query_time=gaze_data_new(:,1); %Time that we want to interpolate to
                if length(sample_time)<2
                    continue;
                end
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
            
                    %quat_start=quaternion(ndi_data_new{i}(ndi_count-1,7),ndi_data_new{i}(ndi_count-1,8),ndi_data_new{i}(ndi_count-1,9),ndi_data_new{i}(ndi_count-1,10));
                    %quat_end=quaternion(ndi_data_new{i}(ndi_count,7),ndi_data_new{i}(ndi_count,8),ndi_data_new{i}(ndi_count,9),ndi_data_new{i}(ndi_count,10));
                    
                    quat_start=[ndi_data_new{i}(ndi_count-1,7),ndi_data_new{i}(ndi_count-1,8),ndi_data_new{i}(ndi_count-1,9),ndi_data_new{i}(ndi_count-1,10)];
                    quat_end=[ndi_data_new{i}(ndi_count,7),ndi_data_new{i}(ndi_count,8),ndi_data_new{i}(ndi_count,9),ndi_data_new{i}(ndi_count,10)];

                    
                    interval_frac=(gaze_data_new(j,1)-ndi_data_new{i}(ndi_count-1,2))/(ndi_data_new{i}(ndi_count,2)-ndi_data_new{i}(ndi_count-1,2));
                    if isnan(interval_frac)
                        continue;
                    end
                     
                    quat_array=quatinterp(quat_start,quat_end,interval_frac,'slerp');
                    %quat_array=compact(qi);
                    ndi_data_interp{i}(j,7:10)=quat_array; %Updates quaternion
                    ndi_data_interp{i}(j,1)=ndi_data_new{i}(ndi_count-1,1); %Updates tool ID
                    ndi_data_interp{i}(j,2)=gaze_data_new(j,1); %Updates time
                    ndi_data_interp{i}(j,3)=ndi_data_new{i}(ndi_count-1,3); %Updates frame number
                    ndi_data_interp{i}(j,11)=ndi_data_new{i}(ndi_count-1,end); %Updates quality measure
                 %   ndi_data_interp{i}=
                end
            end
            
            
            %%Displaying Results to Check Interpolation
            %{
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
            %}
            %%Formating All Interpolated data
            %Data saved as a .csv with gaze_data,tool_1_data,tool_2_data...
            %so: timestamp,video_frame,calib_x,calib_y,calib_valid,tool ID 1, NDI
            %frame, Tx,Ty,Tz,Q0,Qx,Qy,Qz,tracking quality, Tool ID 2, NDI Frame, Tx,...
            [data_rows,~]=size(gaze_data_new);
            full_data=gaze_data_new;
            non_tools=0;
            for i=[1:num_tools]
                if isempty(ndi_data_interp{i})
                    non_tools=non_tools+1;
                    continue;
                end
                full_data=[full_data,ndi_data_interp{i}(:,1),ndi_data_interp{i}(:,3:end)];
            end
            num_tools=num_tools-non_tools;
            
            
            %%Calibration Only Data
            %Extracting rows of full_data that correspond to calibration points only
            calib_ind=find(full_data(:,29)==1); %CHANGE THIS
            calib_only_data=full_data(calib_ind,:);
            
            
            %{
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
            %}
            
            %%Saving Synchronized/Interpolated Data
            %Creating data header
            data_header={'timestamp','framecount','pupil_right_x','pupil_right_y',...
                'pupil_right_width','pupil_right_height','pupil_right_angle','pupil_found_right',...
                'glint0_right_x','glint0_right_y','glint1_right_x','glint1_right_y',...
                'glint2_right_x','glint2_right_y','pupil_left_x','pupil_left_y',...
                'pupil_left_width','pupil_left_height','pupil_left_angle','pupil_found_left',...
                'glint0_left_x','glint0_left_y','glint1_left_x','glint1_left_y',...
                'glint2_left_x','glint2_left_y','calib_x','calib_y','calib_valid'};
            tool_id_ind=30;
            for i=[1:num_tools]
                tool_id_number=num2str(full_data(1,tool_id_ind));
                data_header=[data_header,{['Tool ID',tool_id_number,],['NDI Frame',tool_id_number],['Tx',tool_id_number],['Ty',tool_id_number],['Tz',tool_id_number],['Q0',tool_id_number],['Qx',tool_id_number],['Qy',tool_id_number],['Qz',tool_id_number],['Track Quality',tool_id_number]}];
                tool_id_ind=tool_id_ind+10;
            end
            %[row_dat,col_dat]=size(full_data);
            %if col_dat<49
             %   continue;
            %end
            %Saving Full Data
            full_data_table=array2table(full_data,'VariableNames',data_header);
            %full_data_table.Properties.VariableNames(1:length(data_header))=data_header;
            writetable(full_data_table,FULL_DATAPATH);
            
            %Saving Calibration Only .csv
            calib_only_table=array2table(calib_only_data,'VariableNames',data_header);
            %calib_only_table.VariableNames=data_header;
            writetable(calib_only_table,CALIB_ONLY_DATAPATH);
        end
        end
    end
end

