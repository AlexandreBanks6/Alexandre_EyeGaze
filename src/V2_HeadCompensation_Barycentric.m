%{
Title: Barycentric Interpolation 
Author: Alexandre Banks Gadbois
Description:
We run the head compensation by interpolating polynomial regressions using
barycentric coordinates as the weighting function. 
Algorithm Steps:

Training:
1. Fit 6 regressions f1...f6 (x and y) for each eye (so twelve regressions for each eye).
2. For each regression, record the average inner and outer eye corner position
(C_innerx,C_innery, C_outerx, C_outery) of each eye

Implementation:
1. If two eye corners are currently detected & >3 sets of two averages were detected:
    1.1 use average of both corners
    1.2 use average of calibration corners
2. Else 1 eye corner is currently detected || <3 sets of two averafes were
detected & >3 calibration corners match the current corner type
(inner/outer)
    2.1 Use available corners to solve equation 5 and 3
    2.1 Weight the given polynomial and predict the POG

%}


clear
clc
close all


%Looping Through All Participants
data_root='F:/Alexandre_EyeGazeProject/eyecorner_userstudy2_converted';
%Getting list of subfolders
folder_list=dir(data_root);
dirnames={folder_list([folder_list.isdir]).name};
num_dir=length(dirnames);

%Global Params:
CALIB_THRESHOLD=5;
EVAL_THRESHOLD=3; %Threshold to be considered a valid evaluation trial

mean_acc_results=[];
%Looping for all participants
for m=[1:num_dir]
    if dirnames{m}(1)=='P' %We have a participant and run calibrations and/evaluations
        disp(dirnames{m})
        calib_init_data=readmatrix([data_root,'/',dirnames{m},'/calib_only_merged_Calib_Init.csv']);
        check_calib=checkDetection(calib_init_data,CALIB_THRESHOLD);
        if (check_calib==true)


        end



    end

end
