clear
clc
close all

%Testing initial calibration
calib_init_path='../resources/calib_only_merged_Calib_Init.csv';
calib_init_data=readmatrix(calib_init_path);


%For looping through all
%{

%Setting Parameters
data_root='E:/Alexandre_EyeGazeProject/eyecorner_userstudy_converted';
extensions={'Calib_Init','Eval_Init','Calib_Right','Calib_Left','Calib_Up',...
    'Calib_Down','Eval_Straight','Eval_Right','Eval_Left','Eval_Up','Eval_Down'};



%Getting list of subfolders
folder_list=dir(data_root);
dirnames={folder_list([folder_list.isdir]).name};
num_dir=length(dirnames);

for m=[1:num_dir]
    if dirnames{m}(1)=='P' %We have a participant and run calibrations and/evaluations
        calib_init_path=[data_root,'/',dirnames{m},'/calib_only_merged_Calib_Init.csv'];
        calib_init_data=readmatrix(calib_init_path);
        %Do the initial calibraiton

    end


end
%}
