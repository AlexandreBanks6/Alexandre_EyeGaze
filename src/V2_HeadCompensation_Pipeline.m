clear
clc
close all

%% Running Eye Corner Compensation
%Looping Through All Participants
data_root='E:/Alexandre_EyeGazeProject_Extra/eyecorner_userstudy2_converted';
%data_root='F:/Alexandre_EyeGazeProject/eyecorner_userstudy2_converted';
results_folder='C:/Users/playf/OneDrive/Documents/UBC/Thesis/Paper_FiguresAndResults/Poly_InitEval_RawResults';
%Getting list of subfolders
folder_list=dir(data_root);
dirnames={folder_list([folder_list.isdir]).name};
num_dir=length(dirnames);

%Global Params:
CALIB_THRESHOLD=5;
EVAL_THRESHOLD=3; %Threshold to be considered a valid evaluation trial

mean_acc_results=[];
mean_acc_target_results=[];
%Looping for all participants
for m=[1:num_dir]
    if dirnames{m}(1)=='P' %We have a participant and run calibrations and/evaluations
        disp(dirnames{m})
        %if strcmp(dirnames{m},'P22')||strcmp(dirnames{m},'P23')||strcmp(dirnames{m},'P24')||strcmp(dirnames{m},'P25')||strcmp(dirnames{m},'P26')||strcmp(dirnames{m},'P27')||strcmp(dirnames{m},'P28')
            %Reading In Data
            calib_init_data=readmatrix([data_root,'/',dirnames{m},'/calib_only_merged_Calib_Init.csv']);
            check_calib=checkDetection(calib_init_data,CALIB_THRESHOLD);
                       
            
            if (check_calib==true)
                [train_cell,dist_cell,avg_corners]=getRegressionData(calib_init_data,CALIB_THRESHOLD); %Also gets the average eye corner location at the calibration
                if length(dist_cell)==0
                    continue;
                end
                model_poly=robustRegressor(train_cell);
                
                %training Max's PG Estimator (for PG_hat)
                
                PG_Estimation_Models=maxFitPGRegressor(train_cell);

                %Training Max's POG compensation model
                calib_max=readmatrix([data_root,'/',dirnames{m},'/calib_only_merged_Calib_Comp_Rotate.csv']);

                max_compensation_data=prepMaxCompensationData(calib_max,model_poly,dist_cell,PG_Estimation_Models);
                
                max_compensation_models=maxCompensationTraining(max_compensation_data);



                %Getting the Data To Train the Compensation Model
                %calib_onedot=readmatrix([data_root,'/',dirnames{m},'/calib_only_merged_Calib_Comp_Rotate.csv']);
                %data_cell={calib_onedot};

                
                calib_lift1_dot=readmatrix([data_root,'/',dirnames{m},'/calib_only_merged_Calib_Comp_Lift1_dot.csv']);
                %Only keeps central percentage of data
                calib_lift1_dot=cropTrainingData(calib_lift1_dot);

                calib_lift2_dot=readmatrix([data_root,'/',dirnames{m},'/calib_only_merged_Calib_Comp_Lift2_dot.csv']);
                calib_lift2_dot=cropTrainingData(calib_lift2_dot);

                calib_lift3_dot=readmatrix([data_root,'/',dirnames{m},'/calib_only_merged_Calib_Comp_Lift3_dot.csv']);
                calib_lift3_dot=cropTrainingData(calib_lift3_dot);

                calib_lift4_dot=readmatrix([data_root,'/',dirnames{m},'/calib_only_merged_Calib_Comp_Lift4_dot.csv']);
                calib_lift4_dot=cropTrainingData(calib_lift4_dot);   

                calib_lift5_dot=readmatrix([data_root,'/',dirnames{m},'/calib_only_merged_Calib_Comp_Lift5_dot.csv']);
                calib_lift5_dot=cropTrainingData(calib_lift5_dot);
                data_cell={calib_lift1_dot,calib_lift2_dot,calib_lift3_dot,calib_lift4_dot,calib_lift5_dot};
                
                %{
                calib_lift1_8dot=readmatrix([data_root,'/',dirnames{m},'/calib_only_merged_Calib_Comp_Lift1_8point.csv']);
                calib_lift2_8dot=readmatrix([data_root,'/',dirnames{m},'/calib_only_merged_Calib_Comp_Lift2_8point.csv']);
                calib_lift3_8dot=readmatrix([data_root,'/',dirnames{m},'/calib_only_merged_Calib_Comp_Lift3_8point.csv']);
                calib_lift4_8dot=readmatrix([data_root,'/',dirnames{m},'/calib_only_merged_Calib_Comp_Lift4_8point.csv']);
                calib_lift5_8dot=readmatrix([data_root,'/',dirnames{m},'/calib_only_merged_Calib_Comp_Lift5_8point.csv']);
                data_cell={calib_lift1_8dot,calib_lift2_8dot,calib_lift3_8dot,calib_lift4_8dot,calib_lift5_8dot};
                %}
                
                
                %-------Training My Compensation Model

                compensation_data_onedot=prepCompensationData(data_cell,model_poly,dist_cell,avg_corners);

                right_data=compensation_data_onedot{1};

                left_data=compensation_data_onedot{2};

                del_POG_x_right=right_data(:,1);
                del_POG_y_right=right_data(:,2);
                predictors_right=right_data(:,3:7);
                
                del_POG_x_left=left_data(:,1);
                del_POG_y_left=left_data(:,2);
                predictors_left=left_data(:,3:7);
                mdl_right_x=[];
                mdl_right_y=[];
                mdl_left_x=[];
                mdl_left_y=[];
                if ~(sum(~isnan(predictors_right(:,1)))<4 || sum(~isnan(predictors_right(:,2)))<4 || sum(~isnan(predictors_right(:,3)))<4 || sum(~isnan(predictors_right(:,4)))<4)
                                    
                [mdl_right_x,mdl_right_y]=fitCompensationRegressor(del_POG_x_right,del_POG_y_right,predictors_right);
                end
                if ~(sum(~isnan(predictors_left(:,1)))<4 || sum(~isnan(predictors_left(:,2)))<4 || sum(~isnan(predictors_left(:,3)))<4 || sum(~isnan(predictors_left(:,4)))<4)
                [mdl_left_x,mdl_left_y]=fitCompensationRegressor(del_POG_x_left,del_POG_y_left,predictors_left);
                end



                data_mat=[calib_lift1_dot;calib_lift2_dot;calib_lift3_dot;calib_lift4_dot;calib_lift5_dot];
    
                variance_cell_poly=findPOGVariance(data_mat,model_poly,dist_cell,avg_corners,mdl_right_x,mdl_right_y,mdl_left_x,mdl_left_y,PG_Estimation_Models,max_compensation_models,'poly');
                variance_cell_max=findPOGVariance(data_mat,model_poly,dist_cell,avg_corners,mdl_right_x,mdl_right_y,mdl_left_x,mdl_left_y,PG_Estimation_Models,max_compensation_models,'max');
                variance_cell_comp=findPOGVariance(data_mat,model_poly,dist_cell,avg_corners,mdl_right_x,mdl_right_y,mdl_left_x,mdl_left_y,PG_Estimation_Models,max_compensation_models,'interp');
                %------Evaluating the model
                
                %Evaluation data
                %{
                calib_lift1_dot=readmatrix([data_root,'/',dirnames{m},'/calib_only_merged_Calib_Comp_Lift1_dot.csv']);
                calib_lift2_dot=readmatrix([data_root,'/',dirnames{m},'/calib_only_merged_Calib_Comp_Lift2_dot.csv']);
                calib_lift3_dot=readmatrix([data_root,'/',dirnames{m},'/calib_only_merged_Calib_Comp_Lift3_dot.csv']);
                calib_lift4_dot=readmatrix([data_root,'/',dirnames{m},'/calib_only_merged_Calib_Comp_Lift4_dot.csv']);
                calib_lift5_dot=readmatrix([data_root,'/',dirnames{m},'/calib_only_merged_Calib_Comp_Lift5_dot.csv']);
                data_mat=[calib_lift1_dot;calib_lift2_dot;calib_lift3_dot;calib_lift4_dot;calib_lift5_dot];
                %}
                
                %{
                eval_lift1_8dot=readmatrix([data_root,'/',dirnames{m},'/calib_only_merged_Calib_Comp_Lift1_8point.csv']);
                eval_lift2_8dot=readmatrix([data_root,'/',dirnames{m},'/calib_only_merged_Calib_Comp_Lift2_8point.csv']);
                eval_lift3_8dot=readmatrix([data_root,'/',dirnames{m},'/calib_only_merged_Calib_Comp_Lift3_8point.csv']);
                eval_lift4_8dot=readmatrix([data_root,'/',dirnames{m},'/calib_only_merged_Calib_Comp_Lift4_8point.csv']);
                eval_lift5_8dot=readmatrix([data_root,'/',dirnames{m},'/calib_only_merged_Calib_Comp_Lift5_8point.csv']);

                data_mat=[eval_lift1_8dot;eval_lift2_8dot;eval_lift3_8dot;eval_lift4_8dot;eval_lift5_8dot];
                %}
                
                %{
                eval_lift1_8dot=readmatrix([data_root,'/',dirnames{m},'/calib_only_merged_Eval_Lift1_8Point.csv']);
                eval_lift2_8dot=readmatrix([data_root,'/',dirnames{m},'/calib_only_merged_Eval_Lift2_8Point.csv']);
                eval_lift3_8dot=readmatrix([data_root,'/',dirnames{m},'/calib_only_merged_Eval_Lift3_8Point.csv']);

                data_mat=[eval_lift1_8dot;eval_lift2_8dot;eval_lift3_8dot];
                %}
                
                %eval_lift1_9dot=readmatrix([data_root,'/',dirnames{m},'/calib_only_merged_Eval_Lift1_9Point.csv']);
%                 bool_check=checkEvalData(eval_lift1_9dot);
%                 if ~bool_check
%                     eval_lift1_9dot=[];
%                 end
                %eval_lift2_9dot=readmatrix([data_root,'/',dirnames{m},'/calib_only_merged_Eval_Lift2_9Point.csv']);
%                 bool_check=checkEvalData(eval_lift2_9dot);
%                 if ~bool_check
%                     eval_lift2_9dot=[];
%                 end
                %eval_lift3_9dot=readmatrix([data_root,'/',dirnames{m},'/calib_only_merged_Eval_Lift3_9Point.csv']);

%                 bool_check=checkEvalData(eval_lift3_9dot);
%                 if ~bool_check
%                     eval_lift3_9dot=[];
%                 end
                data_mat=readmatrix([data_root,'/',dirnames{m},'/calib_only_merged_Eval_Init_9Point.csv']);
                %data_mat=[eval_lift1_9dot;eval_lift2_9dot;eval_lift3_9dot];
                
                
                [mean_accuracies,total_results]=evalModelsRegressComp(data_mat,model_poly,dist_cell,avg_corners,mdl_right_x,mdl_right_y,mdl_left_x,mdl_left_y,PG_Estimation_Models,max_compensation_models,variance_cell_poly,variance_cell_max,variance_cell_comp);
                
               
                mean_per_target=findMeanPerTarget(total_results);


                %Participant Number
                part_string=dirnames{m}(2:3);
                part_num=str2double(part_string);
                if length(mean_per_target)==297
                    mean_acc_target_results=[mean_acc_target_results;[part_num,mean_per_target]];
 
                end

                mean_acc_results=[mean_acc_results;[part_num,mean_accuracies]];
            end
        %end
    end

end

%% Saving Results:
%Saving the per-target results
per_target_results_file=[results_folder,'/per_target_results.csv'];
csvwrite(per_target_results_file,mean_acc_target_results);

%Saving the mean acc results
mean_table=array2table(mean_acc_results);
mean_table.Properties.VariableNames(1:10)={'participant','acc_right_poly',...
    'acc_left_poly','acc_combined_poly','acc_right_comp','acc_left_comp',...
    'acc_combined_comp','acc_right_max','acc_left_max','acc_combined_max'};
mean_results_file=[results_folder,'/mean_acc_results.csv'];
writetable(mean_table,mean_results_file);

mean_acc_total=mean(mean_acc_results,1,'omitnan');
disp(mean_acc_total);







%-----------------------------<Function Definitions>------------------
function acc=getAccuracy_mm(tx,ty,pog_x,pog_y)
%Takes in the target and pog locations in the x- and y-directions as a function of the percentage of 
% screen size and returns the accuracy after scaling by the screen size in
% mm
SCREEN_WIDTH_mm=284.48;
SCREEN_HEIGHT_mm=213.36;

err_x=tx-pog_x;
err_y=ty-pog_y;

err_x_mm=SCREEN_WIDTH_mm*(err_x/100);
err_y_mm=SCREEN_HEIGHT_mm*(err_y/100);

acc=sqrt(err_x_mm^2+err_y_mm^2);
end


function bool_check=checkEvalData(eval_data)
    bool_check=true;
    if eval_data(2,27)==50 && eval_data(2,28)==50
        bool_check=false;
    end

end


function mean_per_target=findMeanPerTarget(total_results)
%Returns a vector mean_per_target which has elements: 
% mean_poly, mean_comp, mean_max, target_x, target_y, mean_poly,
% mean_comp,... and repeats for all targets
%Each of the means is for right/left/combined

%{
        total_results: matrix with columns:
        frame_no, accuracy_right_poly, accuracy_left_poly, accuracy_combined_poly,
        accuracy_right_mycomp, accuracy_left_mycomp, accuracy_combined_mycomp,
        actual_del_pog_right_x, actual_del_pog_right_y, actual_del_pog_left_x,
        actual_del_pog_left_y, 
        estimated_del_pog_right_x, estimated_del_pog_right_y,estimated_del_pog_left_x,
        estimated_del_pog_left_y,
        right_inner_x, right_inner_y, right_outer_x, right_outer_y
        left_inner_x, left_inner_y, left_outer_x, left_outer_y
        alpha_right, alpha_left
        t_x, t_y,
        accuracy_right_max, accuracy_left_max, accuracy_combined_max
%}
    mean_per_target=[];
    [row_n,~]=size(total_results);
    old_tx=total_results(1,26);
    old_ty=total_results(1,27);

    curr_seg=[];
    for i=[1:row_n]
        
        curr_seg=[curr_seg;total_results(i,:)];
        if old_tx~=total_results(i,26) || old_ty~=total_results(i,27)   %We have a new target point

            mean_per_target=[mean_per_target,mean(curr_seg(:,[2:7,28:30]),1,'omitnan'),old_tx,old_ty];
            
            old_tx=total_results(i,26);
            old_ty=total_results(i,27);
            curr_seg=[];
        end



    end
    mean_per_target=[mean_per_target,mean(curr_seg(:,[2:7,28:30]),1,'omitnan'),old_tx,old_ty];





end

%------------------<Initial Calibration Functions>------------------
function [trainCell,dist_cell,avg_corners]=getRegressionData(data_matrix,thresh)
%Returns trainCell which has a cell for each of the training types, as well
%as dist_cell which has up to six cells for the distance between each of
%the pg vectors for the left and right eyes such that:
%d_01_right,d_02_right,d_12_right,d_01_left,d_02_left,d_12_left

%Also returns the average corner positions as a cell such that:
%right_inner_x,right_inner_y,right_outer_x,right_outer_y,left_inner_x,...


pg_types={'pg0_left','pg1_left','pg2_left','pg0_right','pg1_right','pg2_right'};
trainCell={};

    for i=[1:length(pg_types)]
        trainMatrix=getIndividualRegressionData(data_matrix,pg_types{i},thresh);
        if all(isnan(trainMatrix))
            continue
    
        else
            trainCell{end+1}={pg_types{i},trainMatrix};
        end
           
    
    
    end
    
    %Finding inter-glint distance
    dist_header={'d_01_right','d_02_right','d_12_right','d_01_left','d_02_left','d_12_left'};
    cell_count=1;
    for i=[1:length(dist_header)]
        dist=findPgDistance(dist_header{i},trainCell);
        if all(isnan(dist))
            continue    
        else
            dist_cell{cell_count,1}=dist_header{i};
            dist_cell{cell_count,2}=dist;
            cell_count=cell_count+1;

        end

    end
    for i=[1:length(trainCell)]

        old_train_cell=trainCell{i}{2};
        trainCell{i}{2}=old_train_cell(~isnan(old_train_cell(:,1)),:);


    end

    if cell_count==1
        dist_cell=cell(0);
    end


    %Finding the average corner locations
    corner_data=data_matrix(:,50:57);
    %Change the corner locations to be in 640x480 not the 1280x480 so
    %corner_data(:,1)=corner_data(:,1);
    %corner_data(:,3)=corner_data(:,3);
    avg_corners=mean(corner_data,1,'omitnan');
    
end



function train_matrix=getIndividualRegressionData(data_matrix,pg_type,thresh)
    %Function tha returns data to train polynomial regressor from the toal
    %data matrix in eye gaze tracking. 
    %Returns: train_matrix with columns:
    %pg(pg_type)_x,pg(pg_type)_y,target_x,target_y,pupil_x,pupil_y
    %Where pg_type is either 0, 1, or 2
    %If train_matrix is not valid then train_matrix=NaN
    
    %Data index indexes for: pupil_x, glint_x, pupily, glint_y,
    %target_x,target_y
    switch pg_type
        case 'pg0_left'
            data_indx=[15,21,16,22,27,28];
        case 'pg0_right'
            data_indx=[3,9,4,10,27,28];
        case 'pg1_left'
            data_indx=[15,23,16,24,27,28];
        case 'pg1_right'
            data_indx=[3,11,4,12,27,28];
        case 'pg2_left'
            data_indx=[15,25,16,26,27,28];
        case 'pg2_right'
            data_indx=[3,13,4,14,27,28];
        otherwise
            disp('Invalid PG Type')
    end
    
    data_raw=data_matrix(:,data_indx);
    [row_n,col_n]=size(data_raw); 
    %We check to see if we have enough unique points, and also remove any
    %NaNs from data_raw
    pupil_detect_count=0; %Counts the number of pupils detected per unique calib point
    calib_pastx=data_matrix(1,27);
    calib_pasty=data_matrix(1,28);
    switched=false;
    train_matrix=[];
    for i=[1:row_n]
        if (data_matrix(i,27)~=calib_pastx)||(data_matrix(i,28)~=calib_pasty)
            switched=false;
            calib_pastx=data_matrix(i,27);
            calib_pasty=data_matrix(i,28);

        end

        if (~anynan(data_raw(i,[1:4]))) && (~switched) %We have a new target point and a valid glint detection
            switched=true;
            train_matrix=[train_matrix;[data_raw(i,2)-data_raw(i,1),data_raw(i,4)-data_raw(i,3),data_raw(i,5),data_raw(i,6),data_raw(i,1),data_raw(i,3)]];
            pupil_detect_count=pupil_detect_count+1;
        elseif ~anynan(data_raw(i,[1:4]))
            train_matrix=[train_matrix;[data_raw(i,2)-data_raw(i,1),data_raw(i,4)-data_raw(i,3),data_raw(i,5),data_raw(i,6),data_raw(i,1),data_raw(i,3)]];
        else
            train_matrix=[train_matrix;[NaN,NaN,NaN,NaN,NaN,NaN]];
        end


    end
    if pupil_detect_count<thresh
        train_matrix=nan; %Not a valid sample of points, too few detections
    end



end



function [dist]=findPgDistance(distance_type,train_cell)
    dist_header={'d_01_right','d_02_right','d_12_right','d_01_left','d_02_left','d_12_left'};
    switch distance_type
        case dist_header{1}
            search_pgs={'pg0_right','pg1_right'};
        case dist_header{2}
            search_pgs={'pg0_right','pg2_right'};
        case dist_header{3}
            search_pgs={'pg1_right','pg2_right'};
        case dist_header{4}
            search_pgs={'pg0_left','pg1_left'};
        case dist_header{5}
            search_pgs={'pg0_left','pg2_left'};
        case dist_header{6}
            search_pgs={'pg1_left','pg2_left'};
    end

    ind_2=NaN;
    ind_1=NaN;

    for i=[1:length(train_cell)]
        if strcmp(train_cell{i}(1),search_pgs(1))
            ind_1=i;
        end
        if strcmp(train_cell{i}(1),search_pgs(2))
            ind_2=i;
        end

    end

    if isnan(ind_1)||isnan(ind_2)
        dist=NaN;

    else
        dist=mean(sqrt((train_cell{ind_2}{2}(:,1)-train_cell{ind_1}{2}(:,1)).^2+(train_cell{ind_2}{2}(:,2)-train_cell{ind_1}{2}(:,2)).^2),"omitnan");
    end
    



end


function valid=checkDetection(calib_data,thresh)
    [row_n,col_n]=size(calib_data);
    pupil_detect_count=0; %Counts the number of pupils detected per unique calib point
    calib_pastx=calib_data(1,27);
    calib_pasty=calib_data(1,28);
    switched=false;
    valid=false;
    for i=[1:row_n] %Loops for the number of rows
        if (calib_data(i,27)~=calib_pastx)||(calib_data(i,28)~=calib_pasty)
            switched=false;
            calib_pastx=calib_data(i,27);
            calib_pasty=calib_data(i,28);

        end
        if ((calib_data(i,8)==1)||(calib_data(i,20)==1)) && (~switched)
            switched=true;
            pupil_detect_count=pupil_detect_count+1;
        end


    end
    if pupil_detect_count>=thresh
        valid=true;
    end

end


function robust_regressor_output=robustRegressor(train_cell)
    %Output is a cell with columns: pg_type, rmse, model parameters
    %pg_type is: pg0_leftx, pg0_lefty,pg1_leftx,pg1_lefty... for pg1->pg3
    %and right/left
    num_pg_detect=length(train_cell);

    if num_pg_detect==0
        robust_regressor_output=nan;
    else %We have detections and proceed
        robust_regressor_output=cell(num_pg_detect*2,4); %Final entry are the model parameters
        for i=[1:num_pg_detect] %Looping for the detections
            train_data=train_cell{i}{2};
            [rmse_x,rmse_y,b_x,b_y]=customRegressor(train_data);
            %Saving x-results
            robust_regressor_output{i*2-1,1}=strcat(train_cell{i}{1},'_x');
            robust_regressor_output{i*2-1,2}=size(train_data,1);
            robust_regressor_output{i*2-1,3}=b_x;
            robust_regressor_output{i*2-1,4}=(rmse_x+rmse_y)/2;

            robust_regressor_output{i*2,1}=strcat(train_cell{i}{1},'_y');
            robust_regressor_output{i*2,2}=size(train_data,1);
            robust_regressor_output{i*2,3}=b_y;
            robust_regressor_output{i*2,4}=(rmse_x+rmse_y)/2;

        end
        

    end



end


function [rmse_x,rmse_y,b_x,b_y]=customRegressor(train_data)
    %Training data is a nx4 vector where col1=pg_x, col2=pg_y, col3=target_x, col4=target_y
    %Output are 6 model parameters and the residual error for x and y
    %The model parameters are such that
    %b(1)+b(2)*pg_x^2+b(3)*pg_x*pg_y+b(4)*pg_y^2+b(5)*pg_x+b(6)*pg_y
    %The tuning constant is set to 4.685 for bisquare
    REGRESS_TUNE=4.2;
    REGRESS_FUNC='huber';
    pg_x=train_data(:,1);
    pg_y=train_data(:,2);

    t_x=train_data(:,3);
    t_y=train_data(:,4);
    

    %--------------Iteratively Weighted Least Squares---------------
    [predictors_x,predictors_y]=customPolynomial(pg_x,pg_y);
    
    

    %Fitting POGx
    %Using iteratively weighted least squares
    [b_x,stats_x]=robustfit(predictors_x,t_x,REGRESS_FUNC,REGRESS_TUNE); %Uses iteratively weighted least squares

    %Fitting POGy
    [b_y,stats_y]=robustfit(predictors_y,t_y,REGRESS_FUNC,REGRESS_TUNE); %Uses iteratively weighted least squares
    
    residual_x=stats_x.resid;
    residual_y=stats_y.resid;

    rmse_x=sqrt(mean(residual_x.^2));
    rmse_y=sqrt(mean(residual_y.^2));
end


function [predictors_x,predictors_y]=customPolynomial(pg_x,pg_y)
    %predictors=[pg_x.^2,pg_x.*pg_y,pg_y.^2,pg_x,pg_y];
    %predictors_x=[pg_x,pg_y,pg_x.^2];
    %predictors_y=[pg_y,pg_x.^2,pg_x.*pg_y,pg_x.^2.*pg_y];
    predictors_x=[pg_x.^2,pg_x.*pg_y,pg_y.^2,pg_x,pg_y];
    predictors_y=[pg_x.^2,pg_x.*pg_y,pg_y.^2,pg_x,pg_y];
    %predictors_x=[pg_x.^2,pg_x.*pg_y,pg_y.^2,pg_x,pg_y];
    %predictors_y=[pg_x.^2,pg_x.*pg_y,pg_y.^2,pg_x,pg_y];

end






%-------------------<Head Compensation Functions>--------------------
function [compensation_data]=prepCompensationData(data_cell,model_cell,dist_cell,avg_corners)
    %{
    Input: data_cell has is the data for  training the compensation model

    compensationData is a cell array with two cells having:
    cell 1: del_POG_x_right,del_POG_y_right,del_corner_inner_x_right,del_corner_inner_y_right,
    del_corner_outer_x_right,del_corner_outer_y_right,alpha_right,t_x,t_y,
    then we append head rotations/poses:
    Tx1,Ty1,Tz1,Q01,Qx1,Qy1,Qz1,Tx2,Ty2,Tz2,Q02,Qx2,Qy2,Qz2

    cell 2: del_POG_x_left,del_POG_y_left,del_corner_inner_x_left,del_corner_inner_y_left,
    del_corner_outer_x_left,del_corner_outer_y_left,alpha_left,t_x,t_y

    values are replaced with nan if they don't exist
    target locations are included for evaluation

    del_corner is defined as: del_corner=corner_calib-corner_curr
    %}
    
    left_headers={'pg0_left_x','pg0_left_y','pg1_left_x','pg1_left_y','pg2_left_x','pg2_left_y'};
    right_headers={'pg0_right_x','pg0_right_y','pg1_right_x','pg1_right_y','pg2_right_x','pg2_right_y'};
    check_model_right=any(ismember(model_cell(:,1),right_headers));
    check_model_left=any(ismember(model_cell(:,1),left_headers));

    %Init compensation data cell
    compensation_data=cell(1,2);
    for i=[1:length(data_cell)]
        curr_mat=data_cell{i}; 
        [reformatted_data_right,reformatted_data_left]=reformatDataEval(curr_mat);
        if check_model_right            
            error_vec_right=findCalibrationErrors(model_cell,reformatted_data_right,right_headers,dist_cell);
        else
            error_vec_right=nan;
        end

        if check_model_left
            error_vec_left=findCalibrationErrors(model_cell,reformatted_data_left,left_headers,dist_cell);
        else
            error_vec_left=nan;
        end

        if ~all(isnan(error_vec_right(:,1)))
            % reformmated_data_right=frame_no,pg0_rightx,pg0_righty,...,pg2_rightx,pg2_righty,target_x,target_y,right_inner_x,right_inner_y,right_outer_x,right_outer_y,..

            del_corner_inner_x=avg_corners(1)-reformatted_data_right(:,end-3);
            del_corner_inner_y=avg_corners(2)-reformatted_data_right(:,end-2);
            del_corner_outer_x=avg_corners(3)-reformatted_data_right(:,end-1);
            del_corner_outer_y=avg_corners(4)-reformatted_data_right(:,end);
            v_calib_x=avg_corners(1)-avg_corners(3);
            v_calib_y=avg_corners(2)-avg_corners(4);
            v_curr_x=reformatted_data_right(:,end-3)-reformatted_data_right(:,end-1);
            v_curr_y=reformatted_data_right(:,end-2)-reformatted_data_right(:,end);

            alpha=2.*atan(sqrt((v_calib_x-v_curr_x).^2+(v_calib_y-v_curr_y).^2)./...
                sqrt((v_calib_x+v_curr_x).^2+(v_calib_y+v_curr_y).^2));

            compensation_data{1}=[compensation_data{1};error_vec_right(:,1),...
                error_vec_right(:,2),del_corner_inner_x,del_corner_inner_y,...
                del_corner_outer_x,del_corner_outer_y,alpha,...
                error_vec_right(:,3),error_vec_right(:,4),...
                curr_mat(:,[[32:38],[42:48]])];

        else
            compensation_data{1}=[compensation_data{1};nan(1,23)];
        end

        if ~all(isnan(error_vec_left(:,1)))
            del_corner_inner_x=avg_corners(5)-reformatted_data_left(:,end-3);
            del_corner_inner_y=avg_corners(6)-reformatted_data_left(:,end-2);
            del_corner_outer_x=avg_corners(7)-reformatted_data_left(:,end-1);
            del_corner_outer_y=avg_corners(8)-reformatted_data_left(:,end);

            v_calib_x=avg_corners(5)-avg_corners(7);
            v_calib_y=avg_corners(6)-avg_corners(8);
            v_curr_x=reformatted_data_left(:,end-3)-reformatted_data_left(:,end-1);
            v_curr_y=reformatted_data_left(:,end-2)-reformatted_data_left(:,end);

            alpha=2.*atan(sqrt((v_calib_x-v_curr_x).^2+(v_calib_y-v_curr_y).^2)./...
                sqrt((v_calib_x+v_curr_x).^2+(v_calib_y+v_curr_y).^2));

            compensation_data{2}=[compensation_data{2};error_vec_left(:,1),...
                error_vec_left(:,2),del_corner_inner_x,del_corner_inner_y,...
                del_corner_outer_x,del_corner_outer_y,alpha,...
                error_vec_left(:,3),error_vec_left(:,4),...
                curr_mat(:,[[32:38],[42:48]])];

        else
            compensation_data{2}=[compensation_data{2};nan(1,23)];
        end
        
    end
   



end



function [reformatted_data_right,reformatted_data_left]=reformatDataEval(eval_data)
    %Returns the data to compute the conventional model in the format of: 
    % reformmated_data_right=frame_no,pg0_rightx,pg0_righty,...,pg2_rightx,pg2_righty,target_x,target_y,right_inner_x,right_inner_y,right_outer_x,right_outer_y,..
    % reformatted_data_left=frame_no,pg0_leftx,pg0_lefty,...,pg2_leftx,pg2_lefty,target_x,target_y,left_inner_x,left_inner_y,left_outer_x,left_outer_y,..
    glintspupils_right_ind=[3,4,9,10,11,12,13,14]; %Contains the glints and pupil positions such that pupil_x,pupil_y,glint0_x,glint0_y...
    glintspupils_left_ind=[15,16,21,22,23,24,25,26];

    glintspupils_right=eval_data(:,glintspupils_right_ind);
    glintspupils_left=eval_data(:,glintspupils_left_ind);

    reformatted_data_right=[eval_data(:,2),glintspupils_right(:,3)-glintspupils_right(:,1),...
        glintspupils_right(:,4)-glintspupils_right(:,2),glintspupils_right(:,5)-glintspupils_right(:,1),...
        glintspupils_right(:,6)-glintspupils_right(:,2),glintspupils_right(:,7)-glintspupils_right(:,1),...
        glintspupils_right(:,8)-glintspupils_right(:,2),eval_data(:,27),eval_data(:,28),eval_data(:,50:53)];

    reformatted_data_left=[eval_data(:,2),glintspupils_left(:,3)-glintspupils_left(:,1),...
        glintspupils_left(:,4)-glintspupils_left(:,2),glintspupils_left(:,5)-glintspupils_left(:,1),...
        glintspupils_left(:,6)-glintspupils_left(:,2),glintspupils_left(:,7)-glintspupils_left(:,1),...
        glintspupils_left(:,8)-glintspupils_left(:,2),eval_data(:,27),eval_data(:,28),eval_data(:,54:57)];
    

end



function error_vec=findCalibrationErrors(model_cell,reformatted_data,header,dist_cell)
    %Returns a vector with the error in the POG estimation for the data in
    %reformatted_data
    %Returns: del_POG_x,del_POG_y,t_x,t_y
    %Error is defined as del_POG_x=t_x-POG_x and del_POG_y=t_y-POG_y

    [row_n,~]=size(reformatted_data);
    error_vec=[];
    for i=[1:row_n]
        curr_row=reformatted_data(i,:); %Current data row  
      % reformmated_data_right=frame_no,pg0_rightx,pg0_righty,...,pg2_rightx,pg2_righty,target_x,target_y,right_inner_x,right_inner_y,right_outer_x,right_outer_y,..

        t_x=curr_row(8);
        t_y=curr_row(9);
        %Index of values in row that are not NaN
        nan_indexs=isnan(curr_row(2:7));
        nan_indx_values=find(nan_indexs);
        if length(nan_indx_values)<3 %At least two x,y pairs are detected
            stripped_header=header(~nan_indexs); %Extracts the pg type that is valid for this frame
            
            valid_header=findPgWithAssociatedDistance(stripped_header,dist_cell);

            if length(valid_header)>2
                model_valid_indexes=ismember(model_cell(:,1),valid_header);
                updated_model_cell=model_cell(model_valid_indexes,:);
                [row_new,~]=size(updated_model_cell);
                %Loops for all the number of points used and returns index of largest
                cur_val=updated_model_cell{1,2};
                cur_ind=1;
                for j=[1:2:row_new]
                    if (updated_model_cell{j,2}>cur_val) %Change > to < if using iteratively least squares
                        cur_val=updated_model_cell{j,2};
                        cur_ind=j;
                    end
                end
                model_x=updated_model_cell{cur_ind,3};
                model_y=updated_model_cell{cur_ind+1,3};
                header_x=valid_header{cur_ind};
                header_y=valid_header{cur_ind+1};
                pg_x_ind=ismember(header,header_x);
                pg_y_ind=ismember(header,header_y);

                pgsonly=curr_row(2:7);

                [d_calib,d_curr]=findScalingFactors(dist_cell,valid_header,header,pgsonly);
                
                if isnan(d_calib)||isnan(d_curr)
                    error_vec=[error_vec;[nan,nan,t_x,t_y]];
                    continue
                end
                pg_x=(d_calib/d_curr).*pgsonly(pg_x_ind);
                pg_y=(d_calib/d_curr).*pgsonly(pg_y_ind);

                [predictors_x,predictors_y]=customPolynomial(pg_x,pg_y);

                POG_x=findPOG(model_x,predictors_x);
                POG_y=findPOG(model_y,predictors_y);
                t_x=curr_row(8);
                t_y=curr_row(9);
    
                error_vec=[error_vec;[t_x-POG_x,t_y-POG_y,t_x,t_y]]; %Appends the error as well as the target locations
            else
                error_vec=[error_vec;[nan,nan,t_x,t_y]];
            end


        else
            error_vec=[error_vec;[nan,nan,t_x,t_y]];
        end

    end

end



function valid_header=findPgWithAssociatedDistance(header,dist_cell)
valid_header=cell(0);
    for i=[1:length(dist_cell(:,1))]
        switch dist_cell{i,1}
            case 'd_01_right'
                check_pgs={'pg0_right_x','pg0_right_y','pg1_right_x','pg1_right_y'};
                check_inds=ismember(header,check_pgs);
                valid_header=[valid_header,header{check_inds}];
            case 'd_02_right'
                check_pgs={'pg0_right_x','pg0_right_y','pg2_right_x','pg2_right_y'};
                check_inds=ismember(header,check_pgs);
                valid_header=[valid_header,header{check_inds}];
            case 'd_12_right'
                check_pgs={'pg1_right_x','pg1_right_y','pg2_right_x','pg2_right_y'};
                check_inds=ismember(header,check_pgs);
                valid_header=[valid_header,header{check_inds}];
            case 'd_01_left'
                check_pgs={'pg0_left_x','pg0_left_y','pg1_left_x','pg1_left_y'};
                check_inds=ismember(header,check_pgs);
                valid_header=[valid_header,header{check_inds}];
            case 'd_02_left'
                check_pgs={'pg0_right_x','pg0_right_y','pg2_left_x','pg2_left_y'};
                check_inds=ismember(header,check_pgs);
                valid_header=[valid_header,header{check_inds}];
            case 'd_12_left'
                check_pgs={'pg1_left_x','pg1_left_y','pg2_left_x','pg2_left_y'};
                check_inds=ismember(header,check_pgs);
                valid_header=[valid_header,header{check_inds}];
        end


    end
    valid_header=unique(valid_header);


end


function [d_calib,d_curr]=findScalingFactors(dist_cell,valid_header,overall_header,pgs_only)
    %Finding distance of calibration

    if any(ismember(valid_header,'pg0_right_x')) && any(ismember(valid_header,'pg1_right_x')) && any(ismember(valid_header,'pg2_right_x'))
        dist_names={'d_01_right','d_02_right','d_12_right'};
    elseif any(ismember(valid_header,'pg0_right_x')) && any(ismember(valid_header,'pg1_right_x'))
        dist_names={'d_01_right'};
    elseif any(ismember(valid_header,'pg0_right_x')) && any(ismember(valid_header,'pg2_right_x'))
        dist_names={'d_02_right'};
    elseif any(ismember(valid_header,'pg1_right_x')) && any(ismember(valid_header,'pg2_right_x'))
        dist_names={'d_12_right'};

    elseif any(ismember(valid_header,'pg0_left_x')) && any(ismember(valid_header,'pg1_left_x')) && any(ismember(valid_header,'pg2_left_x'))
        dist_names={'d_01_left','d_02_left','d_12_left'};
    elseif any(ismember(valid_header,'pg0_left_x')) && any(ismember(valid_header,'pg1_left_x'))
        dist_names={'d_01_left'};
    elseif any(ismember(valid_header,'pg0_left_x')) && any(ismember(valid_header,'pg2_left_x'))
        dist_names={'d_02_left'};
    elseif any(ismember(valid_header,'pg1_left_x')) && any(ismember(valid_header,'pg2_left_x'))
        dist_names={'d_12_left'};

    end
    dist_ind=ismember(dist_cell(:,1),dist_names);
    if all(~dist_ind) %We don't have any corresponding distances in the calibration
        d_calib=nan;
        d_curr=nan;
    else
        dist_vec=cell2mat(dist_cell(dist_ind,2));
        d_calib=mean(dist_vec);
    
        %Finding the current inter-glint distance
    
        valid_inds=ismember(overall_header,valid_header);
        valid_pgs=pgs_only(valid_inds);
        x_vals=[];
        y_vals=[];
        for i=[1:2:length(valid_pgs)]
            x_vals=[x_vals,valid_pgs(i)];
            y_vals=[y_vals,valid_pgs(i+1)];
    
        end
        if length(x_vals)==2
            d_curr=sqrt((y_vals(1)-y_vals(2)).^2+(x_vals(1)-x_vals(2)).^2);
    
        elseif length(x_vals)==3
            diff_1=sqrt((y_vals(1)-y_vals(2)).^2+(x_vals(1)-x_vals(2)).^2);
            diff_2=sqrt((y_vals(1)-y_vals(3)).^2+(x_vals(1)-x_vals(3)).^2);
            diff_3=sqrt((y_vals(2)-y_vals(3)).^2+(x_vals(2)-x_vals(3)).^2);
            d_curr=(diff_1+diff_2+diff_3)/3;
        else
            d_curr=nan;
            d_calib=nan;
    
        end
    end
    

end

function POG=findPOG(model,predictors)
%Generalized function to find the POG at run time
POG=model(1)+sum(model(2:end)'.*predictors,'omitnan');


end



%--------------<Head Compensation Evaluation Functions>--------------------
function [predictors_x,predictors_y]=compensationPolynomial(predictors)
%Predictors are in the format: del_corner_inner_x,del_corner_inner_y, del_corner_outer_x,del_corner_outer_y,alpha

%predictors_x=[predictors(:,1),predictors(:,3),predictors(:,2),predictors(:,4),predictors(:,1).*predictors(:,2),predictors(:,3).*predictors(:,4)];
%predictors_y=[predictors(:,1),predictors(:,3),predictors(:,2),predictors(:,4),predictors(:,1).*predictors(:,2),predictors(:,3).*predictors(:,4)];
%predictors_x=[predictors(:,1).*predictors(:,2),predictors(:,3).*predictors(:,4)];
%predictors_y=[predictors(:,1).*predictors(:,2),predictors(:,3).*predictors(:,4)];
%predictors_x=[predictors(:,1).^2,predictors(:,3).^2,predictors(:,1),predictors(:,3),predictors(:,2),predictors(:,4)];
%predictors_y=[predictors(:,2).^2,predictors(:,4).^2,predictors(:,2),predictors(:,4),predictors(:,1),predictors(:,3)];
%predictors_x=[predictors(:,1:4),predictors(:,1).*predictors(:,2),predictors(:,3).*predictors(:,4)];
%predictors_y=[predictors(:,1:4),predictors(:,1).*predictors(:,2),predictors(:,3).*predictors(:,4)];
%predictors_x=[predictors(:,1:4)];
%predictors_y=[predictors(:,1:4)];
%predictors_x=[predictors(:,1:4)];
%predictors_y=[predictors(:,1:4)];
predictors_x=predictors(:,[1,3]);
predictors_y=predictors(:,[2,4]);

end

function [mdl_x,mdl_y]=fitCompensationRegressor(del_POG_x,del_POG_y,predictors)
%Predictors are in the format: del_corner_inner_x,del_corner_inner_y, del_corner_outer_x,del_corner_outer_y,alpha
REGRESS_TUNE=4.2;
REGRESS_FUNC='huber';

%--------------Iteratively Weighted Least Squares---------------
[predictors_x,predictors_y]=compensationPolynomial(predictors);



%Fitting POGx
%Using iteratively weighted least squares
mdl_x=robustfit(predictors_x,del_POG_x,REGRESS_FUNC,REGRESS_TUNE); %Uses iteratively weighted least squares

%Fitting POGy
mdl_y=robustfit(predictors_y,del_POG_y,REGRESS_FUNC,REGRESS_TUNE); %Uses iteratively weighted least squares


end

function train_data_new=cropTrainingData(train_data)
    %Only keeps a central portion of data
    % The start crop is determined by PERCENT_START and the end crop by
    % PERCENT_END
    % of training points for each target
    %Input:
    PERCENT_START=0.15;
    PERCENT_END=0.85;

    train_data_new=[];
    [row_n,~]=size(train_data);
    old_tx=train_data(1,27);
    old_ty=train_data(1,28);
    seg_data=[];

    for i=[1:row_n]
        %Looks for a change in the target
        

        if train_data(i,27)~=old_tx || train_data(i,28)~=old_ty %The target switched locations and we update the new training data and tracking variables
            old_tx=train_data(i,27);
            old_ty=train_data(i,28);
            [row_seg,~]=size(seg_data);
            start_ind=floor(row_seg*PERCENT_START);
            end_ind=floor(row_seg*PERCENT_END);
            if ~mod(start_ind,1) && ~mod(end_ind,1) && start_ind>0 && end_ind>0
                train_data_new=[train_data_new;seg_data(start_ind:end_ind,:)];
                seg_data=[];
            else
                seg_data=[];
            end



        end
        seg_data=[seg_data;train_data(i,:)]; %Updates current segment data
    
    end

    %One more update, because we don't have a switch at the end
    [row_seg,~]=size(seg_data);
    start_ind=floor(row_seg*PERCENT_START);
    end_ind=floor(row_seg*PERCENT_END);
    if ~mod(start_ind,1) && ~mod(end_ind,1) && start_ind>0 && end_ind>0
        train_data_new=[train_data_new;seg_data(start_ind:end_ind,:)];
    else
        disp('nan vals');
    end


end


function [mean_accuracies,total_results]=evalModelsRegressComp(data_mat,model_cell,dist_cell,avg_corners,comp_model_x_right,comp_model_y_right,comp_model_x_left,comp_model_y_left,PG_Estimation_Models,max_compensation_models,variance_cell_poly,variance_cell_max,variance_cell_comp)
    %{
    Inputs:
        data_mat: matrix containing the evaulation data
        model_cell: contains the original polynomial model
        dist_cell: contains the distance between glints at calibration
        avg_corners: contains the average corner positions at calibration
        tree_models: contains a cell array with three columns: col 1: tree
        model names, col 2: tree models, col 3: tree inputs
    Outputs:
        accuracy is reported as sqrt((POG_x-t_x)^2+(POG_y-t_y)^2)

        mean_accuracies: array with accuracies:
        right_poly,left_poly,combined_poly,
        right_tree,left_tree,combined_tree

        total_results: matrix with columns:
        frame_no, accuracy_right_poly, accuracy_left_poly, accuracy_combined_poly,
        accuracy_right_mycomp, accuracy_left_mycomp, accuracy_combined_mycomp,
        actual_del_pog_right_x, actual_del_pog_right_y, actual_del_pog_left_x,
        actual_del_pog_left_y, 
        estimated_del_pog_right_x, estimated_del_pog_right_y,estimated_del_pog_left_x,
        estimated_del_pog_left_y,
        right_inner_x, right_inner_y, right_outer_x, right_outer_y
        left_inner_x, left_inner_y, left_outer_x, left_outer_y
        alpha_right, alpha_left
        t_x, t_y,
        accuracy_right_max, accuracy_left_max, accuracy_combined_max

    %}
    
    left_headers={'pg0_left_x','pg0_left_y','pg1_left_x','pg1_left_y','pg2_left_x','pg2_left_y'};
    right_headers={'pg0_right_x','pg0_right_y','pg1_right_x','pg1_right_y','pg2_right_x','pg2_right_y'};
    check_model_right=any(ismember(model_cell(:,1),right_headers));
    check_model_left=any(ismember(model_cell(:,1),left_headers));

    %Outputs the data as: 
    reformatted_data=reformatData(data_mat);

    total_results=evalAccuracyComp(model_cell,reformatted_data,right_headers,left_headers,check_model_right,check_model_left,dist_cell,avg_corners,comp_model_x_right,comp_model_y_right,comp_model_x_left,comp_model_y_left,PG_Estimation_Models,max_compensation_models,variance_cell_poly,variance_cell_max,variance_cell_comp);

    mean_accuracies=mean(total_results(:,[2:7,end-2:end]),1,'omitnan');
    



end


function total_results=evalAccuracyComp(model_cell,reformatted_data,right_headers,left_headers,check_model_right,check_model_left,dist_cell,avg_corners,comp_model_x_right,comp_model_y_right,comp_model_x_left,comp_model_y_left,PG_Estimation_Models,max_compensation_models,variance_cell_poly,variance_cell_max,variance_cell_comp)
   %{
    Inputs:
    reformatted_data:
    frame_no, pg0_rightx, pg0_righty, ..., pg2_rightx, pg2_righty, pg0_leftx, pg0_lefty,..., pg2_leftx, pg2_lefty,
    right_inner_x,right_inner_y,right_outer_x,right_outer_y,left_inner_x,left_inner_y,left_outer_x,left_outer_y,
    target_x,target_y, pupil_right_x, pupil_right_y, pupil_left_x,pupil_left_y

    model_cell: contains the original polynomial model
    dist_cell: contains the distance between glints at calibration
    avg_corners: contains the average corner positions at calibration
    tree_models: contains a cell array with three columns: col 1: tree
    model names, col 2: tree models, col 3: tree inputs

    Outputs:
    
    total_results: matrix with columns:
    frame_no, accuracy_right_poly, accuracy_left_poly, accuracy_combined_poly,accuracy_right_tree, accuracy_left_tree, accuracy_combined_tree,
    actual_del_pog_right_x, actual_del_pog_right_y, actual_del_pog_left_x,
    actual_del_pog_left_y, 
    estimated_del_pog_right_x, estimated_del_pog_right_y,estimated_del_pog_left_x,
    estimated_del_pog_left_y,
    del_right_inner_x, del_right_inner_y, del_right_outer_x, del_right_outer_y
    del_left_inner_x, del_left_inner_y, del_left_outer_x, del_left_outer_y
    alpha_right, alpha_left
    t_x, t_y,
    max_del_pog_right_x, max_del_pog_right_y, max_del_pog_left_x,max_del_pog_left_y
    max_del_pg_right_x, max_del_pg_right_y, max_del_pg_left_x,max_del_pg_left_y
    accuracy_right_max, accuracy_left_max, accuracy_combined_max
   %}
    NANTHRESH=0; %Number of nan values we tolerate as input to our tree model
    [row_n,~]=size(reformatted_data);
    total_results=[];
    for i=[1:row_n]
        results_row=nan(1,38);
        curr_row=reformatted_data(i,:); %Current data row

        t_x=curr_row(end-5);    %Targets
        t_y=curr_row(end-4);

        results_row(26)=t_x;
        results_row(27)=t_y;
        results_row(1)=curr_row(1); %Frame number


        max_found_count=0;
        %--------------<Finding the right POG first>--------------

        %Inputs to tree regressor compensation in right eye
        del_corner_inner_x=avg_corners(1)-curr_row(14);
        del_corner_outer_x=avg_corners(3)-curr_row(16);

        del_corner_inner_y=avg_corners(2)-curr_row(15);
        del_corner_outer_y=avg_corners(4)-curr_row(17);

        v_calib_x=avg_corners(1)-avg_corners(3);
        v_calib_y=avg_corners(2)-avg_corners(4);
        v_curr_x=curr_row(14)-curr_row(16);
        v_curr_y=curr_row(15)-curr_row(17);

        alpha=2*atan(sqrt((v_calib_x-v_curr_x)^2+(v_calib_y-v_curr_y)^2)/...
            sqrt((v_calib_x+v_curr_x)^2+(v_calib_y+v_curr_y)^2));

        results_row(16:19)=[del_corner_inner_x,del_corner_inner_y,del_corner_outer_x,del_corner_outer_y];
        results_row(24)=alpha;
        if check_model_right
            right_pgs=curr_row(2:7);
            %Index of values in row that are not NaN
            nan_indexs=isnan(right_pgs);
            nan_indx_values=find(nan_indexs);
            if length(nan_indx_values)<3 %At least two x,y pairs are detected
                stripped_header=right_headers(~nan_indexs); %Extracts the pg type that is valid for this frame
                
                valid_header=findPgWithAssociatedDistance(stripped_header,dist_cell);
    
                if length(valid_header)>2
                    model_valid_indexes=ismember(model_cell(:,1),valid_header);
                    updated_model_cell=model_cell(model_valid_indexes,:);
                    [row_new,~]=size(updated_model_cell);
                    %Loops for all the number of points used and returns index of largest
                    cur_val=updated_model_cell{1,2};
                    cur_ind=1;
                    for j=[1:2:row_new]
                        if (updated_model_cell{j,2}>cur_val) %Change > to < if using iteratively least squares
                            cur_val=updated_model_cell{j,2};
                            cur_ind=j;
                        end
                    end
                    model_x=updated_model_cell{cur_ind,3};
                    model_y=updated_model_cell{cur_ind+1,3};
                    header_x=valid_header{cur_ind};
                    header_y=valid_header{cur_ind+1};
                    pg_x_ind=ismember(right_headers,header_x);
                    pg_y_ind=ismember(right_headers,header_y);

                    %Getting the pupil position (right_x, and right_y)
                    pupil_x=curr_row(end-3);
                    pupil_y=curr_row(end-2);

                    %Getting Correct PG Estimation Model
                    PG_Estimation_Headers=PG_Estimation_Models(:,1);
                    pg_estimation_indx=ismember(PG_Estimation_Headers,header_x);
                    pg_estimation_indy=ismember(PG_Estimation_Headers,header_y);


    
                    [d_calib,d_curr]=findScalingFactors(dist_cell,valid_header,right_headers,right_pgs);
                    
                    if ~isnan(d_calib) && ~isnan(d_curr)
                        
                    
                        pg_x=(d_calib/d_curr).*right_pgs(pg_x_ind);
                        pg_y=(d_calib/d_curr).*right_pgs(pg_y_ind);

        
                        [predictors_x,predictors_y]=customPolynomial(pg_x,pg_y);
        
                        POG_x_poly_right=findPOG(model_x,predictors_x);
                        POG_y_poly_right=findPOG(model_y,predictors_y);

                        %-----------<Running Max's approach>--------------
                        if sum(pg_estimation_indx)>0 && sum(pg_estimation_indy)>0
                            PG_model_x=PG_Estimation_Models{pg_estimation_indx,2};
                            PG_model_y=PG_Estimation_Models{pg_estimation_indy,2};
                        if ~isempty(PG_model_x) && ~isempty(PG_model_y)

                            %Estimating PG
                            PG_estim_x=PG_model_x(1)+PG_model_x(2)*pupil_x+PG_model_x(3)*pupil_y;
                            PG_estim_y=PG_model_x(1)+PG_model_y(2)*pupil_x+PG_model_y(3)*pupil_y;

                            %Finding delta PG 
                            delta_pg_x=right_pgs(pg_x_ind)-PG_estim_x;
                            delta_pg_y=right_pgs(pg_y_ind)-PG_estim_y;

                            %Finding appropriate POG compensation model
                            %(max)
                            POG_MaxModels_Headers=max_compensation_models(:,1);
                            pog_estimation_max_indx=ismember(POG_MaxModels_Headers,header_x);
                            pog_estimation_max_indy=ismember(POG_MaxModels_Headers,header_y);

                            pog_max_modelx=max_compensation_models{pog_estimation_max_indx,2};
                            pog_max_modely=max_compensation_models{pog_estimation_max_indy,2};

                            if ~isempty(pog_max_modelx) && ~isempty(pog_max_modely)
                                max_found_count=max_found_count+1;
                                del_POG_max_x_right=pog_max_modelx(1)+pog_max_modelx(2)*delta_pg_x+pog_max_modelx(3)*delta_pg_y;
                                del_POG_max_y_right=pog_max_modely(1)+pog_max_modely(2)*delta_pg_x+pog_max_modely(3)*delta_pg_y;
                                
                                POG_x_max_right=del_POG_max_x_right+POG_x_poly_right;
                                POG_y_max_right=del_POG_max_y_right+POG_y_poly_right;

                                results_row(28)=del_POG_max_x_right;
                                results_row(29)=del_POG_max_y_right;
                                results_row(32)=delta_pg_x;
                                results_row(33)=delta_pg_y;
                                accuracy_max_right=getAccuracy_mm(t_x,t_y,POG_x_max_right,POG_y_max_right);
                                results_row(36)=accuracy_max_right;

                             end

                            
                        end
                        end
                        
                        del_POG_x_actual=t_x-POG_x_poly_right;
                        del_POG_y_actual=t_y-POG_y_poly_right;
                        accuracy_poly=getAccuracy_mm(t_x,t_y,POG_x_poly_right,POG_y_poly_right);
                        results_row(2)=accuracy_poly;
                        results_row(8)=del_POG_x_actual;
                        results_row(9)=del_POG_y_actual;
                        predictors=[del_corner_inner_x,del_corner_inner_y,del_corner_outer_x,del_corner_outer_y,alpha];
                        [predictors_x,predictors_y]=compensationPolynomial(predictors);
                        %Compensating in x-direction
                        accuracy_get=0;
                        if length(comp_model_x_right)>0 && ~any(isnan(predictors_x))
                                accuracy_get=accuracy_get+1;
                                del_POG_x_tree=findCompensation(comp_model_x_right,predictors_x);
                                del_POG_x_tree=del_POG_x_tree;
                                results_row(12)=del_POG_x_tree;
                                POG_x_tree_right=del_POG_x_tree+POG_x_poly_right;

                        end

                        if length(comp_model_y_right)>0 && ~any(isnan(predictors_y))
                                accuracy_get=accuracy_get+1;
                                del_POG_y_tree=findCompensation(comp_model_y_right,predictors_y);
                                del_POG_y_tree=del_POG_y_tree;
                                results_row(13)=del_POG_y_tree;
                                POG_y_tree_right=del_POG_y_tree+POG_y_poly_right;

                        end

                        if accuracy_get>=2
                                accuracy_tree=getAccuracy_mm(t_x,t_y,POG_x_tree_right,POG_y_tree_right);
                                results_row(5)=accuracy_tree;
                        end
    
                        
                    end
                      
                end
        
            end

        end


        %-------------<Finding the left POG next>--------------

        %Inputs to tree regressor compensation in left eye
        del_corner_inner_x=avg_corners(5)-curr_row(18);
        del_corner_outer_x=avg_corners(7)-curr_row(20);

        del_corner_inner_y=avg_corners(6)-curr_row(19);
        del_corner_outer_y=avg_corners(8)-curr_row(21);

        v_calib_x=avg_corners(5)-avg_corners(7);
        v_calib_y=avg_corners(6)-avg_corners(8);
        v_curr_x=curr_row(18)-curr_row(20);
        v_curr_y=curr_row(19)-curr_row(21);

        alpha=2*atan(sqrt((v_calib_x-v_curr_x)^2+(v_calib_y-v_curr_y)^2)/...
            sqrt((v_calib_x+v_curr_x)^2+(v_calib_y+v_curr_y)^2));

        results_row(20:23)=[del_corner_inner_x,del_corner_inner_y,del_corner_outer_x,del_corner_outer_y];
        results_row(25)=alpha;
        if check_model_left
            left_pgs=curr_row(8:13);
            %Index of values in row that are not NaN
            nan_indexs=isnan(left_pgs);
            nan_indx_values=find(nan_indexs);
            if length(nan_indx_values)<3 %At least two x,y pairs are detected
                stripped_header=left_headers(~nan_indexs); %Extracts the pg type that is valid for this frame
                
                valid_header=findPgWithAssociatedDistance(stripped_header,dist_cell);
    
                if length(valid_header)>2
                    model_valid_indexes=ismember(model_cell(:,1),valid_header);
                    updated_model_cell=model_cell(model_valid_indexes,:);
                    [row_new,~]=size(updated_model_cell);
                    %Loops for all the number of points used and returns index of largest
                    cur_val=updated_model_cell{1,2};
                    cur_ind=1;
                    for j=[1:2:row_new]
                        if (updated_model_cell{j,2}>cur_val) %Change > to < if using iteratively least squares
                            cur_val=updated_model_cell{j,2};
                            cur_ind=j;
                        end
                    end
                    model_x=updated_model_cell{cur_ind,3};
                    model_y=updated_model_cell{cur_ind+1,3};
                    header_x=valid_header{cur_ind};
                    header_y=valid_header{cur_ind+1};
                    pg_x_ind=ismember(left_headers,header_x);
                    pg_y_ind=ismember(left_headers,header_y);
    

                    %Getting the pupil position (left_x, and left_y)
                    pupil_x=curr_row(end-1);
                    pupil_y=curr_row(end);


                    %Getting Correct PG Estimation Model
                    PG_Estimation_Headers=PG_Estimation_Models(:,1);
                    pg_estimation_indx=ismember(PG_Estimation_Headers,header_x);
                    pg_estimation_indy=ismember(PG_Estimation_Headers,header_y);



                    [d_calib,d_curr]=findScalingFactors(dist_cell,valid_header,left_headers,left_pgs);
                    
                    if ~isnan(d_calib) && ~isnan(d_curr)
                        pg_x=(d_calib/d_curr).*left_pgs(pg_x_ind);
                        pg_y=(d_calib/d_curr).*left_pgs(pg_y_ind);
        
                        [predictors_x,predictors_y]=customPolynomial(pg_x,pg_y);
        
                        POG_x_poly_left=findPOG(model_x,predictors_x);
                        POG_y_poly_left=findPOG(model_y,predictors_y);


                        %-----------<Running Max's approach>--------------
                        if sum(pg_estimation_indx)>0 && sum(pg_estimation_indy)>0
                            PG_model_x=PG_Estimation_Models{pg_estimation_indx,2};
                            PG_model_y=PG_Estimation_Models{pg_estimation_indy,2};
                        if ~isempty(PG_model_x) && ~isempty(PG_model_y)
   
                            %Estimating PG
                            PG_estim_x=PG_model_x(1)+PG_model_x(2)*pupil_x+PG_model_x(3)*pupil_y;
                            PG_estim_y=PG_model_x(1)+PG_model_y(2)*pupil_x+PG_model_y(3)*pupil_y;

                            %Finding delta PG 
                            delta_pg_x=left_pgs(pg_x_ind)-PG_estim_x;
                            delta_pg_y=left_pgs(pg_y_ind)-PG_estim_y;

                            %Finding appropriate POG compensation model
                            %(max)
                            POG_MaxModels_Headers=max_compensation_models(:,1);
                            pog_estimation_max_indx=ismember(POG_MaxModels_Headers,header_x);
                            pog_estimation_max_indy=ismember(POG_MaxModels_Headers,header_y);

                            pog_max_modelx=max_compensation_models{pog_estimation_max_indx,2};
                            pog_max_modely=max_compensation_models{pog_estimation_max_indy,2};

                            if ~isempty(pog_max_modelx) && ~isempty(pog_max_modely)
                                max_found_count=max_found_count+1;
                                del_POG_max_x_left=pog_max_modelx(1)+pog_max_modelx(2)*delta_pg_x+pog_max_modelx(3)*delta_pg_y;
                                del_POG_max_y_left=pog_max_modely(1)+pog_max_modely(2)*delta_pg_x+pog_max_modely(3)*delta_pg_y;
                                
                                POG_x_max_left=del_POG_max_x_left+POG_x_poly_left;
                                POG_y_max_left=del_POG_max_y_left+POG_y_poly_left;


                                results_row(30)=del_POG_max_x_left;
                                results_row(31)=del_POG_max_y_left;
                                results_row(34)=delta_pg_x;
                                results_row(34)=delta_pg_y;
                                accuracy_max_left=getAccuracy_mm(t_x,t_y,POG_x_max_left,POG_y_max_left);
                                results_row(37)=accuracy_max_left;

                             end

                            
                        end
                        end




                        del_POG_x_actual=t_x-POG_x_poly_left;
                        del_POG_y_actual=t_y-POG_y_poly_left;
                        accuracy_poly=getAccuracy_mm(t_x,t_y,POG_x_poly_left,POG_y_poly_left);
                        results_row(3)=accuracy_poly;
                        results_row(10)=del_POG_x_actual;
                        results_row(11)=del_POG_y_actual;

                        predictors=[del_corner_inner_x,del_corner_inner_y,del_corner_outer_x,del_corner_outer_y,alpha];
                        [predictors_x,predictors_y]=compensationPolynomial(predictors);
                        %Compensating in x-direction
                        accuracy_get=0;
                        if length(comp_model_x_left)>0 && ~any(isnan(predictors_x))
                                accuracy_get=accuracy_get+1;
                                del_POG_x_tree=findCompensation(comp_model_x_left,predictors_x);
                                del_POG_x_tree=del_POG_x_tree;
                                results_row(14)=del_POG_x_tree;
                                POG_x_tree_left=del_POG_x_tree+POG_x_poly_left;

                        end

                        if length(comp_model_y_left)>0 && ~any(isnan(predictors_y))
                                accuracy_get=accuracy_get+1;
                                del_POG_y_tree=findCompensation(comp_model_y_left,predictors_y);
                                del_POG_y_tree=del_POG_y_tree;
                                results_row(15)=del_POG_y_tree;
                                POG_y_tree_left=del_POG_y_tree+POG_y_poly_left;

                        end

                        if accuracy_get>=2
                                accuracy_tree=getAccuracy_mm(t_x,t_y,POG_x_tree_left,POG_y_tree_left);                           
                                results_row(6)=accuracy_tree;
                        end
                        
    
                        
                    end
                      
                end
        
            end

        end
        


        %-------------------<Getting Combined Results>-----------------
        if exist('POG_x_poly_right','var') && exist('POG_y_poly_right','var') && exist('POG_x_poly_left','var') && exist('POG_y_poly_left','var') 
            [POG_combined_x,POG_combined_y]=findCombinedWithVariance(variance_cell_poly,POG_x_poly_right,POG_x_poly_left,POG_y_poly_right,POG_y_poly_left);
            accuracy_combined=getAccuracy_mm(t_x,t_y,POG_combined_x,POG_combined_y);  
            results_row(4)=accuracy_combined;

        end

        if exist('POG_x_tree_right','var') && exist('POG_y_tree_right','var') && exist('POG_x_tree_left','var') && exist('POG_y_tree_left','var') 
            [POG_combined_x,POG_combined_y]=findCombinedWithVariance(variance_cell_comp,POG_x_tree_right,POG_x_tree_left,POG_y_tree_right,POG_y_tree_left);   
            accuracy_combined=getAccuracy_mm(t_x,t_y,POG_combined_x,POG_combined_y); 
            results_row(7)=accuracy_combined;

        end

        %Getting combined max results

        if max_found_count>=2
            [POG_combined_max_x,POG_combined_max_y]=findCombinedWithVariance(variance_cell_max,POG_x_max_right,POG_x_max_left,POG_y_max_right,POG_y_max_left);
            accuracy_combined_max=getAccuracy_mm(t_x,t_y,POG_combined_max_x,POG_combined_max_y); 
            results_row(end)=accuracy_combined_max;



        end


        total_results=[total_results;results_row];
    end

end


function reformatted_data=reformatData(eval_data)
    %Returns the data to evaluate the model in the format of: 
    % frame_no, pg0_rightx, pg0_righty, ..., pg2_rightx, pg2_righty, pg0_leftx, pg0_lefty,..., pg2_leftx, pg2_lefty,
    % right_inner_x,right_inner_y,right_outer_x,right_outer_y,left_inner_x,left_inner_y,left_outer_x,left_outer_y,
    % target_x,target_y, pupil_right_x, pupil_right_y, pupil_left_x,pupil_left_y


    glintspupils_right_ind=[3,4,9,10,11,12,13,14]; %Contains the glints and pupil positions such that pupil_x,pupil_y,glint0_x,glint0_y...
    glintspupils_left_ind=[15,16,21,22,23,24,25,26];

    glintspupils_right=eval_data(:,glintspupils_right_ind);
    glintspupils_left=eval_data(:,glintspupils_left_ind);

    reformatted_data=[eval_data(:,2),glintspupils_right(:,3)-glintspupils_right(:,1),...
        glintspupils_right(:,4)-glintspupils_right(:,2),glintspupils_right(:,5)-glintspupils_right(:,1),...
        glintspupils_right(:,6)-glintspupils_right(:,2),glintspupils_right(:,7)-glintspupils_right(:,1),...
        glintspupils_right(:,8)-glintspupils_right(:,2),...
        glintspupils_left(:,3)-glintspupils_left(:,1),...
        glintspupils_left(:,4)-glintspupils_left(:,2),glintspupils_left(:,5)-glintspupils_left(:,1),...
        glintspupils_left(:,6)-glintspupils_left(:,2),glintspupils_left(:,7)-glintspupils_left(:,1),...
        glintspupils_left(:,8)-glintspupils_left(:,2),...
        eval_data(:,50:57),...
        eval_data(:,27),eval_data(:,28),glintspupils_right(:,1),glintspupils_right(:,2),...
        glintspupils_left(:,1),glintspupils_left(:,2)];

end

function del_POG=findCompensation(model,predictors)
%Generalized function to find the POG at run time
del_POG=model(1)+sum(model(2:end)'.*predictors,2,'omitnan');


end



%----------------<Max Head Compensation Functions>--------------------
function [predictors_x,predictors_y]=maxPGPolynomial(predictors)
%Predictors are in the format: pupil_x, pupil_y
predictors_x=[predictors(:,1),predictors(:,2)]; %Model is just b0+b1*pupil_x+b2*pupil_y
predictors_y=[predictors(:,1),predictors(:,2)];

end

function [PG_Estimation_Models]=maxFitPGRegressor(train_cell)
%{
Inputs:
train_cell: has cells corresponding to the pg vector found. Each cell has
two sub cells: 
cell 1: name of the pg_vector
cell 2: the data in format: pg_vector_x, pg_y,pupil_x,pupil_y


Output:
PG_Estimation_Models:
A cell array with two columns: 
column 1: the PG estimation model type (e.g. 'pg0_right_x','pg0_right_y','pg1_right_x',
'pg1_right_y',...'pg0_left_x','pg0_left_y'...
column 2: the PG estimation model values
%} 


REGRESS_TUNE=4.2;
REGRESS_FUNC='huber';

%We loop for all the PG vectors found and find the PG estimation model
num_pg_detect=length(train_cell);

if num_pg_detect==0
    PG_Estimation_Models=nan;
else %We have detections and proceed
    PG_Estimation_Models=cell(num_pg_detect*2,2); %Will store our PG models
    for i=[1:num_pg_detect]

        pg_type=train_cell{i}{1}; %What is the type of pg vector we are getting
        pg_estimation_types={[pg_type,'_x'],[pg_type,'_y']};
        
        train_data=train_cell{i}{2};        

        
        PG_x=train_data(:,1);
        PG_y=train_data(:,2);

        pupils=train_data(:,[5:6]);


        
        [predictors_x,predictors_y]=maxPGPolynomial(pupils);

        %Using iteratively weighted least squares
        mdl_x=robustfit(predictors_x,PG_x,REGRESS_FUNC,REGRESS_TUNE); %Uses iteratively weighted least squares

        %Fitting POGy
        mdl_y=robustfit(predictors_y,PG_y,REGRESS_FUNC,REGRESS_TUNE); %Uses iteratively weighted least squares
        PG_Estimation_Models(i*2-1,1)=pg_estimation_types(1);
        PG_Estimation_Models{i*2-1,2}=mdl_x;

        PG_Estimation_Models(i*2,1)=pg_estimation_types(2);
        PG_Estimation_Models{i*2,2}=mdl_y;

    end
end




end



%Used to prep the data to train Max's compensation model
function compensation_data=prepMaxCompensationData(data_matrix,model_cell,dist_cell,PG_Estimation_Models)
    %{
    Input: 

    data_matrix: data for  training the compensation model
    model_cell: the cell containing the polynomial regression models
    dist_cell: the cell with the distances between calibration glints
    PG_Estimation_Models: cell array with the models to find the PG at the
    calibrated pupil position
    
 

    Output:

    compensationData is a cell array with two columns having:
    col 1: the pg type (e.g. pg0_left_x, pg0_left_y, etc.)
    col 2: the actual corresponding data with: del_POG (x or y),
    del_PG_x, del_PG_y
    
    %}
    
    left_headers={'pg0_left_x','pg0_left_y','pg1_left_x','pg1_left_y','pg2_left_x','pg2_left_y'};
    right_headers={'pg0_right_x','pg0_right_y','pg1_right_x','pg1_right_y','pg2_right_x','pg2_right_y'};
    check_model_right=any(ismember(model_cell(:,1),right_headers));
    check_model_left=any(ismember(model_cell(:,1),left_headers));

    %Initializing Results Data
    compensation_data=cell(12,2);
    compensation_data(:,1)=[left_headers';right_headers'];
    if iscell(PG_Estimation_Models)
        reformatted_data=reformatMaxData(data_matrix);
    
    
        %------------------<Right Data First>-------------------
        [row_n,~]=size(reformatted_data);
    
        for i=[1:row_n]
            curr_row=reformatted_data(i,:); %Current data row 
            
            t_x=curr_row(end-1);
            t_y=curr_row(end);
    
            nan_indexs=isnan(curr_row(1:6));
            nan_indx_values=find(nan_indexs);
    
            if length(nan_indx_values)<3 %At least two x,y pairs are detected
                stripped_header=right_headers(~nan_indexs); %Extracts the pg type that is valid for this frame
                valid_header=findPgWithAssociatedDistance(stripped_header,dist_cell);
                
                if length(valid_header)>2
                    model_valid_indexes=ismember(model_cell(:,1),valid_header);
                    updated_model_cell=model_cell(model_valid_indexes,:);
                    [row_new,~]=size(updated_model_cell);
    
                    for j=[1:floor(row_new/2)]
    
                        %Prepping correct indexing
                        model_x=updated_model_cell{j*2-1,3};
                        model_y=updated_model_cell{j*2,3};
                        header_x=valid_header{j*2-1};
                        header_y=valid_header{j*2};
                        pg_x_ind=ismember(right_headers,header_x);
                        pg_y_ind=ismember(right_headers,header_y);
    
                        %Getting pupil position
                        pupil_x=curr_row(13);
                        pupil_y=curr_row(14);
    
                        %Getting Correct PG Estimation Model
                        PG_Estimation_Headers=PG_Estimation_Models(:,1);
                        pg_estimation_indx=ismember(PG_Estimation_Headers,header_x);
                        pg_estimation_indy=ismember(PG_Estimation_Headers,header_y);
    
                        if sum(pg_estimation_indx)>0 && sum(pg_estimation_indy)>0
                            PG_model_x=PG_Estimation_Models{pg_estimation_indx,2};
                            PG_model_y=PG_Estimation_Models{pg_estimation_indy,2};
                            if ~isempty(PG_model_x) && ~isempty(PG_model_y)

                                
                                %Estimating PG
                                PG_estim_x=PG_model_x(1)+PG_model_x(2)*pupil_x+PG_model_x(3)*pupil_y;
                                PG_estim_y=PG_model_x(1)+PG_model_y(2)*pupil_x+PG_model_y(3)*pupil_y;
                                
                                %Getting delta PG 
                                pgsonly=curr_row(1:6);
                                
                                pg_x=pgsonly(pg_x_ind);
                                pg_y=pgsonly(pg_y_ind);
                                delta_pg_x=pg_x-PG_estim_x;
                                delta_pg_y=pg_y-PG_estim_y;
            
                                %Getting estimated POG
                                
                                [d_calib,d_curr]=findScalingFactors(dist_cell,valid_header,right_headers,pgsonly);
                            
                                if isnan(d_calib)||isnan(d_curr)
                                    %error_vec=[error_vec;[nan,nan,t_x,t_y]];
                                    continue
                                end
                                
                                pg_x=(d_calib/d_curr).*pg_x;
                                pg_y=(d_calib/d_curr).*pg_y;
                
                                [predictors_x,predictors_y]=customPolynomial(pg_x,pg_y);
                
                                POG_x=findPOG(model_x,predictors_x);
                                POG_y=findPOG(model_y,predictors_y);
                                
            
                                delta_POG_x=t_x-POG_x;
                                delta_POG_y=t_y-POG_y;
    
                                %Saving Results
                                results_headers=compensation_data(:,1);
                                results_ind_x=ismember(results_headers,header_x);
                                results_ind_y=ismember(results_headers,header_y);
    
                                data_x=compensation_data{results_ind_x,2};
                                data_y=compensation_data{results_ind_y,2};
    
                                data_x=[data_x;delta_POG_x,delta_pg_x,delta_pg_y];
                                data_y=[data_y;delta_POG_y,delta_pg_x,delta_pg_y];
    
                                compensation_data{results_ind_x,2}=data_x;
                                compensation_data{results_ind_y,2}=data_y;
    
    
                            end   
                        end
    
                    end
                end
            end
    
        end




        %------------------<Left Data Next>-------------------
        [row_n,~]=size(reformatted_data);
    
        for i=[1:row_n]
            curr_row=reformatted_data(i,:); %Current data row 
            
            t_x=curr_row(end-1);
            t_y=curr_row(end);
    
            nan_indexs=isnan(curr_row(7:12));
            nan_indx_values=find(nan_indexs);
    
            if length(nan_indx_values)<3 %At least two x,y pairs are detected
                stripped_header=left_headers(~nan_indexs); %Extracts the pg type that is valid for this frame
                valid_header=findPgWithAssociatedDistance(stripped_header,dist_cell);
                
                if length(valid_header)>2
                    model_valid_indexes=ismember(model_cell(:,1),valid_header);
                    updated_model_cell=model_cell(model_valid_indexes,:);
                    [row_new,~]=size(updated_model_cell);
    
                    for j=[1:floor(row_new/2)]
    
                        %Prepping correct indexing
                        model_x=updated_model_cell{j*2-1,3};
                        model_y=updated_model_cell{j*2,3};
                        header_x=valid_header{j*2-1};
                        header_y=valid_header{j*2};
                        pg_x_ind=ismember(left_headers,header_x);
                        pg_y_ind=ismember(left_headers,header_y);
    
                        %Getting pupil position
                        pupil_x=curr_row(15);
                        pupil_y=curr_row(16);
    
                        %Getting Correct PG Estimation Model
                        PG_Estimation_Headers=PG_Estimation_Models(:,1);
                        pg_estimation_indx=ismember(PG_Estimation_Headers,header_x);
                        pg_estimation_indy=ismember(PG_Estimation_Headers,header_y);
    

                        if sum(pg_estimation_indx)>0 && sum(pg_estimation_indy)>0
                            PG_model_x=PG_Estimation_Models{pg_estimation_indx,2};
                            PG_model_y=PG_Estimation_Models{pg_estimation_indy,2};
                            if ~isempty(PG_model_x) && ~isempty(PG_model_y)
                            
                                %Estimating PG
                                PG_estim_x=PG_model_x(1)+PG_model_x(2)*pupil_x+PG_model_x(3)*pupil_y;
                                PG_estim_y=PG_model_x(1)+PG_model_y(2)*pupil_x+PG_model_y(3)*pupil_y;
                                
                                %Getting delta PG 
                                pgsonly=curr_row(7:12);
                                
                                pg_x=pgsonly(pg_x_ind);
                                pg_y=pgsonly(pg_y_ind);
                                delta_pg_x=pg_x-PG_estim_x;
                                delta_pg_y=pg_y-PG_estim_y;
            
                                %Getting estimated POG
                                
                                [d_calib,d_curr]=findScalingFactors(dist_cell,valid_header,left_headers,pgsonly);
                            
                                if isnan(d_calib)||isnan(d_curr)
                                    %error_vec=[error_vec;[nan,nan,t_x,t_y]];
                                    continue
                                end
                                
                                pg_x=(d_calib/d_curr).*pg_x;
                                pg_y=(d_calib/d_curr).*pg_y;
                
                                [predictors_x,predictors_y]=customPolynomial(pg_x,pg_y);
                
                                POG_x=findPOG(model_x,predictors_x);
                                POG_y=findPOG(model_y,predictors_y);
                                
            
                                delta_POG_x=t_x-POG_x;
                                delta_POG_y=t_y-POG_y;
    
                                %Saving Results
                                results_headers=compensation_data(:,1);
                                results_ind_x=ismember(results_headers,header_x);
                                results_ind_y=ismember(results_headers,header_y);
    
                                data_x=compensation_data{results_ind_x,2};
                                data_y=compensation_data{results_ind_y,2};
    
                                data_x=[data_x;delta_POG_x,delta_pg_x,delta_pg_y];
                                data_y=[data_y;delta_POG_y,delta_pg_x,delta_pg_y];
    
                                compensation_data{results_ind_x,2}=data_x;
                                compensation_data{results_ind_y,2}=data_y;


                            end   
                        end
    
                    end
                end
            end
    
        end

    end


   
    
    data_vec=compensation_data(:,2);
    rm_ind=[];
    for i=1:length(data_vec)
        if isempty(data_vec{i})
            rm_ind=[rm_ind,i];
        end
    end

    compensation_data(rm_ind,:)=[];



end

%Subfunction used to reformat the data
function reformatted_data=reformatMaxData(eval_data)
    %Returns the data to evaluate the model in the format of: 
    % pg0_rightx, pg0_righty, ..., pg2_rightx, pg2_righty, pg0_leftx, pg0_lefty,..., pg2_leftx, pg2_lefty,
    % pupil_right_x,pupil_right_y,pupil_left_x,pupil_left_y
    % target_x,target_y


    glintspupils_right_ind=[3,4,9,10,11,12,13,14]; %Contains the glints and pupil positions such that pupil_x,pupil_y,glint0_x,glint0_y...
    glintspupils_left_ind=[15,16,21,22,23,24,25,26];

    glintspupils_right=eval_data(:,glintspupils_right_ind);
    glintspupils_left=eval_data(:,glintspupils_left_ind);

    reformatted_data=[glintspupils_right(:,3)-glintspupils_right(:,1),...
        glintspupils_right(:,4)-glintspupils_right(:,2),glintspupils_right(:,5)-glintspupils_right(:,1),...
        glintspupils_right(:,6)-glintspupils_right(:,2),glintspupils_right(:,7)-glintspupils_right(:,1),...
        glintspupils_right(:,8)-glintspupils_right(:,2),...
        glintspupils_left(:,3)-glintspupils_left(:,1),...
        glintspupils_left(:,4)-glintspupils_left(:,2),glintspupils_left(:,5)-glintspupils_left(:,1),...
        glintspupils_left(:,6)-glintspupils_left(:,2),glintspupils_left(:,7)-glintspupils_left(:,1),...
        glintspupils_left(:,8)-glintspupils_left(:,2),...
        glintspupils_right(:,1),glintspupils_right(:,2),glintspupils_left(:,1),glintspupils_left(:,2),...
        eval_data(:,27),eval_data(:,28)];

end


%Max's Compensation Training
function max_compensation_models=maxCompensationTraining(max_compensation_data)
    %Output is a cell with columns: pg_type, compensation_model

    [num_pg_cells,~]=size(max_compensation_data);

    if num_pg_cells==0
        max_compensation_models=nan;
    else %We have detections and proceed
        max_compensation_models=cell(num_pg_cells,2); %Final entry are the model parameters
        max_compensation_models(:,1)=max_compensation_data(:,1);
        for i=[1:num_pg_cells] %Looping for the detections

            train_data=max_compensation_data{i,2};
            return_model=maxCompensationRegressor(train_data);
            max_compensation_models{i,2}=return_model;
        end
        

    end



end


function [return_model]=maxCompensationRegressor(train_data)

    REGRESS_TUNE=4.2;
    REGRESS_FUNC='huber';
    %--------------Iteratively Weighted Least Squares---------------

    %Fitting POGx
    %Using iteratively weighted least squares
    [return_model,~]=robustfit(train_data(:,[2:3]),train_data(:,1),REGRESS_FUNC,REGRESS_TUNE); %Uses iteratively weighted least squares

end

%-----------------------<Variance Weighting Functions>---------------------

function variance_cell=findPOGVariance(data_mat,model_cell,dist_cell,avg_corners,comp_model_x_right,comp_model_y_right,comp_model_x_left,comp_model_y_left,PG_Estimation_Models,max_compensation_models,variance_type)
%Finds the variance of the POG estimate for the left and right eyes in the
%x, and y directions when looking at the central dot

%{
model_poly=model array from the initial calibration
dist_cell=corner distance from the intial calibration
avg_corners_init=average corner positions at the intial calibration
poly_functions_array=5 additional fitted polynomials for the interpolation
approac

output: cell with two columns:
    col1:               col2:
    variance_right_x    value
    variance_right_y    value
    variance_left_x     value
    variance_left_y     value

%}

    left_headers={'pg0_left_x','pg0_left_y','pg1_left_x','pg1_left_y','pg2_left_x','pg2_left_y'};
    right_headers={'pg0_right_x','pg0_right_y','pg1_right_x','pg1_right_y','pg2_right_x','pg2_right_y'};
    check_model_right=any(ismember(model_cell(:,1),right_headers));
    check_model_left=any(ismember(model_cell(:,1),left_headers));

    %Outputs the data as: 
    reformatted_data=reformatData(data_mat);

    %Returns POG results as a matrix with four columns (right_x, right_y,
    %left_x, left_y)
    POG_Results=evalAccuracyCompVariance(model_cell,reformatted_data,right_headers,left_headers,check_model_right,check_model_left,dist_cell,avg_corners,comp_model_x_right,comp_model_y_right,comp_model_x_left,comp_model_y_left,PG_Estimation_Models,max_compensation_models,variance_type);
    variance_results=var(POG_Results,0,1,'omitnan');
    %variance_results=mean(POG_Results,1,'omitnan');

    variance_headers={'variance_right_x','variance_right_y','variance_left_x','variance_left_y'};

    for i=[1:4]
        variance_cell{i,1}=variance_headers{i};
        variance_cell{i,2}=variance_results(i);
    end

end


function total_results=evalAccuracyCompVariance(model_cell,reformatted_data,right_headers,left_headers,check_model_right,check_model_left,dist_cell,avg_corners,comp_model_x_right,comp_model_y_right,comp_model_x_left,comp_model_y_left,PG_Estimation_Models,max_compensation_models,variance_type)
   %{
    Inputs:
    reformatted_data:
    frame_no, pg0_rightx, pg0_righty, ..., pg2_rightx, pg2_righty, pg0_leftx, pg0_lefty,..., pg2_leftx, pg2_lefty,
    right_inner_x,right_inner_y,right_outer_x,right_outer_y,left_inner_x,left_inner_y,left_outer_x,left_outer_y,
    target_x,target_y, pupil_right_x, pupil_right_y,
    pupil_left_x,pupil_left_y

    model_cell: contains the original polynomial model
    dist_cell: contains the distance between glints at calibration
    avg_corners: contains the average corner positions at calibration

    Outputs:
    
    total_results: matrix with columns:
    %accuracy_right_poly, accuracy_left_poly, accuracy_combined_poly (3), accuracy_right_interp, accuracy_left_interp, accuracy_combined_interp (6),
    %accuracy_right_max, accuracy_left_max, accuracy_combined_max (9),t_x, t_y (11)
   %}

    %Interpolation parameters
    %k=4; %Number of functions which we interpolate (best for gaussian)
    k=3; %(best for idw)
    p=1.5; %Order of interpolation weight
    sigma=0.75;
    weighting_type='idw'; %Using inverse distance weighting
    closest_type='euclidean'; %Using euclidean distance as the initial similarity measure


 
    NANTHRESH=0; %Number of nan values we tolerate as input to our tree model
    [row_n,~]=size(reformatted_data);
    total_results=[];
    for i=[1:row_n]
        results_row=nan(1,4);
        curr_row=reformatted_data(i,:); %Current data row

        t_x=curr_row(22);    %Targets
        t_y=curr_row(23);



        %--------------<Finding the right POG first>--------------
        del_corner_inner_x=avg_corners(1)-curr_row(14);
        del_corner_outer_x=avg_corners(3)-curr_row(16);

        del_corner_inner_y=avg_corners(2)-curr_row(15);
        del_corner_outer_y=avg_corners(4)-curr_row(17);
        if check_model_right
            right_pgs=curr_row(2:7);
            %Index of values in row that are not NaN
            nan_indexs=isnan(right_pgs);
            nan_indx_values=find(nan_indexs);
            if length(nan_indx_values)<3 %At least two x,y pairs are detected
                stripped_header=right_headers(~nan_indexs); %Extracts the pg type that is valid for this frame
                
                valid_header=findPgWithAssociatedDistance(stripped_header,dist_cell);
    
                if length(valid_header)>2
                    model_valid_indexes=ismember(model_cell(:,1),valid_header);
                    updated_model_cell=model_cell(model_valid_indexes,:);
                    [row_new,~]=size(updated_model_cell);
                    %Loops for all the number of points used and returns index of largest
                    cur_val=updated_model_cell{1,2};
                    cur_ind=1;
                    for j=[1:2:row_new]
                        if (updated_model_cell{j,2}>cur_val) %Change > to < if using residuals
                            cur_val=updated_model_cell{j,2};
                            cur_ind=j;
                        end
                    end
                    model_x=updated_model_cell{cur_ind,3};
                    model_y=updated_model_cell{cur_ind+1,3};
                    header_x=valid_header{cur_ind};
                    header_y=valid_header{cur_ind+1};
                    pg_x_ind=ismember(right_headers,header_x);
                    pg_y_ind=ismember(right_headers,header_y);

                    %Getting the pupil position (right_x, and right_y)
                    pupil_x=curr_row(24);
                    pupil_y=curr_row(25);

                    %Getting Correct PG Estimation Model
                    PG_Estimation_Headers=PG_Estimation_Models(:,1);
                    pg_estimation_indx=ismember(PG_Estimation_Headers,header_x);
                    pg_estimation_indy=ismember(PG_Estimation_Headers,header_y);

    
                    [d_calib,d_curr]=findScalingFactors(dist_cell,valid_header,right_headers,right_pgs);
                    
                    if ~isnan(d_calib) && ~isnan(d_curr)
                        
                        %----------Running Classic Polynomial
                    
                        pg_x=(d_calib/d_curr).*right_pgs(pg_x_ind);
                        pg_y=(d_calib/d_curr).*right_pgs(pg_y_ind);

        
                        [predictors_x,predictors_y]=customPolynomial(pg_x,pg_y);
        
                        POG_x_poly_right=findPOG(model_x,predictors_x);
                        POG_y_poly_right=findPOG(model_y,predictors_y);
                        if strcmp(variance_type,'poly')
                            results_row(1)=POG_x_poly_right;
                            results_row(2)=POG_y_poly_right;
%                             results_row(1)=abs(t_x-POG_x_poly_right);
%                             results_row(2)=abs(t_y-POG_y_poly_right);
                        end

                        %-----------Running Max's approach
                        if sum(pg_estimation_indx)>0 && sum(pg_estimation_indy)>0
                            PG_model_x=PG_Estimation_Models{pg_estimation_indx,2};
                            PG_model_y=PG_Estimation_Models{pg_estimation_indy,2};
                        if ~isempty(PG_model_x) && ~isempty(PG_model_y)

                            %Estimating PG
                            PG_estim_x=PG_model_x(1)+PG_model_x(2)*pupil_x+PG_model_x(3)*pupil_y;
                            PG_estim_y=PG_model_x(1)+PG_model_y(2)*pupil_x+PG_model_y(3)*pupil_y;

                            %Finding delta PG 
                            delta_pg_x=right_pgs(pg_x_ind)-PG_estim_x;
                            delta_pg_y=right_pgs(pg_y_ind)-PG_estim_y;

                            %Finding appropriate POG compensation model
                            %(max)
                            POG_MaxModels_Headers=max_compensation_models(:,1);
                            pog_estimation_max_indx=ismember(POG_MaxModels_Headers,header_x);
                            pog_estimation_max_indy=ismember(POG_MaxModels_Headers,header_y);

                            pog_max_modelx=max_compensation_models{pog_estimation_max_indx,2};
                            pog_max_modely=max_compensation_models{pog_estimation_max_indy,2};

                            if ~isempty(pog_max_modelx) && ~isempty(pog_max_modely)
                                del_POG_max_x_right=pog_max_modelx(1)+pog_max_modelx(2)*delta_pg_x+pog_max_modelx(3)*delta_pg_y;
                                del_POG_max_y_right=pog_max_modely(1)+pog_max_modely(2)*delta_pg_x+pog_max_modely(3)*delta_pg_y;
                                
                                POG_x_max_right=del_POG_max_x_right+POG_x_poly_right;
                                POG_y_max_right=del_POG_max_y_right+POG_y_poly_right;
                                if strcmp(variance_type,'max')
                                    results_row(1)=POG_x_max_right;
                                    results_row(2)=POG_y_max_right;
%                                     results_row(1)=abs(t_x-POG_x_max_right);
%                                     results_row(2)=abs(t_y-POG_y_max_right);
                                end


                             end

                            
                        end 
                        end
                        predictors=[del_corner_inner_x,del_corner_inner_y,del_corner_outer_x,del_corner_outer_y];
                        [predictors_x,predictors_y]=compensationPolynomial(predictors);
                        %Compensating in x-direction
                        accuracy_get=0;
                        if length(comp_model_x_right)>0 && ~any(isnan(predictors_x))
                                accuracy_get=accuracy_get+1;
                                del_POG_x_tree=findCompensation(comp_model_x_right,predictors_x);
                                del_POG_x_tree=del_POG_x_tree;
                                POG_x_tree_right=del_POG_x_tree+POG_x_poly_right;

                        end

                        if length(comp_model_y_right)>0 && ~any(isnan(predictors_y))
                                accuracy_get=accuracy_get+1;
                                del_POG_y_tree=findCompensation(comp_model_y_right,predictors_y);
                                del_POG_y_tree=del_POG_y_tree;                             
                                POG_y_tree_right=del_POG_y_tree+POG_y_poly_right;

                        end
                        if accuracy_get>=2 && strcmp(variance_type,'comp')
                            results_row(1)=POG_x_tree_right;
                            results_row(2)=POG_y_tree_right;
                        end
                    end
                      
      
   
                       
                                   
                end
        
            end

        end


        %-------------<Finding the left POG next>--------------

        if check_model_left
            left_pgs=curr_row(8:13);
            %Index of values in row that are not NaN
            nan_indexs=isnan(left_pgs);
            nan_indx_values=find(nan_indexs);
            if length(nan_indx_values)<3 %At least two x,y pairs are detected
                stripped_header=left_headers(~nan_indexs); %Extracts the pg type that is valid for this frame
                
                valid_header=findPgWithAssociatedDistance(stripped_header,dist_cell);
    
                if length(valid_header)>2
                    model_valid_indexes=ismember(model_cell(:,1),valid_header);
                    updated_model_cell=model_cell(model_valid_indexes,:);
                    [row_new,~]=size(updated_model_cell);
                    %Loops for all the number of points used and returns index of largest
                    cur_val=updated_model_cell{1,2};
                    cur_ind=1;
                    for j=[1:2:row_new]
                        if (updated_model_cell{j,2}>cur_val) %Change > to < if using iteratively least squares
                            cur_val=updated_model_cell{j,2};
                            cur_ind=j;
                        end
                    end
                    model_x=updated_model_cell{cur_ind,3};
                    model_y=updated_model_cell{cur_ind+1,3};
                    header_x=valid_header{cur_ind};
                    header_y=valid_header{cur_ind+1};
                    pg_x_ind=ismember(left_headers,header_x);
                    pg_y_ind=ismember(left_headers,header_y);
    

                    %Getting the pupil position (left_x, and left_y)
                    pupil_x=curr_row(26);
                    pupil_y=curr_row(27);


                    %Getting Correct PG Estimation Model
                    PG_Estimation_Headers=PG_Estimation_Models(:,1);
                    pg_estimation_indx=ismember(PG_Estimation_Headers,header_x);
                    pg_estimation_indy=ismember(PG_Estimation_Headers,header_y);



                    [d_calib,d_curr]=findScalingFactors(dist_cell,valid_header,left_headers,left_pgs);
                    
                    if ~isnan(d_calib) && ~isnan(d_curr)

                        %------------Running Typicaly POG Approach
                        pg_x=(d_calib/d_curr).*left_pgs(pg_x_ind);
                        pg_y=(d_calib/d_curr).*left_pgs(pg_y_ind);
        
                        [predictors_x,predictors_y]=customPolynomial(pg_x,pg_y);
        
                        POG_x_poly_left=findPOG(model_x,predictors_x);
                        POG_y_poly_left=findPOG(model_y,predictors_y);

                        if strcmp(variance_type,'poly')
                            results_row(3)=POG_x_poly_left;
                            results_row(4)=POG_y_poly_left;
%                             results_row(3)=abs(t_x-POG_x_poly_left);
%                             results_row(4)=abs(t_y-POG_y_poly_left);
                        end


                                               
                        %-----------<Running Max's approach>--------------
                        if sum(pg_estimation_indx)>0 && sum(pg_estimation_indy)>0
                            PG_model_x=PG_Estimation_Models{pg_estimation_indx,2};
                            PG_model_y=PG_Estimation_Models{pg_estimation_indy,2};
                        if ~isempty(PG_model_x) && ~isempty(PG_model_y)
   
                            %Estimating PG
                            PG_estim_x=PG_model_x(1)+PG_model_x(2)*pupil_x+PG_model_x(3)*pupil_y;
                            PG_estim_y=PG_model_x(1)+PG_model_y(2)*pupil_x+PG_model_y(3)*pupil_y;

                            %Finding delta PG 
                            delta_pg_x=left_pgs(pg_x_ind)-PG_estim_x;
                            delta_pg_y=left_pgs(pg_y_ind)-PG_estim_y;

                            %Finding appropriate POG compensation model
                            %(max)
                            POG_MaxModels_Headers=max_compensation_models(:,1);
                            pog_estimation_max_indx=ismember(POG_MaxModels_Headers,header_x);
                            pog_estimation_max_indy=ismember(POG_MaxModels_Headers,header_y);

                            pog_max_modelx=max_compensation_models{pog_estimation_max_indx,2};
                            pog_max_modely=max_compensation_models{pog_estimation_max_indy,2};

                            if ~isempty(pog_max_modelx) && ~isempty(pog_max_modely)
                                del_POG_max_x_left=pog_max_modelx(1)+pog_max_modelx(2)*delta_pg_x+pog_max_modelx(3)*delta_pg_y;
                                del_POG_max_y_left=pog_max_modely(1)+pog_max_modely(2)*delta_pg_x+pog_max_modely(3)*delta_pg_y;
                                
                                POG_x_max_left=del_POG_max_x_left+POG_x_poly_left;
                                POG_y_max_left=del_POG_max_y_left+POG_y_poly_left;

                                if strcmp(variance_type,'max')
                                    results_row(3)=POG_x_max_left;
                                    results_row(4)=POG_y_max_left;
%                                     results_row(3)=abs(t_x-POG_x_max_left);
%                                     results_row(4)=abs(t_y-POG_y_max_left);
                                end

                             end

                            
                        end
                        end

                        predictors=[del_corner_inner_x,del_corner_inner_y,del_corner_outer_x,del_corner_outer_y];
                        [predictors_x,predictors_y]=compensationPolynomial(predictors);
                        %Compensating in x-direction
                        accuracy_get=0;
                        if length(comp_model_x_left)>0 && ~any(isnan(predictors_x))
                                accuracy_get=accuracy_get+1;
                                del_POG_x_tree=findCompensation(comp_model_x_left,predictors_x);
                                del_POG_x_tree=del_POG_x_tree;
                                POG_x_tree_left=del_POG_x_tree+POG_x_poly_left;

                        end

                        if length(comp_model_y_left)>0 && ~any(isnan(predictors_y))
                                accuracy_get=accuracy_get+1;
                                del_POG_y_tree=findCompensation(comp_model_y_left,predictors_y);
                                del_POG_y_tree=del_POG_y_tree;
                                POG_y_tree_left=del_POG_y_tree+POG_y_poly_left;

                        end
                        if accuracy_get>=2 && strcmp(variance_type,'comp')
                            results_row(3)=POG_x_tree_left;
                            results_row(4)=POG_y_tree_left;
                        end


                                      
    
                        
                    end
                  
                      
                end
        
            end

        end
        
        total_results=[total_results;results_row];

    end

end

function [POG_combined_x,POG_combined_y]=findCombinedWithVariance(variance_cell,POG_x_right,POG_x_left,POG_y_right,POG_y_left)
        right_x_ind=ismember(variance_cell(:,1),'variance_right_x');
        right_y_ind=ismember(variance_cell(:,1),'variance_right_y');
        left_x_ind=ismember(variance_cell(:,1),'variance_left_x');
        left_y_ind=ismember(variance_cell(:,1),'variance_left_y');

        var_right_x=variance_cell{right_x_ind,2};
        var_right_y=variance_cell{right_y_ind,2};
        var_left_x=variance_cell{left_x_ind,2};
        var_left_y=variance_cell{left_y_ind,2};
        %-------------------<Getting Combined Results>-----------------
        if ~isnan(var_right_x) && ~isnan(var_right_y) && ~isnan(var_left_x) && ~isnan(var_left_y)
            POG_combined_x=(POG_x_right*(1/var_right_x)+POG_x_left*(1/var_left_x))/((1/var_right_x)+(1/var_left_x));
            POG_combined_y=(POG_y_right*(1/var_right_y)+POG_y_left*(1/var_left_y))/((1/var_right_y)+(1/var_left_y));

        else

            POG_combined_x=(POG_x_right+POG_x_left)/2;
            POG_combined_y=(POG_y_right+POG_y_left)/2;
        end



end
