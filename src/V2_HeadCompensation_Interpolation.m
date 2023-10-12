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

%%
clear
clc
close all

%% Testing multivariate interpolation approach
%{
curr_corners=[351,145,201,250];
poly_functions={[2,2.8,3,3.8,5],[nan,nan,195,253];...
    nan(1,5),[340,144,198,251];...
    [1.9,2.8,3.035,3.82,5.12],[nan,nan,192,243];...
    [2.2,2.75,2.95,3.83,4.82],[345,138,nan,nan];...
    [2.35,2.8,3.05,3.79,4.92],[nan,nan,204,252];...
    [1.99,2.85,3.1,3.78,5.01],nan(1,4);...
    };
closest_type='euclidean';
weighting_type='idw';
k=3;
p=1;
weighted_poly=multivariateInterp(poly_functions,curr_corners,closest_type,weighting_type,k,p);
%}

%% Testing on Actual Data

%Looping Through All Participants
data_root='E:/Alexandre_EyeGazeProject_Extra/eyecorner_userstudy2_converted';
results_folder='C:/Users/playf/OneDrive/Documents/UBC/Thesis/Paper_FiguresAndResults/Interpolation_IDW_WithResiduals_Results';
%Getting list of subfolders
folder_list=dir(data_root);
dirnames={folder_list([folder_list.isdir]).name};
num_dir=length(dirnames);

%Global Params:
CALIB_THRESHOLD=5;
EVAL_THRESHOLD=3; %Threshold to be considered a valid evaluation trial

mean_acc_target_results=[];
mean_acc_results=[];
%Looping for all participants
for m=[1:num_dir]
    if dirnames{m}(1)=='P' %We have a participant and run calibrations and/evaluations
        disp(dirnames{m})
        calib_init_data=readmatrix([data_root,'/',dirnames{m},'/calib_only_merged_Calib_Init.csv']);
        check_calib=checkDetection(calib_init_data,CALIB_THRESHOLD);
        if (check_calib==true)
            [train_cell_init,dist_cell_init,avg_corners_init]=getRegressionData(calib_init_data,CALIB_THRESHOLD); %Also gets the average eye corner location at the calibration
            if length(dist_cell_init)==0
                continue;
            end
            model_poly_init=robustRegressor(train_cell_init); %Get the vanilla model from the initial calibration

            %poly_function is a cell array where 
            % col1: has the subarray with: col 1: model type
            % (pg0_left_x, pg0_left_y, etc.), col2: fitted polynomials,
            % col3: residual errors, col4: # of calibration points
            %col2 has the average corner locations
            poly_functions_array=cell(6,2); %Each cell has the above array for each calibration
            poly_functions_array{1,1}=model_poly_init;
            poly_functions_array{1,2}=avg_corners_init;
            
            for i=1:5 %Loops for the other 5 8-point calibrations
                calib_data=readmatrix([data_root,'/',dirnames{m},'/calib_only_merged_Calib_Comp_Lift',num2str(i),'_8point.csv']);
                check_calib=checkDetection(calib_data,CALIB_THRESHOLD);
                if (check_calib==true)
                    [train_cell,dist_cell,avg_corners]=getRegressionDataForInterp(calib_data,CALIB_THRESHOLD,dist_cell_init); %Also gets the average eye corner location at the calibration
                    if length(dist_cell)==0
                        continue;
                    end
                    model_poly=robustRegressor(train_cell); %Get the vanilla model from the initial calibration
                    poly_functions_array{i+1,1}=model_poly;
                    poly_functions_array{i+1,2}=avg_corners;

                end
            end

            %training Max's PG Estimator (for PG_hat)
            
            PG_Estimation_Models=maxFitPGRegressor(train_cell_init);

            %Training Max's POG compensation model
            calib_max=readmatrix([data_root,'/',dirnames{m},'/calib_only_merged_Calib_Comp_Rotate.csv']);

            max_compensation_data=prepMaxCompensationData(calib_max,model_poly_init,dist_cell_init,PG_Estimation_Models);
            
            max_compensation_models=maxCompensationTraining(max_compensation_data);


            %---------Using 1-dot data to find variance in POG estimates for
            %combining the POG estimates

            %Extracting one-dot data:
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
            data_mat=[calib_lift1_dot;calib_lift2_dot;calib_lift3_dot;calib_lift4_dot;calib_lift5_dot];

            variance_cell=findPOGVariance(data_mat,model_poly_init,dist_cell_init,avg_corners_init,poly_functions_array);





            %avg_corners in the format: right_inner_x,right_inner_y,right_outer_x,right_outer_y,left_inner_x,...
                

            eval_lift1_9dot=readmatrix([data_root,'/',dirnames{m},'/calib_only_merged_Eval_Lift1_9Point.csv']);
            eval_lift2_9dot=readmatrix([data_root,'/',dirnames{m},'/calib_only_merged_Eval_Lift2_9Point.csv']);
            eval_lift3_9dot=readmatrix([data_root,'/',dirnames{m},'/calib_only_merged_Eval_Lift3_9Point.csv']);

            data_mat=[eval_lift1_9dot;eval_lift2_9dot;eval_lift3_9dot];
            
            
            [mean_accuracies,total_results]=evalModelsRegressComp(data_mat,model_poly_init,dist_cell_init,avg_corners_init,PG_Estimation_Models,max_compensation_models,poly_functions_array,variance_cell);
            
            mean_per_target=findMeanPerTarget(total_results);



            %Participant Number
            part_string=dirnames{m}(2:3);
            part_num=str2double(part_string);

            mean_acc_target_results=[mean_acc_target_results;[part_num,mean_per_target]];
            mean_acc_results=[mean_acc_results;[part_num,mean_accuracies]];

        end



    end

end

%Saving Results:
%Saving the per-target results
per_target_results_file=[results_folder,'/per_target_results.csv'];
csvwrite(per_target_results_file,mean_acc_target_results);

%Saving the mean acc results
mean_table=array2table(mean_acc_results);
mean_table.Properties.VariableNames(1:10)={'participant','acc_right_poly',...
    'acc_left_poly','acc_combined_poly','acc_right_interp','acc_left_interp',...
    'acc_combined_interp','acc_right_max','acc_left_max','acc_combined_max'};
mean_results_file=[results_folder,'/mean_acc_results.csv'];
writetable(mean_table,mean_results_file);



mean_acc_total=mean(mean_acc_results,1,'omitnan');
disp(mean_acc_total);












%##########################Function Definitions############################
function mean_per_target=findMeanPerTarget(total_results)
%Returns a vector mean_per_target which has elements: 
% mean_poly, mean_comp, mean_max, target_x, target_y, mean_poly,
% mean_comp,... and repeats for all targets
%Each of the means is for right/left/combined

%{
        total_results: matrix with columns:
    accuracy_right_poly, accuracy_left_poly, accuracy_combined_poly (3), accuracy_right_interp, accuracy_left_interp, accuracy_combined_interp (6),
    accuracy_right_max, accuracy_left_max, accuracy_combined_max (9),t_x, t_y (11),
%}
    mean_per_target=[];
    [row_n,~]=size(total_results);
    old_tx=total_results(1,10);
    old_ty=total_results(1,11);

    curr_seg=[];
    for i=[1:row_n]
        
        curr_seg=[curr_seg;total_results(i,:)];
        if old_tx~=total_results(i,10) || old_ty~=total_results(i,11)   %We have a new target point

            mean_per_target=[mean_per_target,mean(curr_seg(:,[1:9]),1,'omitnan'),old_tx,old_ty];
            
            old_tx=total_results(i,10);
            old_ty=total_results(i,11);
            curr_seg=[];
        end



    end
    mean_per_target=[mean_per_target,mean(curr_seg(:,[1:9]),1,'omitnan'),old_tx,old_ty];





end




%---------------------<Initial Calibration Functions>----------------------
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



function [predictors_x,predictors_y]=customPolynomial(pg_x,pg_y)
    %predictors=[pg_x.^2,pg_x.*pg_y,pg_y.^2,pg_x,pg_y];
    %predictors_x=[pg_x,pg_y,pg_x.^2];
    %predictors_y=[pg_y,pg_x.^2,pg_x.*pg_y,pg_x.^2.*pg_y];
    predictors_x=[pg_x.^2,pg_x.*pg_y,pg_y.^2,pg_x,pg_y];
    predictors_y=[pg_x.^2,pg_x.*pg_y,pg_y.^2,pg_x,pg_y];
    %predictors_x=[pg_x.^2,pg_x.*pg_y,pg_y.^2,pg_x,pg_y];
    %predictors_y=[pg_x.^2,pg_x.*pg_y,pg_y.^2,pg_x,pg_y];

end

function robust_regressor_output=robustRegressor(train_cell)
    %Output is a cell with columns:col 1: model type
    % (pg0_left_x, pg0_left_y, etc.), col2: fitted polynomials,
    % col3: residual errors, col4: # of calibration points
    %model type is: pg0_leftx, pg0_lefty,pg1_leftx,pg1_lefty... for pg1->pg3
    %and right/left
    num_pg_detect=length(train_cell);

    if num_pg_detect==0
        robust_regressor_output=nan;
    else %We have detections and proceed
        robust_regressor_output=cell(num_pg_detect*2,4);
        for i=[1:num_pg_detect] %Looping for the detections
            train_data=train_cell{i}{2};
            [rmse_x,rmse_y,b_x,b_y]=customRegressor(train_data);
            %Saving x-results
            robust_regressor_output{i*2-1,1}=strcat(train_cell{i}{1},'_x');
            robust_regressor_output{i*2-1,2}=b_x;
            robust_regressor_output{i*2-1,3}=(rmse_x+rmse_y)/2;
            robust_regressor_output{i*2-1,4}=size(train_data,1);


            robust_regressor_output{i*2,1}=strcat(train_cell{i}{1},'_y');
            robust_regressor_output{i*2,2}=b_y;
            robust_regressor_output{i*2,3}=(rmse_x+rmse_y)/2;
            robust_regressor_output{i*2,4}=size(train_data,1);


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
    %train_data=cropTrainingData(train_data);

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

%------------------<Multivariable Interpolation Functions>-----------------

function weighted_poly=multivariateInterp(poly_functions,curr_corners,closest_type,weighting_type,k,p,sigma)
%{
Description: Performs multivariate inerpolation where we are given the
calibrated polynomials f1...fn, and the corresponding inner/outer eye
corner locations c1...cn such that c={inner_x,inner_y,outer_x,outer_y}
and the current corner location. We then pick the top k closest corners
c1...cn to our current corner ccurr using a similarity function.
We then find weights wi where i=1..k for each calibrated function using a
weighting function. We then normalize the weights


Inputs:
poly_functions: cell array where: col1=fitted polynomials (if it is nan, make sure to put all 
coefficients to nan), col2= avg corner
locations (inner_x, inner_y, outer_x, outer_y)
col3= rmse error at calibration

col4= number of calibration points used

curr_corners: current corner location

closest_type: type of similarity measure to use options are: 'euclidean'

weighting_type: type of weighting function to use options are: 'idw' for
inverse distance weighting

k: the k-closest eye corner positions that we use to weight the function
p: order of the exponent term in inverse distance weighting function
(typically 1)

%}




    %Extracting values
    calibrated_functions=cell2mat(poly_functions(:,1)); %Matrix where each row are the coefficients of the fitted polynomial
    calibrated_corners=cell2mat(poly_functions(:,2));
    rmse_errors=cell2mat(poly_functions(:,3));
    
    %Check for nan values in current inner/outer corner locations
    %if only one corner is currently found (other one is nan), only use that same corresponding
    %corner from the calibrated corner locations
    %We also check for nan values in the calibrated corner locations 
    
    [calibrated_corners,calibrated_functions,curr_corners,rmse_errors,bool_check]=checkCalibratedCorners(calibrated_corners,calibrated_functions,curr_corners,rmse_errors,k);
    if bool_check %We have an acceptable number of calibration corner locations

        %Finding closesness between current corners and calibrated corners 
        score=closenessScore(calibrated_corners,curr_corners,rmse_errors,closest_type);
        
        %Arrange scores from lowest to highest (highest means further away)
        [ascend_scores,inds]=sort(score);
        ascend_functions=calibrated_functions(inds,:);
        ascend_corners=calibrated_corners(inds,:);
        
        %Truncating to k-closest
        k_functions=ascend_functions(1:k,:); %matrix where each row are the top k fitted polynomials
        k_corners=ascend_corners(1:k,:);
        

        weights=weightingFunction(k_corners,curr_corners,weighting_type,p,sigma);
        
        %we multiply every row by the corresponding weight (row 1*weights(1)+row
        %2*weights(2) and final polynomial weights are the sum along the columns
        
        scaled_functions=weights.*k_functions; 
        
        weighted_poly=sum(scaled_functions,1); %Sums the polynomial coefficients to give our final interpolated polynomial


    else
        weighted_poly=nan(1,length(calibrated_functions(1,:))); %Returns nan for the current polynomial
    
    end




end

function score=closenessScore(calibrated_corners,current_corners,rmse_errors,closest_type)
    %input: calibrated_corners which is a matrix with each row corresponding to
    %the corner locations (inner_x, inner_y, outer_x, outer_y)for a given polynomial
    
    %output=vector with the similarity score between each of the calibrated
    %corner locations and the current corner locations
    
    %Higher score means worse
    
    switch closest_type
        case 'euclidean'
            score=sqrt(sum((calibrated_corners-current_corners).^2,2));
        case 'rmse_error'
            score=rmse_errors;
        otherwise
            error('Incorrect Closeness Score Passed')
    
    
    
    end

end


function weights=weightingFunction(calibrated_corners,current_corners,weighting_type,p,sigma)
    %input: top 'k' calibrated_corners which is a matrix with each row corresponding to
    %the corner locations (inner_x, inner_y, outer_x, outer_y)for a given polynomial
    %current_corners: which is the location of the current eye corners
    %weighting_type: 'idw' for inverse distance weighting
    %p is a power exponetial that controls weight decay, typically left at
    %1
    %sigma is a positive parameter controling the influence of calibration
    %points (smaller sigma makes the weight more localized)
    %Also normalizes the weights to sum to 1

    %output: the corresponding weights found from the weighting metric

    switch weighting_type
        case 'idw'
            weights=1./(sqrt(sum((calibrated_corners-current_corners).^2,2)).^p);
        case 'gaussian'
            weights=exp(-((sqrt(sum((calibrated_corners-current_corners).^2,2))/sigma).^2));
        otherwise
            error('incorrect weighting function type');
            
    end


    %Normalize the weights to sum to 1
    
    weights_sum=sum(weights);
    weights=weights/weights_sum;



end


function [new_calib_corners,new_calib_functions,new_curr_corners,rmse_error_new,bool_check]=checkCalibratedCorners(calibrated_corners,calibrated_functions,curr_corners,rmse_errors,k)

    inds_innercorners_notnan=~isnan(calibrated_corners(:,1)); %Gets row index where inner corners are not nan
    inds_outercorners_notnan=~isnan(calibrated_corners(:,3)); %Gets row index where outer corners are not nan
    inds_functions_notnan=~isnan(calibrated_functions(:,1)); %Gets row index where polynomial functions are not nan

    %Init parameters
    new_calib_corners=nan;
    new_calib_functions=nan;
    new_curr_corners=nan;
    rmse_error_new=nan;
    bool_check=false;

    if ~isnan(curr_corners(1)) && ~isnan(curr_corners(3)) %We have both inner and outer current eye corners
        if sum(inds_innercorners_notnan&inds_functions_notnan)>=k && sum(inds_outercorners_notnan&inds_functions_notnan)>=k %We have enough calibrated for both corners
            joint_notnan=inds_innercorners_notnan&inds_outercorners_notnan&inds_functions_notnan; %Indexes where we have only the function
            new_calib_functions=calibrated_functions(joint_notnan,:);
            rmse_error_new=rmse_errors(joint_notnan);
            new_calib_corners=calibrated_corners(joint_notnan,:);
            new_curr_corners=curr_corners;
            bool_check=true;
        elseif sum(inds_innercorners_notnan&inds_functions_notnan)>=k %We have enough inner corners from our calibrated corners
            joint_notnan=inds_innercorners_notnan&inds_functions_notnan; %Indexes where we have only the function
            new_calib_functions=calibrated_functions(joint_notnan,:);
            rmse_error_new=rmse_errors(joint_notnan);
            new_calib_corners=calibrated_corners(joint_notnan,[1:2]);
            new_curr_corners=curr_corners([1:2]);
            bool_check=true;

        elseif sum(inds_outercorners_notnan&inds_functions_notnan)>=k %We have enough outer corners from our calibrated corners
            joint_notnan=inds_outercorners_notnan&inds_functions_notnan; %Indexes where we have only the function
            new_calib_functions=calibrated_functions(joint_notnan,:);
            rmse_error_new=rmse_errors(joint_notnan);
            new_calib_corners=calibrated_corners(joint_notnan,[3:4]);
            new_curr_corners=curr_corners([3:4]);
            bool_check=true;
        end

    elseif ~isnan(curr_corners(1)) && isnan(curr_corners(3)) %We have the current inner corner, but not the outer
            if sum(inds_innercorners_notnan&inds_functions_notnan)>=k %We have enough inner corners from our calibrated corners
                joint_notnan=inds_innercorners_notnan&inds_functions_notnan; %Indexes where we have only the function
                new_calib_functions=calibrated_functions(joint_notnan,:);
                rmse_error_new=rmse_errors(joint_notnan);
                new_calib_corners=calibrated_corners(joint_notnan,[1:2]);
                new_curr_corners=curr_corners([1:2]);
                bool_check=true;
            end
    elseif isnan(curr_corners(1)) && ~isnan(curr_corners(3)) %We don't have current inner corner, but we have the current outer
            
        if sum(inds_outercorners_notnan&inds_functions_notnan)>=k %We have enough outer corners from our calibrated corners
            joint_notnan=inds_outercorners_notnan&inds_functions_notnan; %Indexes where we have only the function
            new_calib_functions=calibrated_functions(joint_notnan,:);
            rmse_error_new=rmse_errors(joint_notnan);
            new_calib_corners=calibrated_corners(joint_notnan,[3:4]);
            new_curr_corners=curr_corners([3:4]);
            bool_check=true;
        end

    end

end

function poly_functions=reformatDataMultivariateInterp(poly_functions_array,header,side_type)
%{
output:
poly_functions: cell array where: col1=fitted polynomials (if it is nan, make sure to put all 
coefficients to nan), col2= avg corner
locations (inner_x, inner_y, outer_x, outer_y)
col3= rmse error at calibration

col4= number of calibration points used
%}


    [num_row,~]=size(poly_functions_array);
    
    poly_functions=cell(0);
    poly_functions_count=1;
    for i=[1:num_row]
        corners=poly_functions_array{i,2};
        switch side_type
            case 'right'
                corners=corners(1:4);
            case 'left'
                corners=corners(5:8);
            otherwise
                error('wrong side type when reformatting data for interpolation function')
        end

        curr_occurence=poly_functions_array{i,1};
        headers=curr_occurence(:,1);
        poly_ind=ismember(headers,header);
        if sum(poly_ind)>=1
            poly_functions{poly_functions_count,1}=curr_occurence{poly_ind,2}';
            poly_functions{poly_functions_count,2}=corners;
            poly_functions{poly_functions_count,3}=curr_occurence{poly_ind,3};
            poly_functions{poly_functions_count,4}=curr_occurence{poly_ind,4};
            poly_functions_count=poly_functions_count+1;
            
        end

            
    end


end


function [trainCell,dist_cell,avg_corners]=getRegressionDataForInterp(data_matrix,thresh,dist_cell_calib)
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
        dist=findPgDistanceForInterp(dist_header{i},trainCell);
        if all(isnan(dist))
            continue    
        else
            dist_cell{cell_count,1}=dist_header{i};
            dist_cell{cell_count,2}=dist;
            cell_count=cell_count+1;

        end

    end

    %-----------------<Rescaling the PG vectors here>---------------------
    trainCell=rescalePGVectorsForInterp(dist_cell,dist_cell_calib,trainCell);

    %Getting rid of nan values    
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

function [dist]=findPgDistanceForInterp(distance_type,train_cell)
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
        dist=sqrt((train_cell{ind_2}{2}(:,1)-train_cell{ind_1}{2}(:,1)).^2+(train_cell{ind_2}{2}(:,2)-train_cell{ind_1}{2}(:,2)).^2);
    end
    



end

function [trainCell]=rescalePGVectorsForInterp(dist_cell,dist_cell_calib,trainCell_old)
%{
Inputs 
    dist_cell: cell which contains the distance between vectors for the
    current 8-point calibration
    dist_cell_calib: cell which contains the distance between vectors for
    the initial 8-point calibration
    trainCell: cell with the training data
Outputs
    trainCell: rescaled training data based off of inter-glint distance
    between interpolation calibrations and initial calibration

%}
    train_cell_pgtypes=cell(0);
    for i=[1:length(trainCell_old)]
        train_cell_pgtypes{i}=trainCell_old{i}(1); 
    
    
    end
    trainCell={};
    for i=[1:length(trainCell_old)]
        pg_type=trainCell_old{i}{1};
        old_pg_data=trainCell_old{i}{1,2};
        [row_n,~]=size(old_pg_data);

        %Getting the possible dist types
        if strcmp(pg_type,'pg0_right') || strcmp(pg_type,'pg1_right') || strcmp(pg_type,'pg2_right')
            dist_names={'d_01_right','d_02_right','d_12_right'};
        elseif strcmp(pg_type,'pg0_left') || strcmp(pg_type,'pg1_left') || strcmp(pg_type,'pg2_left')
            dist_names={'d_01_left','d_02_left','d_12_left'};
        end
        are_names_curr=ismember(dist_names,dist_cell(:,1));
        are_names_calib=ismember(dist_names,dist_cell_calib(:,1));
        joint_names_inds=are_names_curr&are_names_calib;
        dist_names=dist_names(joint_names_inds);
        dist_curr_ind=ismember(dist_cell(:,1),dist_names);
        dist_calib_ind=ismember(dist_cell_calib(:,1),dist_names);



        %Scaling the pg vectors
        new_pg_data=[];
        for j=[1:row_n]

            %Getting distance between current vectors
            dist_curr_extracted=dist_cell(dist_curr_ind,2);
            dist_curr=[];
            for m=[1:length(dist_curr_extracted)]
                dist_curr=[dist_curr,dist_curr_extracted{m}(j)];
            end

            %Getting distance between calibrated vectors
            dist_calib=cell2mat(dist_cell_calib(dist_calib_ind,2));


            %Getting rid of nan values
            nan_inds_curr=~isnan(dist_curr);
            nan_inds_calib=~isnan(dist_calib);
        
            joint_inds=nan_inds_curr&nan_inds_calib';
            %size_joint=['joint: ',num2str(size(joint_inds))];
            %size_curr=['curr: ',num2str(size(dist_curr))];
            %size_calib=['calib: ',num2str(size(dist_calib))];
            %disp(size_joint);
            %disp(size_curr);
            %disp(size_calib);
            %[row_calib,col_calib]=size(dist_calib);
            %size_dist_calib=row_calib*col_calib;
            dist_curr=dist_curr(joint_inds);
            dist_calib=dist_calib(joint_inds);
            
            
            %Rescaling the train_cell data
            if sum(joint_inds)>=1 %We have distances that we can scale it with
                scale_factor=mean(dist_calib)/mean(dist_curr);
                new_pg_data=[new_pg_data;scale_factor.*old_pg_data(j,[1:2]),old_pg_data(j,[3:6])];
            else
                new_pg_data=[new_pg_data;nan(1,2),old_pg_data(j,[3:6])];

            end
            
                       

        end
        trainCell{end+1}={pg_type,new_pg_data};


    end



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
                        model_x=updated_model_cell{j*2-1,2};
                        model_y=updated_model_cell{j*2,2};
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
                        model_x=updated_model_cell{j*2-1,2};
                        model_y=updated_model_cell{j*2,2};
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





%---------------------------<Evaluation Functions>-------------------------
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



function [mean_accuracies,total_results]=evalModelsRegressComp(data_mat,model_cell,dist_cell,avg_corners,PG_Estimation_Models,max_compensation_models,poly_functions_array,variance_cell)
    %{
    Inputs:
        data_mat: matrix containing the evaulation data
        model_cell: contains the original polynomial model
        dist_cell: contains the distance between glints at initial calibration
        avg_corners: contains the average corner positions at initial calibration
        PG_Estimation_Models: models for max's approach to estimate the PG
        vector at calibration
        max_compensation_models: models for delta_POG from max's approach
        poly_functions_arrays: array used by the inerpolation approach,
           with format:             poly_function is a cell array where 
            col1: has the subarray with: col 1: model type
             (pg0_left_x, pg0_left_y, etc.), col2: fitted polynomials,
             col3: residual errors, col4: # of calibration points
            col2 has the average corner locations
    Outputs:
        accuracy is reported as sqrt((POG_x-t_x)^2+(POG_y-t_y)^2)

        mean_accuracies: array with accuracies:
        right_poly,left_poly,combined_poly,
        right_interp,left_interp,combined_interp
        right_max, left_max, combined_max

        total_results: matrix with columns:
    accuracy_right_poly, accuracy_left_poly, accuracy_combined_poly (3), accuracy_right_interp, accuracy_left_interp, accuracy_combined_interp (6),
    accuracy_right_max, accuracy_left_max, accuracy_combined_max (9),t_x, t_y (11),

    %}
    
    left_headers={'pg0_left_x','pg0_left_y','pg1_left_x','pg1_left_y','pg2_left_x','pg2_left_y'};
    right_headers={'pg0_right_x','pg0_right_y','pg1_right_x','pg1_right_y','pg2_right_x','pg2_right_y'};
    check_model_right=any(ismember(model_cell(:,1),right_headers));
    check_model_left=any(ismember(model_cell(:,1),left_headers));

    %Outputs the data as: 
    reformatted_data=reformatData(data_mat);

    total_results=evalAccuracyComp(model_cell,reformatted_data,right_headers,left_headers,check_model_right,check_model_left,dist_cell,avg_corners,PG_Estimation_Models,max_compensation_models,poly_functions_array,variance_cell);

    mean_accuracies=mean(total_results(:,[1:9]),1,'omitnan');
    



end


function total_results=evalAccuracyComp(model_cell,reformatted_data,right_headers,left_headers,check_model_right,check_model_left,dist_cell,avg_corners,PG_Estimation_Models,max_compensation_models,poly_functions_array,variance_cell)
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

    %Filter parameters
    %MAV_Length=15;
    %past_tx=reformatted_data(1,22);
    %past_ty=reformatted_data(1,23);
    
    
    %Arrays for interp POG filtering
    %POG_x_interp_right_array=[];
    %POG_y_interp_right_array=[];

    %POG_x_interp_left_array=[];
    %POG_y_interp_left_array=[];



 
    NANTHRESH=0; %Number of nan values we tolerate as input to our tree model
    [row_n,~]=size(reformatted_data);
    total_results=[];
    for i=[1:row_n]
        results_row=nan(1,11);
        curr_row=reformatted_data(i,:); %Current data row

        t_x=curr_row(22);    %Targets
        t_y=curr_row(23);

        results_row(10)=t_x;
        results_row(11)=t_y;


        max_found_count=0;


        %--------------<Finding the right POG first>--------------
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
                    cur_val=updated_model_cell{1,4};
                    cur_ind=1;
                    for j=[1:2:row_new]
                        if (updated_model_cell{j,4}>cur_val) %Change > to < if using residuals
                            cur_val=updated_model_cell{j,4};
                            cur_ind=j;
                        end
                    end
                    model_x=updated_model_cell{cur_ind,2};
                    model_y=updated_model_cell{cur_ind+1,2};
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

                    PG_model_x=PG_Estimation_Models{pg_estimation_indx,2};
                    PG_model_y=PG_Estimation_Models{pg_estimation_indy,2};
    
                    [d_calib,d_curr]=findScalingFactors(dist_cell,valid_header,right_headers,right_pgs);
                    
                    if ~isnan(d_calib) && ~isnan(d_curr)
                        
                        %----------Running Classic Polynomial
                    
                        pg_x=(d_calib/d_curr).*right_pgs(pg_x_ind);
                        pg_y=(d_calib/d_curr).*right_pgs(pg_y_ind);

        
                        [predictors_x,predictors_y]=customPolynomial(pg_x,pg_y);
        
                        POG_x_poly_right=findPOG(model_x,predictors_x);
                        POG_y_poly_right=findPOG(model_y,predictors_y);
                        accuracy_poly=sqrt((POG_x_poly_right-t_x)^2+(POG_y_poly_right-t_y)^2);
                        results_row(1)=accuracy_poly;
                        %-----------Running Max's approach
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

                                accuracy_max_right=sqrt((t_x-POG_x_max_right)^2+(t_y-POG_y_max_right)^2);
                                results_row(7)=accuracy_max_right;

                             end

                            
                        end 
                    end
                        
                    %-----------Running Polynomial Interpolation Approcah
                    %We use PG's with smallest residuals for the
                    %calibrated polynomial when doing the
                    %interpolation approach (gives best results)
                    cur_val=updated_model_cell{1,3};
                    cur_ind=1;
                    for j=[1:2:row_new]
                        if (updated_model_cell{j,3}<cur_val) %Change > to < if using residuals
                            cur_val=updated_model_cell{j,3};
                            cur_ind=j;
                        end
                    end
                    header_x=valid_header{cur_ind};
                    header_y=valid_header{cur_ind+1};
                    pg_x_ind=ismember(right_headers,header_x);
                    pg_y_ind=ismember(right_headers,header_y);

                    [d_calib,d_curr]=findScalingFactors(dist_cell,valid_header,right_headers,right_pgs);
                
                    if ~isnan(d_calib) && ~isnan(d_curr)
                 
                        pg_x=(d_calib/d_curr).*right_pgs(pg_x_ind);
                        pg_y=(d_calib/d_curr).*right_pgs(pg_y_ind);

        
                        [predictors_x,predictors_y]=customPolynomial(pg_x,pg_y);

                        right_corners=curr_row(14:17); %Gets the values for the right corners

                        %Weighted Functions in the x-direction
                        poly_functions_x=reformatDataMultivariateInterp(poly_functions_array,header_x,'right');
                        weighted_poly_x=multivariateInterp(poly_functions_x,right_corners,closest_type,weighting_type,k,p,sigma);

                        poly_functions_y=reformatDataMultivariateInterp(poly_functions_array,header_y,'right');
                        weighted_poly_y=multivariateInterp(poly_functions_y,right_corners,closest_type,weighting_type,k,p,sigma);

                        %Finding interpolation results
                        if ~all(isnan(weighted_poly_x)) && ~all(isnan(weighted_poly_y))
                            POG_x_interp_right=findPOG(weighted_poly_x',predictors_x);
                            POG_y_interp_right=findPOG(weighted_poly_y',predictors_y);
                            %{
                            %Filtering
                            if t_x~=past_tx||t_y~=past_ty
                                    POG_x_interp_right_array=[];
                                    POG_y_interp_right_array=[];
                            end

                            
                            POG_x_interp_right_array=[POG_x_interp_right_array,POG_x_interp_right];
                            POG_y_interp_right_array=[POG_y_interp_right_array,POG_y_interp_right];

                            if length(POG_x_interp_right_array)>MAV_Length
                                POG_x_interp_right_array=POG_x_interp_right_array(2:end);
                                POG_y_interp_right_array=POG_y_interp_right_array(2:end);
                            end
                            POG_x_interp_right=mean(POG_x_interp_right_array,'omitnan');
                            POG_y_interp_right=mean(POG_y_interp_right_array,'omitnan');
                            %}
                            
                            accuracy_interp_right=sqrt((POG_x_interp_right-t_x)^2+(POG_y_interp_right-t_y)^2);
                            results_row(4)=accuracy_interp_right;
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
                    cur_val=updated_model_cell{1,4};
                    cur_ind=1;
                    for j=[1:2:row_new]
                        if (updated_model_cell{j,4}>cur_val) %Change > to < if using iteratively least squares
                            cur_val=updated_model_cell{j,4};
                            cur_ind=j;
                        end
                    end
                    model_x=updated_model_cell{cur_ind,2};
                    model_y=updated_model_cell{cur_ind+1,2};
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

                    PG_model_x=PG_Estimation_Models{pg_estimation_indx,2};
                    PG_model_y=PG_Estimation_Models{pg_estimation_indy,2};


                    [d_calib,d_curr]=findScalingFactors(dist_cell,valid_header,left_headers,left_pgs);
                    
                    if ~isnan(d_calib) && ~isnan(d_curr)

                        %------------Running Typicaly POG Approach
                        pg_x=(d_calib/d_curr).*left_pgs(pg_x_ind);
                        pg_y=(d_calib/d_curr).*left_pgs(pg_y_ind);
        
                        [predictors_x,predictors_y]=customPolynomial(pg_x,pg_y);
        
                        POG_x_poly_left=findPOG(model_x,predictors_x);
                        POG_y_poly_left=findPOG(model_y,predictors_y);

                            
                        accuracy_poly=sqrt((POG_x_poly_left-t_x)^2+(POG_y_poly_left-t_y)^2);
                        results_row(2)=accuracy_poly;


                                               
                        %-----------<Running Max's approach>--------------
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


                                accuracy_max_left=sqrt((t_x-POG_x_max_left)^2+(t_y-POG_y_max_left)^2);
                                results_row(8)=accuracy_max_left;

                             end

                            
                        end

                                      
    
                        
                    end
                    %-----------Running Polynomial Interpolation Approcah
                    cur_val=updated_model_cell{1,3};
                    cur_ind=1;
                    for j=[1:2:row_new]
                        if (updated_model_cell{j,3}<cur_val) %Change > to < if using iteratively least squares
                            cur_val=updated_model_cell{j,3};
                            cur_ind=j;
                        end
                    end

                    header_x=valid_header{cur_ind};
                    header_y=valid_header{cur_ind+1};
                    pg_x_ind=ismember(left_headers,header_x);
                    pg_y_ind=ismember(left_headers,header_y);

                    [d_calib,d_curr]=findScalingFactors(dist_cell,valid_header,left_headers,left_pgs);
                    
                    if ~isnan(d_calib) && ~isnan(d_curr)

                        pg_x=(d_calib/d_curr).*left_pgs(pg_x_ind);
                        pg_y=(d_calib/d_curr).*left_pgs(pg_y_ind);
        
                        [predictors_x,predictors_y]=customPolynomial(pg_x,pg_y);


                        left_corners=curr_row(18:21); %Gets the calues for the right corners
    
                        %Weighted Functions in the x-direction
                        poly_functions_x=reformatDataMultivariateInterp(poly_functions_array,header_x,'left');
                        weighted_poly_x=multivariateInterp(poly_functions_x,left_corners,closest_type,weighting_type,k,p,sigma);
    
                        poly_functions_y=reformatDataMultivariateInterp(poly_functions_array,header_y,'left');
                        weighted_poly_y=multivariateInterp(poly_functions_y,left_corners,closest_type,weighting_type,k,p,sigma);
                        if ~all(isnan(weighted_poly_x)) && ~all(isnan(weighted_poly_y))
                        %Finding interpolation results
                            POG_x_interp_left=findPOG(weighted_poly_x',predictors_x);
                            POG_y_interp_left=findPOG(weighted_poly_y',predictors_y);
                            
                            %Filtering
                            %{
                                                        
                            if t_x~=past_tx||t_y~=past_ty
                                    POG_x_interp_left_array=[];
                                    POG_y_interp_left_array=[];
                            end
                            
                            POG_x_interp_left_array=[POG_x_interp_left_array,POG_x_interp_left];
                            POG_y_interp_left_array=[POG_y_interp_left_array,POG_y_interp_left];
    
                            if length(POG_x_interp_left_array)>MAV_Length
                                POG_x_interp_left_array=POG_x_interp_left_array(2:end);
                                POG_y_interp_left_array=POG_y_interp_left_array(2:end);
                            end
                            POG_x_interp_left=mean(POG_x_interp_left_array,'omitnan');
                            POG_y_interp_left=mean(POG_y_interp_left_array,'omitnan');
                            
                            %}
                            accuracy_interp_left=sqrt((POG_x_interp_left-t_x)^2+(POG_y_interp_left-t_y)^2);
                            results_row(5)=accuracy_interp_left;
                        end
                    end
                      
                end
        
            end

        end
        
        %-------------------<Getting Combined Results>-----------------
        if exist('POG_x_poly_right','var') && exist('POG_y_poly_right','var') && exist('POG_x_poly_left','var') && exist('POG_y_poly_left','var') 
          
            POG_combined_x=(POG_x_poly_right+POG_x_poly_left)/2;
            POG_combined_y=(POG_y_poly_right+POG_y_poly_left)/2;
            accuracy_combined=sqrt((t_x-POG_combined_x)^2+(t_y-POG_combined_y)^2);
            results_row(3)=accuracy_combined;

        end

        if exist('POG_x_interp_right','var') && exist('POG_y_interp_right','var') && exist('POG_x_interp_left','var') && exist('POG_y_interp_left','var') 
           %Getting variance of corresponding POGs (var_right_x,
            %var_right_y, var_left_x, var_left_y)
            
            right_x_ind=ismember(variance_cell(:,1),'variance_right_x');
            right_y_ind=ismember(variance_cell(:,1),'variance_right_y');
            left_x_ind=ismember(variance_cell(:,1),'variance_left_x');
            left_y_ind=ismember(variance_cell(:,1),'variance_left_y');

            var_right_x=variance_cell{right_x_ind,2};
            var_right_y=variance_cell{right_y_ind,2};
            var_left_x=variance_cell{left_x_ind,2};
            var_left_y=variance_cell{left_y_ind,2};

            if ~isnan(var_right_x) && ~isnan(var_right_y) && ~isnan(var_left_x) && ~isnan(var_left_y)
                POG_combined_x=(POG_x_interp_right*(1/var_right_x)+POG_x_interp_left*(1/var_left_x))/((1/var_right_x)+(1/var_left_x));
                POG_combined_y=(POG_y_interp_right*(1/var_right_y)+POG_y_interp_left*(1/var_left_y))/((1/var_right_y)+(1/var_left_y));

            else

                POG_combined_x=(POG_x_interp_right+POG_x_interp_left)/2;
                POG_combined_y=(POG_y_interp_right+POG_y_interp_left)/2;
            end
            
            
            
            accuracy_combined=sqrt((t_x-POG_combined_x)^2+(t_y-POG_combined_y)^2);
            results_row(6)=accuracy_combined;

        end

        %Getting combined max results

        if max_found_count>=2
            POG_combined_max_x=(POG_x_max_right+POG_x_max_left)/2;
            POG_combined_max_y=(POG_y_max_right+POG_y_max_left)/2;
            accuracy_combined_max=sqrt((t_x-POG_combined_max_x)^2+(t_y-POG_combined_max_y)^2);

            results_row(9)=accuracy_combined_max;


        end


        total_results=[total_results;results_row];
        past_tx=t_x;
        past_ty=t_y;
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
        glintspupils_left(:,1),glintspupils_left(:,2)];%Adds the pupil positions

end



%-----------------------<Variance Weighting Functions>---------------------

function variance_cell=findPOGVariance(data_mat,model_cell,dist_cell,avg_corners,poly_functions_array)

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
    POG_Results=evalAccuracyCompVariance(model_cell,reformatted_data,right_headers,left_headers,check_model_right,check_model_left,dist_cell,avg_corners,poly_functions_array);
    variance_results=var(POG_Results,0,1,'omitnan');

    variance_headers={'variance_right_x','variance_right_y','variance_left_x','variance_left_y'};

    for i=[1:4]
        variance_cell{i,1}=variance_headers{i};
        variance_cell{i,2}=variance_results(i);
    end

end


function total_results=evalAccuracyCompVariance(model_cell,reformatted_data,right_headers,left_headers,check_model_right,check_model_left,dist_cell,avg_corners,poly_functions_array)
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
                      
                    %-----------Running Polynomial Interpolation Approcah
                    %We use PG's with smallest residuals for the
                    %calibrated polynomial when doing the
                    %interpolation approach (gives best results)
                    cur_val=updated_model_cell{1,3};
                    cur_ind=1;
                    for j=[1:2:row_new]
                        if (updated_model_cell{j,3}<cur_val) %Change > to < if using residuals
                            cur_val=updated_model_cell{j,3};
                            cur_ind=j;
                        end
                    end
                    header_x=valid_header{cur_ind};
                    header_y=valid_header{cur_ind+1};
                    pg_x_ind=ismember(right_headers,header_x);
                    pg_y_ind=ismember(right_headers,header_y);

                    [d_calib,d_curr]=findScalingFactors(dist_cell,valid_header,right_headers,right_pgs);
                
                    if ~isnan(d_calib) && ~isnan(d_curr)
                 
                        pg_x=(d_calib/d_curr).*right_pgs(pg_x_ind);
                        pg_y=(d_calib/d_curr).*right_pgs(pg_y_ind);

        
                        [predictors_x,predictors_y]=customPolynomial(pg_x,pg_y);

                        right_corners=curr_row(14:17); %Gets the values for the right corners

                        %Weighted Functions in the x-direction
                        poly_functions_x=reformatDataMultivariateInterp(poly_functions_array,header_x,'right');
                        weighted_poly_x=multivariateInterp(poly_functions_x,right_corners,closest_type,weighting_type,k,p,sigma);

                        poly_functions_y=reformatDataMultivariateInterp(poly_functions_array,header_y,'right');
                        weighted_poly_y=multivariateInterp(poly_functions_y,right_corners,closest_type,weighting_type,k,p,sigma);

                        %Finding interpolation results
                        if ~all(isnan(weighted_poly_x)) && ~all(isnan(weighted_poly_y))
                            POG_x_interp_right=findPOG(weighted_poly_x',predictors_x);
                            POG_y_interp_right=findPOG(weighted_poly_y',predictors_y);
                            results_row(1)=POG_x_interp_right;
                            results_row(2)=POG_y_interp_right;
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
                   
                    %-----------Running Polynomial Interpolation Approcah
                    cur_val=updated_model_cell{1,3};
                    cur_ind=1;
                    for j=[1:2:row_new]
                        if (updated_model_cell{j,3}<cur_val) %Change > to < if using iteratively least squares
                            cur_val=updated_model_cell{j,3};
                            cur_ind=j;
                        end
                    end

                    header_x=valid_header{cur_ind};
                    header_y=valid_header{cur_ind+1};
                    pg_x_ind=ismember(left_headers,header_x);
                    pg_y_ind=ismember(left_headers,header_y);

                    [d_calib,d_curr]=findScalingFactors(dist_cell,valid_header,left_headers,left_pgs);
                    
                    if ~isnan(d_calib) && ~isnan(d_curr)

                        pg_x=(d_calib/d_curr).*left_pgs(pg_x_ind);
                        pg_y=(d_calib/d_curr).*left_pgs(pg_y_ind);
        
                        [predictors_x,predictors_y]=customPolynomial(pg_x,pg_y);


                        left_corners=curr_row(18:21); %Gets the calues for the right corners
    
                        %Weighted Functions in the x-direction
                        poly_functions_x=reformatDataMultivariateInterp(poly_functions_array,header_x,'left');
                        weighted_poly_x=multivariateInterp(poly_functions_x,left_corners,closest_type,weighting_type,k,p,sigma);
    
                        poly_functions_y=reformatDataMultivariateInterp(poly_functions_array,header_y,'left');
                        weighted_poly_y=multivariateInterp(poly_functions_y,left_corners,closest_type,weighting_type,k,p,sigma);
                        if ~all(isnan(weighted_poly_x)) && ~all(isnan(weighted_poly_y))
                        %Finding interpolation results
                            POG_x_interp_left=findPOG(weighted_poly_x',predictors_x);
                            POG_y_interp_left=findPOG(weighted_poly_y',predictors_y);                       
                            results_row(3)=POG_x_interp_left;
                            results_row(4)=POG_y_interp_left;
                        end
                    end
                      
                end
        
            end

        end
        
        total_results=[total_results;results_row];

    end

end