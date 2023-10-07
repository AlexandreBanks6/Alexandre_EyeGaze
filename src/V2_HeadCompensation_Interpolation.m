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
                [train_cell_init,dist_cell_init,avg_corners_init]=getRegressionData(calib_init_data,CALIB_THRESHOLD); %Also gets the average eye corner location at the calibration
                if length(dist_cell_init)==0
                    continue;
                end
                model_poly_init=robustRegressor(train_cell_init); %Get the vanilla model from the initial calibration

                %poly_function_array will contain 6 rows for each of the
                %calibrated polynomials. There are 4 columns with:
                %col1: fitted polynomials, col2: avg corners, col3:
                %residual error, col4=#of points
                poly_function_array=cell(6,1); %Each cell has the above array for each calibration


        end



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










%##########################Function Definitions############################

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

function robust_regressor_output=robustRegressor(train_cell)
    %Output is a cell with columns: pg_type, rmse, model parameters
    %pg_type is: pg0_leftx, pg0_lefty,pg1_leftx,pg1_lefty... for pg1->pg3
    %and right/left
    num_pg_detect=length(train_cell);

    if num_pg_detect==0
        robust_regressor_output=nan;
    else %We have detections and proceed
        robust_regressor_output=cell(num_pg_detect*2,3); %Final entry are the model parameters
        for i=[1:num_pg_detect] %Looping for the detections
            train_data=train_cell{i}{2};
            [rmse_x,rmse_y,b_x,b_y]=customRegressor(train_data);
            %Saving x-results
            robust_regressor_output{i*2-1,1}=strcat(train_cell{i}{1},'_x');
            %robust_regressor_output{i*2-1,2}=(rmse_x+rmse_y)/2;
            robust_regressor_output{i*2-1,2}=size(train_data,1);
            robust_regressor_output{i*2-1,3}=b_x;

            robust_regressor_output{i*2,1}=strcat(train_cell{i}{1},'_y');
            %robust_regressor_output{i*2,2}=(rmse_x+rmse_y)/2;
            robust_regressor_output{i*2,2}=size(train_data,1);
            robust_regressor_output{i*2,3}=b_y;

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

%------------------<Multivariable Interpolation Functions>-----------------

function weighted_poly=multivariateInterp(poly_functions,curr_corners,closest_type,weighting_type,k,p)
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
    
    %Check for nan values in current inner/outer corner locations
    %if only one corner is currently found (other one is nan), only use that same corresponding
    %corner from the calibrated corner locations
    %We also check for nan values in the calibrated corner locations 
    
    [calibrated_corners,calibrated_functions,curr_corners,bool_check]=checkCalibratedCorners(calibrated_corners,calibrated_functions,curr_corners,k);
    if bool_check %We have an acceptable number of calibration corner locations

        %Finding closesness between current corners and calibrated corners 
        score=closenessScore(calibrated_corners,curr_corners,closest_type);
        
        %Arrange scores from lowest to highest (highest means further away)
        [ascend_scores,inds]=sort(score);
        ascend_functions=calibrated_functions(inds,:);
        ascend_corners=calibrated_corners(inds,:);
        
        %Truncating to k-closest
        k_functions=ascend_functions(1:k,:); %matrix where each row are the top k fitted polynomials
        k_corners=ascend_corners(1:k,:);
        

        weights=weightingFunction(k_corners,curr_corners,weighting_type,p);
        
        %we multiply every row by the corresponding weight (row 1*weights(1)+row
        %2*weights(2) and final polynomial weights are the sum along the columns
        
        scaled_functions=weights.*k_functions; 
        
        weighted_poly=sum(scaled_functions,1); %Sums the polynomial coefficients to give our final interpolated polynomial


    else
        weighted_poly=nan(1,length(calibrated_functions(1,:))); %Returns nan for the current polynomial
    
    end




end

function score=closenessScore(calibrated_corners,current_corners,closest_type)
    %input: calibrated_corners which is a matrix with each row corresponding to
    %the corner locations (inner_x, inner_y, outer_x, outer_y)for a given polynomial
    
    %output=vector with the similarity score between each of the calibrated
    %corner locations and the current corner locations
    
    %Higher score means worse
    
    switch closest_type
        case 'euclidean'
            score=sqrt(sum((calibrated_corners-current_corners).^2,2));
        otherwise
            error('Incorrect Closeness Score Passed')
    
    
    
    end

end


function weights=weightingFunction(calibrated_corners,current_corners,weighting_type,p)
    %input: top 'k' calibrated_corners which is a matrix with each row corresponding to
    %the corner locations (inner_x, inner_y, outer_x, outer_y)for a given polynomial
    %current_corners: which is the location of the current eye corners
    %weighting_type: 'idw' for inverse distance weighting
    %p is a power exponetial that controls weight decay, typically left at
    %1
    %Also normalizes the weights to sum to 1

    %output: the corresponding weights found from the weighting metric

    switch weighting_type
        case 'idw'
            weights=1./(sqrt(sum((calibrated_corners-current_corners).^2,2)).^p);
            
        otherwise
            error('incorrect weighting function type');
            
    end


    %Normalize the weights to sum to 1
    
    weights_sum=sum(weights);
    weights=weights/weights_sum;



end


function [new_calib_corners,new_calib_functions,new_curr_corners,bool_check]=checkCalibratedCorners(calibrated_corners,calibrated_functions,curr_corners,k)
    inds_innercorners_notnan=~isnan(calibrated_corners(:,1)); %Gets row index where inner corners are not nan
    inds_outercorners_notnan=~isnan(calibrated_corners(:,3)); %Gets row index where outer corners are not nan
    inds_functions_notnan=~isnan(calibrated_functions(:,1)); %Gets row index where polynomial functions are not nan

    %Init parameters
    new_calib_corners=nan;
    new_calib_functions=nan;
    new_curr_corners=nan;
    bool_check=false;

    if ~isnan(curr_corners(1)) && ~isnan(curr_corners(3)) %We have both inner and outer current eye corners
        if sum(inds_innercorners_notnan&inds_functions_notnan)>=k && sum(inds_outercorners_notnan&inds_functions_notnan)>=k %We have enough calibrated for both corners
            joint_notnan=inds_innercorners_notnan&inds_outercorners_notnan&inds_functions_notnan; %Indexes where we have only the function
            new_calib_functions=calibrated_functions(joint_notnan,:);
            new_calib_corners=calibrated_corners(joint_notnan,:);
            new_curr_corners=curr_corners;
            bool_check=true;
        elseif sum(inds_innercorners_notnan&inds_functions_notnan)>=k %We have enough inner corners from our calibrated corners
            joint_notnan=inds_innercorners_notnan&inds_functions_notnan; %Indexes where we have only the function
            new_calib_functions=calibrated_functions(joint_notnan,:);
            new_calib_corners=calibrated_corners(joint_notnan,[1:2]);
            new_curr_corners=curr_corners([1:2]);
            bool_check=true;

        elseif sum(inds_outercorners_notnan&inds_functions_notnan)>=k %We have enough outer corners from our calibrated corners
            joint_notnan=inds_outercorners_notnan&inds_functions_notnan; %Indexes where we have only the function
            new_calib_functions=calibrated_functions(joint_notnan,:);
            new_calib_corners=calibrated_corners(joint_notnan,[3:4]);
            new_curr_corners=curr_corners([3:4]);
            bool_check=true;
        end

    elseif ~isnan(curr_corners(1)) && isnan(curr_corners(3)) %We have the current inner corner, but not the outer
            joint_notnan=inds_innercorners_notnan&inds_functions_notnan; %Indexes where we have only the function
            new_calib_functions=calibrated_functions(joint_notnan,:);
            new_calib_corners=calibrated_corners(joint_notnan,[1:2]);
            new_curr_corners=curr_corners([1:2]);
            bool_check=true;
    elseif isnan(curr_corners(1)) && ~isnan(curr_corners(3)) %We don't have current inner corner, but we have the current outer
            joint_notnan=inds_outercorners_notnan&inds_functions_notnan; %Indexes where we have only the function
            new_calib_functions=calibrated_functions(joint_notnan,:);
            new_calib_corners=calibrated_corners(joint_notnan,[3:4]);
            new_curr_corners=curr_corners([3:4]);
            bool_check=true;
    end

end