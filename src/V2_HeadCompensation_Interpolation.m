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
curr_corners=[342,145,201,250];
poly_functions={[2,2.8,3,3.8,5],[350,142,195,253];...
    [2.4,2.55,3.01,3.81,5.2],[340,144,198,251];...
    [1.9,2.8,3.035,3.82,5.12],[352,141,192,243];...
    [2.2,2.75,2.95,3.83,4.82],[345,138,197,249];...
    [2.35,2.8,3.05,3.79,4.92],[339,148,204,252];...
    [1.99,2.85,3.1,3.78,5.01],[347,152,207,255];...
    };
closest_type='euclidean';
weighting_type='idw';
k=3;
weighted_poly=multivariateInterp(poly_functions,curr_corners,closest_type,weighting_type,k);


%%

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











%##########################Function Definitions############################




%------------------<Multivariable Interpolation Functions>-----------------

function weighted_poly=multivariateInterp(poly_functions,curr_corners,closest_type,weighting_type,k)
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

%}




    %Extracting values
    calibrated_functions=cell2mat(poly_functions(:,1)); %Matrix where each row are the coefficients of the fitted polynomial
    calibrated_corners=cell2mat(poly_functions(:,2));
    
    %Check for nan values in current inner/outer corner locations
    %if only one corner is currently found (other one is nan), only use that same corresponding
    %corner from the calibrated corner locations
    %We also check for nan values in the calibrated corner locations 
    
    [num_notnan,calibrated_corners,calibrated_functions]=checkCalibratedCorners(calibrated_corners,calibrated_functions);
    if num_notnan>=k %We have an acceptable number of calibration corner locations
        bool_check=false;
        if ~isnan(curr_corners(1)) && ~isnan(curr_corners(3)) %We have both inner and outer current eye corners
            bool_check=true;
            %so we just continue without changing anything
        elseif ~isnan(curr_corners(1)) && isnan(curr_corners(3)) %We have the inner corner, but not the outer
            %Only use the inner corners
            calibrated_corners=calibrated_corners(:,[1:2]);
            curr_corners=curr_corners([1:2]);
            bool_check=true;
        elseif isnan(curr_corners(1)) && ~isnan(curr_corners(3)) %We don't have inner corner, but we have the outer
            %Only use the outer corner
            calibrated_corners=calibrated_corners(:,[3:4]);
            curr_corners=curr_corners([3:4]);
            bool_check=true;
        else %We dont have either corners currently
            weighted_poly=nan(1,length(calibrated_functions(1,:))); %Returns nan for the current polynomial
        end
        
        if bool_check
            %Finding closesness between current corners and calibrated corners 
            score=closenessScore(calibrated_corners,curr_corners,closest_type);
            
            %Arrange scores from lowest to highest (highest means further away)
            [ascend_scores,inds]=sort(score);
            ascend_functions=calibrated_functions(inds,:);
            ascend_corners=calibrated_corners(inds,:);
            
            %Truncating to k-closest
            k_functions=ascend_functions(1:k,:); %matrix where each row are the top k fitted polynomials
            k_corners=ascend_corners(1:k,:);
            
            p=1; %order of weighting
            weights=weightingFunction(k_corners,curr_corners,weighting_type,p);
            
            %we multiply every row by the corresponding weight (row 1*weights(1)+row
            %2*weights(2) and final polynomial weights are the sum along the columns
            
            scaled_functions=weights.*k_functions; 
            
            weighted_poly=sum(scaled_functions,1); %Sums the polynomial coefficients to give our final interpolated polynomial
        end

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


function [num_notnan,new_corners,new_functions]=checkCalibratedCorners(calibrated_corners,calibrated_functions)
    inds_innercorners_notnan=~isnan(calibrated_corners(:,1)); %Gets row index where inner corners are not nan
    inds_outercorners_notnan=~isnan(calibrated_corners(:,3)); %Gets row index where outer corners are not nan
    inds_functions_notnan=~isnan(calibrated_functions(:,1)); %Gets row index where polynomial functions are not nan

    joint_notnan=inds_innercorners_notnan&inds_outercorners_notnan&inds_functions_notnan; %Indexes where we have only the function
    num_notnan=sum(joint_notnan);
    new_functions=calibrated_functions(joint_notnan,:);
    new_corners=calibrated_corners(joint_notnan,:);
end