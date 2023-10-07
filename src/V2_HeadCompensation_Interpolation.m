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











%##########################Function Definitions############################




%------------------<Multivariable Interpolation Functions>-----------------

function multivariateInterp(poly_functions,curr_corners,closest_type,weighting_type,k)
%{
Description: Performs multivariate inerpolation where we are given the
calibrated polynomials f1...fn, and the corresponding inner/outer eye
corner locations c1...cn such that c={inner_x,inner_y,outer_x,outer_y}
and the current corner location. We then pick the top k closest corners
c1...cn to our current corner ccurr using a similarity function.
We then find weights wi where i=1..k for each calibrated function using a
weighting function. We then normalize the weights


Inputs:
poly_functions: cell array where: col1=fitted polynomials, col2= avg corner
locations (inner_x, inner_y, outer_x, outer_y)

curr_corners: current corner location

closest_type: type of similarity measure to use options are: 'euclidean'

weighting_type: type of weighting function to use options are: 'idw' for
inverse distance weighting

k: the k-closest eye corner positions that we use to weight the function

%}

%Finding closesness between current corners and calibrated corners 
calibrated_functions=poly_functions{:,1};
calibrated_corners=poly_functions{:,2};

score=closenessScore(calibrated_corners,curr_corners,closest_type);

%Arrange scores from lowest to highest (highest means further away)
[ascend_scores,inds]=sort(score);
ascend_functions=calibrated_functions(inds,:);
ascend_corners=calibrated_corners(inds,:);

%Truncating to k-closest
k_functions=ascend_functions(1:k,:);
k_corners=ascend_corners(1:k,:);






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
            warning('Incorrect Closeness Score Passed')
    
    
    
    end

end


function weights=weightingFunction(calibrated_corners,current_corners,weighting_type)
    %input: top 'k' calibrated_corners which is a matrix with each row corresponding to
    %the corner locations (inner_x, inner_y, outer_x, outer_y)for a given polynomial
    %current_corners: which is the location of the current eye corners
    %weighting_type: 'idw' for inverse distance weighting

    %output: the corresponding weights found from the weighting metric

    switch weighting_type
        case 'idw'
            
        otherwise
            warning('incorrect weighting function type');
    end





end
