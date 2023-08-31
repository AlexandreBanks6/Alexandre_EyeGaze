clear
clc
close all

%Testing initial calibration
data_root='../../data/eyecorner_userstudy_converted';
%Getting list of subfolders
folder_list=dir(data_root);
dirnames={folder_list([folder_list.isdir]).name};
num_dir=length(dirnames);

%Global Params:
CALIB_THRESHOLD=5;
EVAL_THRESHOLD=3; %Threshold to be considered a valid evaluation trial

%Looping for all participants
for m=[1:num_dir]
    %if dirnames{m}(1)=='P' %We have a participant and run calibrations and/evaluations
    if strcmp(dirnames{m},'P02')
        %Getting Calibration Data
        calib_init_path=[data_root,'/',dirnames{m},'/calib_only_merged_Calib_Init.csv'];
        calib_init_data=readmatrix(calib_init_path);

        calib_up_path=[data_root,'/',dirnames{m},'/calib_only_merged_Calib_Up.csv'];
        calib_up_data=readmatrix(calib_up_path);

        calib_down_path=[data_root,'/',dirnames{m},'/calib_only_merged_Calib_Down.csv'];
        calib_down_data=readmatrix(calib_down_path);

        calib_right_path=[data_root,'/',dirnames{m},'/calib_only_merged_Calib_Right.csv'];
        calib_right_data=readmatrix(calib_right_path);

        calib_left_path=[data_root,'/',dirnames{m},'/calib_only_merged_Calib_Left.csv'];
        calib_left_data=readmatrix(calib_left_path);

        check_calib=checkDetection(calib_init_data,CALIB_THRESHOLD);
        if (check_calib==true)
            [train_cell,dist_cell,avg_corners]=getRegressionData(calib_init_data,CALIB_THRESHOLD); %Also gets the average eye corner location at the calibration
            if length(dist_cell)==0
                continue
            end
            model_poly=robustRegressor(train_cell); %Robust Regressor model params
            
            old_data_cell={calib_up_data,calib_down_data,calib_right_data,calib_left_data};
            data_cell=cell(0);
            for i=[1:length(old_data_cell)]

                curr_data=old_data_cell{i};
                valid_dat=checkDetection(curr_data,1); %We use data as long as one unique detection happens
                if valid_dat
                    data_cell=[data_cell,curr_data];

                end

            end
            if ~isempty(data_cell)
                compensation_data=prepCompensationData(data_cell,model_poly,dist_cell,avg_corners);
                [tree_mdl_right,input_var_names_right]=fitTreeModel(compensation_data{1});
                [tree_mdl_left,input_var_names_left]=fitTreeModel(compensation_data{2});
            end
        
        end


    end
end


%##########################<Function Definitions>##########################

%-----------------<Initial Data Prep Functions>-----------------
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
    avg_corners=mean(corner_data,1,'omitnan');
    
end

function train_matrix=getIndividualRegressionData(data_matrix,pg_type,thresh)
    %Function tha returns data to train polynomial regressor from the toal
    %data matrix in eye gaze tracking. 
    %Returns: train_matrix with columns: pg(pg_type)_x,pg(pg_type)_y,target_x,target_y
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
            train_matrix=[train_matrix;[data_raw(i,2)-data_raw(i,1),data_raw(i,4)-data_raw(i,3),data_raw(i,5),data_raw(i,6)]];
            pupil_detect_count=pupil_detect_count+1;
        elseif ~anynan(data_raw(i,[1:4]))
            train_matrix=[train_matrix;[data_raw(i,2)-data_raw(i,1),data_raw(i,4)-data_raw(i,3),data_raw(i,5),data_raw(i,6)]];
        else
            train_matrix=[train_matrix;[NaN,NaN,NaN,NaN]];
        end


    end
    if pupil_detect_count<thresh
        train_matrix=nan; %Not a valid sample of points, too few detections
    end



end
%---------------<Polynomial Regression Functions>---------------------

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
    Input: data_cell has up to five cells where each cell is a matrix
    containing the data to train the compensation model.
    Each cell has: Calib_Init,Calib_Up,Calib_Down,Calib_Right,Calib_Left

    compensationData is a cell array with two cells having:
    cell 1: del_POG_x_right,del_POG_y_right,del_corner_inner_x_right,del_corner_inner_y_right,
    del_corner_outer_x_right,del_corner_outer_y_right,alpha_right,t_x,t_y

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
        end

        if check_model_left
            error_vec_left=findCalibrationErrors(model_cell,reformatted_data_left,left_headers,dist_cell);
        end

        if ~all(isnan(error_vec_right(:,1)))
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
                error_vec_right(:,3),error_vec_right(:,4)];

        else
            compensation_data{1}=[compensation_data{1},nan,nan,nan,nan,nan,nan,nan,nan,nan];
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
                error_vec_left(:,3),error_vec_left(:,4)];

        else
            compensation_data{2}=[compensation_data{2},nan,nan,nan,nan,nan,nan,nan,nan,nan];
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
POG=model(1)+sum(model(2:end)'.*predictors);


end

function [tree_mdl,input_var_names]=fitTreeModel(train_data)
    
    %We do this four times for the x,y and for both eyes
       
    
    if all(isnan(train_data(:,1)))    %The output variable is all nan so we can't train
        tree_mdl=nan;
        input_var_names=nan;
    else
        input_var_names=cell(0);
        predictors=[];

        if ~all(isnan(train_data(:,3)))
            predictors=[predictors,d_curr_inner];
            input_var_names=[input_var_names,'d_curr_inner_x'];

        end

        if ~all(isnan(train_data(:,4)))
            predictors=[predictors,d_curr_inner];
            input_var_names=[input_var_names,'d_curr_inner_y'];

        end

        if ~all(isnan(train_data(:,5)))
            predictors=[predictors,d_curr_outer];
            input_var_names=[input_var_names,'d_curr_outer_x'];

        end

        if ~all(isnan(train_data(:,6)))
            predictors=[predictors,d_curr_outer];
            input_var_names=[input_var_names,'d_curr_outer_y'];

        end

        if ~all(isnan(train_data(:,7)))
            predictors=[predictors,alpha];
            input_var_names=[input_var_names,'alpha'];

        end
        
        if ~isempty(input_var_names)
            rng default
            tree_mdl=fitrtree(predictors,d_POG,'Surrogate','on',OptimizeHyperparameters','auto',...
                'HyperparameterOptimizationOptions',struct('AcquisitionFunctionName',...
                'expected-improvement-plus','MaxTime',600)); %Optimizes the hyperparameters that
                                                    % minimize five-fold
                                                    % cross-validation loss
                                                    % using baesian
                                                    % optimization. Maximum
                                                    % time for optimization
                                                    % is 10 minutes. Also
                                                    % uses surrogate splits
                                                    % because there is
                                                    % missing data

        else
            tree_mdl=nan;
            input_var_names=nan;
        end





    end


end


