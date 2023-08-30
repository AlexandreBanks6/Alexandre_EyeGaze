clear
clc
close all

%Testing initial calibration
data_root='../../data/eyecorner_userstudy_converted';
%Getting list of subfolders
folder_list=dir(data_root);
dirnames={folder_list([folder_list.isdir]).name};
num_dir=length(dirnames);

%Looping for all participants
for m=[1:num_dir]
    if dirnames{m}(1)=='P' %We have a participant and run calibrations and/evaluations
        %Testing initial calibration
        calib_init_path=[data_root,'/',dirnames{m},'/calib_only_merged_Calib_Init.csv'];
        calib_init_data=readmatrix(calib_init_path);
        eval_init_path=[data_root,'/',dirnames{m},'/calib_only_merged_Eval_Init.csv'];
        eval_init_data=readmatrix(eval_init_path);



    end
end


%-------------------------<Function Definitions>---------------------------

function [compensationData]=prepCompensationData(data_cell,model_cell,dist_cell)
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
    %}
    
    left_headers={'pg0_left_x','pg0_left_y','pg1_left_x','pg1_left_y','pg2_left_x','pg2_left_y'};
    right_headers={'pg0_right_x','pg0_right_y','pg1_right_x','pg1_right_y','pg2_right_x','pg2_right_y'};
    check_model_right=any(ismember(model_cell(:,1),right_headers));
    check_model_left=any(ismember(model_cell(:,1),left_headers));
    for i=[1:length(data_cell)]
        curr_mat=data_cell{i}; 
        [reformatted_data_right,reformatted_data_left]=reformatDataEval(curr_mat)
        if check_model_right            
            error_vec_right=findCalibrationErrors(model_cell,reformatted_data_right,right_headers,dist_cell);
        end

        if check_model_left
            %error_vec_left=
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
                    if (updated_model_cell{j,2}>cur_val) && (strcmp(type,'robust')) %Change > to < if using iteratively least squares
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
                continue
            end


        else
            continue
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

function [predictors_x,predictors_y]=customPolynomial(pg_x,pg_y)
    %predictors=[pg_x.^2,pg_x.*pg_y,pg_y.^2,pg_x,pg_y];
    %predictors_x=[pg_x,pg_y,pg_x.^2];
    %predictors_y=[pg_y,pg_x.^2,pg_x.*pg_y,pg_x.^2.*pg_y];
    predictors_x=[pg_x.^2,pg_x.*pg_y,pg_y.^2,pg_x,pg_y];
    predictors_y=[pg_x.^2,pg_x.*pg_y,pg_y.^2,pg_x,pg_y];
    %predictors_x=[pg_x.^2,pg_x.*pg_y,pg_y.^2,pg_x,pg_y];
    %predictors_y=[pg_x.^2,pg_x.*pg_y,pg_y.^2,pg_x,pg_y];

end

function [tree_mdl,input_var_names]=fitTreeModel(d_POG,d_curr_outer,d_curr_inner,alpha)
    
    if all(isnan(d_POG))    %The output variable is all nan so we can't train
        tree_mdl=nan;
        input_var_names=nan;
    else
        input_var_names=cell(0);
        predictors=[];
        if ~all(isnan(d_curr_outer))
            predictors=[predictors,d_curr_outer];
            input_var_names=[input_var_names,'d_curr_outer'];

        end

        if ~all(isnan(d_curr_inner))
            predictors=[predictors,d_curr_inner];
            input_var_names=[input_var_names,'d_curr_inner'];

        end

        if ~all(isnan(d_curr_inner))
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


