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

function [compensationData]=prepCompensationData(data_cell)
    %{
    Input: data_cell has up to five cells where each cell is a matrix
    containing the calibration data for the compensation model.
    Each cell has: Calib_Init,Calib_Up,Calib_Down,Calib_Right,Calib_Left

    compensationData has:
    del_POG_x_right,del_POG_y_right,del_corner_inner_x_right,del_corner_inner_y_right,
    del_corner_outer_x_right,del_corner_outer_y_right,alpha_inner_right,alpha_outer_right,
    del_POG_x_left,del_POG_y_left,del_corner_inner_x_left,del_corner_inner_y_left,
    del_corner_outer_x_left,del_corner_outer_y_left,alpha_inner_left,alpha_outer_left

    values are replaced with nan if they don't exist
    %}
    
    for i=[1:length(data_cell)]
        curr_mat=data_cell{i};   
    
    
    end


end


function [reformatted_data_right,reformatted_data_left]=reformatDataEval(eval_data)
    %Returns the data to compute the conventional model in the format of:
    % reformatted_data_right=frame_no,pg0_leftx,pg0_lefty,...,pg2_leftx,pg2_lefty,target_x,target_y,right_inner_x,right_inner_y,right_outer_x,right_outer_y,..
    % reformmated_data_left=frame_no,pg0_rightx,pg0_righty,...,pg2_rightx,pg2_righty,target_x,target_y,left_inner_x,left_inner_y,left_outer_x,left_outer_y,..
    glintspupils_right_ind=[3,4,9,10,11,12,13,14]; %Contains the glints and pupil positions such that pupil_x,pupil_y,glint0_x,glint0_y...
    glintspupils_left_ind=[15,16,21,22,23,24,25,26];

    glintspupils_right=eval_data(:,glintspupils_right_ind);
    glintspupils_left=eval_data(:,glintspupils_left_ind);

    reformatted_data_right=[eval_data(:,2),glintspupils_right(:,3)-glintspupils_right(:,1),...
        glintspupils_right(:,4)-glintspupils_right(:,2),glintspupils_right(:,5)-glintspupils_right(:,1),...
        glintspupils_right(:,6)-glintspupils_right(:,2),glintspupils_right(:,7)-glintspupils_right(:,1),...
        glintspupils_right(:,8)-glintspupils_right(:,2),eval_data(:,27),eval_data(:,28),eval_data(:,29:end)];

    reformatted_data_left=[eval_data(:,2),glintspupils_left(:,3)-glintspupils_left(:,1),...
        glintspupils_left(:,4)-glintspupils_left(:,2),glintspupils_left(:,5)-glintspupils_left(:,1),...
        glintspupils_left(:,6)-glintspupils_left(:,2),glintspupils_left(:,7)-glintspupils_left(:,1),...
        glintspupils_left(:,8)-glintspupils_left(:,2),eval_data(:,27),eval_data(:,28),eval_data(:,29:end)];
    

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


