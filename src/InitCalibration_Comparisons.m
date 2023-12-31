%clear
%clc
%close all

%Note: We define the PG vector as Glint-Pupil Glint "minus" Pupil


data_root='../../data/eyecorner_userstudy_converted';
%Getting list of subfolders
folder_list=dir(data_root);
dirnames={folder_list([folder_list.isdir]).name};
num_dir=length(dirnames);

%Getting accuracy of right,left, and combined per participant for robust
%and classic approaches
%'accuracy_robust_subjects' and 'accuracy_classic_subjects' contains:
%participant number, mean accuracy right, mean accuracy left, combined
%accuracy

%%Getting Accuracy
accuracy_robust_subjects=[];
accuracy_classic_subjects=[];

for m=[1:num_dir]
    if dirnames{m}(1)=='P' %We have a participant and run calibrations and/evaluations
        %Testing initial calibration
        calib_init_path=[data_root,'/',dirnames{m},'/calib_only_merged_Calib_Init.csv'];
        calib_init_data=readmatrix(calib_init_path);
        eval_init_path=[data_root,'/',dirnames{m},'/calib_only_merged_Eval_Init.csv'];
        eval_init_data=readmatrix(eval_init_path);
        CALIB_THRESHOLD=5;
        EVAL_THRESHOLD=3; %Threshold to be considered a valid evaluation trial
        
        check_calib=checkDetection(calib_init_data,CALIB_THRESHOLD);
        check_eval=checkDetection(eval_init_data,EVAL_THRESHOLD);
        if (check_calib==true) && (check_eval==true)
            %Setting up to three PG vector predictor variables with corresponding
            %targets
            train_cell=getRegressionData(calib_init_data,CALIB_THRESHOLD);
            model_robust=robustRegressor(train_cell); %Robust Regressor model params
            model_least_square=leastSquaresRegressor(train_cell);

            [mean_accuracy_robust,total_right_robust,total_left_robust,total_combined_robust]=evalModel(model_robust,eval_init_data,'robust');
            [mean_accuracy_classic,total_right_classic,total_left_classic,total_combined_classic]=evalModel(model_least_square,eval_init_data,'classic');
            
            accuracy_robust_subjects=[accuracy_robust_subjects;str2num(dirnames{m}(2:3)),mean_accuracy_robust];
            accuracy_classic_subjects=[accuracy_classic_subjects;str2num(dirnames{m}(2:3)),mean_accuracy_classic];
        end
    end
end


%%Evaluating Results


%-------------Descriptive Stats
%mean and standard deviation of classic results, with col1=right,
%col2=left, col3=combined
sig_level=0.05/3;

avg_classic=mean(accuracy_classic_subjects(:,[2:end]),1,"omitnan");
avg_robust=mean(accuracy_robust_subjects(:,[2:end]),1,"omitnan");

std_classic=std(accuracy_classic_subjects(:,[2:end]),1,"omitnan");
std_robust=std(accuracy_robust_subjects(:,[2:end]),1,"omitnan");

%-----------Paired t-Test for right,left, and combined
classic_right=accuracy_classic_subjects(:,2);
classic_right=classic_right(~isnan(classic_right));

robust_right=accuracy_robust_subjects(:,2);
robust_right=robust_right(~isnan(robust_right));

[h_right,p_right,ci_right,stats_right]=ttest(classic_right,robust_right,'Alpha',sig_level);


classic_left=accuracy_classic_subjects(:,3);
classic_left=classic_left(~isnan(classic_left));

robust_left=accuracy_robust_subjects(:,3);
robust_left=robust_left(~isnan(robust_left));

[h_left,p_left,ci_left,stats_left]=ttest(classic_left,robust_left,'Alpha',sig_level);


classic_combined=accuracy_classic_subjects(:,4);
classic_combined=classic_combined(~isnan(classic_combined));

robust_combined=accuracy_robust_subjects(:,4);
robust_combined=robust_combined(~isnan(robust_combined));

[h_combined,p_combined,ci_combined,stats_combined]=ttest(classic_combined,robust_combined,'Alpha',sig_level);


p_results=[p_right,p_left,p_combined];
disp(p_results);









%----------------------------<Function Definitions>------------------------
function [predictors_x,predictors_y]=customPolynomial(pg_x,pg_y)
    %predictors=[pg_x.^2,pg_x.*pg_y,pg_y.^2,pg_x,pg_y];
%     predictors_x=[pg_x,pg_y,pg_x.^2];
%     predictors_y=[pg_y,pg_x.^2,pg_x.*pg_y,pg_x.^2.*pg_y];
    predictors_x=[pg_x.^2,pg_x.*pg_y,pg_y.^2,pg_x,pg_y];
    predictors_y=[pg_x.^2,pg_x.*pg_y,pg_y.^2,pg_x,pg_y];

end

function [predictors_x,predictors_y]=classicPolynomial(pg_x,pg_y)
    predictors_x=[pg_x.^2,pg_x.*pg_y,pg_y.^2,pg_x,pg_y];
    predictors_y=[pg_x.^2,pg_x.*pg_y,pg_y.^2,pg_x,pg_y];

end

function POG=findPOG(model,predictors)
%Generalized function to find the POG at run time
POG=model(1)+sum(model(2:end)'.*predictors);


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


function trainCell=getRegressionData(data_matrix,thresh)
pg_types={'pg0_left','pg1_left','pg2_left','pg0_right','pg1_right','pg2_right'};
trainCell={};
for i=[1:length(pg_types)]
    trainMatrix=getIndividualRegressionData(data_matrix,pg_types{i},thresh);
    if anynan(trainMatrix)
        continue

    else
        trainCell{end+1}={pg_types{i},trainMatrix};
    end
       


end


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
        end


    end
    if pupil_detect_count<thresh
        train_matrix=nan; %Not a valid sample of points, too few detections
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




    %Use RANSAC
    %sampleSize=5;
    %maxDistance=20;
    %train_vals_x=[pg_x,pg_y,t_x];
    %train_vals_y=[pg_x,pg_y,t_y];
    %fitLineFcn=@(points) ([ones(size(points,1),1),points(:,1).^2,points(:,1).*points(:,2),points(:,2).^2,points(:,1),points(:,2)]\points(:,3));
    %evalLineFcn=@(model,points) (sqrt((points(:,3)-model(1).*ones(size(points,1),1)+model(2).*points(:,1).^2+model(3).*points(:,1).*points(:,2)+model(4).*points(:,2).^2+model(5).*points(:,1)+model(6).*points(:,2)).^2)); 
    
    %[b_x,~]=ransac(train_vals_x,fitLineFcn,evalLineFcn,sampleSize,maxDistance);
    %[b_y,~]=ransac(train_vals_y,fitLineFcn,evalLineFcn,sampleSize,maxDistance);
    
    %rmse_x=[];
    %rmse_y=[];



    
    
    



    %------------------------------
    %tune_end=8;
    %tune_start=2;
    %tune_step=20;
    %tuning_constants=[tune_start:(tune_end-tune_start)/tune_step:tune_end];
    %tuning_constants=[tuning_constants,4.685];






    %fittypes={'andrews','bisquare','cauchy','fair','huber','logistic','ols','talwar','welsch'};

    %for m=[1:length(fittypes)]

    %We try various tuning constants and use the one with the smallest
    %residual for each x and y
    %b_x_cell=cell(length(tuning_constants));
    %b_y_cell=cell(length(tuning_constants));

    %for i=[1:length(tuning_constants)]
     %   [b_x_cell{i},stats]=robustfit(predictors,t_x,'bisquare',tuning_constants(i));
      %  resids(:,i)=stats.resid;
    %end
    %Compute the mean squared error
    %rmse_resids_x=sqrt(mean(resids.^2,2));
    %[~,min_ind]=min(rmse_resids_x);
    %b_x=b_x_cell{min_ind};
    %rmse_x=rmse_resids_x(min_ind);

    %for i=[1:length(tuning_constants)]
     %   [b_y_cell{i},stats]=robustfit(predictors,t_y,'bisquare',tuning_constants(i));
      %  resids(:,i)=stats.resid;
    %end
    %Compute the mean squared error
    %rmse_resids_y=sqrt(mean(resids.^2,1));
    %[~,min_ind]=min(rmse_resids_y);
    %b_y=b_y_cell{min_ind};
    %rmse_y=rmse_resids_y(min_ind);

    


    
    %Checking results
    
    %{
    test_results=[b_x(1)+b_x(2).*pg_x.^2+b_x(3).*pg_x.*pg_y+b_x(4).*pg_y.^2+b_x(5).*pg_x+b_x(6).*pg_y,...
        b_y(1)+b_y(2).*pg_x.^2+b_y(3).*pg_x.*pg_y+b_y(4).*pg_y.^2+b_y(5).*pg_x+b_y(6).*pg_y];
    figure;
    plot(t_x,t_y,'bo','LineWidth',3,'MarkerSize',12);
    hold on
    plot(test_results(:,1),test_results(:,2),'rx');
    hold off;
    title('Robust Regressor');
    xlabel('screen x');
    ylabel('screen y');
    legend('Data','Fit');
    %}

    %end

    %{
    figure;
    plot(pg_x,t_x,'bo',pg_x,b_x(1)+b_x(2).*pg_x.^2+b_x(3).*pg_x.*pg_y+b_x(4).*pg_y.^2+b_x(5).*pg_x+b_x(6).*pg_y,'r*');
    title('Data and Fit on x-values');
    legend(['Data','Fit']);
    %}

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
            robust_regressor_output{i*2-1,2}=(rmse_x+rmse_y)/2;
            %robust_regressor_output{i*2-1,2}=size(train_data,1);
            robust_regressor_output{i*2-1,3}=b_x;

            robust_regressor_output{i*2,1}=strcat(train_cell{i}{1},'_y');
            robust_regressor_output{i*2,2}=(rmse_x+rmse_y)/2;
            %robust_regressor_output{i*2,2}=size(train_data,1);
            robust_regressor_output{i*2,3}=b_y;

        end
        

    end



end

function [b_x,b_y]=customLeastSquares(train_data)
    %Training data is a nx4 vector where col1=pg_x, col2=pg_y, col3=target_x, col4=target_y
    %Output are 6 model parameters
    %The model parameters are such that
    %b(1)+b(2)*pg_x^2+b(3)*pg_x*pg_y+b(4)*pg_y^2+b(5)*pg_x+b(6)*pg_y
    
    pg_x=train_data(:,1);
    pg_y=train_data(:,2);

    t_x=train_data(:,3);
    t_y=train_data(:,4);
    [return_x,return_y]=classicPolynomial(pg_x,pg_y);
    predictors_x=[ones(length(pg_x),1),return_x];
    predictors_y=[ones(length(pg_y),1),return_y];

    b_x=predictors_x\t_x;
    b_y=predictors_y\t_y;
    
    %{
    figure;
    plot(pg_x,t_x,'bo')
    hold on
    plot(pg_x,b_x(1)+b_x(2).*pg_x.^2+b_x(3).*pg_x.*pg_y+b_x(4).*pg_y.^2+b_x(5).*pg_x+b_x(6).*pg_y,'r*');
    hold off
    title('Data and Fit on x-values');
    legend('Data','Fit');
    %}
    
    %{
    test_results=[b_x(1)+b_x(2).*pg_x.^2+b_x(3).*pg_x.*pg_y+b_x(4).*pg_y.^2+b_x(5).*pg_x+b_x(6).*pg_y,...
        b_y(1)+b_y(2).*pg_x.^2+b_y(3).*pg_x.*pg_y+b_y(4).*pg_y.^2+b_y(5).*pg_x+b_y(6).*pg_y];
    figure;
    plot(t_x,t_y,'bo','LineWidth',3,'MarkerSize',12);
    hold on
    plot(test_results(:,1),test_results(:,2),'rx');
    hold off;
    title('Least Square Fit');
    xlabel('screen x');
    ylabel('screen y');
    legend('Data','Fit');
    %}

end


function ls_regressor_output=leastSquaresRegressor(train_cell)
    %Output is a cell with columns: pg_type, num_points, model parameters
    %pg_type is: pg0_leftx, pg0_lefty,pg1_leftx,pg1_lefty... for pg1->pg3
    %and right/left
    num_pg_detect=length(train_cell);

    if num_pg_detect==0
        ls_regressor_output=nan;
    else %We have detections and proceed
        ls_regressor_output=cell(num_pg_detect*2,3); %Final six columns are the model parameters
        for i=[1:num_pg_detect] %Looping for the detections
            train_data=train_cell{i}{2};
            [b_x,b_y]=customLeastSquares(train_data);
            %Saving x-results
            ls_regressor_output{i*2-1,1}=strcat(train_cell{i}{1},'_x');
            ls_regressor_output{i*2-1,2}=size(train_data,1);
            ls_regressor_output{i*2-1,3}=b_x;
            %Saving y-results
            ls_regressor_output{i*2,1}=strcat(train_cell{i}{1},'_y');
            ls_regressor_output{i*2,2}=size(train_data,1);
            ls_regressor_output{i*2,3}=b_y;

        end
        

    end



end

function [reformatted_data_right,reformatted_data_left]=reformatDataEval(eval_data)
    %Returns the data to evaluate the models in the format of:
    % reformatted_data_right=frame_no,pg0_leftx,pg0_lefty,...,pg2_leftx,pg2_lefty,target_x,target_y
    % reformmated_data_left=frame_no,pg0_rightx,pg0_righty,...,pg2_rightx,pg2_righty,target_x,target_y
    glintspupils_right_ind=[3,4,9,10,11,12,13,14]; %Contains the glints and pupil positions such that pupil_x,pupil_y,glint0_x,glint0_y...
    glintspupils_left_ind=[15,16,21,22,23,24,25,26];

    glintspupils_right=eval_data(:,glintspupils_right_ind);
    glintspupils_left=eval_data(:,glintspupils_left_ind);

    reformatted_data_right=[eval_data(:,2),glintspupils_right(:,3)-glintspupils_right(:,1),...
        glintspupils_right(:,4)-glintspupils_right(:,2),glintspupils_right(:,5)-glintspupils_right(:,1),...
        glintspupils_right(:,6)-glintspupils_right(:,2),glintspupils_right(:,7)-glintspupils_right(:,1),...
        glintspupils_right(:,8)-glintspupils_right(:,2),eval_data(:,27),eval_data(:,28)];

    reformatted_data_left=[eval_data(:,2),glintspupils_left(:,3)-glintspupils_left(:,1),...
        glintspupils_left(:,4)-glintspupils_left(:,2),glintspupils_left(:,5)-glintspupils_left(:,1),...
        glintspupils_left(:,6)-glintspupils_left(:,2),glintspupils_left(:,7)-glintspupils_left(:,1),...
        glintspupils_left(:,8)-glintspupils_left(:,2),eval_data(:,27),eval_data(:,28)];
    

end

function [accuracy_results]=evalAccuracy(model_cell,reformatted_data,header,type)
    [row_n,~]=size(reformatted_data);
    accuracy_results=[];
    for i=[1:row_n]
        curr_row=reformatted_data(i,:); %Current data row    
        
        %Index of values in row that are not NaN
        nan_indexs=isnan(curr_row(2:7));
        nan_indx_values=find(nan_indexs);
        if length(nan_indx_values)<5 %At least one x,y pair are detected
            valid_header=header(~nan_indexs); %Extracts the pg type that is valid for this frame
            model_valid_indexes=ismember(model_cell(:,1),valid_header);
            updated_model_cell=model_cell(model_valid_indexes,:);
            [row_new,~]=size(updated_model_cell);
            %Loops for all the RMSE accuracies and returns index of largest
            cur_val=updated_model_cell{1,2};
            cur_ind=1;
            for j=[1:2:row_new]
                if (updated_model_cell{j,2}<cur_val) && (strcmp(type,'robust')) %Change > to < if using iteratively least squares
                    cur_val=updated_model_cell{j,2};
                    cur_ind=j;
                elseif (updated_model_cell{j,2}>cur_val) && (strcmp(type,'classic'))
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
            pg_x=pgsonly(pg_x_ind);
            pg_y=pgsonly(pg_y_ind);
            if strcmp(type,'robust')
                [predictors_x,predictors_y]=customPolynomial(pg_x,pg_y);
            else
                [predictors_x,predictors_y]=classicPolynomial(pg_x,pg_y);
            end
            
            POG_x=findPOG(model_x,predictors_x);
            POG_y=findPOG(model_y,predictors_y);
%             POG_x=model_x(1)+model_x(2)*predictors_x(1)+model_x(3)*predictors_x(2)+model_x(4)*predictors_x(3)+...
%                 model_x(5)*predictors_x(4)+model_x(6)*predictors_x(5);
%             POG_y=model_y(1)+model_y(2)*predictors_y(1)+model_y(3)*predictors_y(2)+model_y(4)*predictors_y(3)+...
%                 model_y(5)*predictors_y(4)+model_y(6)*predictors_y(5);
            t_x=curr_row(8);
            t_y=curr_row(9);

            accuracy_results=[accuracy_results;[sqrt((t_x-POG_x)^2+(t_y-POG_y)^2),t_x,t_y]]; %Appends the accuracy as well as the target locations



        else
            continue
        end

    end

end

function [accuracy_results]=evalAccuracyCombined(model_cell,right_data,left_data,header_right,header_left,type)
    [row_right,~]=size(right_data);
    [row_left,~]=size(left_data);
    row_n=row_right;
    if row_left<row_right
        row_n=row_left;
    end

    accuracy_results=[];
    for i=[1:row_n]
        curr_row_right=right_data(i,:); %Current data row    
        curr_row_left=left_data(i,:);
        
        %-----------<Do Right First>----------
        %Index of values in row that are not NaN
        nan_indexs=isnan(curr_row_right(2:7));
        nan_indx_values=find(nan_indexs);
        if length(nan_indx_values)<5 %At least one x,y pair are detected
            valid_header=header_right(~nan_indexs); %Extracts the pg type that is valid for this frame
            model_valid_indexes=ismember(model_cell(:,1),valid_header);
            updated_model_cell=model_cell(model_valid_indexes,:);
            [row_new,~]=size(updated_model_cell);
            %Loops for all the RMSE accuracies and returns index of largest
            cur_val=updated_model_cell{1,2};
            cur_ind=1;
            for j=[1:2:row_new]
                if (updated_model_cell{j,2}<cur_val) && (strcmp(type,'robust')) %Change > to < if using iteratively least squares
                    cur_val=updated_model_cell{j,2};
                    cur_ind=j;
                elseif (updated_model_cell{j,2}>cur_val) && (strcmp(type,'classic'))
                    cur_val=updated_model_cell{j,2};
                    cur_ind=j;
                end
            end
            model_x=updated_model_cell{cur_ind,3};
            model_y=updated_model_cell{cur_ind+1,3};
            header_x=valid_header{cur_ind};
            header_y=valid_header{cur_ind+1};
            pg_x_ind=ismember(header_right,header_x);
            pg_y_ind=ismember(header_right,header_y);
            pgsonly=curr_row_right(2:7);
            pg_x=pgsonly(pg_x_ind);
            pg_y=pgsonly(pg_y_ind);
            
            if strcmp(type,'robust')
                [predictors_x,predictors_y]=customPolynomial(pg_x,pg_y);
            else
                [predictors_x,predictors_y]=classicPolynomial(pg_x,pg_y);
            end
            POG_x_right=findPOG(model_x,predictors_x);
            POG_y_right=findPOG(model_y,predictors_y);
            %POG_x_right=model_x(1)+model_x(2)*predictors_x(1)+model_x(3)*predictors_x(2)+model_x(4)*predictors_x(3)+...
                %model_x(5)*predictors_x(4)+model_x(6)*predictors_x(5);
            %POG_y_right=model_y(1)+model_y(2)*predictors_y(1)+model_y(3)*predictors_y(2)+model_y(4)*predictors_y(3)+...
                %model_y(5)*predictors_y(4)+model_y(6)*predictors_y(5);
            t_x=curr_row_right(8);
            t_y=curr_row_right(9);

            


        else
            continue
        end

        
        %-----------<Do Left Next>----------
        %Index of values in row that are not NaN
        nan_indexs=isnan(curr_row_left(2:7));
        nan_indx_values=find(nan_indexs);
        if length(nan_indx_values)<5 %At least one x,y pair are detected
            valid_header=header_left(~nan_indexs); %Extracts the pg type that is valid for this frame
            model_valid_indexes=ismember(model_cell(:,1),valid_header);
            updated_model_cell=model_cell(model_valid_indexes,:);
            [row_new,~]=size(updated_model_cell);
            %Loops for all the RMSE accuracies and returns index of largest
            cur_val=updated_model_cell{1,2};
            cur_ind=1;
            for j=[1:2:row_new]
                if (updated_model_cell{j,2}<cur_val) && (strcmp(type,'robust')) %Change > to < if using iteratively least squares
                    cur_val=updated_model_cell{j,2};
                    cur_ind=j;
                elseif (updated_model_cell{j,2}>cur_val) && (strcmp(type,'classic'))
                    cur_val=updated_model_cell{j,2};
                    cur_ind=j;
                end
            end
            model_x=updated_model_cell{cur_ind,3};
            model_y=updated_model_cell{cur_ind+1,3};
            header_x=valid_header{cur_ind};
            header_y=valid_header{cur_ind+1};
            pg_x_ind=ismember(header_left,header_x);
            pg_y_ind=ismember(header_left,header_y);
            pgsonly=curr_row_left(2:7);
            pg_x=pgsonly(pg_x_ind);
            pg_y=pgsonly(pg_y_ind);
            if strcmp(type,'robust')
                [predictors_x,predictors_y]=customPolynomial(pg_x,pg_y);
            else
                [predictors_x,predictors_y]=classicPolynomial(pg_x,pg_y);
            end
            
            POG_x_left=findPOG(model_x,predictors_x);
            POG_y_left=findPOG(model_y,predictors_y);

%             POG_x_left=model_x(1)+model_x(2)*predictors_x(1)+model_x(3)*predictors_x(2)+model_x(4)*predictors_x(3)+...
%                 model_x(5)*predictors_x(4)+model_x(6)*predictors_x(5);
%             POG_y_left=model_y(1)+model_y(2)*predictors_y(1)+model_y(3)*predictors_y(2)+model_y(4)*predictors_y(3)+...
%                 model_y(5)*predictors_y(4)+model_y(6)*predictors_y(5);
        else
            continue
        end
    
        POG_x=(POG_x_left+POG_x_right)/2;
        POG_y=(POG_y_left+POG_y_right)/2;
        accuracy_results=[accuracy_results;[sqrt((t_x-POG_x)^2+(t_y-POG_y)^2),t_x,t_y]]; %Appends the accuracy as well as the target locations


    end

end

function [mean_accuracy,total_right,total_left,total_combined]=evalModel(model_cell,eval_data,type)
    %Function that takes in a cell of the fitted model and data to evaluate it
    %on and returns:
    %mean_accuracy: mean right, mean left, mean combined
    %total_right=accuracy, target x, target y (same for total_left, and
    %total_combined
    %Accuracy is reported as sqrt((POG_x-t_x)^2+(POG_y-t_y)^2)
    
    %pixel_accuracy=[right_acc,left_acc,combined_acc]
    [reformatted_right,reformatted_left]=reformatDataEval(eval_data);
    left_headers={'pg0_left_x','pg0_left_y','pg1_left_x','pg1_left_y','pg2_left_x','pg2_left_y'};
    right_headers={'pg0_right_x','pg0_right_y','pg1_right_x','pg1_right_y','pg2_right_x','pg2_right_y'};
    check_model_right=any(ismember(model_cell(:,1),right_headers));
    check_model_left=any(ismember(model_cell(:,1),left_headers));
    %Gets the results: pixel accuracy, target_x, target_y; for the right,left,
    %and combined
    if check_model_right && check_model_left

        total_right=evalAccuracy(model_cell,reformatted_right,right_headers,type); %Returns array with the accuracy and target locations
        total_left=evalAccuracy(model_cell,reformatted_left,left_headers,type);
        total_combined=evalAccuracyCombined(model_cell,reformatted_right,reformatted_left,right_headers,left_headers,type);
    elseif check_model_right
        total_right=evalAccuracy(model_cell,reformatted_right,right_headers,type); %Returns array with the accuracy and target locations
        total_left=NaN;
        total_combined=NaN;
    elseif check_model_left
        total_left=evalAccuracy(model_cell,reformatted_left,left_headers,type);
        total_right=NaN;
        total_combined=NaN;
    else
        total_right=NaN;
        total_left=NaN;
        total_combined=NaN;
    end
    if isempty(total_right)
        total_right=NaN;
    end
    if isempty(total_left)
        total_left=NaN;
    end
    if isempty(total_combined)
        total_combined=NaN;
    end

    mean_accuracy=[mean(total_right(:,1)),mean(total_left(:,1)),mean(total_combined(:,1))];



end
