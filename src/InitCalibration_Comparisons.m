clear
clc
close all

%Note: We define the PG vector as Glint-Pupil

%Testing initial calibration
calib_init_path='../resources/calib_only_merged_Calib_Init.csv';
calib_init_data=readmatrix(calib_init_path);



check_val=checkDetection(calib_init_data,5);
if check_val==true
    %Setting up to three PG vector predictor variables with corresponding
    %targets
    train_cell=getRegressionData(calib_init_data);
    model_robust=robustRegressor(train_cell); %Robust Regressor model params
    model_least_square=leastSquaresRegressor(train_cell);
    

end




function valid=checkDetection(calib_data,thresh)
    [row_n,col_n]=size(calib_data);
    pupil_detect_count=0; %Counts the number of pupils detected per unique calib point
    calib_pastx=calib_data(1,27);
    calib_pasty=calib_data(1,28);
    switched=false;
    valid=false;
    for i=[1:row_n] %Loops for the number of rows
        if ((calib_data(i,8)==1)||(calib_data(i,20)==1)) && (~switched)
            switched=true;
            pupil_detect_count=pupil_detect_count+1;
        end
        if (calib_data(i,27)~=calib_pastx)||(calib_data(i,28)~=calib_pasty)
            switched=false;
            calib_pastx=calib_data(i,27);
            calib_pasty=calib_data(i,28);

        end

    end
    if pupil_detect_count>=thresh
        valid=true;
    end

end


function trainCell=getRegressionData(data_matrix)
pg_types={'pg0_left','pg1_left','pg2_left','pg0_right','pg1_right','pg2_right'};
trainCell={};
for i=[1:length(pg_types)]
    trainMatrix=getIndividualRegressionData(data_matrix,pg_types{i});
    if anynan(trainMatrix)
        continue

    else
        trainCell{end+1}={pg_types{i},trainMatrix};
    end
       


end


end

function train_matrix=getIndividualRegressionData(data_matrix,pg_type)
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
        if (~anynan(data_raw(i,[1:4]))) && (~switched) %We have a new target point and a valid glint detection
            switched=true;
            train_matrix=[train_matrix;[data_raw(i,2)-data_raw(i,1),data_raw(i,4)-data_raw(i,3),data_raw(i,5),data_raw(i,6)]];
            pupil_detect_count=pupil_detect_count+1;
        elseif ~anynan(data_raw(i,[1:4]))
            train_matrix=[train_matrix;[data_raw(i,2)-data_raw(i,1),data_raw(i,4)-data_raw(i,3),data_raw(i,5),data_raw(i,6)]];
        end
        if (data_matrix(i,27)~=calib_pastx)||(data_matrix(i,28)~=calib_pasty)
            switched=false;
            calib_pastx=data_matrix(i,27);
            calib_pasty=data_matrix(i,28);

        end

    end
    if pupil_detect_count<5
        train_matrix=nan; %Not a valid sample of points, too few detections
    end



end


function [rmse_x,rmse_y,b_x,b_y]=customRegressor(train_data)
    %Training data is a nx4 vector where col1=pg_x, col2=pg_y, col3=target_x, col4=target_y
    %Output are 6 model parameters and the residual error for x and y
    %The model parameters are such that
    %b(1)+b(2)*pg_x^2+b(3)*pg_x*pg_y+b(4)*pg_y^2+b(5)*pg_x+b(6)*pg_y
    %The tuning constant is set to 4.685 for bisquare
    pg_x=train_data(:,1);
    pg_y=train_data(:,2);

    t_x=train_data(:,3);
    t_y=train_data(:,4);
    
    predictors=[pg_x.^2,pg_x.*pg_y,pg_y.^2,pg_x,pg_y];

        %Fitting POGx
    [b_x,stats_x]=robustfit(predictors,t_x,'bisquare');

    %Fitting POGy
    [b_y,stats_y]=robustfit(predictors,t_y,'bisquare');
    
    residual_x=stats_x.resid;
    residual_y=stats_y.resid;

    rmse_x=sqrt(mean(residual_x.^2));
    rmse_y=sqrt(mean(residual_y.^2));

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
    title(fittypes{m});
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
        robust_regressor_output=cell(num_pg_detect*2,3); %Final six columns are the model parameters
        for i=[1:num_pg_detect] %Looping for the detections
            train_data=train_cell{i}{2};
            [rmse_x,rmse_y,b_x,b_y]=customRegressor(train_data);
            %Saving x-results
            robust_regressor_output{i*2-1,1}=strcat(train_cell{i}{1},'_x');
            robust_regressor_output{i*2-1,2}=(rmse_x+rmse_y)/2;
            robust_regressor_output{i*2-1,3}=b_x;

            robust_regressor_output{i*2,1}=strcat(train_cell{i}{1},'_y');
            robust_regressor_output{i*2,2}=(rmse_x+rmse_y)/2;
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
    
    predictors=[pg_x.^2,pg_x.*pg_y,pg_y.^2,pg_x,pg_y,ones(length(pg_x),1)];

    b_x=predictors\t_x;
    b_y=predictors\t_y;
    b_x=[b_x(end);b_x];
    b_x(end)=[];

    b_y=[b_y(end);b_y];
    b_y(end)=[];
    
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

