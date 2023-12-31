
%% Clearing
clear
clc
close all

%% Training
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
    if strcmp(dirnames{m},'P11')
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


        %Getting evalulation data
        eval_init_path=[data_root,'/',dirnames{m},'/calib_only_merged_Eval_Init.csv'];
        eval_init_data=readmatrix(eval_init_path);

        eval_up_path=[data_root,'/',dirnames{m},'/calib_only_merged_Eval_Up.csv'];
        eval_up_data=readmatrix(eval_up_path);

        eval_down_path=[data_root,'/',dirnames{m},'/calib_only_merged_Eval_Down.csv'];
        eval_down_data=readmatrix(eval_down_path);

        eval_right_path=[data_root,'/',dirnames{m},'/calib_only_merged_Eval_Right.csv'];
        eval_right_data=readmatrix(eval_right_path);

        eval_left_path=[data_root,'/',dirnames{m},'/calib_only_merged_Eval_Left.csv'];
        eval_left_data=readmatrix(eval_left_path);

        eval_straight_path=[data_root,'/',dirnames{m},'/calib_only_merged_Eval_Straight.csv'];
        eval_straight_data=readmatrix(eval_straight_path);


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
                %Compensation data cell 1= right, cell 2=left
                %{
                    cell 1: del_POG_x_right,del_POG_y_right,del_corner_inner_x_right,del_corner_inner_y_right,
    del_corner_outer_x_right,del_corner_outer_y_right,alpha_right,t_x,t_y,
    then we append head rotations/poses:
    Tx1,Ty1,Tz1,Q01,Qx1,Qy1,Qz1,Tx2,Ty2,Tz2,Q02,Qx2,Qy2,Qz2

    cell 2: del_POG_x_left,del_POG_y_left,del_corner_inner_x_left,del_corner_inner_y_left,
    del_corner_outer_x_left,del_corner_outer_y_left,alpha_left,t_x,t_y
                %}
                compensation_data=prepCompensationData(data_cell,model_poly,dist_cell,avg_corners);
                right_data=compensation_data{1};
                left_data=compensation_data{2};

                del_POG_x_right=right_data(:,1);
                del_POG_y_right=right_data(:,2);
                predictors_right=right_data(:,3:7);
                
                del_POG_x_left=left_data(:,1);
                del_POG_y_left=left_data(:,2);
                predictors_left=left_data(:,3:7);

                [mdl_right_x,mdl_right_y]=fitCompensationRegressor(del_POG_x_right,del_POG_y_right,predictors_right);
                [mdl_left_x,mdl_left_y]=fitCompensationRegressor(del_POG_x_left,del_POG_y_left,predictors_left);
                
                %[tree_mdl_right_x,tree_mdl_right_y,input_var_names_right_x,input_var_names_right_y]=fitTreeModel(compensation_data{1});
                %[tree_mdl_left_x,tree_mdl_left_y,input_var_names_left_x,input_var_names_left_y]=fitTreeModel(compensation_data{2});
                
                %Evaluate our new head compensation, old head comp, and
                %polynomial
                
            else
                %Evaluate only the polynomial
            end
        
        end


    end
end
%% Checking linear regressor fit
figure;
[predictors_x,predictors_y]=compensationPolynomial(predictors_right);
fit_right_x=findCompensation(mdl_right_x,predictors_x);
plot(predictors_x(:,1),fit_right_x,'rx',predictors_x(:,1),del_POG_x_right,'bo');


%% Checking  models
data_mat=calib_right_data;
%data_mat=eval_straight_data;
%data_mat=[calib_up_data;calib_down_data;calib_right_data;calib_left_data];
%data_mat=[eval_straight_data;eval_up_data;eval_down_data;eval_right_data;eval_left_data];
%calib_data=[calib_up_data;calib_down_data;calib_right_data;calib_left_data];
%tree_models=[{'right_x'},{tree_mdl_right_x},{input_var_names_right_x};...
    %{'right_y'},{tree_mdl_right_y},{input_var_names_right_y};...
    %{'left_x'},{tree_mdl_left_x},{input_var_names_left_x};...
    %{'left_y'},{tree_mdl_left_y},{input_var_names_left_y}];

%[mean_accuracies,total_results]=evalModelsRegressComp(calib_right_data,model_poly,dist_cell,avg_corners,tree_models);
[mean_accuracies,total_results]=evalModelsRegressComp(data_mat,model_poly,dist_cell,avg_corners,mdl_right_x,mdl_right_y,mdl_left_x,mdl_left_y);

%{
        total_results: matrix with columns:
        frame_no, accuracy_right_poly, accuracy_left_poly, accuracy_combined_poly,accuracy_right_tree, accuracy_left_tree, accuracy_combined_tree,
        actual_del_pog_right_x, actual_del_pog_right_y, actual_del_pog_left_x,
        actual_del_pog_left_y, 
        estimated_del_pog_right_x, estimated_del_pog_right_y,estimated_del_pog_left_x,
        estimated_del_pog_left_y,
        right_inner_x, right_inner_y, right_outer_x, right_outer_y
        left_inner_x, left_inner_y, left_outer_x, left_outer_y
        alpha_right, alpha_left
        t_x, t_y        

%}
figure;
plot(total_results(:,16),total_results(:,8),'rx',total_results(:,16),total_results(:,12),'bo');
legend('actual','estimated')
title('Actual del POG and Estimated del POG')

figure;
plot(total_results(:,16),total_results(:,8)-total_results(:,12),'rx');
title('Error in del POG')

%% Doing some random tests
data_cell={calib_right_data};
compensation_data=prepCompensationData(data_cell,model_poly,dist_cell,avg_corners);
right_results=compensation_data{1};

data_cell={calib_left_data};
compensation_data=prepCompensationData(data_cell,model_poly,dist_cell,avg_corners);
left_results=compensation_data{1};

data_cell={calib_up_data};
compensation_data=prepCompensationData(data_cell,model_poly,dist_cell,avg_corners);
up_results=compensation_data{1};

data_cell={calib_down_data};
compensation_data=prepCompensationData(data_cell,model_poly,dist_cell,avg_corners);
down_results=compensation_data{1};


del_pog_x_right=right_results(:,1);
del_pog_y_right=right_results(:,2);
del_pog_x_left=left_results(:,1);
del_pog_y_left=left_results(:,2);


%Checking corner movements
right_rotate_inner_right_x=right_results(:,3);
right_rotate_inner_right_y=right_results(:,4);
left_rotate_inner_right_x=left_results(:,3);
left_rotate_inner_right_y=left_results(:,4);
up_rotate_inner_right_x=up_results(:,3);
up_rotate_inner_right_y=up_results(:,4);
down_rotate_inner_right_x=down_results(:,3);
down_rotate_inner_right_y=down_results(:,4);

figure;
plot(right_rotate_inner_right_x,right_rotate_inner_right_y,'bo',left_rotate_inner_right_x,left_rotate_inner_right_y,'ro',...
    up_rotate_inner_right_x,up_rotate_inner_right_y,'go',down_rotate_inner_right_x,down_rotate_inner_right_y,'ko');
legend('rotate right','rotate left','rotate up','rotate down');
xlabel('corner shift x');
ylabel('corner shift y');
%{
figure;
plot(del_pog_x_right,del_pog_y_right,'bo',del_pog_x_left,del_pog_y_left,'ro');
%}

%{
figure;
histogram(del_pog_x_right);
hold on
histogram(del_pog_x_left);
hold off
%}


%{
del_pog_x=right_results(:,1);
t_x=right_results(:,end-1);
t_y=right_results(:,end);
figure;
plot3(del_pog_x,t_x,t_y,'bo');
title('del-pog-right-x vs. t_x');
ylabel('t_x');
xlabel('del pog_x');
zlabel('t_y');
%}
%% Plotting correlation between eye corners and head rotation
clear
clc

%Note that we reverse the sign of del_corner to reflect what is intuitive
%such that del_corner=corner_curr-corner_calib
CALIB_THRESHOLD=5;


data_root='../../data/eyecorner_userstudy_converted';
part_num='P04';
calib_init_path=[data_root,'/',part_num,'/calib_only_merged_Calib_Init.csv'];
calib_init_data=readmatrix(calib_init_path);

calib_up_path=[data_root,'/',part_num,'/calib_only_merged_Calib_Up.csv'];
calib_up_data=readmatrix(calib_up_path);

calib_down_path=[data_root,'/',part_num,'/calib_only_merged_Calib_Down.csv'];
calib_down_data=readmatrix(calib_down_path);

calib_right_path=[data_root,'/',part_num,'/calib_only_merged_Calib_Right.csv'];
calib_right_data=readmatrix(calib_right_path);

calib_left_path=[data_root,'/',part_num,'/calib_only_merged_Calib_Left.csv'];
calib_left_data=readmatrix(calib_left_path);


check_calib=checkDetection(calib_init_data,CALIB_THRESHOLD);

eval_init_path=[data_root,'/',part_num,'/calib_only_merged_Eval_Init.csv'];
eval_init_data=readmatrix(eval_init_path);

eval_up_path=[data_root,'/',part_num,'/calib_only_merged_Eval_Up.csv'];
eval_up_data=readmatrix(eval_up_path);

eval_down_path=[data_root,'/',part_num,'/calib_only_merged_Eval_Down.csv'];
eval_down_data=readmatrix(eval_down_path);

eval_right_path=[data_root,'/',part_num,'/calib_only_merged_Eval_Right.csv'];
eval_right_data=readmatrix(eval_right_path);

eval_left_path=[data_root,'/',part_num,'/calib_only_merged_Eval_Left.csv'];
eval_left_data=readmatrix(eval_left_path);

eval_straight_path=[data_root,'/',part_num,'/calib_only_merged_Eval_Straight.csv'];
eval_straight_data=readmatrix(eval_straight_path);

data_cell={calib_up_data,calib_down_data,calib_right_data,calib_left_data};
%data_cell={eval_straight_data,eval_right_data,eval_left_data,eval_up_data,eval_down_data};
if (check_calib==true)
    [train_cell,dist_cell,avg_corners]=getRegressionData(calib_init_data,CALIB_THRESHOLD); %Also gets the average eye corner location at the calibration
    if length(dist_cell)==0
        disp('dist cell empty')
    else

        model_poly=robustRegressor(train_cell); %Robust Regressor model params

        compensation_data=prepCompensationData(data_cell,model_poly,dist_cell,avg_corners);
        %{
            compensationData is a cell array with two cells having:
            cell 1: del_POG_x_right,del_POG_y_right,del_corner_inner_x_right,del_corner_inner_y_right,
            del_corner_outer_x_right,del_corner_outer_y_right,alpha_right,t_x,t_y
        
            cell 2: del_POG_x_left,del_POG_y_left,del_corner_inner_x_left,del_corner_inner_y_left,
            del_corner_outer_x_left,del_corner_outer_y_left,alpha_left,t_x,t_y
        %}
        right_eye_data=compensation_data{1};
        left_eye_data=compensation_data{2};
        del_POG_x_right=right_eye_data(:,1);
        del_POG_y_right=right_eye_data(:,2);

        right_inner_x=-1*right_eye_data(:,3);
        right_inner_y=-1*right_eye_data(:,4);
        right_outer_x=-1*right_eye_data(:,5);
        right_outer_y=-1*right_eye_data(:,6);
        right_alpha=right_eye_data(:,7).*(180/pi);

        left_inner_x=-1*left_eye_data(:,3);
        left_inner_y=-1*left_eye_data(:,4);
        left_outer_x=-1*left_eye_data(:,5);
        left_outer_y=-1*left_eye_data(:,6);
        left_alpha=left_eye_data(:,7).*(180/pi);
        
        calib_pose_tool1=calib_init_data(:,32:38);
        calib_pose_tool2=calib_init_data(:,42:48);


        %Change to left_eye_data if doing left eye corner
        pose_data_rotated_tool1=right_eye_data(:,10:16);
        pose_data_rotated_tool2=right_eye_data(:,17:23);

        
        %method 1 with reference frame
        %Getting rotation/translation for initial calibration
        calib_q1=calib_pose_tool1(:,4:end);
        calib_q2=calib_pose_tool2(:,4:end);
        q12=quatmultiply(quatconj(calib_q1),calib_q2);
        q12=quaternion(q12);
        calib_quat_avg=meanrot(q12,1);
        calib_t12=calib_pose_tool2(:,1:3)-calib_pose_tool1(:,1:3);
        calib_t12_avg=mean(calib_t12,1);


        %Getting rotation/translation for head each sample during head
        %rotation

        %Rotation
        rotated_q1=pose_data_rotated_tool1(:,4:end);
        rotated_q2=pose_data_rotated_tool2(:,4:end);
        rotated_q12=quatmultiply(quatconj(rotated_q1),rotated_q2); %Gets current quaternion
        calib_q12_avg=compact(calib_quat_avg);
        q_calib_rotated=quatmultiply(quatconj(calib_q12_avg),rotated_q12);
        [yaw,pitch,roll]=quat2angle(q_calib_rotated);
        yaw=yaw.*(180/pi);
        pitch=pitch.*(180/pi);
        roll=roll.*(180/pi);
        %Translation

        rotated_t12=pose_data_rotated_tool2(:,1:3)-pose_data_rotated_tool1(:,1:3);
        translation=rotated_t12-calib_t12_avg;
        translation=translation.*1000;

        
        
        %{
        %Method 2 with no reference
        q1=quaternion(calib_pose_tool2(:,4:end));
        calib_q1_avg=meanrot(q1,1);
        calib_translation_avg=mean(calib_pose_tool2(:,1:3));

        %Getting rotation amount
        calib_q1_avg=compact(calib_q1_avg);
        q_calib_rotated=quatmultiply(quatconj(calib_q1_avg),pose_data_rotated_tool2(:,4:end));
        [yaw,pitch,roll]=quat2angle(q_calib_rotated);
        yaw=yaw.*(180/pi);
        pitch=pitch.*(180/pi);
        roll=roll.*(180/pi);

        %Translation
        translation=pose_data_rotated_tool2(:,1:3)-calib_translation_avg;
        %}
       

    end
end
%%Plotting coorelation between head pose and corner location
%We plot each dependent variable (inner_x, inner_y,outer_x,outer_y etc) vs
%the 6 independent variables (roll, pitch, yaw, translation...)



%#################<Displaying error-corner Correlation>################### 



%~~~~~~~Displaying del_pog_x
MM_PER_PIXEL=0.27781;
fit_offset=2;
%Fitting data with best fit line
[b_inner_x,stats]=robustfit(right_inner_x,del_POG_x_right,'huber',4.2);
right_inner_line_x=b_inner_x(1)+b_inner_x(2).*right_inner_x;
R2_right_inner_x=sum((right_inner_line_x-mean(del_POG_x_right,'omitnan')).^2,'omitnan')./sum((del_POG_x_right-mean(del_POG_x_right,'omitnan')).^2,'omitnan');
right_inner_line_x=b_inner_x(1)+b_inner_x(2).*[min(right_inner_x)-fit_offset:0.5:max(right_inner_x)+fit_offset]; %For displaying

[b_outer_x,stats]=robustfit(right_outer_x,del_POG_x_right,'huber',4.2);
right_outer_line_x=b_outer_x(1)+b_outer_x(2).*right_outer_x;
R2_right_outer_x=sum((right_outer_line_x-mean(del_POG_x_right,'omitnan')).^2,'omitnan')./sum((del_POG_x_right-mean(del_POG_x_right,'omitnan')).^2,'omitnan');
right_outer_line_x=b_outer_x(1)+b_outer_x(2).*[min(right_outer_x)-fit_offset:0.5:max(right_outer_x)+fit_offset]; %For displaying

[b_inner_y,stats]=robustfit(right_inner_y,del_POG_x_right,'huber',4.2);
right_inner_line_y=b_inner_y(1)+b_inner_y(2).*right_inner_y;
R2_right_inner_y=sum((right_inner_line_y-mean(del_POG_x_right,'omitnan')).^2,'omitnan')./sum((del_POG_x_right-mean(del_POG_x_right,'omitnan')).^2,'omitnan');
right_inner_line_y=b_inner_y(1)+b_inner_y(2).*[min(right_inner_y)-fit_offset:0.5:max(right_inner_y)+fit_offset]; %For displaying

[b_outer_y,stats]=robustfit(right_outer_y,del_POG_x_right,'huber',4.2);
right_outer_line_y=b_outer_y(1)+b_outer_y(2).*right_outer_y;
R2_right_outer_y=sum((right_outer_line_y-mean(del_POG_x_right,'omitnan')).^2,'omitnan')./sum((del_POG_x_right-mean(del_POG_x_right,'omitnan')).^2,'omitnan');
right_outer_line_y=b_outer_y(1)+b_outer_y(2).*[min(right_outer_y)-fit_offset:0.5:max(right_outer_y)+fit_offset]; %For displaying

%{
[b_alpha,stats]=robustfit(right_alpha,del_POG_x_right,'huber',4.2);
right_alpha_line=b_alpha(1)+b_alpha(2).*right_alpha;
R2_right_alpha=sum((right_alpha_line-mean(del_POG_x_right,'omitnan')).^2,'omitnan')./sum((del_POG_x_right-mean(del_POG_x_right,'omitnan')).^2,'omitnan');
right_alpha_line=b_alpha(1)+b_alpha(2).*[min(right_alpha)-fit_offset:0.5:max(right_alpha)+fit_offset]; %For displaying
%}

marker_size=5;
marker_inner_type='+';
marker_outer_type='x';
marker_inner_color='#4A7BB7';
marker_outer_color='#A50026';
ylim_range=[-6,7];
fig=figure;
t=tiledlayout(2,2,'TileSpacing','compact');

%inner x
nexttile
plot(right_inner_x,del_POG_x_right.*MM_PER_PIXEL,'color',marker_inner_color,'Marker',marker_inner_type,'LineWidth',1,'MarkerSize',marker_size,'LineStyle',"none");
hold on
plot([min(right_inner_x)-fit_offset:0.5:max(right_inner_x)+fit_offset],right_inner_line_x.*MM_PER_PIXEL,'LineWidth',2,'Color',marker_outer_color,'LineStyle','--');
hold off
ylim(ylim_range);

title('\deltaC Inner x','FontName','Times New Roman','FontSize',13);
a = get(gca,'XTickLabel');  
set(gca,'XTickLabel',a,'fontsize',10)
set(gca,'XTickLabelMode','auto')

%Showing best fit
eq=sprintf('y=%.2f x + %.2f \nR^2=%.2f',b_inner_x(2),b_inner_x(1),R2_right_inner_x);
text(-25,5,eq,'FontSize',8);


%inner y
nexttile

plot(right_inner_y,del_POG_x_right.*MM_PER_PIXEL,'color',marker_inner_color,'Marker',marker_inner_type,'LineWidth',1,'MarkerSize',marker_size,'LineStyle',"none");
hold on
plot([min(right_inner_y)-fit_offset:0.5:max(right_inner_y)+fit_offset],right_inner_line_y.*MM_PER_PIXEL,'LineWidth',2,'Color',marker_outer_color,'LineStyle','--');
hold off
ylim(ylim_range);

title('\deltaC Inner y','FontName','Times New Roman','FontSize',13);
a = get(gca,'XTickLabel');  
set(gca,'XTickLabel',a,'fontsize',10)
set(gca,'XTickLabelMode','auto')

%Showing best fit
eq=sprintf('y=%.2f x + %.2f \nR^2=%.2f',b_inner_y(2),b_inner_y(1),R2_right_inner_y);
text(-19,5,eq,'FontSize',8);

%outer x
nexttile

plot(right_outer_x,del_POG_x_right.*MM_PER_PIXEL,'color',marker_inner_color,'Marker',marker_inner_type,'LineWidth',1,'MarkerSize',marker_size,'LineStyle',"none");
hold on
plot([min(right_outer_x)-fit_offset:0.5:max(right_outer_x)+fit_offset],right_outer_line_x.*MM_PER_PIXEL,'LineWidth',2,'Color',marker_outer_color,'LineStyle','--');
hold off
ylim(ylim_range);

title('\deltaC Outer x','FontName','Times New Roman','FontSize',13);
a = get(gca,'XTickLabel');  
set(gca,'XTickLabel',a,'fontsize',10)
set(gca,'XTickLabelMode','auto')

%Showing best fit
eq=sprintf('y=%.2f x + %.2f \nR^2=%.2f',b_outer_x(2),b_outer_x(1),R2_right_outer_x);
text(-24,5,eq,'FontSize',8);

%outer y
nexttile

plot(right_outer_y,del_POG_x_right.*MM_PER_PIXEL,'color',marker_inner_color,'Marker',marker_inner_type,'LineWidth',1,'MarkerSize',marker_size,'LineStyle',"none");
hold on
plot([min(right_outer_y)-fit_offset:0.5:max(right_outer_y)+fit_offset],right_outer_line_y.*MM_PER_PIXEL,'LineWidth',2,'Color',marker_outer_color,'LineStyle','--');
hold off
ylim(ylim_range);

title('\deltaC Outer y','FontName','Times New Roman','FontSize',13);
a = get(gca,'XTickLabel');  
set(gca,'XTickLabel',a,'fontsize',10)
set(gca,'XTickLabelMode','auto');

%Showing best fit
eq=sprintf('y=%.2f x + %.2f \nR^2=%.2f',b_outer_y(2),b_outer_y(1),R2_right_outer_y);
text(-40,5,eq,'FontSize',8);


%alpha
%{
nexttile

plot(right_alpha,del_POG_x_right,'color',marker_inner_color,'Marker',marker_inner_type,'LineWidth',1,'MarkerSize',marker_size,'LineStyle',"none");
hold on
plot([min(right_alpha)-fit_offset:0.5:max(right_alpha)+fit_offset],right_alpha_line,'LineWidth',2,'Color',marker_outer_color,'LineStyle','--');
hold off
ylim(ylim_range);

title('Alpha','FontName','Times New Roman','FontSize',13);
a = get(gca,'XTickLabel');  
set(gca,'XTickLabel',a,'fontsize',10)
set(gca,'XTickLabelMode','auto')
%}

leg=legend('Error Data','Regression Line');
title(t,['POG Error vs Right Eye Corners Shifts for ',part_num],'FontName','Times New Roman','FontSize',17,'FontWeight','bold');


% Left label
ylabel(t,'\deltaPOG x (mm)','FontName','Times New Roman','FontSize',15,'Color','k')

xlabel(t,'(mm)','FontName','Times New Roman','FontSize',15)



%#################<Displaying Corner-Head Correlation>################### 
%--------------<Displaying results for "right eye" inner_x>---------------
marker_size=5;
marker_inner_type='+';
marker_outer_type='x';
marker_inner_color='#4A7BB7';
marker_outer_color='#A50026';
ylim_range=[-55,70];
fig=figure;
t=tiledlayout(3,2,'TileSpacing','compact');
%x-translation
nexttile
yyaxis left
plot(translation(:,2),right_inner_x,'color',marker_inner_color,'Marker',marker_inner_type,'LineWidth',1,'MarkerSize',marker_size,'LineStyle',"none");
ylim(ylim_range);
yyaxis right
plot(translation(:,2),right_outer_x,'color',marker_outer_color,'Marker',marker_outer_type,'LineWidth',1,'MarkerSize',marker_size,'LineStyle',"none");
ylim(ylim_range);
title('translation (x)','FontName','Times New Roman','FontSize',13);
a = get(gca,'XTickLabel');  
set(gca,'XTickLabel',a,'fontsize',10)
set(gca,'XTickLabelMode','auto')


%roll
nexttile
yyaxis left
plot(roll,right_inner_x,'color',marker_inner_color,'Marker',marker_inner_type,'LineWidth',1,'MarkerSize',marker_size,'LineStyle',"none");
ylim(ylim_range);
yyaxis right
plot(roll,right_outer_x,'color',marker_outer_color,'Marker',marker_outer_type,'LineWidth',1,'MarkerSize',marker_size,'LineStyle',"none");
ylim(ylim_range);
title('roll','FontName','Times New Roman','FontSize',13);
a = get(gca,'XTickLabel');  
set(gca,'XTickLabel',a,'fontsize',10)
set(gca,'XTickLabelMode','auto')

%y-translation
nexttile

yyaxis left
plot(translation(:,3),right_inner_x,'color',marker_inner_color,'Marker',marker_inner_type,'LineWidth',1,'MarkerSize',marker_size,'LineStyle',"none");
ylim(ylim_range);
yyaxis right

plot(translation(:,3),right_outer_x,'color',marker_outer_color,'Marker',marker_outer_type,'LineWidth',1,'MarkerSize',marker_size,'LineStyle',"none");
ylim(ylim_range);
title('translation (y)','FontName','Times New Roman','FontSize',13);
a = get(gca,'XTickLabel');  
set(gca,'XTickLabel',a,'fontsize',10)
set(gca,'XTickLabelMode','auto')


%pitch
nexttile
yyaxis left
plot(pitch,right_inner_x,'color',marker_inner_color,'Marker',marker_inner_type,'LineWidth',1,'MarkerSize',marker_size,'LineStyle',"none");
ylim(ylim_range);
yyaxis right
plot(pitch,right_outer_x,'color',marker_outer_color,'Marker',marker_outer_type,'LineWidth',1,'MarkerSize',marker_size,'LineStyle',"none");
ylim(ylim_range);
title('pitch','FontName','Times New Roman','FontSize',13);
a = get(gca,'XTickLabel');  
set(gca,'XTickLabel',a,'fontsize',10)
set(gca,'XTickLabelMode','auto')



%z-translation
nexttile
yyaxis left
plot(translation(:,1),right_inner_x,'color',marker_inner_color,'Marker',marker_inner_type,'LineWidth',1,'MarkerSize',marker_size,'LineStyle',"none");
ylim(ylim_range);
title('translation (z)');
yyaxis right
plot(translation(:,1),right_outer_x,'color',marker_outer_color,'Marker',marker_outer_type,'LineWidth',1,'MarkerSize',marker_size,'LineStyle',"none");
ylim(ylim_range);
title('translation (z)','FontName','Times New Roman','FontSize',13);
xlabel('translation (mm)')
a = get(gca,'XTickLabel');  
set(gca,'XTickLabel',a,'fontsize',10)
set(gca,'XTickLabelMode','auto')

%yaw
nexttile
yyaxis left
plot(yaw,-right_inner_x,'color',marker_inner_color,'Marker',marker_inner_type,'LineWidth',1,'MarkerSize',marker_size,'LineStyle',"none");
ylim(ylim_range);
yyaxis right
plot(yaw,-right_outer_x,'color',marker_outer_color,'Marker',marker_outer_type,'LineWidth',1,'MarkerSize',marker_size,'LineStyle',"none");
ylim(ylim_range);
title('yaw','FontName','Times New Roman','FontSize',13);
xlabel('angle (degrees)')

a = get(gca,'XTickLabel');  
set(gca,'XTickLabel',a,'fontsize',10)
set(gca,'XTickLabelMode','auto')


leg=legend('Inner x','Outer x','FontName','Times New Roman','FontSize',10);
leg.Layout.Tile='south';
title(t,['Right Eye Corner vs Head Pose for ',part_num],'FontName','Times New Roman','FontSize',17,'FontWeight','bold');

ax = axes(fig);
han = gca;
han.Visible = 'off';

% Left label
yyaxis(ax, 'left');
ylabel('\deltaC inner x (px)','FontName','Times New Roman','FontSize',15)
han.YLabel.Visible = 'on';
% Right label
yyaxis(ax, 'right');
ylabel('\deltaC outer x (px)','FontName','Times New Roman','FontSize',15)
han.YLabel.Visible = 'on';


%--------------<Displaying results for "right eye" inner_y>---------------
%{
figure;
t=tiledlayout(3,2);
%x-translation
nexttile
plot(translation(:,1),right_inner_y,'ro','LineWidth',1);
title('translation (x)');

%roll
nexttile
plot(roll,right_inner_y,'ro','LineWidth',1);
title('roll');

%y-translation
nexttile
plot(translation(:,3),right_inner_y,'ro','LineWidth',1);
title('translation (y)');

%pitch
nexttile
plot(pitch,right_inner_y,'ro','LineWidth',1);
title('pitch');

%z-translation
nexttile
plot(translation(:,2),right_inner_y,'ro','LineWidth',1);
title('translation (z)');

%yaw
nexttile
plot(yaw,right_inner_y,'ro','LineWidth',1);
title('yaw');





title(t,['Right Eye Inner Corner Y vs Head Pose for ',part_num]);

%}


%%########################<Displaying Alpha-Head Coorelation>################

%{
figure;
t=tiledlayout(3,2);
%x-translation
nexttile
plot(translation(:,1),right_alpha,'ro','LineWidth',1);
title('translation (x)');

%roll
nexttile
plot(roll,right_alpha,'ro','LineWidth',1);
title('roll');

%y-translation
nexttile
plot(translation(:,3),right_alpha,'ro','LineWidth',1);
title('translation (y)');


%pitch
nexttile
plot(pitch,right_alpha,'ro','LineWidth',1);
title('pitch');



%z-translation
nexttile
plot(translation(:,2),right_alpha,'ro','LineWidth',1);
title('translation (z)');

%yaw
nexttile
plot(yaw,right_alpha,'ro','LineWidth',1);
title('yaw');


title(t,['Right Eye Alpha vs Head Pose for ',part_num]);

%}
%{

figure;
t=tiledlayout(3,2);
%x-translation
nexttile
plot(translation(:,1),left_alpha,'ro','LineWidth',1);
title('translation (x)');

%roll
nexttile
plot(roll,left_alpha,'ro','LineWidth',1);
title('roll');

%y-translation
nexttile
plot(translation(:,3),left_alpha,'ro','LineWidth',1);
title('translation (y)');

%pitch
nexttile
plot(pitch,left_alpha,'ro','LineWidth',1);
title('pitch');

%z-translation
nexttile
plot(translation(:,2),left_alpha,'ro','LineWidth',1);
title('translation (z)');

%yaw
nexttile
plot(yaw,left_alpha,'ro','LineWidth',1);
title('yaw');





title(t,['Left Eye Alpha vs Head Pose for ',part_num]);
%}

%% Plotting Some Results
%{
    total_results: matrix with columns:
    frame_no, accuracy_right_poly, accuracy_left_poly, accuracy_combined_poly,accuracy_right_tree, accuracy_left_tree, accuracy_combined_tree,
    actual_del_pog_right_x, actual_del_pog_right_y, actual_del_pog_left_x,
    actual_del_pog_left_y, 
    estimated_del_pog_right_x, estimated_del_pog_right_y,estimated_del_pog_left_x,
    estimated_del_pog_left_y,
    del_right_inner_x, del_right_inner_y, del_right_outer_x, del_right_outer_y
    del_left_inner_x, del_left_inner_y, del_left_outer_x, del_left_outer_y
    alpha_right, alpha_left
    t_x, t_y

    compensationData is a cell array with two cells having:
    cell 1: del_POG_x_right,del_POG_y_right,del_corner_inner_x_right,del_corner_inner_y_right,
    del_corner_outer_x_right,del_corner_outer_y_right,alpha_right,t_x,t_y

    cell 2: del_POG_x_left,del_POG_y_left,del_corner_inner_x_left,del_corner_inner_y_left,
    del_corner_outer_x_left,del_corner_outer_y_left,alpha_left,t_x,t_y
%}

%Plot del_pog_right_x vs. del_right_inner_x
comp_data_right=compensation_data{1};

%{
figure;
%plot3(total_results(:,8),total_results(:,16),total_results(:,17),'bo');
plot3(comp_data_right(:,1),comp_data_right(:,5),comp_data_right(:,6),'bo');
title('del-pog-right-x vs. del-right-inner-x')
xlabel('del POG')
ylabel('del corner inner x')
zlabel('del corner inner y')
%}
%{
figure;
plot(comp_data_right(:,1),comp_data_right(:,7),'bo');
title('del-pog-right-x vs. del-right-inner-x')
xlabel('del POG')
ylabel('del corner outer')
%}



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
    %Change the corner locations to be in 640x480 not the 1280x480 so
    %corner_data(:,1)=corner_data(:,1);
    %corner_data(:,3)=corner_data(:,3);
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
            %[rmse_x,rmse_y,b_x,b_y]=customRegressor(train_data);
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
    Input: data_cell has up to four cells where each cell is a matrix
    containing the data to train the compensation model.
    Each cell has: Calib_Up,Calib_Down,Calib_Right,Calib_Left

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

function [tree_mdl_x,tree_mdl_y,input_var_names_x,input_var_names_y]=fitTreeModel(train_data)
    
    %We return the model for compensation in the x-direction as well as the
    %model for compensation in the y-direction: tree_mdl_x & tree_mdl_y
    %To train mdl_x we use: del_pog_x (output), del_inner_x,del_outer_x,alpha
    %To train mdl_y we use: del_pog_y (output), del_inner_y,del_outer_y,alpha
    %We use surrogate splits to deal with missing data
       
    %--------------------------<First find predictors>------------------------
    input_var_names_x=cell(0);
    predictors_x=[];

    input_var_names_y=cell(0);
    predictors_y=[];
    %cell 1: del_POG_x_right,del_POG_y_right,del_corner_inner_x_right,del_corner_inner_y_right,
    %del_corner_outer_x_right,del_corner_outer_y_right,alpha_right,t_x,t_y
    

    if ~all(isnan(train_data(:,3))) %inner_x
            predictors_x=[predictors_x,train_data(:,3)];
            input_var_names_x=[input_var_names_x,'d_corner_inner_x'];

            predictors_y=[predictors_y,train_data(:,3)];
            input_var_names_y=[input_var_names_y,'d_corner_inner_x'];

    end

    if ~all(isnan(train_data(:,4))) %inner y
        predictors_y=[predictors_y,train_data(:,4)];
        input_var_names_y=[input_var_names_y,'d_corner_inner_y'];

        predictors_x=[predictors_x,train_data(:,4)];
        input_var_names_x=[input_var_names_x,'d_corner_inner_y'];

    end

    if ~all(isnan(train_data(:,5))) %Outer x
        predictors_x=[predictors_x,train_data(:,5)];
        input_var_names_x=[input_var_names_x,'d_corner_outer_x'];

        predictors_y=[predictors_y,train_data(:,5)];
        input_var_names_y=[input_var_names_y,'d_corner_outer_x'];

    end

    if ~all(isnan(train_data(:,6))) %outer y
        predictors_y=[predictors_y,train_data(:,6)];
        input_var_names_y=[input_var_names_y,'d_corner_outer_y'];

        predictors_x=[predictors_x,train_data(:,6)];
        input_var_names_x=[input_var_names_x,'d_corner_outer_y'];

    end
    
    %{
    if ~all(isnan(train_data(:,7)))
        predictors_x=[predictors_x,train_data(:,7)];
        input_var_names_x=[input_var_names_x,'alpha'];

        predictors_y=[predictors_y,train_data(:,7)];
        input_var_names_y=[input_var_names_y,'alpha'];

    end
    %}
    
    %---------------------<Check Outcomes and Train>-------------------

    %Train x-model
    if all(isnan(train_data(:,1))) %Check del_pog_x is valid
        tree_mdl_x=nan;
        input_var_names_x=nan;
    else
        if ~isempty(input_var_names_x)
            rng default
            tree_mdl_x=fitrtree(predictors_x,train_data(:,1),'Surrogate','on','OptimizeHyperparameters','auto',...
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
            tree_mdl_x=nan;
            input_var_names_x=nan;
        end

    end


    %Train y-model
    if all(isnan(train_data(:,2))) %Check del_pog_y is valid
        tree_mdl_y=nan;
        input_var_names_y=nan;
    else
        if ~isempty(input_var_names_y)
            rng default
            tree_mdl_y=fitrtree(predictors_y,train_data(:,2),'Surrogate','on','OptimizeHyperparameters','auto',...
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
            tree_mdl_y=nan;
            input_var_names_y=nan;
        end

    end

end

function [mdl_x,mdl_y,input_var_names_x,input_var_names_y]=fitSVRModel(train_data)
    
    %We return the model for compensation in the x-direction as well as the
    %model for compensation in the y-direction: tree_mdl_x & tree_mdl_y
    %To train mdl_x we use: del_pog_x (output), del_inner_x,del_outer_x,alpha
    %To train mdl_y we use: del_pog_y (output), del_inner_y,del_outer_y,alpha
    %We use surrogate splits to deal with missing data


    
    %Update to compensation data that we remove columns with too many nans
    %(more than 10%) then we get rid of respective rows with nans

    %{
        cell 2: del_POG_x_left,del_POG_y_left,del_corner_inner_x_left,del_corner_inner_y_left,
    del_corner_outer_x_left,del_corner_outer_y_left,alpha_left,t_x,t_y
    %}
    NANPERCENTAGE=0.4;
    input_var_names={'d_corner_inner_x','d_corner_inner_y','d_corner_outer_x','d_corner_outer_y','alpha'};
    predictors_only=train_data(:,3:7);
    new_predictors=[];
    new_input_var_names=cell(0);
    for i=[1:5]
        curr_predictor=predictors_only(:,i);
        num_pred=length(curr_predictor);
        num_nan=sum(isnan(curr_predictor));
        if (num_nan/num_pred)<NANPERCENTAGE
            new_predictors=[new_predictors,curr_predictor];
            new_input_var_names=[new_input_var_names,input_var_names{i}];
        end
    end

    if size(new_predictors,2)>0
        new_train=[train_data(:,[1,2]),new_predictors];
        [rows,cols]=size(new_train);

        %Training x-model
        if sum(isnan(new_train(:,1)))/length(new_train(:,1))<NANPERCENTAGE %x-outputs are valid
            input_var_names_x=new_input_var_names;
            
            %Remove rows with nan values 
            train_x=[];

            for i=[1:rows]
                no_nan=sum(isnan(new_train(i,:)));
                if no_nan<=0
                    train_x=[train_x;new_train(i,:)];
                end

            end

            rng default
            mdl_x=fitrsvm(train_x(:,[3:7]),train_x(:,1),'OptimizeHyperparameters','auto',...
    'HyperparameterOptimizationOptions',struct('AcquisitionFunctionName',...
    'expected-improvement-plus','MaxTime',1200));
    
        else
            mdl_x=nan;
            input_var_names_x=nan;
        end

        
        %Training y-model
        if sum(isnan(new_train(:,2)))/length(new_train(:,2))<NANPERCENTAGE %y-outputs are valid
            input_var_names_y=new_input_var_names;
            %Remove rows with nan values 
            train_y=[];
            for i=[1:rows]
                no_nan=sum(isnan(new_train(i,:)));
                if no_nan<=0
                    train_y=[train_y;new_train(i,:)];
                end

            end

            %Training the support vector regression
            rng default
            mdl_y=fitrsvm(train_x(:,[3:7]),train_x(:,2),'OptimizeHyperparameters','auto',...
    'HyperparameterOptimizationOptions',struct('AcquisitionFunctionName',...
    'expected-improvement-plus','MaxTime',1200));
        else
            mdl_y=nan;
            input_var_names_y=nan;
        end

    else
            mdl_x=nan;
            input_var_names_x=nan;
            mdl_y=nan;
            input_var_names_y=nan;
    end

end

function [predictors_x,predictors_y]=compensationPolynomial(predictors)
%Predictors are in the format: del_corner_inner_x,del_corner_inner_y, del_corner_outer_x,del_corner_outer_y,alpha
%predictors_x=[predictors(:,1),predictors(:,3),predictors(:,2),predictors(:,4),predictors(:,1).*predictors(:,2),predictors(:,3).*predictors(:,4)];
%predictors_y=[predictors(:,1),predictors(:,3),predictors(:,2),predictors(:,4),predictors(:,1).*predictors(:,2),predictors(:,3).*predictors(:,4)];
%predictors_x=[predictors(:,1).*predictors(:,2),predictors(:,3).*predictors(:,4)];
%predictors_y=[predictors(:,1).*predictors(:,2),predictors(:,3).*predictors(:,4)];
predictors_x=[predictors(:,1)];
predictors_y=[predictors(:,2)];

end

function [mdl_x,mdl_y]=fitCompensationRegressor(del_POG_x,del_POG_y,predictors)
%Predictors are in the format: del_corner_inner_x,del_corner_inner_y, del_corner_outer_x,del_corner_outer_y,alpha
REGRESS_TUNE=4.2;
REGRESS_FUNC='huber';

%--------------Iteratively Weighted Least Squares---------------
[predictors_x,predictors_y]=compensationPolynomial(predictors);



%Fitting POGx
%Using iteratively weighted least squares
[mdl_x,~]=robustfit(predictors_x,del_POG_x,REGRESS_FUNC,REGRESS_TUNE); %Uses iteratively weighted least squares

%Fitting POGy
[mdl_y,~]=robustfit(predictors_y,del_POG_y,REGRESS_FUNC,REGRESS_TUNE); %Uses iteratively weighted least squares


end

function del_POG=findCompensation(model,predictors)
%Generalized function to find the POG at run time
del_POG=model(1)+sum(model(2:end)'.*predictors,2,'omitnan');


end
%-------------------------<Evaluation Functions>------------------------
function [mean_accuracies,total_results]=evalModels(data_mat,model_cell,dist_cell,avg_corners,tree_models)
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
        frame_no, accuracy_right_poly, accuracy_left_poly, accuracy_combined_poly,accuracy_right_tree, accuracy_left_tree, accuracy_combined_tree,
        actual_del_pog_right_x, actual_del_pog_right_y, actual_del_pog_left_x,
        actual_del_pog_left_y, 
        estimated_del_pog_right_x, estimated_del_pog_right_y,estimated_del_pog_left_x,
        estimated_del_pog_left_y,
        right_inner_x, right_inner_y, right_outer_x, right_outer_y
        left_inner_x, left_inner_y, left_outer_x, left_outer_y
        alpha_right, alpha_left
        t_x, t_y        

    %}
    
    left_headers={'pg0_left_x','pg0_left_y','pg1_left_x','pg1_left_y','pg2_left_x','pg2_left_y'};
    right_headers={'pg0_right_x','pg0_right_y','pg1_right_x','pg1_right_y','pg2_right_x','pg2_right_y'};
    check_model_right=any(ismember(model_cell(:,1),right_headers));
    check_model_left=any(ismember(model_cell(:,1),left_headers));

    %Outputs the data as: 
    reformatted_data=reformatData(data_mat);

    total_results=evalAccuracy(model_cell,reformatted_data,right_headers,left_headers,check_model_right,check_model_left,dist_cell,avg_corners,tree_models);

    mean_accuracies=mean(total_results(:,[2:7]),1,'omitnan');
    



end

function total_results=evalAccuracy(model_cell,reformatted_data,right_headers,left_headers,check_model_right,check_model_left,dist_cell,avg_corners,tree_models)
   %{
    Inputs:
    reformatted_data:
    frame_no, pg0_rightx, pg0_righty, ..., pg2_rightx, pg2_righty, pg0_leftx, pg0_lefty,..., pg2_leftx, pg2_lefty,
    right_inner_x,right_inner_y,right_outer_x,right_outer_y,left_inner_x,left_inner_y,left_outer_x,left_outer_y,
    target_x,target_y

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
    t_x, t_y
   %}
    NANTHRESH=0; %Number of nan values we tolerate as input to our tree model
    [row_n,~]=size(reformatted_data);
    total_results=[];
    for i=[1:row_n]
        results_row=nan(1,27);
        curr_row=reformatted_data(i,:); %Current data row

        t_x=curr_row(end-1);    %Targets
        t_y=curr_row(end);

        results_row(end-1)=t_x;
        results_row(end)=t_y;
        results_row(1)=curr_row(1); %Frame number


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
    
                    [d_calib,d_curr]=findScalingFactors(dist_cell,valid_header,right_headers,right_pgs);
                    
                    if ~isnan(d_calib) && ~isnan(d_curr)
                        
                    
                        pg_x=(d_calib/d_curr).*right_pgs(pg_x_ind);
                        pg_y=(d_calib/d_curr).*right_pgs(pg_y_ind);
        
                        [predictors_x,predictors_y]=customPolynomial(pg_x,pg_y);
        
                        POG_x_poly_right=findPOG(model_x,predictors_x);
                        POG_y_poly_right=findPOG(model_y,predictors_y);
                        
                        del_POG_x_actual=t_x-POG_x_poly_right;
                        del_POG_y_actual=t_y-POG_y_poly_right;
    
                        accuracy_poly=sqrt((POG_x_poly_right-t_x)^2+(POG_y_poly_right-t_y)^2);
                        results_row(2)=accuracy_poly;
                        results_row(8)=del_POG_x_actual;
                        results_row(9)=del_POG_y_actual;
    
                        %Compensating
                        ind_x=ismember(tree_models(:,1),'right_x');
                        ind_y=ismember(tree_models(:,1),'right_y');
                        if any(ind_x) && any(ind_y) && iscell(tree_models{1,3}) && iscell(tree_models{2,3}) %We have a model for compensation
                            %Compensating in x-direction
                            input_vars_x=tree_models{ind_x,3};
                            tree_model_x=tree_models{ind_x,2};
                            predictors=[del_corner_inner_x,del_corner_inner_y,del_corner_outer_x,del_corner_outer_y,alpha];
                            accuracy_get=0;  
                            if sum(isnan(predictors))<=NANTHRESH %If we have only NANTHRESH nans in our predictor variable we continue
                                accuracy_get=accuracy_get+1;  
                                check_title={'d_corner_inner_x','d_corner_inner_y','d_corner_outer_x','d_corner_outer_y','alpha'};
                                [~,check_inds]=ismember(input_vars_x,check_title);
                                predictors=predictors(check_inds);
                                del_POG_x_tree=predict(tree_model_x,predictors);
                                results_row(12)=del_POG_x_tree;
                                POG_x_tree_right=del_POG_x_tree+POG_x_poly_right;
                            end

    
                            
                            input_vars_y=tree_models{ind_y,3};
                            tree_model_y=tree_models{ind_y,2};
                            predictors=[del_corner_inner_x,del_corner_inner_y,del_corner_outer_x,del_corner_outer_y,alpha];
                            if sum(isnan(predictors))<=NANTHRESH %If we have only NANTHRESH nans in our predictor variable we continue

                                accuracy_get=accuracy_get+1;  
                                check_title={'d_corner_inner_x','d_corner_inner_y','d_corner_outer_x','d_corner_outer_y','alpha'};
                                [~,check_inds]=ismember(input_vars_y,check_title);
                                predictors=predictors(check_inds);
                                del_POG_y_tree=predict(tree_model_y,predictors);
                                results_row(13)=del_POG_y_tree;
                                POG_y_tree_right=del_POG_y_tree+POG_y_poly_right;

                            end
                            if accuracy_get>=2
                                accuracy_tree=sqrt((POG_x_tree_right-t_x)^2+(POG_y_tree_right-t_y)^2);
                                results_row(5)=accuracy_tree;
                            end


    
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
    
                    [d_calib,d_curr]=findScalingFactors(dist_cell,valid_header,left_headers,left_pgs);
                    
                    if ~isnan(d_calib) && ~isnan(d_curr)
                        pg_x=(d_calib/d_curr).*left_pgs(pg_x_ind);
                        pg_y=(d_calib/d_curr).*left_pgs(pg_y_ind);
        
                        [predictors_x,predictors_y]=customPolynomial(pg_x,pg_y);
        
                        POG_x_poly_left=findPOG(model_x,predictors_x);
                        POG_y_poly_left=findPOG(model_y,predictors_y);
                        
                        del_POG_x_actual=t_x-POG_x_poly_left;
                        del_POG_y_actual=t_y-POG_y_poly_left;
    
                        accuracy_poly=sqrt((POG_x_poly_left-t_x)^2+(POG_y_poly_left-t_y)^2);
                        results_row(3)=accuracy_poly;
                        results_row(10)=del_POG_x_actual;
                        results_row(11)=del_POG_y_actual;
    
                        %Compensating
                        ind_x=ismember(tree_models(:,1),'left_x');
                        ind_y=ismember(tree_models(:,1),'left_y');
                        if any(ind_x) && any(ind_y) && iscell(tree_models{3,3}) && iscell(tree_models{4,3}) %We have a model for compensation
                            %Compensating in x-direction
                            input_vars_x=tree_models{ind_x,3};
                            tree_model_x=tree_models{ind_x,2};
                            predictors=[del_corner_inner_x,del_corner_inner_y,del_corner_outer_x,del_corner_outer_y,alpha];
                            accuracy_get=0;
                            if sum(isnan(predictors))<=NANTHRESH %If we have only NANTHRESH nans in our predictor variable we continue
                                accuracy_get=accuracy_get+1;
                                check_title={'d_corner_inner_x','d_corner_inner_y','d_corner_outer_x','d_corner_outer_y','alpha'};
                                [~,check_inds]=ismember(input_vars_x,check_title);
                                predictors=predictors(check_inds);
                                del_POG_x_tree=predict(tree_model_x,predictors);

                                results_row(14)=del_POG_x_tree;
                                POG_x_tree_left=del_POG_x_tree+POG_x_poly_left;
                            end
                            
                            %Compensating in y-direction
                            
                            input_vars_y=tree_models{ind_y,3};
                            tree_model_y=tree_models{ind_y,2};
                            predictors=[del_corner_inner_x,del_corner_inner_y,del_corner_outer_x,del_corner_outer_y]; %alpha];
                            if sum(isnan(predictors))<=NANTHRESH %If we have only NANTHRESH nans in our predictor variable we continue
                                accuracy_get=accuracy_get+1;
                                check_title={'d_corner_inner_x','d_corner_inner_y','d_corner_outer_x','d_corner_outer_y'}; %'alpha'};
                                [~,check_inds]=ismember(input_vars_y,check_title);
                                predictors=predictors(check_inds);
                                del_POG_y_tree=predict(tree_model_y,predictors);
                                results_row(15)=del_POG_y_tree;
                                POG_y_tree_left=del_POG_y_tree+POG_y_poly_left;
                            end
                            if accuracy_get>=2
                                accuracy_tree=sqrt((POG_x_tree_left-t_x)^2+(POG_y_tree_left-t_y)^2);
                            
                                results_row(6)=accuracy_tree;
                            end


    
                        end
                    end
                      
                end
        
            end

        end
        
        %-------------------<Getting Combined Results>-----------------
        if exist('POG_x_poly_right','var') && exist('POG_y_poly_right','var') && exist('POG_x_poly_left','var') && exist('POG_y_poly_left','var') 
            POG_combined_x=(POG_x_poly_right+POG_x_poly_left)/2;
            POG_combined_y=(POG_y_poly_right+POG_y_poly_left)/2;
            accuracy_combined=sqrt((POG_combined_x-t_x)^2+(POG_combined_y-t_y)^2);
            results_row(4)=accuracy_combined;

        end

        if exist('POG_x_tree_right','var') && exist('POG_y_tree_right','var') && exist('POG_x_tree_left','var') && exist('POG_y_tree_left','var') 
            POG_combined_x=(POG_x_tree_right+POG_x_tree_left)/2;
            POG_combined_y=(POG_y_tree_right+POG_y_tree_left)/2;
            accuracy_combined=sqrt((POG_combined_x-t_x)^2+(POG_combined_y-t_y)^2);
            results_row(7)=accuracy_combined;

        end


        total_results=[total_results;results_row];
    end

end


function reformatted_data=reformatData(eval_data)
    %Returns the data to evaluate the model in the format of: 
    % frame_no, pg0_rightx, pg0_righty, ..., pg2_rightx, pg2_righty, pg0_leftx, pg0_lefty,..., pg2_leftx, pg2_lefty,
    % right_inner_x,right_inner_y,right_outer_x,right_outer_y,left_inner_x,left_inner_y,left_outer_x,left_outer_y,
    % target_x,target_y


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
        eval_data(:,27),eval_data(:,28)];

end


function [mean_accuracies,total_results]=evalModelsRegressComp(data_mat,model_cell,dist_cell,avg_corners,comp_model_x_right,comp_model_y_right,comp_model_x_left,comp_model_y_left)
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
        frame_no, accuracy_right_poly, accuracy_left_poly, accuracy_combined_poly,accuracy_right_tree, accuracy_left_tree, accuracy_combined_tree,
        actual_del_pog_right_x, actual_del_pog_right_y, actual_del_pog_left_x,
        actual_del_pog_left_y, 
        estimated_del_pog_right_x, estimated_del_pog_right_y,estimated_del_pog_left_x,
        estimated_del_pog_left_y,
        right_inner_x, right_inner_y, right_outer_x, right_outer_y
        left_inner_x, left_inner_y, left_outer_x, left_outer_y
        alpha_right, alpha_left
        t_x, t_y        

    %}
    
    left_headers={'pg0_left_x','pg0_left_y','pg1_left_x','pg1_left_y','pg2_left_x','pg2_left_y'};
    right_headers={'pg0_right_x','pg0_right_y','pg1_right_x','pg1_right_y','pg2_right_x','pg2_right_y'};
    check_model_right=any(ismember(model_cell(:,1),right_headers));
    check_model_left=any(ismember(model_cell(:,1),left_headers));

    %Outputs the data as: 
    reformatted_data=reformatData(data_mat);

    total_results=evalAccuracyComp(model_cell,reformatted_data,right_headers,left_headers,check_model_right,check_model_left,dist_cell,avg_corners,comp_model_x_right,comp_model_y_right,comp_model_x_left,comp_model_y_left);

    mean_accuracies=mean(total_results(:,[2:7]),1,'omitnan');
    



end

function total_results=evalAccuracyComp(model_cell,reformatted_data,right_headers,left_headers,check_model_right,check_model_left,dist_cell,avg_corners,comp_model_x_right,comp_model_y_right,comp_model_x_left,comp_model_y_left)
   %{
    Inputs:
    reformatted_data:
    frame_no, pg0_rightx, pg0_righty, ..., pg2_rightx, pg2_righty, pg0_leftx, pg0_lefty,..., pg2_leftx, pg2_lefty,
    right_inner_x,right_inner_y,right_outer_x,right_outer_y,left_inner_x,left_inner_y,left_outer_x,left_outer_y,
    target_x,target_y

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
    t_x, t_y
   %}
    NANTHRESH=0; %Number of nan values we tolerate as input to our tree model
    [row_n,~]=size(reformatted_data);
    total_results=[];
    for i=[1:row_n]
        results_row=nan(1,27);
        curr_row=reformatted_data(i,:); %Current data row

        t_x=curr_row(end-1);    %Targets
        t_y=curr_row(end);

        results_row(end-1)=t_x;
        results_row(end)=t_y;
        results_row(1)=curr_row(1); %Frame number


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
    
                    [d_calib,d_curr]=findScalingFactors(dist_cell,valid_header,right_headers,right_pgs);
                    
                    if ~isnan(d_calib) && ~isnan(d_curr)
                        
                    
                        pg_x=(d_calib/d_curr).*right_pgs(pg_x_ind);
                        pg_y=(d_calib/d_curr).*right_pgs(pg_y_ind);
        
                        [predictors_x,predictors_y]=customPolynomial(pg_x,pg_y);
        
                        POG_x_poly_right=findPOG(model_x,predictors_x);
                        POG_y_poly_right=findPOG(model_y,predictors_y);
                        
                        del_POG_x_actual=t_x-POG_x_poly_right;
                        del_POG_y_actual=t_y-POG_y_poly_right;
    
                        accuracy_poly=sqrt((POG_x_poly_right-t_x)^2+(POG_y_poly_right-t_y)^2);
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
                                accuracy_tree=sqrt((t_x-POG_x_tree_right)^2+(t_y-POG_y_tree_right)^2);
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
    
                    [d_calib,d_curr]=findScalingFactors(dist_cell,valid_header,left_headers,left_pgs);
                    
                    if ~isnan(d_calib) && ~isnan(d_curr)
                        pg_x=(d_calib/d_curr).*left_pgs(pg_x_ind);
                        pg_y=(d_calib/d_curr).*left_pgs(pg_y_ind);
        
                        [predictors_x,predictors_y]=customPolynomial(pg_x,pg_y);
        
                        POG_x_poly_left=findPOG(model_x,predictors_x);
                        POG_y_poly_left=findPOG(model_y,predictors_y);
                        
                        del_POG_x_actual=t_x-POG_x_poly_left;
                        del_POG_y_actual=t_y-POG_y_poly_left;
    
                        accuracy_poly=sqrt((POG_x_poly_left-t_x)^2+(POG_y_poly_left-t_y)^2);
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
                                accuracy_tree=sqrt((t_x-POG_x_tree_left)^2+(t_y-POG_y_tree_left)^2);
                            
                                results_row(6)=accuracy_tree;
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
            results_row(4)=accuracy_combined;

        end

        if exist('POG_x_tree_right','var') && exist('POG_y_tree_right','var') && exist('POG_x_tree_left','var') && exist('POG_y_tree_left','var') 
            POG_combined_x=(POG_x_tree_right+POG_x_tree_left)/2;
            POG_combined_y=(POG_y_tree_right+POG_y_tree_left)/2;
            accuracy_combined=sqrt((t_x-POG_combined_x)^2+(t_y-POG_combined_y)^2);
            results_row(7)=accuracy_combined;

        end


        total_results=[total_results;results_row];
    end

end
