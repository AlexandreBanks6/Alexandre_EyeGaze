clear
clc
close all

%%Init Parameters and Reading Data
data_root='E:/Alexandre_EyeGazeProject_Extra/eyecorner_userstudy2_converted';
part_num='P01';

calib_init_data=readmatrix([data_root,'/',part_num,'/calib_only_merged_Calib_Init.csv']); %Initial calibration data


calib_onedot_data=readmatrix([data_root,'/',part_num,'/calib_only_merged_Calib_Comp_Rotate.csv']); %One Dot Rotation Calibration data

calib_comp_lifts_data=[]; %Lift and replace calibration data
lift_names={'Lift1_8point.csv','Lift2_8point.csv','Lift3_8point.csv','Lift4_8point.csv','Lift5_8point.csv'};
lift_root='calib_only_merged_Calib_Comp_';

for i=[1:length(lift_names)]
    calib_comp_lifts_data=[calib_comp_lifts_data;readmatrix([data_root,'/',part_num,'/',lift_root,lift_names{i}])];
end

%%Computing delta corner and delta head
WIDTH_CONVERSION=284.48;
HEIGHT_CONVERSION=213.36;
%----------<Delta Corners>----------
%Corners are in format: right_inner_x,right_inner_y,
% right_outer_x,right_outer_y,left_inner_x,...

avg_corners=mean(calib_init_data(:,[50:57]),1,'omitnan');
corners_onedot=calib_onedot_data(:,[50:57]); 
corners_lifts=calib_comp_lifts_data(:,[50:57]);
delta_c_onedot=avg_corners-corners_onedot;  %delta_c=corner_calib-corner_movement
delta_c_onedot(:,[1,3,5,7])=(delta_c_onedot(:,[1,3,5,7])./100).*WIDTH_CONVERSION;
delta_c_onedot(:,[2,4,6,8])=(delta_c_onedot(:,[2,4,6,8])./100).*HEIGHT_CONVERSION;
delta_c_lifts=avg_corners-corners_lifts;
delta_c_lifts(:,[1,3,5,7])=(delta_c_lifts(:,[1,3,5,7])./100).*WIDTH_CONVERSION;
delta_c_lifts(:,[2,4,6,8])=(delta_c_lifts(:,[2,4,6,8])./100).*HEIGHT_CONVERSION;

%------------<Delta Head>------------
%Tool 1 is not moving, tool 2 is moving

%Getting average head position at calibration
q1_calib=calib_init_data(:,[35:38]);
q2_calib=calib_init_data(:,[45:48]);

t1_calib=calib_init_data(:,[32:34]);
t2_calib=calib_init_data(:,[42:44]);

[calib_quat_avg,calib_t12_avg]=getCalibPose(q1_calib,q2_calib,t1_calib,t2_calib);


%Getting head movement for dot and rotation

q1_dot=calib_onedot_data(:,[35:38]);
q2_dot=calib_onedot_data(:,[45:48]);

t1_dot=calib_onedot_data(:,[32:34]);
t2_dot=calib_onedot_data(:,[42:44]);

[dot_rotations,dot_translations]=getMovedPositions(q1_dot,q2_dot,t1_dot,t2_dot,calib_quat_avg,calib_t12_avg);

%Getting head movement for lifts


q1_lifts=calib_comp_lifts_data(:,[35:38]);
q2_lifts=calib_comp_lifts_data(:,[45:48]);

t1_lifts=calib_comp_lifts_data(:,[32:34]);
t2_lifts=calib_comp_lifts_data(:,[42:44]);

[lifts_rotations,lifts_translations]=getMovedPositions(q1_lifts,q2_lifts,t1_lifts,t2_lifts,calib_quat_avg,calib_t12_avg);


%%Plotting Coorelation with head rotations for all corners
%save_dir='C:/Users/playf/OneDrive/Documents/UBC/Thesis/Paper_FiguresAndResults/';
titles={'Right Eye Inner','Right Eye Outer','Left Eye Inner','Left Eye Outer'};
%save_titles={'InnerRight','OuterRight','InnerLeft','OuterLeft'};
%{
for i=[1:4]
    fig_handle_x=headCornerPlotter(dot_rotations,dot_translations,delta_c_onedot(:,i*2-1),[titles{i},' Corner Change vs Head Pose For ',part_num],'x');
    fig_handle_y=headCornerPlotter(dot_rotations,dot_translations,delta_c_onedot(:,i*2),[titles{i},' Corner Change vs Head Pose For ',part_num],'y');
    %fig_handle_x=headCornerPlotter(lifts_rotations,lifts_translations,delta_c_lifts(:,i*2-1),[titles{i},' Corner Change vs Head Pose For ',part_num],'x');
    %fig_handle_y=headCornerPlotter(lifts_rotations,lifts_translations,delta_c_lifts(:,i*2),[titles{i},' Corner Change vs Head Pose For ',part_num],'y');

    %saveas(fig_handle_x,[save_dir,'V2_CornerToHead_Rough_',save_titles{i},'x.png']);
    %saveas(fig_handle_y,[save_dir,'V2_CornerToHead_Rough_',save_titles{i},'y.png']);

end
%}

fig_handle_x=headCornerPlotter(dot_rotations,dot_translations,delta_c_onedot(:,5),[titles{3},' Corner Change vs Head Pose For ',part_num],'x');
fig_handle_y=headCornerPlotter(dot_rotations,dot_translations,delta_c_onedot(:,6),[titles{3},' Corner Change vs Head Pose For ',part_num],'y');



%% Plotting a single head-corner coorelation
save_dir='C:/Users/playf/OneDrive/Documents/UBC/Thesis/Paper_FiguresAndResults/';
marker_size=5;
marker_type='o';
marker_color='#4A7BB7';

%right outer corner x vs trans x
fig1=figure;
plot(dot_translations(:,1),delta_c_onedot(:,3),'color',marker_color,'Marker',marker_type,'LineWidth',1,'MarkerSize',marker_size,'LineStyle',"none");

title('Shift in Right Outer x Corner vs Translation (x)','FontName','Times New Roman','FontSize',13);
xlabel('translation x (mm)','FontName','Times New Roman','FontSize',15);
ylabel('\deltaCx (px)','FontName','Times New Roman','FontSize',15)


%right outer corner x vs pitch
fig2=figure;
plot(dot_rotations(:,3),delta_c_onedot(:,3),'color',marker_color,'Marker',marker_type,'LineWidth',1,'MarkerSize',marker_size,'LineStyle',"none");

title('Shift in Right Outer x Corner vs rotation (pitch)','FontName','Times New Roman','FontSize',13);
xlabel('pitch (degrees)','FontName','Times New Roman','FontSize',15);
ylabel('\deltaCx (px)','FontName','Times New Roman','FontSize',15);

%right outer corner y vs trans y
fig3=figure;
plot(dot_translations(:,2),delta_c_onedot(:,4),'color',marker_color,'Marker',marker_type,'LineWidth',1,'MarkerSize',marker_size,'LineStyle',"none");

title('Shift in Right Outer y Corner vs Translation (y)','FontName','Times New Roman','FontSize',13);
xlabel('translation y (mm)','FontName','Times New Roman','FontSize',15);
ylabel('\deltaCy (px)','FontName','Times New Roman','FontSize',15)


%right outer corner y vs roll
fig4=figure;
plot(dot_rotations(:,1),delta_c_onedot(:,4),'color',marker_color,'Marker',marker_type,'LineWidth',1,'MarkerSize',marker_size,'LineStyle',"none");

title('Shift in Right Outer y Corner vs rotation (roll)','FontName','Times New Roman','FontSize',13);
xlabel('roll (degrees)','FontName','Times New Roman','FontSize',15);
ylabel('\deltaCy (px)','FontName','Times New Roman','FontSize',15);

%Saving results
saveas(fig1,[save_dir,'RightOuterX_vs_TranslationX.png']);
saveas(fig2,[save_dir,'RightOuterX_vs_Pitch.png']);
saveas(fig3,[save_dir,'RightOuterY_vs_TranslationY.png']);
saveas(fig4,[save_dir,'RightOuterY_vs_Roll.png']);


%% Plotting Coorelation with lift and replace for all corners
headCornerPlotter(lifts_rotations,lifts_translations,delta_c_lifts(:,4),'Change in Inner Corner Position vs Head Pose','x')

%% Function Definitions

function [calib_quat_avg,calib_t12_avg]=getCalibPose(q1,q2,t1,t2)
    %Function that gets the average head rotation (quaterion) and translation at the
    %initial calibration
    % 1 and 2 means tool 1 and 2; q1=quaternion for tool 1
    q12=quatmultiply(quatconj(q1),q2);
    q12=quaternion(q12);
    calib_quat_avg=meanrot(q12,1);
    
    calib_t12=t2-t1;
    calib_t12_avg=mean(calib_t12,1,'omitnan');
end

function [moved_rotations,moved_translations]=getMovedPositions(q1,q2,t1,t2,calib_quat_avg,calib_t12_avg)
%Returns the rotation and translation from the average calibration position
%to the current position. Current position is given as arrays of the tool 1
%and tool 2 pose's
CONVERSION_FACTOR=(180/pi);

%Rotation from calibration to current
rotated_q12=quatmultiply(quatconj(q1),q2); %Finds rotation from q1 to q2
calib_q12_avg=compact(calib_quat_avg);

q_calib_to_moved=quatmultiply(quatconj(calib_q12_avg),rotated_q12); %Rotation from calibration position to current position

[yaw,pitch,roll]=quat2angle(q_calib_to_moved);
yaw=yaw.*CONVERSION_FACTOR;
pitch=pitch.*CONVERSION_FACTOR;
roll=roll.*CONVERSION_FACTOR;
moved_rotations=[yaw,pitch,roll];

%Translation part
moved_t12=t2-t1; 
moved_translations=moved_t12-calib_t12_avg;
moved_translations=moved_translations.*1000;

end


function fig=headCornerPlotter(moved_rotations,moved_translations,delta_corner,title_name,delta_c_direction)
%Title is what we want the overall title of the plot to be
%Plot the corner vs the change in pose for all 6 different poses
%moved_rotations=array with three columns corresponding to roll, pitch, and
%yaw
%moved_translations=array with three columns corresponding to translation
%in x,y, and z

marker_size=5;
marker_type='o';
marker_color='#4A7BB7';

fig=figure;
t=tiledlayout(3,2,'TileSpacing','compact');

%x-translation
nexttile
plot(moved_translations(:,1),delta_corner,'color',marker_color,'Marker',marker_type,'LineWidth',1,'MarkerSize',marker_size,'LineStyle',"none");
hold on
showBestFitLine(moved_translations(:,1),delta_corner,5);
hold off


title('translation (x)','FontName','Times New Roman','FontSize',13);
a = get(gca,'XTickLabel');  
set(gca,'XTickLabel',a,'fontsize',10)
set(gca,'XTickLabelMode','auto')


%roll
nexttile

plot(moved_rotations(:,1),delta_corner,'color',marker_color,'Marker',marker_type,'LineWidth',1,'MarkerSize',marker_size,'LineStyle',"none");
hold on
showBestFitLine(moved_rotations(:,1),delta_corner,0.025);
hold off


title('roll','FontName','Times New Roman','FontSize',13);
a = get(gca,'XTickLabel');  
set(gca,'XTickLabel',a,'fontsize',10)
set(gca,'XTickLabelMode','auto')

%y-translation
nexttile

plot(moved_translations(:,2),delta_corner,'color',marker_color,'Marker',marker_type,'LineWidth',1,'MarkerSize',marker_size,'LineStyle',"none");
hold on
showBestFitLine(moved_translations(:,2),delta_corner,5);
hold off


title('translation (y)','FontName','Times New Roman','FontSize',13);
a = get(gca,'XTickLabel');  
set(gca,'XTickLabel',a,'fontsize',10)
set(gca,'XTickLabelMode','auto')


%pitch
nexttile

plot(moved_rotations(:,3),delta_corner,'color',marker_color,'Marker',marker_type,'LineWidth',1,'MarkerSize',marker_size,'LineStyle',"none");
hold on
showBestFitLine(moved_rotations(:,3),delta_corner,0.025);
hold off


title('pitch','FontName','Times New Roman','FontSize',13);
a = get(gca,'XTickLabel');  
set(gca,'XTickLabel',a,'fontsize',10)
set(gca,'XTickLabelMode','auto')



%z-translation
nexttile

plot(moved_translations(:,3),delta_corner,'color',marker_color,'Marker',marker_type,'LineWidth',1,'MarkerSize',marker_size,'LineStyle',"none");
hold on
showBestFitLine(moved_translations(:,3),delta_corner,5);
hold off

title('translation (z)','FontName','Times New Roman','FontSize',13);
xlabel('translation (mm)','FontName','Times New Roman','FontSize',20,'FontWeight','bold')
a = get(gca,'XTickLabel');  
set(gca,'XTickLabel',a,'fontsize',10)
set(gca,'XTickLabelMode','auto')

%yaw
nexttile

plot(moved_rotations(:,2),delta_corner,'color',marker_color,'Marker',marker_type,'LineWidth',1,'MarkerSize',marker_size,'LineStyle',"none");
hold on
showBestFitLine(moved_rotations(:,2),delta_corner,0.025);
hold off

title('yaw','FontName','Times New Roman','FontSize',13);
xlabel('angle (degrees)','FontName','Times New Roman','FontSize',20,'FontWeight','bold')

a = get(gca,'XTickLabel');  
set(gca,'XTickLabel',a,'fontsize',10)
set(gca,'XTickLabelMode','auto')



title(t,title_name,'FontName','Times New Roman','FontSize',17,'FontWeight','bold');

ax = axes(fig);
han = gca;
han.Visible = 'off';

% Left label
yyaxis(ax, 'left');
ylabel(['\deltaC ',delta_c_direction,' (mm)'],'FontName','Times New Roman','FontSize',15,'Color','k','FontWeight','bold');
han.YLabel.Visible = 'on';

plottools %Opening in plot tools




end


function [y_line,x_line,R_squared,line_model]=showBestFitLine(x_data,y_data,fit_offset)
marker_lin_color='#A50026';
line_model=polyfit(x_data,y_data,2);
r_squared_y=line_model(1).*(x_data.^2)+line_model(2).*x_data+line_model(3);
R_squared=1-sum((y_data-r_squared_y).^2,'omitnan')./sum((y_data-mean(y_data,'omitnan')).^2,'omitnan');

x_line=[min(x_data)-fit_offset:0.001:max(x_data)+fit_offset];
y_line=line_model(1).*(x_line.^2)+line_model(2).*x_line+line_model(3);



eq=sprintf('y=%.2f x^{2} + %.2f x + %.2f \nR^2=%.2f',line_model(1),line_model(2),line_model(3),R_squared);
xl = xlim;
yl = ylim;
xt = xl(1);
yt = yl(1);
plot(x_line,y_line,'LineWidth',2,'Color',marker_lin_color,'LineStyle','--');
text(xt,yt,eq,'FontSize',12);

end