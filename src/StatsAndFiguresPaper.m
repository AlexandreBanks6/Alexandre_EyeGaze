clear
clc
close all

%% Loading in Accuracy Data
%Reading in the model-based data
path_modelbased='C:/Users/playf/OneDrive/Documents/UBC/Thesis/Paper_FiguresAndResults/Model_Based_Results';
mean_acc_modelbased=readmatrix([path_modelbased,'/mean_acc_results.csv']);
pertarget_acc_modelbased=readmatrix([path_modelbased,'/per_target_results.csv']);

%Reading in the interpolation-based data
path_modelbased='C:/Users/playf/OneDrive/Documents/UBC/Thesis/Paper_FiguresAndResults/Interpolation_IDW_WithMostPoints_RawResults';
mean_acc_interpolation=readmatrix([path_modelbased,'/mean_acc_results.csv']);
pertarget_acc_interpolation=readmatrix([path_modelbased,'/per_target_results.csv']);

%--------------<First we do things for the model-based paper (IPCAI)>------

%% Excluding Data
%{
for i=[2:10]
figure(i);
plot(mean_acc_modelbased(:,1),mean_acc_modelbased(:,i),'ob');
end
%}
%Excluding participants from the experiment:
%Excluding p13 from all
mean_acc_modelbased(13,:)=[];
p11_ind=mean_acc_modelbased(:,1)==11;
mean_acc_modelbased(p11_ind,:)=[];


p20_ind=mean_acc_modelbased(:,1)==20;
mean_acc_modelbased(p20_ind,[8:10])=nan(1,3);

p21_ind=mean_acc_modelbased(:,1)==21;
mean_acc_modelbased(p21_ind,[5:7])=nan(1,3);

p22_ind=mean_acc_modelbased(:,1)==22;
mean_acc_modelbased(p22_ind,[5:7])=nan(1,3);

p26_ind=mean_acc_modelbased(:,1)==26;
mean_acc_modelbased(p26_ind,[7])=nan;

mean_acc_modelbased(12,5)=nan;
mean_acc_modelbased(26,5)=nan;
%% Converting Data to degrees of visual angle
mean_acc_modelbased=atan(mean_acc_modelbased./457.2).*(180/pi);

%% Initial plotting
%Test bargraph
boxplot(mean_acc_modelbased(:,[2,5,8]));

%% Running anova and post-hoc analysis for model-based (IPCAI)
%Doing it for right eyes
grp_right={'poly_right','model_right','max_right'};
[p_right,tbl_right,stats_right]=anova1(mean_acc_modelbased(:,[2,5,8]));

%Doing it for left eyes
grp_right={'poly_left','model_left','max_left'};
[p_left,tbl_left,stats_left]=anova1(mean_acc_modelbased(:,[3,6,9]));

[posthoc_res_right,~,~,~]=multcompare(stats_right,'Alpha',0.05,'CriticalValueType','tukey-kramer');
posthoc_res_right = array2table(posthoc_res_right,"VariableNames", ...
    ["Group A","Group B","Lower Limit","A-B","Upper Limit","P-value"]);


[posthoc_res_left,~,~,~]=multcompare(stats_left,'Alpha',0.05,'CriticalValueType','tukey-kramer');
posthoc_res_left = array2table(posthoc_res_left,"VariableNames", ...
    ["Group A","Group B","Lower Limit","A-B","Upper Limit","P-value"]);
%% Running Descriptive Stats
acc_means=mean(mean_acc_modelbased(:,[2:end]),1,'omitnan');
acc_stds=std(mean_acc_modelbased(:,[2:end]),0,1,'omitnan');

%% Plotting Model BasedMean acc's with violin plot
grp_right={'Poly','Li','CCHC','Poly','Li','CCHC'};
customViolin(mean_acc_modelbased(:,[3,9,6,2,8,5]),grp_right,'POG Error Comparison for Estimation Algorithms');

%% Fixing per target results data
pertarget_acc_modelbased(13,:)=[];

p20_ind=pertarget_acc_modelbased(:,1)==20;
pertarget_acc_modelbased(p20_ind,:)=[];

p21_ind=pertarget_acc_modelbased(:,1)==21;
pertarget_acc_modelbased(p21_ind,:)=[];

p22_ind=pertarget_acc_modelbased(:,1)==22;
pertarget_acc_modelbased(p22_ind,:)=[];

p26_ind=pertarget_acc_modelbased(:,1)==26;
pertarget_acc_modelbased(p26_ind,:)=[];


%% Processing per-target results data
right_results=[];
left_results=[];
targets=[];

for i=[2:11:288]
    right_results=[right_results,pertarget_acc_modelbased(:,i+3)];
    left_results=[left_results,pertarget_acc_modelbased(:,i+4)];
    targets=[targets;mean(pertarget_acc_modelbased(:,[i+9,i+10]),1)];
end

%% Descriptive Stats on Right/Left target results

%Right target accuracies
right_results=[right_results(:,1:9);right_results(:,10:18);right_results(:,19:27)];
%Converting to degrees of visual angle
right_means=mean(right_results,1,'omitnan');
right_stds=std(right_results,1,'omitnan');

%Left targets accuracies
left_results=[left_results(:,1:9);left_results(:,10:18);left_results(:,19:27)];
%Converting to degrees of visual angle
left_results=atan(left_results./457.2).*(180/pi);
left_means=mean(left_results,1,'omitnan');
left_stds=std(left_results,1,'omitnan');

%targets
targets=targets(1:9,:);
%Converting targets to mm
targets([1,4,7],1)=targets([1,4,7],1)-10;
targets([3,6,9],1)=targets([3,6,9],1)+10;

targets([1,2,3],2)=targets([1,2,3],2)-10;
targets([7,8,9],2)=targets([7,8,9],2)+10;

targets(:,1)=(targets(:,1)./100)*284.48;
targets(:,2)=(targets(:,2)./100)*213.36;


%% Plotting acc per target for right eye
%We plot each center as the target location, then a solid circle around
%this as the mean accuracy, and a dashed filled circle around this as the
%standard deviation
mean_radius=right_means./2;
outer_radius=right_means./2+right_stds./4;
inner_radius=right_means./2-right_stds./4;
figure;
hold on
for i=[1:9]
    patchcolor='b'; %Doesn't influence anything, but needed
    
    dash_type='-';
    facecolor='none';
    edgecolor="#0072BD";
    facealpha=0;
    linewidth=1;
    circle(targets(i,1),targets(i,2),mean_radius(i),patchcolor,facecolor,edgecolor,facealpha,linewidth,dash_type); %Mean line

    dash_type='-';
    facecolor='#98cce5';
    edgecolor=facecolor;
    facealpha=0.5;
    linewidth=1;
    circle(targets(i,1),targets(i,2),outer_radius(i),patchcolor,facecolor,edgecolor,facealpha,linewidth,dash_type); %Outer radius
    
    facecolor='w';
    edgecolor='#98cce5';
    facealpha=1;
    linewidth=1;
    
    circle(targets(i,1),targets(i,2),inner_radius(i),patchcolor,facecolor,edgecolor,facealpha,linewidth,dash_type); %Inner radius

    
end
hold off
axis equal
%viscircles(targets,right_means./2+right_stds./4,'Color','r','LineWidth',1); %Outer radius
%viscircles(targets,right_means./2-right_stds./4,'Color','r','LineWidth',1,'ObjectPolarity','r'); %Inner radius
%viscircles(targets,right_means./2,'Color','k','LineWidth',1); %Mean line
%xaxis([0,284.48]);
%yaxis([0,213.36]);



%---------------------------<Helper Function Definitions>-----------------
function circles = circle(x,y,r,patchcolor,facecolor,edgecolor,facealpha,linewidth,line_type)

th = 0:pi/50:2*pi;
x_circle = r * cos(th) + x;
y_circle = r * sin(th) + y;
circles = plot(x_circle, y_circle);
fill(x_circle, y_circle, patchcolor,'FaceColor',facecolor,'EdgeColor',edgecolor,'FaceAlpha',facealpha,'LineWidth',linewidth,'LineStyle',line_type);

end