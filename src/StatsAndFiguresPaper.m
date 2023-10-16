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
for i=[2:10]
figure(i);
plot(mean_acc_modelbased(:,1),mean_acc_modelbased(:,i),'ob');
end

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


%% Plotting Model BasedMean acc's with violin plot
grp_right={'Poly_{r}','Li_{r}','CCHC_{r}'};
%vs_right=violinplot(mean_acc_modelbased(:,[2,8,5]),grp_right,'ShowData',false,'Bandwidth',range(mean_acc_modelbased(:,2))*0.1,'Width',0.3);
%data=num2cell(mean_acc_modelbased(:,[2,8,5]));
%customViolin(mean_acc_modelbased(:,2),0.7,2,'k',parula(numel(data)))
%KernelBoxPlot(mean_acc_modelbased(:,5),'Poly_r');
customViolin(mean_acc_modelbased(:,[2,8,5]),0.5,grp_right,'violin plot');