clear
clc
close all

%% Loading Data
normal_pol_dat=load('ModifiedPolynomialResults.mat');
modified_pol_dat=load('NormalPolynomialResults.mat');

accuracy_classic=normal_pol_dat.accuracy_classic_subjects;

accuracy_robust_norm_pol=normal_pol_dat.accuracy_robust_subjects;
accuracy_robust_mod_pol=modified_pol_dat.accuracy_robust_subjects;

%% Descriptive Stats
avg_classic=mean(accuracy_classic(:,[2:end]),1,"omitnan");
avg_robust_norm_pol=mean(accuracy_robust_norm_pol(:,[2:end]),1,"omitnan");
avg_robust_mod_pol=mean(accuracy_robust_mod_pol(:,[2:end]),1,"omitnan");

std_classic=std(accuracy_classic(:,[2:end]),1,"omitnan");
std_robust_norm_pol=std(accuracy_robust_norm_pol(:,[2:end]),1,"omitnan");
std_robust_mod_pol=std(accuracy_robust_mod_pol(:,[2:end]),1,"omitnan");