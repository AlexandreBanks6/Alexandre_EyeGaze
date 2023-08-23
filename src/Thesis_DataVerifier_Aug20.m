clear
clc
close all


%Setting Parameters
data_root='E:/Alexandre_EyeGazeProject/eyecorner_userstudy_converted';
extensions={'Calib_Init','Eval_Init','Calib_Right','Calib_Left','Calib_Up',...
    'Calib_Down','Eval_Straight','Eval_Right','Eval_Left','Eval_Up','Eval_Down'};
CALIB_THRESHOLD=5;
EVAL_THRESHOLD=3; %Threshold to be considered a valid evaluation trial
NUM_EVAL_THRESHOLD=3; %The number of valid eval trials for all eval trials to be good for participant
HEAD_COMP_NUMBER_THRESHOLD=10;  %Number of unique head position/target combinations to be good for participant
results_header=[{'P_Number'},extensions,{'IsGoodParticipant'}];
%results_tables=cell2table(cell(0,12),'VariableNames',results_header);
results_matrix=zeros(26,13); %Contains the results of whether it is valid or not

%Getting list of subfolders
folder_list=dir(data_root);
dirnames={folder_list([folder_list.isdir]).name};
num_dir=length(dirnames);

results_count=1;
for m=[1:num_dir]
    if dirnames{m}(1)=='P'
        for j=[1:length(extensions)]
            if (strcmp(dirnames{m},'P12')) && (strcmp(extensions{j},'Eval_Right')||strcmp(extensions{j},'Eval_Left')||strcmp(extensions{j},'Eval_Down'))
                continue
            end
            csv_datapath=[data_root,'/',dirnames{m},'/','calib_only_merged_',extensions{j},'.csv'];
            data=readmatrix(csv_datapath);
            switch extensions{j}
                case extensions{1}
                    results_matrix(results_count,2)=checkDetection(data,CALIB_THRESHOLD);
                case extensions{2}
                    results_matrix(results_count,3)=checkDetection(data,EVAL_THRESHOLD);
                case extensions{3}
                    results_matrix(results_count,4)=numUniqueDetections(data);
                case extensions{4}
                    results_matrix(results_count,5)=numUniqueDetections(data);
                case extensions{5}
                    results_matrix(results_count,6)=numUniqueDetections(data);
                case extensions{6}
                    results_matrix(results_count,7)=numUniqueDetections(data);
                case extensions{7}
                    results_matrix(results_count,8)=checkDetection(data,EVAL_THRESHOLD);
                case extensions{8}
                    results_matrix(results_count,9)=checkDetection(data,EVAL_THRESHOLD);
                case extensions{9}
                    results_matrix(results_count,10)=checkDetection(data,EVAL_THRESHOLD);
                case extensions{10}
                    results_matrix(results_count,11)=checkDetection(data,EVAL_THRESHOLD);
                case extensions{11}
                    results_matrix(results_count,12)=checkDetection(data,EVAL_THRESHOLD);
                otherwise
                    warning('Issue with Switch Case')
            end

        end
        results_matrix(results_count,1)=str2num(dirnames{m}(2:end));
        results_count=results_count+1;
    end




end

%Fill in the final column of the results_matrix which tells us if we have
%good data or bad for that participant
[num_row,~]=size(results_matrix);
for i=[1:num_row]
    calib_init_good=results_matrix(i,2);
    evals_good=(results_matrix(i,3)+results_matrix(i,8)+results_matrix(i,9)+...
        results_matrix(i,10)+results_matrix(i,11)+results_matrix(i,12))>=NUM_EVAL_THRESHOLD;
    comps_good=(results_matrix(i,4)+results_matrix(i,5)+results_matrix(i,6)+results_matrix(i,7))>=HEAD_COMP_NUMBER_THRESHOLD;
    
    results_matrix(i,end)=(calib_init_good&&evals_good&&comps_good);
end

%Saving results
results_table=array2table(results_matrix,'VariableNames',results_header);
writetable(results_table,[data_root,'/','results_check.csv'])

%Function Definitions
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

function pupil_detect_count=numUniqueDetections(calib_data)
    [row_n,col_n]=size(calib_data);
    pupil_detect_count=0; %Counts the number of pupils detected per unique calib point
    calib_pastx=calib_data(1,27);
    calib_pasty=calib_data(1,28);
    switched=false;
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

end

