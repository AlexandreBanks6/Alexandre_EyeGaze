clear
clc
close all

data_root='E:/Alexandre_EyeGazeProject_Extra/eyecorner_userstudy2_converted';
%Getting list of subfolders
folder_list=dir(data_root);
dirnames={folder_list([folder_list.isdir]).name};
num_dir=length(dirnames);

for m=[1:num_dir]
    if strcmp(dirnames{m}(1),'P')
        p_num=str2double(dirnames{m}(2:end));
        if p_num>22
        participant_root=[data_root,'/',dirnames{m}];
        file_list=dir(participant_root);
        file_names={file_list(~[file_list.isdir]).name};
        num_names=length(file_names);
        for p=[1:num_names]
            current_file=file_names{p};
            if strcmp(current_file(1:10),'calib_only') %The current file starts with calib_only so we find the corresponding eyecorner file
               
                for n=[1:num_names]
                    candidate_file=file_names{n};
                    if strcmp(candidate_file(1:10),'eyecorners') && strcmp(candidate_file(12:end),current_file(19:end)) %Found associated eyecorner file
                        %Opening the two files and updating the calib_only
                        %file
                        current_file_fullpath=[participant_root,'/',current_file];
                        candidate_file_fullpath=[participant_root,'/',candidate_file];

                        calib_only_data_table=readtable(current_file_fullpath);
                        calib_only_data=table2array(calib_only_data_table);

                        eyecorner_data=readmatrix(candidate_file_fullpath);
                        
                        %Initializing eyecorner matrix
                        [row_calib,~]=size(calib_only_data);
                        eyecorner_mat=nan(row_calib,8);
                        for i=[1:row_calib]
                            eyecorner_indx=find(eyecorner_data(:,1)==calib_only_data(i,2));
                            if ~isempty(eyecorner_indx)
                                eyecorner_mat(i,:)=eyecorner_data(eyecorner_indx,[2:9]);
                            end
                            
                        end
                        calib_only_data(:,50:57)=eyecorner_mat;
                        old_names=calib_only_data_table.Properties.VariableNames;
                        old_names(50:57)={'right_inner_x','right_inner_y',...
                            'right_outer_x','right_outer_y','left_inner_x','left_inner_y',...
                            'left_outer_x','left_outer_y'};
                        for i=[1:length(old_names)]
                            if(isempty(old_names{i}))
                                old_names{i}=['Blank',num2str(i)];
                            end
                        end

                        output_table=array2table(calib_only_data,'VariableNames',old_names);
                        writetable(output_table,current_file_fullpath)


                    end

                end
                
            elseif strcmp(current_file(1:9),'full_data')
                for n=[1:num_names]
                    candidate_file=file_names{n};
                    if strcmp(candidate_file(1:10),'eyecorners') && strcmp(candidate_file(end-8:end),current_file(end-8:end)) %Found associated eyecorner file
                        %Opening the two files and updating the calib_only
                        %file
                        current_file_fullpath=[participant_root,'/',current_file];
                        candidate_file_fullpath=[participant_root,'/',candidate_file];

                        full_data_table=readtable(current_file_fullpath);
                        full_data=table2array(full_data_table);
                        eyecorner_data=readmatrix(candidate_file_fullpath);
                        
                        %Initializing eyecorner matrix
                        [row_calib,~]=size(full_data);
                        eyecorner_mat=nan(row_calib,8);
                        for i=[1:row_calib]
                            eyecorner_indx=find(eyecorner_data(:,1)==full_data(i,2));
                            if ~isempty(eyecorner_indx)
                                eyecorner_mat(i,:)=eyecorner_data(eyecorner_indx,[2:9]);
                            end
                            
                        end
                        full_data(:,50:57)=eyecorner_mat;
                        old_names=calib_only_data_table.Properties.VariableNames;
                        old_names(50:57)={'right_inner_x','right_inner_y',...
                            'right_outer_x','right_outer_y','left_inner_x','left_inner_y',...
                            'left_outer_x','left_outer_y'};
                        output_table=array2table(full_data,'VariableNames',old_names);
                        writetable(output_table,current_file_fullpath)


                    end

                end


            end
       
        
        end
        end
    end

end