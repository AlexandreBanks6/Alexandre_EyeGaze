# Reference

This analysis code was used for publication of: 

Banks A, Eldin Abdelaal A, Salcudean S. Head motion-corrected eye gaze tracking with the da Vinci surgical system. Int J Comput Assist Radiol Surg. 2024 Jul;19(7):1459-1467. doi: 10.1007/s11548-024-03173-4. Epub 2024 Jun 18. PMID: 38888820. 

[https://pubmed.ncbi.nlm.nih.gov/38888820/](https://pubmed.ncbi.nlm.nih.gov/38888820/)

# Eye Gaze Calibration Steps

- Do an 8-point calibration to fit the standard polynomial regressor.
- Do 5x1-point calibration to fit the corner-contingent head compensation model. During this calibration, get the user to look at a single dot in the center of the screen, then lift there head, and then return their head and look at the dot again. Do this five times.
  
# Running the Code

**Note that this is development code and may need to be modified to remove analysis and display functionality for an application setting.**

## Data Organization Code

Follow the following steps to organize the data and get the correct data structure:
- Create a _DataCollection_ParticipantList.csv_ in the _data_root_ directory that specifies the folders (participant numbers) in the order that we wish to process.
- Run _CollectedData_RenamerAndConverter.py_ which re-names three files (the caliblog .txt, the gazelog.txt, and the eye video .avi) in each participant folder under EyeGaze_Data to have the format: caliblog_{caliblog_count}.txt, gazelog_{gazelog_count}.txt, and _{avi_count}.txt where caliblog_count, gazelog_count, and avi_count is an incrementer for the number of corresponding files for that participant. If not using NDI, can comment out lines 74-83 (looping for NDI data)
- Run _Data_Merger_eyeNdi.m_ which takes in .csv with eye gaze calibration and .csv from the NDI tool measurements. Syncrhonizes the two and creates a file called _eye_NDI_merged.csv_ and saves it under a directory specified by "DATA_DIR" in the code (called the _converted_ directory). Creates a merged csv with only the calibration information and trims the corresponding video. If not using NDI data, only need lines 393 and 394 where we are extracting rows of the data that correspond to calibration points only. Need to still run lines 420 (creating a new data header) and lines 443 and 445 (saving the calibration only .csv). The other lines can be commented out (except the looping lines) as we are not synchronizing with NDI.

## Eye Corner Detection Code

- Run _EyeGaze_Main_Aug9.py_. Make sure that the _YOLOV7_MODEL_PATH_ points to the desired pre-trained weights. We already have pre-trained weights at _resources/eyegaze_model_new_aug18.pt_. Make sure that _data_root_ points to the folder where the _converted_ data is. Set the _dest_path_ on line 1145 to where we want to save the _CornerResults.csv_.
- Run _Data_Merger_corners.m_ to merge the corner data with the NDI and eye gaze data. Can ignore lines 65 to 102 which are the "full_data" and are only needed when we are using the NDI.

## Head Compensation Code

- Run V2_HeadCompensation_Pipeline.m, may need to 

# Data Structure
