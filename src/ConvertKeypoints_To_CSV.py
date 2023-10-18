import json

JSON_KEYPOINTS='E:/Alexandre_EyeGazeProject_Extra/EvaluatingCornerDetection_Accuracy/annotated_corners.json'
OUTPUT_CSV='E:/Alexandre_EyeGazeProject_Extra/EvaluatingCornerDetection_Accuracy/annotated_corners.csv'

s=json.load(open(JSON_KEYPOINTS))
csv_eyecorner=open(OUTPUT_CSV,mode='w')
for ann in s['annotations']:
    #Goes, right_inner_x, right_inner_y, right_outer_x, right_outer_y, left_inner_x,...
    curr_keypoint=[ann['keypoints'][0],ann['keypoints'][1],ann['keypoints'][3],ann['keypoints'][4],\
                   ann['keypoints'][6],ann['keypoints'][7],ann['keypoints'][9],ann['keypoints'][10]]
    csv_eyecorner.write('{},{},{},{},{},{},{},{}\n'.format(curr_keypoint[0],curr_keypoint[1],curr_keypoint[2],curr_keypoint[3],curr_keypoint[4],curr_keypoint[5],curr_keypoint[6],curr_keypoint[7]))
csv_eyecorner.close()