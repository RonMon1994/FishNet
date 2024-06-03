import pandas as pd
from utils import process_data, process_video_frames
from pathlib import Path
import torch


import os
main_path = os.getcwd()

processed_data = os.path.join(main_path, '2022-0509_med_campaign_stereoBRUVs_ImagePtPair.txt')
data_3DPoints = pd.read_csv(os.path.join(main_path,'2022-0509_med_campaign_stereoBRUVs_3DPoints.txt'), sep='\t', on_bad_lines='skip')
data_Lengths = pd.read_csv(os.path.join(main_path,'2022-0509_med_campaign_stereoBRUVs_Lengths.txt'), sep='\t', on_bad_lines='skip')
output_frames_dir = Path(os.path.join(main_path,'Dataset'))
# This is from the hard disk 
videos_base_dir = Path('/media/ronm/Crucial X6/chunk_4')
frame_rate = 60

processed_data = process_data(processed_data, data_3DPoints, data_Lengths)
processed_data.to_csv('med_sea_data.txt', sep='\t', index=False)

dict_left, images_path_left = process_video_frames(processed_data, 'Left', videos_base_dir, output_frames_dir, frame_rate)
dict_right , images_path_right = process_video_frames(processed_data, 'Right', videos_base_dir, output_frames_dir, frame_rate)

