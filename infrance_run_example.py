from pathlib import Path
from utils import process_video_frames_sample
import os
from ultralytics import YOLO


main_path = os.getcwd()
videos_base_dir = Path('/media/ronm/Crucial X6/chunk_4/israchz091121A/videos')
frame_rate = 60
output_frames_dir = Path(os.path.join(main_path, 'Dataset'))

images_path = process_video_frames_sample(videos_base_dir, output_frames_dir, frame_rate)
print(f"Extracted {len(images_path)} images.")
videos_base_dir = Path('/media/ubuntu/Crucial X6/chunk_4/israchz091121A/videos')
frame_rate = 60
output_frames_dir = Path(os.path.join(main_path, 'Dataset'))

# images_path = process_video_frames_sample(videos_base_dir, output_frames_dir, frame_rate)
print(f"Extracted {len(images_path)} images.")
model = YOLO(os.path.join(main_path, "weights/best.pt"))

model.predict(source = os.path.join(main_path, '/home/ubuntu/FishNet/Dataset/images_inference/lGOPR3305'), save=True)