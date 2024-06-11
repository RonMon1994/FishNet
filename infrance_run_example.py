import argparse
from pathlib import Path
from utils import process_video_frames_sample
import os
from ultralytics import YOLO
import pandas as pd

def process_videos(main_path, videos_base_dir, frame_rate, output_frames_dir):
    images_path = process_video_frames_sample(videos_base_dir, output_frames_dir, frame_rate)
    print(f"Extracted {len(images_path)} images.")
    return images_path

def run_inference(main_path, images_dir, model):
    results = model.predict(source=images_dir, save=True)

    data = []
    for result in results:
        image_name = Path(result.path).name  
        for box in result.boxes:
            bbx = box.xyxy.tolist()[0]
            classification = box.cls.tolist()[0]
            probability = box.conf.tolist()[0]
            data.append({
                'image_name': image_name,
                'bbx': bbx,
                'classification': classification,
                'probability': probability
            })

    return data

def save_to_excel(data, output_path):
    df = pd.DataFrame(data)
    df.to_excel(output_path, index=False, engine='openpyxl')
    print(f"Saved predictions to {output_path}.")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--videos_base_dir", help="Base directory for videos", default='/media/ronm/Crucial X6/chunk_4/israchz091121A/videos')
    args = parser.parse_args()

    main_path = os.getcwd()
    videos_base_dir = Path(args.videos_base_dir)
    frame_rate = 60
    output_frames_dir = Path(os.path.join(main_path, 'Dataset'))

    images_path = process_videos(main_path, videos_base_dir, frame_rate, output_frames_dir)

    # Load the YOLO model
    model = YOLO(os.path.join(main_path, "weights/best.pt"))

    # Perform predictions on images
    images_dir = os.path.join(main_path, 'Dataset/images_inference/lGOPR3305')
    inference_data = run_inference(main_path, images_dir, model)

    # Save results to Excel
    output_excel_path = os.path.join(main_path, 'model_predictions.xlsx')
    save_to_excel(inference_data, output_excel_path)

if __name__ == "__main__":
    main()