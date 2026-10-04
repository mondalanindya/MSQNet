#!/usr/bin/env python3
"""
Create a minimal synthetic toy dataset for testing the pipeline end-to-end.
Generates small video frame directories and valid annotation CSVs for AnimalKingdom.
"""
import os
import csv
import argparse
from PIL import Image

def generate_toy_animalkingdom(output_dir, num_videos=6, num_frames=16):
    dataset_path = os.path.join(output_dir, 'AnimalKingdom', 'action_recognition')
    anno_dir = os.path.join(dataset_path, 'annotation')
    img_dir = os.path.join(dataset_path, 'dataset', 'image')
    os.makedirs(anno_dir, exist_ok=True)
    os.makedirs(img_dir, exist_ok=True)

    rows_train = []
    rows_val = []

    for i in range(num_videos):
        vid_id = f"toy_video_{i:03d}"
        vid_path = os.path.join(img_dir, vid_id)
        os.makedirs(vid_path, exist_ok=True)

        # Generate simple synthetic frames
        color = ((i * 45) % 255, (i * 75) % 255, (i * 115) % 255)
        for f in range(num_frames):
            img = Image.new('RGB', (224, 224), color=color)
            img.save(os.path.join(vid_path, f"{f+1:05d}.jpg"))

        labels = f"{i % 140},{(i + 1) % 140}"
        if i < num_videos - 2:
            rows_train.append([vid_id, labels])
        else:
            rows_val.append([vid_id, labels])

    # Write train_light.csv
    train_csv = os.path.join(anno_dir, 'train_light.csv')
    with open(train_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f, delimiter=';')
        writer.writerow(['video_id', 'labels'])
        writer.writerows(rows_train)

    # Write val_light.csv
    val_csv = os.path.join(anno_dir, 'val_light.csv')
    with open(val_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f, delimiter=';')
        writer.writerow(['video_id', 'labels'])
        writer.writerows(rows_val)

    print(f"[INFO] Synthetic AnimalKingdom toy dataset generated at: {dataset_path}")
    print(f"  - Train videos: {len(rows_train)}")
    print(f"  - Val videos  : {len(rows_val)}")
    print(f"  - Frames/video: {num_frames}")
    print(f"  - Annotations : {train_csv}, {val_csv}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate synthetic toy dataset for verification")
    parser.add_argument("--output", "-o", default="./datasets", help="Root directory for datasets")
    parser.add_argument("--num_videos", type=int, default=6, help="Number of toy videos to create")
    parser.add_argument("--num_frames", type=int, default=16, help="Number of frames per video")
    args = parser.parse_args()

    generate_toy_animalkingdom(args.output, args.num_videos, args.num_frames)
