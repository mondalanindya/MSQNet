import os
import cv2
import time
import glob
import argparse

def video_to_frames(input_loc, output_loc):
    """Function to extract frames from input video file
    and save them as separate frames in an output directory.
    Args:
        input_loc: Input video file.
        output_loc: Output directory to save the frames.
    Returns:
        count: Number of extracted frames
    """
    os.makedirs(output_loc, exist_ok=True)
    time_start = time.time()
    cap = cv2.VideoCapture(input_loc)
    video_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"[INFO] Extracting: {os.path.basename(input_loc)} ({video_length} frames)")
    count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_name = f"{count + 1:05d}.jpg"
        cv2.imwrite(os.path.join(output_loc, frame_name), frame)
        count += 1
        if video_length > 0 and count >= video_length:
            break
    cap.release()
    time_end = time.time()
    print(f"[INFO] {count} frames extracted in {time_end - time_start:.2f}s -> {output_loc}")
    return count

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract video frames into image directories")
    parser.add_argument("--video_dir", "-d", type=str, required=True, help="Directory containing input videos")
    parser.add_argument("--pattern", "-p", type=str, default="*.avi", help="Glob pattern for video files (default: *.avi)")
    parser.add_argument("--output_dir", "-o", type=str, default=None, help="Root directory to save frames (default: same as video_dir)")
    args = parser.parse_args()

    video_paths = sorted(glob.glob(os.path.join(args.video_dir, args.pattern)))
    if not video_paths:
        print(f"[WARNING] No videos matched '{args.pattern}' in {args.video_dir}")
    else:
        print(f"[INFO] Found {len(video_paths)} videos to process")
        for vpath in video_paths:
            vname = os.path.splitext(os.path.basename(vpath))[0]
            out_loc = os.path.join(args.output_dir or args.video_dir, vname)
            video_to_frames(vpath, out_loc)