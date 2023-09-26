import os
import cv2
import time
import glob

def video_to_frames(input_loc, output_loc):
    """Function to extract frames from input video file
    and save them as separate frames in an output directory.
    Args:
        input_loc: Input video file.
        output_loc: Output directory to save the frames.
    Returns:
        None
    """
    try:
        os.mkdir(output_loc)
    except OSError:
        pass
    # Log the time
    time_start = time.time()
    # Start capturing the feed
    cap = cv2.VideoCapture(input_loc)
    # Find the number of frames
    video_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print ("Number of frames: ", video_length)
    count = 0
    print ("Converting video...")
    # Start converting the video
    while cap.isOpened():
        # Extract the frame
        ret, frame = cap.read()
        if not ret:
            continue
        # Write the results back to output location.
        cv2.imwrite(output_loc + "/%#05d.jpg" % (count + 1), frame)
        count = count + 1
        # If there are no more frames left
        if (count > (video_length - 1)):
            # Log the time again
            time_end = time.time()
            # Release the feed
            cap.release()
            # Print stats
            print ("Done extracting frames.\n%d frames extracted" % count)
            print ("It took %d seconds for conversion.\n" % (time_end - time_start))
            break

if __name__=="__main__":
    root_path = '/home/adutta/Workspace/Datasets/Hockey'
    video_paths = sorted(glob.glob(os.path.join(root_path, 'period*-gray.avi')))
    video_names = [video_paths[i].split('/')[-1].split('.')[0] for i in range(len(video_paths))]
    for i in range(len(video_names)):
        print(video_names[i])
        input_loc = os.path.join(root_path, video_names[i] + '.avi')
        output_loc = os.path.join(root_path, video_names[i])
        video_to_frames(input_loc, output_loc)