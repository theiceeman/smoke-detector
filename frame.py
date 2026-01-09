# python3 09-project/frame.py
import cv2
import os

# Get the directory where this script is located
script_dir = os.path.dirname(os.path.abspath(__file__))

# Path to your video (relative to script location)
video_path = os.path.join(script_dir, "drone_fire_video.mp4")

# Where to save frames
output_folder = os.path.join(script_dir, "extracted_frames")
os.makedirs(output_folder, exist_ok=True)

# Open video
cap = cv2.VideoCapture(video_path)

# Check if video opened successfully
if not cap.isOpened():
    print(f"Error: Could not open video file: {video_path}")
    print("Please check that the file exists and is a valid video file.")
    exit(1)

fps = int(cap.get(cv2.CAP_PROP_FPS))
print(f"Video FPS: {fps}")
frame_idx = 0
saved_idx = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Save exactly one frame per second
    if frame_idx % fps == 0:
        filename = os.path.join(output_folder, f"frame_{saved_idx}.jpg")
        cv2.imwrite(filename, frame)
        saved_idx += 1
    
    frame_idx += 1

cap.release()
print(f"Done! Extracted {saved_idx} frames (1 frame per second) to {output_folder}")
