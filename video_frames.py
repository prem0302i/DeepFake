import cv2
import os

def extract_frames(video_path, output_dir, progress_callback=None):
    """
    Extracts all frames from a video and saves them to an output directory.
    Reports progress using a callback function.
    """
    os.makedirs(output_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_id = 0
    print(f"✅ Starting frame extraction for {video_path}...")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imwrite(os.path.join(output_dir, f"frame_{frame_id}.jpg"), frame)
        frame_id += 1
        if progress_callback and total_frames > 0:
            progress = (frame_id / total_frames) * 100
            progress_callback(progress)

    cap.release()
    if progress_callback:
        progress_callback(100)
    print("✅ Frame extraction completed.")
