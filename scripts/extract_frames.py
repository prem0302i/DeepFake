import cv2
import os

def extract_frames(video_path, output_dir, every_n=5):
    """
    Extract frames from video and save them to output_dir
    """

    os.makedirs(output_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    frame_idx = 0
    saved = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % every_n == 0:
            frame_path = os.path.join(output_dir, f"{saved}.jpg")
            cv2.imwrite(frame_path, frame)
            saved += 1

        frame_idx += 1

    cap.release()
    return output_dir
