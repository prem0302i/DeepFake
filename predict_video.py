import torch
import cv2
import os
import shutil
import logging
from scripts.extract_frames import extract_frames
from scripts.face_crop import crop_face

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def predict_video(model, video_path, device, progress_callback=None):
    frame_dir = "temp_frames"
    if os.path.exists(frame_dir):
        shutil.rmtree(frame_dir)
    os.makedirs(frame_dir)

    frame_paths = extract_frames(video_path, frame_dir) # This now returns a list of paths

    fake_votes = 0
    total = 0
    logging.info(f"Starting prediction on {len(frame_paths)} frames.")

    for i, frame_path in enumerate(frame_paths):
        frame = cv2.imread(frame_path)
        if frame is None:
            logging.warning(f"Could not read frame {frame_path}")
            continue

        faces = crop_face(frame)
        if not faces:
            logging.warning(f"No faces detected in frame {i}.")

        for face in faces:
            face = cv2.resize(face, (224, 224))
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face = face / 255.0

            tensor = torch.tensor(face).permute(2, 0, 1)
            tensor = tensor.unsqueeze(0).float().to(device)

            with torch.no_grad():
                output = model(tensor)
                pred = output.argmax(1).item()

            fake_votes += pred
            total += 1

        if progress_callback:
            progress = (i + 1) / len(frame_paths) * 100
            progress_callback(progress)

    # Clean up the temporary frame directory
    if os.path.exists(frame_dir):
        shutil.rmtree(frame_dir)

    logging.info(f"Prediction complete. Total faces processed: {total}.")
    if total == 0:
        logging.error("No faces were detected in any frame.")
        return "No face detected"

    return "FAKE" if fake_votes / total > 0.5 else "REAL"
