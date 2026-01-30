import cv2
import os

def create_annotated_video(frame_dir, annotated_frame_details, output_video_path, original_video_path):
    """
    Creates a video from annotated frames.
    """
    # Get video properties from the original video
    cap = cv2.VideoCapture(original_video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    frame_files = sorted(os.listdir(frame_dir), key=lambda x: int(os.path.splitext(x)[0]))

    for frame_filename in frame_files:
        frame_path = os.path.join(frame_dir, frame_filename)
        frame = cv2.imread(frame_path)

        if frame_filename in annotated_frame_details:
            details = annotated_frame_details[frame_filename]
            for detail in details:
                box, pred, prob = detail
                x, y, w, h = box
                color = (0, 0, 255) if pred == 1 else (0, 255, 0) # Red for FAKE, Green for REAL
                label = f"{'FAKE' if pred == 1 else 'REAL'}: {prob:.2f}"
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        out.write(frame)

    out.release()
