from flask import Flask, render_template, request, jsonify, url_for, redirect
import os
import sys
import torch
import threading
from werkzeug.utils import secure_filename

# Add project root to system path to allow module imports
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(PROJECT_ROOT)

from predict_video import predict_video
from models.cnn import DeepFakeCNN

app = Flask(__name__, static_folder='static')

# --- CONFIGURATION ---
UPLOAD_FOLDER = os.path.join(PROJECT_ROOT, "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# --- GLOBAL VARIABLES ---
video_processing_status = {"status": "idle"}  # idle, processing, done, error

# --- MODEL LOADING ---
device = "cuda" if torch.cuda.is_available() else "cpu"
model = DeepFakeCNN().to(device)
model_path = os.path.join(PROJECT_ROOT, 'models', 'deepfake_cnn.pth')
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

def process_video_in_background(video_path):
    global video_processing_status

    def update_progress(progress):
        video_processing_status['progress'] = progress

    try:
        video_processing_status = {"status": "processing", "progress": 0, "message": "Analyzing video..."}
        
        prediction = predict_video(model, video_path, device, progress_callback=update_progress)
        
        video_processing_status = {
            "status": "done",
            "prediction": prediction
        }

    except Exception as e:
        import traceback
        traceback.print_exc()
        video_processing_status = {"status": "error", "message": f"An error occurred: {e}"}
    finally:
        # Clean up the uploaded file
        if os.path.exists(video_path): 
            os.remove(video_path)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    global video_processing_status
    if 'video' not in request.files:
        return redirect(request.url)
    file = request.files['video']
    if file.filename == '' or video_processing_status.get('status') == 'processing':
        return redirect(request.url)
    if file:
        filename = secure_filename(file.filename)
        video_path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(video_path)

        thread = threading.Thread(target=process_video_in_background, args=(video_path,))
        thread.start()

        return redirect(url_for('index'))

@app.route('/status')
def status():
    return jsonify(video_processing_status)

@app.route('/reset', methods=['POST'])
def reset():
    global video_processing_status
    video_processing_status = {"status": "idle"}
    return jsonify({'status': 'ok'})

if __name__ == "__main__":
    app.run(debug=True)
