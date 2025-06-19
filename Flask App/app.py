from flask import Flask, render_template, request, send_file
import os
from werkzeug.utils import secure_filename
from ultralytics import YOLO
import cv2
import numpy as np
import io
import time
from PIL import Image

UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'mp4'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load the YOLO model for vehicle damage detection
model = YOLO("best1.pt")

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def resize_image(image_path, output_path, size=(300, 300)):
    """Resize the image to the specified size."""
    with Image.open(image_path) as img:
        img = img.resize(size)
        img.save(output_path)

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    uploaded_file_urls = []  # List to store URLs of uploaded images
    damage_detected_urls = []  # List to store URLs of damage-detected images

    if request.method == 'POST':
        files = request.files.getlist('media')  # Allow multiple file uploads
        for file in files:
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(filepath)

                # Determine file type
                ext = filename.rsplit('.', 1)[1].lower()
                if ext in {'png', 'jpg', 'jpeg'}:  # Only resize images
                    # Resize the image
                    resized_path = os.path.join(app.config['UPLOAD_FOLDER'], f"resized_{filename}")
                    resize_image(filepath, resized_path)

                    # Generate file URL for resized image
                    uploaded_file_url = f"{app.config['UPLOAD_FOLDER']}/resized_{filename}"
                    uploaded_file_urls.append(uploaded_file_url)
                else:  # For videos, use the original file path
                    uploaded_file_url = f"{app.config['UPLOAD_FOLDER']}/{filename}"
                    uploaded_file_urls.append(uploaded_file_url)

                # Perform damage detection
                damage_detected_url = detect_damage(filepath)
                damage_detected_urls.append(damage_detected_url)

    return render_template('index.html', 
                           file_urls=uploaded_file_urls, 
                           damage_detected_urls=damage_detected_urls)

def detect_damage(file_path):
    ext = file_path.rsplit('.', 1)[1].lower()
    timestamp = int(time.time())

    if ext in {'png', 'jpg', 'jpeg'}:
        img = cv2.imread(file_path)
        results = model(img)[0]
        annotated = results.plot()
        damage_detected_path = os.path.join(app.config['UPLOAD_FOLDER'], f"damage_detected_{timestamp}.jpg")
        cv2.imwrite(damage_detected_path, annotated)
        return f"{app.config['UPLOAD_FOLDER']}/damage_detected_{timestamp}.jpg"

    elif ext == 'mp4':
        cap = cv2.VideoCapture(file_path)
        if not cap.isOpened():
            print("Error: Unable to open video file.")
            return None

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30  # fallback FPS if 0
        damage_detected_path = os.path.join(app.config['UPLOAD_FOLDER'], f"damage_detected_{timestamp}.mp4")
        out = cv2.VideoWriter(damage_detected_path, cv2.VideoWriter_fourcc(*'X264'), 30, (width, height))

        frame_index = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            try:
                results = model(frame)[0]
                annotated_frame = results.plot()
                out.write(annotated_frame)
            except Exception as e:
                print(f"Error on frame {frame_index}: {e}")
            
            frame_index += 1

        cap.release()
        out.release()

        print(f"Video saved at: {damage_detected_path}")  # Added print statement

        return f"{app.config['UPLOAD_FOLDER']}/damage_detected_{timestamp}.mp4"


if __name__ == '__main__':
    app.run(debug=True)