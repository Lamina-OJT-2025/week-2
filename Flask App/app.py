from flask import Flask, render_template, request, send_file
import os
from werkzeug.utils import secure_filename
from ultralytics import YOLO
import cv2
import numpy as np
import io
import time

UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'mp4'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load the YOLO model for vehicle damage detection
model = YOLO("best.pt")

filename = f"damage_detected_{int(time.time())}.jpg"
damage_detected_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    uploaded_file_url = None
    file_type = None
    damage_detected_url = None

    if request.method == 'POST':
        file = request.files.get('media')
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            uploaded_file_url = f"{app.config['UPLOAD_FOLDER']}/{filename}"
            ext = filename.rsplit('.', 1)[1].lower()
            file_type = 'image' if ext in {'png', 'jpg', 'jpeg'} else 'video'

            # If the uploaded file is an image, perform damage detection
            if file_type == 'image':
                damage_detected_url = detect_damage(filepath)

    return render_template('index.html', 
                           file_url=uploaded_file_url, 
                           file_type=file_type, 
                           damage_detected_url=damage_detected_url)

def detect_damage(image_path):
    # Read the image
    img = cv2.imread(image_path)

    # Perform damage detection using the YOLO model
    results = model(img)[0]
    annotated = results.plot()

    # Save the annotated image
    damage_detected_path = os.path.join(app.config['UPLOAD_FOLDER'], 'damage_detected.jpg')
    cv2.imwrite(damage_detected_path, annotated)

    return f"{app.config['UPLOAD_FOLDER']}/damage_detected.jpg"

if __name__ == '__main__':
    app.run(debug=True)

