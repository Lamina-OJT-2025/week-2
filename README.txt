# Vehicle Damage Detection Web App

A Flask-based web application for detecting vehicle damage in images and videos using a YOLO model.

## Features:

- Upload images (PNG, JPG, JPEG) or videos (MP4)
- Detect and highlight vehicle damage using a trained YOLO model
- View original and annotated (damage-detected) media side by side
- Modern, responsive UI

## Setup:

1. Download the Repository via Google Drive or clone on github.
	```
	https://github.com/Lamina-OJT-2025/week-2.git
 	```
2. Create and Activate a Virtual Environment (Recommended).
	```
	python -m venv venv  # Create virtual evironment
	Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass  # Bypass the system
	venv\Scripts\activate  # On Windows
	# or
	source venv/bin/activate  # On macOS/Linux
3. Install Dependencies.
  	```
   	pip install -r requirements.txt
  	```
## How to Run:

1. Navigate to the folder.
   	```
   	cd "c:\LocationOfTheFolder\Flask_App"
   	```
3. Proceed to run the app.
	```
 	python app.py
	```
4. Open in browser.
  	```
	Go to http://127.0.0.1:5000/ in your web browser.
 	```
5. Upload Media.
  	```
  	-  Click "Choose Files" to select images or videos.
  	-  Click "Upload" to process.
  	-  View the original and damage-detected results side by side.
  	```

## Notes
  - Make sure you have a compatible YOLO model (best.pt) in the project directory.
  - For video processing, ensure FFmpeg is installed and added to your system PATH.
  - Large videos may take longer to process
  - Photo and Video upload has a limit of 50mb.
