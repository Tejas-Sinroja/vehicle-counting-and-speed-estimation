# vehicle-counting-and-speed-estimation

## Introduction
This project implements a vehicle counting and speed estimation system using YOLOv8 for object detection and a simple centroid tracking algorithm. The system processes video frames to detect vehicles, track their movement, and estimate their speed based on reference distance and frame rate.

## Dataset Overview
The input data for this project is a video file containing moving vehicles. The script processes each frame to detect and track vehicles.

## Data Preprocessing
- The video frames are captured using OpenCV.
- YOLOv8 is used to detect vehicles in each frame.
- A centroid tracking method is used to assign unique IDs to vehicles.
- Vehicle positions are stored and updated across frames to estimate speed and count crossings.

## Model Building
- The YOLOv8 model (`yolov8n.pt`) is loaded to perform real-time vehicle detection.
- The detected vehicles are tracked using a simple distance-based matching algorithm.
- Speed is estimated using pixel displacement, frame rate, and a predefined reference distance.

## Results
The system outputs:
- The number of vehicles moving in upward and downward directions.
- The estimated speed of each detected vehicle.
- A visual representation of detected vehicles with bounding boxes and labels.

## Requirements
- Python
- OpenCV
- NumPy
- Ultralytics YOLO

Creating virtual environment:
```bash
python -m venv venv
```
Activating the environment:
```bash
venv\Scripts\activate
```
Install dependencies using:
```bash
pip install -r requirements.txt
```
Run the main file 
```bash
python main.py
```

## Contributor
- **Tejas Sinroja**

## Reference
- [Ultralytics YOLO](https://github.com/ultralytics/ultralytics)

