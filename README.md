# Object Trajectory Prediction with Kalman Filter
(currently work in progress)

## Overview
Welcome to my shot tracking computer vision project! In this project I implement basketball detection and tracking utilising a YOLOv8 model to track ball trajectory with the help of OpenCV and it's Kalman filter properties. The implementation takes in video footage and is able to visualise the prediction in real-time.
Colour Key:
- Green is the actual ball trajectory
- Red is the predicted trajectory in front of the ball
- Pink is the predicted polynomial trajectory


Enjoy the demo below, featuring myself as the test dummy. This definitely didn't take 20 shots to make a basket...


![Trajectory Example](Videos/shot3_output.gif)

## Features
- Object Detection: Utilizes the YOLOv8 model to detect basketballs in video frames.
- Background Subtraction: Applies background subtraction to isolate moving objects, enhancing detection accuracy.
- Kalman Filter: Employs a Kalman Filter for predicting the basketball's future positions based on its current and previous locations.
- Video Processing: Processes video input to detect and track the basketball, displaying the tracking and prediction results in real-time.
- Video Output: Saves the processed video with tracking and prediction annotations to a file.
## Dependencies
- Python 3.8 or later
- OpenCV
- NumPy
- Ultralytics YOLOv8 (Ensure that you have the correct model file best.pt)
## Installation
1. Ensure you have Python 3.8 or later installed.
2. Install the required Python packages:

``pip install opencv-python-headless numpy``

3. For YOLOv8, follow the installation instructions provided by Ultralytics. Typically, this involves cloning their repository and setting it up as per their documentation.
## Usage
1. Place your video file in the Videos directory and update the video_path variable in balldetection.py to point to your video file.
2. Run the script:
``python balldetection.py``

4. Processed video will be displayed in real-time and saved to Videos/shot3_output.mp4 (or your specified output path).
## Files
*balldetection.py*: Main script for ball detection, tracking, and video processing.

*kalman.py*: Implements the Kalman Filter for motion prediction.
## Customization
Feel free to:
- adjust the detection confidence threshold and other YOLO settings as needed.
- adjust the Kalman Filter parameters for different types of motion or to improve prediction accuracy.

## References
https://github.com/SriramEmarose/Motion-Prediction-with-Kalman-Filter/blob/master/KalmanFilter.py

