import cv2
from ultralytics import YOLO
import numpy as np
from kalman import KalmanFilter
from draw import Detect

# Load the YOLOv8 model
model = YOLO('best.pt')

# Open the video file
video_path = "shot1.mp4"
cap = cv2.VideoCapture(video_path)

b = [] # Store bounding box

kf = KalmanFilter()
detect = Detect()

# Loop through the video frames
while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()

    if success:
        # Run YOLOv8 inference on the frame
        results = model(frame)

        # Visualize the results on the frame
        annotated_frame = results[0].plot()

        #b = results[0].boxes.xyxy[0].cpu().numpy()
        #print(b)
        #print(results[0].boxes.cls)

        for result in results:

            detection_count = result.boxes.shape[0]

            for i in range(detection_count):
                cls = int(result.boxes.cls[i].item())
                name = result.names[cls]
                print(name)
                if (name=="basketball"):
                    confidence = float(result.boxes.conf[i].item())
                    print(confidence)
                    bounding_box = result.boxes.xyxy[i].cpu().numpy()
                    print(bounding_box)
                    if confidence > 0.7:
                        b = bounding_box
                
        if b is not None:
            detect.bbox(frame, int(b[0]), int(b[1]), int(b[2]), int(b[3]))

        # Display the annotated frame
        # cv2.imshow("YOLOv8 Inference", annotated_frame)
        cv2.imshow("YOLOv8 Inference", frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(100) & 0xFF == ord("q"):
            break
    else:
        # Break the loop if the end of the video is reached
        break

# Release the video capture object and close the display window
cap.release()
cv2.destroyAllWindows()