# balldetection.py
import cv2
from ultralytics import YOLO
import numpy as np
from kalman import KalmanFilter

model = YOLO('model.pt')

video_path = "Videos/shot3.mp4"
cap = cv2.VideoCapture(video_path)
output_video_path = "Videos/shot3_output.mp4"

width = 1200
height = 700
fps = (cap.get(cv2.CAP_PROP_FPS)/2)

# Initialize - video writer
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  
out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

b = [] # Store bounding box
xList = [item for item in range(0, int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)))]

# Initialize - Kalman Filter
kf = KalmanFilter()

posListX = []
posListY = []
predListX = []
predListY = []

while cap.isOpened(): # Main Video Loop
    success, frame = cap.read()

    if success:
        frame = cv2.resize(frame, (width, height))
        results = model(frame)

        # Visualize the results on the frame
        #annotated_frame = results[0].plot()

        #b = results[0].boxes.xyxy[0].cpu().numpy()
        #print(b)
        #print(results[0].boxes.cls)

        for result in results:
            detection_count = result.boxes.shape[0]
            # print("results: ", result)
            for i in range(detection_count):
                bbox = result.boxes.xyxy[i].cpu().numpy()
                confidence = float(result.boxes.conf[i].item())
                cls = int(result.boxes.cls[i].item())
                name = result.names[cls]

                # Bounding Boxes
                w, h = bbox[2] - bbox[0], bbox[3] - bbox[1] 
                # Center Points
                cx = int((bbox[0] + bbox[2]) / 2)
                cy = int((bbox[1] + bbox[3]) / 2)

                #print(name)
                if (name=="basketball") and confidence > 0.4:
                    posListX.append(cx)
                    posListY.append(cy)
                    # Kalman Filter prediction
                    px, py = kf.predict(cx, cy)
                    predListX.append(px)
                    predListY.append(py)
                    
                if name == "rim" and confidence > 0.1:
                    # Draw rim bounding box
                    cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 3)



        if posListX:
            # Polynomial Regression y = ax^2 + bx + c
            A, B, C = np.polyfit(posListX, posListY, 2)

            for i, (posX, posY) in enumerate(zip(posListX, posListY)):
                pos = (posX, posY)
                cv2.circle(frame, pos, 10, (0,255,0), cv2.FILLED)
                if i == 0:
                    cv2.line(frame, pos, pos, (0,255,0), 2)
                else:
                    cv2.line(frame, pos,(posListX[i-1], posListY[i-1]), (0,255,0), 2) # Green line for tracked trajectory
            
            # Draw predicted positions with red circles
            for i, (predX, predY) in enumerate(zip(predListX, predListY)):
                cv2.circle(frame, (predX, predY), 10, (0, 0, 255), 2) # Red dots for predicted trajectory

            for x in xList:
                y = int(A * x ** 2 + B * x + C)
                cv2.circle(frame, (x,y), 1, (255, 0, 255), cv2.FILLED) # Pink line for predicted parabola trajectory
        
        out.write(frame)

        cv2.imshow("YOLOv8 Inference", frame)

        if cv2.waitKey(50) & 0xFF == ord("q"):
            break
    else:
        break 
cap.release()
cv2.destroyAllWindows()