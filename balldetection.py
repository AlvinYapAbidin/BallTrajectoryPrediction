import cv2
from ultralytics import YOLO
import numpy as np

class KalmanFilter:
    # initializes Kalman Filter object with 4 dynamic parameters and 2 measured dimensions in a 2D space 
    #(x position, y position, x velocity, y velocity)
    kf = cv2.KalmanFilter(4,2)
    # measurement matrix, measurement directly corresponds to the first state variables (x, y) and ignores the velocities
    kf.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
    # transition matrix, how the state evolves from one time step to the next
    # matrix indicates that positions are updated by the velocities ([1, 0, 1, 0], [0, 1, 0, 1])
    kf.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)

    # this function uses the kalman filter to predict the next state based on the current measurement (coordX, coordY) of the position
    def predict(self, coordX, coordY):
        # creates measurement array from current observed x and y position
        measured = np.array([[np.float32(coordX)], [np.float32(coordY)]])
        # correction step of the Kalman Filter with current measurement 
        # updates the filter's internal state estimation to be more accurate based on observed measurement
        self.kf.correct(measured)
        # perform prediction step of Kalman Filter, sing internal state and model of the system to predict the next state
        predicted = self.kf.predict()
        # extract the predicted x and y positions from the predicted result
        x, y = int(predicted[0]), int(predicted[1])
        return x, y

# Load the YOLOv8 model
model = YOLO('best.pt')

# Open the video file
video_path = "shot1.mp4"
cap = cv2.VideoCapture(video_path)

b = []

kf = KalmanFilter()

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
                #x = int(bounding_box[0])
                #y = int(bounding_box[1])
                #width = int(bounding_box[2] - x)
                #height = int(bounding_box[3] - y)

        cv2.rectangle(frame, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (0, 0, 255), 2)
        cx = int(int(b[0] + int(b[2])) / 2)
        cy = int(int(b[1] + int(b[3])) / 2)

        predicted = kf.predict(cx, cy)
        cv2.circle(frame, (cx, cy), 10, (0, 255, 255), -1)
        cv2.circle(frame, (predicted[0], predicted[1]), 10, (255, 0, 0), 4)

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