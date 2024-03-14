import cv2
import numpy as np

class KalmanFilter:
    def __init__(self):
        self.kf = cv2.KalmanFilter(4, 2)# initializes Kalman Filter object with 4 dynamic parameters and 2 measured dimensions in a 2D space 
                                        #(x position, y position, x velocity, y velocity)
        self.kf.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32) # measurement matrix, measurement directly corresponds to the first state variables (x, y) and ignores the velocities
        self.kf.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)
        self.kf.processNoiseCov = np.eye(4, dtype=np.float32) * 0.03
        self.kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 1
        self.kf.errorCovPost = np.eye(4, dtype=np.float32)

    def predict(self, coordX, coordY):
        measured = np.array([[np.float32(coordX)], [np.float32(coordY)]])
        self.kf.correct(measured)
        predicted = self.kf.predict()
        return int(predicted[0]), int(predicted[1])

# Initialize Kalman Filter
kf = KalmanFilter()

# Window settings
window_name = "Mouse Tracking with Kalman Filter"
cv2.namedWindow(window_name)
frame = np.zeros((480, 640, 3), np.uint8)

def mouse_move(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE:
        frame.fill(255)
        cx, cy = x, y
        px, py = kf.predict(cx, cy)
        cv2.circle(frame, (cx, cy), 10, (0, 255, 0), -1)
        cv2.circle(frame, (px, py), 10, (0, 0, 255), 2)
        cv2.imshow(window_name, frame)

cv2.setMouseCallback(window_name, mouse_move)

while True:
    if cv2.waitKey(1) & 0xFF == 27: # ESC key to exit
        break

cv2.destroyAllWindows()
