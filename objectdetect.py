import cv2
import numpy as np

class OrangeDetector:
    def __init__(self):
        # Create mask for orange color
        self.low_orange = np.array([11, 128, 90])
        self.high_orange = np.array([179, 255, 255])

    def detect(self, frame):
        hsv_img = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Create masks with color ranges
        mask = cv2.inRange(hsv_img, self.low_orange, self.high_orange)

        # Find Contours
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=lambda x: cv2.contourArea(x), reverse=True)

        box = (0, 0, 0, 0)
        for cnt in contours:
            (x, y, w, h) = cv2.boundingRect(cnt)
            box = (x, y, x + w, y + h)
            break

        return box

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

dispW = 640
dispH = 480
flip = 2

kf = KalmanFilter()

od = OrangeDetector()

cap = cv2.VideoCapture("pysource/orange.mp4")

while True:
    ret, frame = cap.read()
    if ret is False:
        break

    boundingbox = od.detect(frame)
    x, y, x2, y2 = boundingbox
    cx = int((x + x2) / 2)
    cy = int((y + y2) / 2)

    predicted = kf.predict(cx, cy)
    cv2.circle(frame, (cx, cy), 10, (0, 255, 255), -1)
    cv2.circle(frame, (predicted[0], predicted[1]), 10, (255, 0, 0), 4)

    cv2.imshow("Frame", frame)
    key = cv2.waitKey(150)
    if key == 27:
        break