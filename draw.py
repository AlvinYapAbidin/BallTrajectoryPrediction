import cv2
import numpy as np
from kalman import KalmanFilter

class Detect:
    def __init__(self):
        self.kf = KalmanFilter()

    def bbox(self,frame, x, y, x2, y2):
        cv2.rectangle(frame, (int(x), int(y)), (int(x2), int(y2)), (0, 0, 255), 2)
        cx = int(int(x + x2) / 2)
        cy = int(int(y + y2) / 2)

        predicted = self.kf.predict(cx, cy)
        cv2.circle(frame, (cx, cy), 10, (0, 255, 255), -1)
        cv2.circle(frame, (predicted[0], predicted[1]), 10, (255, 0, 0), 4)
