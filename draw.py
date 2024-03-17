# Draw.py
import cv2
import numpy as np
from kalman import KalmanFilter

class Detect:
    def __init__(self):
        self.kf = KalmanFilter()
        self.posListX = []
        self.posListY = []
        self.predListX = []  # List to store predicted X coordinates
        self.predListY = [] 

    def update(self, frame, xList, x, y, x2, y2):
        cv2.rectangle(frame, (int(x), int(y)), (int(x2), int(y2)), (0, 0, 255), 2)
        cx = int(int(x + x2) / 2)
        cy = int(int(y + y2) / 2)

        # append the center coordinates to lists
        self.posListX.append(cx)
        self.posListY.append(cy)

        # Kalman filter prediction
        px, py = self.kf.predict(cx, cy)
        cv2.circle(frame, (px, py), 2, (0, 0, 255), 2)
        cv2.circle(frame, (cx, cy), 10, (0, 255, 255), 2)

        # Draw parabola
        if self.posListX:
            A, B, C = np.polyfit(self.posListX, self.posListY, 2)
            for i, (posX, posY) in enumerate(zip(self.posListX, self.posListY)):
                pos = (posX, posY)
                cv2.circle(frame, pos, 5, (0,255,0), cv2.FILLED)
                if i == 0:
                    cv2.line(frame, pos, pos, (0,255,0), 2)
                else:
                    cv2.line(frame, pos,(self.posListX[i-1], self.posListY[i-1]), (0,255,0), 2)

            for x in xList:
                y = int(A * x ** 2 + B * x + C)
                cv2.circle(frame, (x,y), 5, (255, 0, 255), cv2.FILLED)

