import cv2
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