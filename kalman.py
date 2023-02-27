import numpy as np
from numpy.linalg import inv



class KalmanFilter1D:

    def __init__(self, dt, u, std_accel, std_meas):
        self.u = u
        
        self.A = np.array([[1, dt],
                           [0, 1]])

        self.B = np.array([[1, (dt**2)/2],
                           [0, 1]])

        self.H = np.array([[1, 0]])

        self.Q = np.array([[(dt**4)/4, (dt**3)/2],
                           [(dt**3)/2, dt**2]]) * std_accel**2

        self.R = std_meas**2

        self.P = np.eye(self.A.shape[0])

        self.x = np.array([[0],
                           [0]])


    def predict(self):
        self.x = (self.A @ self.x) + (self.B * self.u)

        self.P = (self.A @ self.P @ self.A.T) + self.Q


    def update(self, z):
        S = (self.H @ (self.P @ self.H.T)) + self.R

        K = self.P @ self.H.T @ np.linalg.inv(S)

        self.x = self.x + K @ (z - (self.H @ self.x))

        I = np.eye(self.H.shape[1])

        self.P = (I - (K @ self.H)) @ self.P

        return self.x





class KalmanFilter2D:

    def __init__(self, init_state, dt, u_x, u_y, std_accel, x_std_meas, y_std_meas):
        '''
        Parameters
        ----------
        init_state : Object's initial state
        dt : Sampling time
        u_x : Acceleration in x-direction
        u_y : Acceleration in y-direction
        std_accel : Process noise
        x_std_meas : Standard deviation of the measurement in x-direction
        y_std_meas : Standard deviation of the measurement in the y-direction
        '''

        # Control input variables
        self.u = np.array([[u_x], [u_y]])

        # Initial state
        self.x = init_state

        # State transition matrix
        self.A = np.array([[1, 0, dt, 0],
                           [0, 1, 0, dt],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]])

        # Control input matrix
        self.B = np.array([[(dt**2)/2, 0],
                           [0, (dt**2)/2],
                           [dt, 0],
                           [0, dt]])

        # Measurement mapping matrix 
        # - transforms state vars into measurements that can be obtained through sensors
        self.H = np.array([[1, 0, 0, 0],
                           [0, 1, 0, 0]])

        # Initial process noise covariance matrix
        self.Q = np.array([[(dt**4)/4, 0, (dt**3)/2, 0],
                           [0, (dt**4)/4, 0, (dt**3)/2],
                           [(dt**3)/2, 0, dt**2, 0],
                           [0, (dt**3)/2, 0, dt**2]]) * std_accel**2

        # Initial measurement noise covariance matrix
        self.R = np.array([[x_std_meas**2, 0],
                           [0, y_std_meas**2]])

        # Initial covariance matrix
        self.P = np.eye(self.A.shape[1])



    def predict(self):
        '''
        Performs prediction of the state estimate and the error covariance.
        '''

        # Project the state ahead
        self.x = self.A @ self.x + self.B @ self.u

        # Project the error covariance ahead
        self.P = self.A @ self.P @ self.A.T + self.Q



    def update(self, z):
        '''
        Updates the predicted state estimate by integrating a sensor measurement.
        '''

        # Compute the Kalman gain
        S = (self.H @ self.P @ self.H.T) + self.R
        K = self.P @ self.H.T @ inv(S)

        # Update estimate with measurement
        self.x = self.x + K @ (z - self.H @ self.x)

        # Update the error covariance
        I = np.eye(self.H.shape[1])
        self.P = (I - K @ self.H) @ self.P

        return self.x[:2, 0]
