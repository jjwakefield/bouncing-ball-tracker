import numpy as np
import matplotlib.pyplot as plt
from kalman import KalmanFilter1D


if __name__ == '__main__':
    dt = 0.1
    t = np.arange(0, 100, dt)

    real_track = 0.1 * (t**2 - t)

    u = 2
    std_accel = 0.25    # standard deviation of acceleration
    std_meas = 1.2      # standard deviation of measurement

    kf = KalmanFilter1D(dt, u, std_accel, std_meas)

    predictions = []
    measurements = []

    for x in real_track:
        # Measurement
        z = kf.H * x + np.random.normal(0, 50)
        z = z.flatten()
        measurements.append(z[0])

        # Predict
        kf.predict()

        # Update
        x_est = kf.update(z[0])
        predictions.append(x_est[0, 0])

    fig = plt.figure()

    plt.scatter(t, measurements, label='Measurements', c='b', s=0.1)
    plt.plot(t, real_track, label='Real track', color='k', linewidth=1.5)
    plt.plot(t, predictions, label='Filtered predictions', color='r', linewidth=1.5)

    plt.xlabel('Time', fontsize=20)
    plt.ylabel('Position', fontsize=20)
    plt.title('1D Kalman Filter Tracker')
    plt.legend()
    plt.show()
