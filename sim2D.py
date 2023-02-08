import numpy as np
from kalman import KalmanFilter2D
from plot import live_plot


if __name__ == '__main__':
    dt = 0.1
    t = np.arange(0, 100, dt)
    n_timesteps = t.shape[0]

    v_y = 10
    v_x = 1
    
    x_accel = 0
    y_accel = -1

    std_x = 10
    std_y = 10

    x_true = (v_x * t) + (0.5 * x_accel * t**2) # + np.random.normal(0, 1, size=(n_timesteps,))
    y_true = (v_y * t) + (0.5 * y_accel * t**2) # + np.random.normal(0, 1, size=(n_timesteps,))

    x_meas = x_true + np.random.normal(0, std_x, size=(n_timesteps,))
    y_meas = y_true + np.random.normal(0, std_y, size=(n_timesteps,))

    true_track = np.column_stack((x_true, y_true))
    measurements = np.column_stack((x_meas, y_meas))

    init_state = np.array([[measurements[0, 0]], 
                           [measurements[0, 1]],
                           [v_x],
                           [v_y]])

    kf = KalmanFilter2D(init_state, dt=dt, u_x=x_accel, u_y=y_accel, std_accel=0.1, x_std_meas=std_x, y_std_meas=std_y)

    filtered_est = np.zeros((n_timesteps, 2))

    for i, z in enumerate(measurements):
        # Predict
        kf.predict()
        # Update
        z = np.vstack(z)
        est = kf.update(z)
        filtered_est[i] = est


live_plot(true_track, measurements, filtered_est)
