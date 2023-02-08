from random import choice, randint
import numpy as np
from kalman import KalmanFilter2D
from plot import live_plot



def compute_motion_data(width, height, t, v_x, v_y, y_accel, std_x, std_y, energy_loss):
    # Initial state
    x = randint(0, width)
    y = randint(height/2, height)

    actual = np.zeros(shape=(t.shape[0], 2))
    measurements = np.zeros(shape=(t.shape[0], 2))

    for i in range(t.shape[0]):
        actual[i, 0] = x
        actual[i, 1] = y

        # Add noise and measure ball's position
        measurements[i, 0] = x + np.random.normal(0, std_x)
        measurements[i, 1] = y + np.random.normal(0, std_y)

        # Move ball
        x += v_x
        y += v_y

        # Acceleration due to gravity
        v_y += y_accel

        # Check for collision with upper boundary
        if y > height:
            v_y = -v_y * energy_loss
            # Set ball to ceiling level to avoid ball getting 'stuck'
            y = height

        # Check for collision with lower boundary
        if y < 0:
            v_y = -v_y * energy_loss
            # Set ball to ground level to avoid ball getting 'stuck'
            y = 0

        # Check for collision with right wall
        if x > width:
            v_x = -v_x
            x = width

        # Check for collision with left wall
        if x < 0:
            v_x = -v_x
            x = 0

    return actual, measurements



if __name__=='__main__':
    width = 1200
    height = 700

    v_x = 7 * choice([-1, 1])
    v_y = 10
    
    x_accel = 0
    y_accel = -1

    std_x = 10
    std_y = 10
    std_accel = 20

    energy_loss = 0.9

    dt = 0.1
    t = np.arange(0, 50, dt)
    
    actual, measurements = compute_motion_data(width, height, t, v_x, v_y, y_accel, std_x, std_y, energy_loss)

    # np.savetxt(f'data/actual{std_accel}.csv', actual, delimiter=',')
    # np.savetxt(f'data/measurements{std_accel}.csv', measurements, delimiter=',')

    # actual = np.genfromtxt(f'data/actual{std_accel}.csv', delimiter=',')
    # measurements = np.genfromtxt(f'data/measurements{std_accel}.csv', delimiter=',')

    init_state = np.array([[measurements[0, 0]], 
                           [measurements[0, 1]],
                           [v_x],
                           [v_y]])

    kf = KalmanFilter2D(init_state, 
                        dt=dt, 
                        u_x=x_accel, u_y=y_accel, 
                        std_accel=std_accel, 
                        x_std_meas=std_x, y_std_meas=std_y)

    filtered_est = np.zeros((t.shape[0], 2))

    for i, z in enumerate(measurements):
        # Predict
        kf.predict()
        # Update
        z = np.vstack(z)
        est = kf.update(z)
        filtered_est[i] = est


    live_plot(actual, measurements, filtered_est, 
              x_range=[0, width], y_range=[0, height], 
              update_interval=50)
    