from random import choice, randint
import numpy as np
from kalman import KalmanFilter2D
from particle_filter import ParticleFilter
from plot import animated_plot_kalman
from plot import animated_plot_particle



def compute_motion_data(width, height, t, v_x, v_y, y_accel, std_x, std_y, energy_loss, init_state):
    # Initial state
    x = init_state[0]
    y = init_state[1]

    path = np.zeros(shape=(t.shape[0], 2))
    measurements = np.zeros(shape=(t.shape[0], 2))

    for i in range(t.shape[0]):
        path[i, 0] = x
        path[i, 1] = y

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

    return path, measurements



def use_kalman_filter(dt, v_x, v_y, x_accel, y_accel, std_x, std_y, std_accel, t, actual, measurements, full_plot):
    init_state = np.array([[measurements[0, 0]], 
                           [measurements[0, 1]],
                           [v_x],
                           [v_y]])

    kf = KalmanFilter2D(init_state=init_state, 
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

    animated_plot_kalman(actual, measurements, filtered_est, 
              x_range=[0, width], y_range=[0, height], 
              update_interval=50, full_plot=full_plot)
    


def use_particle_filter(width, height, v_x, v_y, y_accel, std_x, std_y, t, actual, measurements, full_plot):
    n_particles = 50

    pf = ParticleFilter(n_particles, actual[0], std_x, std_y)

    particle_paths = []

    for i in range(n_particles):
        path, _ = compute_motion_data(width, height, t, v_x, v_y, y_accel, std_x, std_y, energy_loss, pf.particles[i])
        particle_paths.append(path)

    particle_paths = np.array(particle_paths)
    pf.particles = particle_paths

    animated_plot_particle(actual, measurements, pf, 
              x_range=[0, width], y_range=[0, height], 
              update_interval=50, full_plot=full_plot)




if __name__=='__main__':
    width = 1200
    height = 700

    v_x = 7 * choice([-1, 1])
    v_y = 10
    
    x_accel = 0
    y_accel = -1

    std_x = 20
    std_y = 20
    std_accel = 10

    energy_loss = 0.9

    dt = 0.1
    t = np.arange(0, 50, dt)

    init_state = np.array([randint(0, width), randint(height/2, height)])
    
    actual, measurements = compute_motion_data(width, height, t, v_x, v_y, y_accel, std_x, std_y, energy_loss, init_state)

    

    # use_kalman_filter(dt, v_x, v_y, x_accel, y_accel, std_x, std_y, std_accel, t, actual, measurements, full_plot=False)

    use_particle_filter(width, height, v_x, v_y, y_accel, std_x, std_y, t, actual, measurements, full_plot=False)
    
