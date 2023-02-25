import numpy as np
import matplotlib.pyplot as plt
from numpy.random import normal
from kalman import KalmanFilter2D

# Set up the initial state of the ball
pos = np.array([0, 10]) # Starting position
vel = np.array([5, -9.81]) # Starting velocity
acc = np.array([0, 9.81]) # Acceleration due to gravity
energy_loss = 0.9 # Energy loss on collision
dt = 0.01 # Time step

x_range = [0, 10] # x range
y_range = [0, 20] # y range
offset = 2 # Offset for plotting

std_x = 1.0 # Standard deviation of x measurement noise
std_y = 1.0 # Standard deviation of y measurement noise
std_accel = 10 # Standard deviation of acceleration noise
n_iters = 100 # Number of iterations


# Define the observation function
def observe(pos):
    # Simulate measurement noise
    noise = [normal(scale=std_x), normal(scale=std_y)]
    return pos + noise

# Define the transition function
def transition(pos, vel, acc, dt):
    # Apply gravity
    vel = vel + acc * dt
    # Update position
    pos = pos + vel * dt
    # Bounce off the ground or ceiling
    if pos[1] < y_range[0] or pos[1] > y_range[1]:
        vel[1] = -vel[1] * energy_loss
    # Bounce off the walls
    if pos[0] < x_range[0] or pos[0] > x_range[1]:
        vel[0] = -vel[0] * energy_loss
    return pos, vel


positions = np.array([pos]) # True positions
measurements = np.array([observe(pos)]) # Measured positions
filtered = np.empty((0, 2)) # Filtered positions

# Set up the initial state of the Kalman filter
init_state = np.array([[measurements[0, 0]],
                       [measurements[0, 1]],
                       [vel[0]],
                       [vel[1]]])

# Create the Kalman filter
kf = KalmanFilter2D(init_state=init_state,
                    dt=dt,
                    u_x=acc[0], u_y=acc[1],
                    std_accel=std_accel,
                    x_std_meas=std_x, y_std_meas=std_y)


# Run the simulation
for i in range(1, n_iters):
    # Move the ball
    pos, vel = transition(pos, vel, acc, dt)
    positions = np.concatenate((positions, np.array([pos])))

    # Predict
    kf.predict()

    # Measure
    measured_pos = observe(pos)
    measurements = np.concatenate((measurements, np.array([measured_pos])))

    # Update
    z = np.vstack(measured_pos)
    est = kf.update(z)
    filtered = np.concatenate((filtered, np.array([est])))

    # Plot the results
    plt.clf()
    plt.plot(positions[:, 0], positions[:, 1], 'r', lw=3, label='True') # True position
    plt.scatter(measurements[:, 0], measurements[:, 1], c='b', marker='x', label='Measured') # Measured position
    plt.plot(filtered[:, 0], filtered[:, 1], 'g', lw=3, label='Filtered') # Filtered position
    plt.xlim([x_range[0] - offset, x_range[1] + offset])
    plt.ylim([y_range[0] - offset, y_range[1] + offset])
    plt.gca().invert_yaxis()
    plt.legend()
    plt.grid()
    plt.title('Bouncing Ball')
    plt.pause(0.01)

    # Remove old data to clean up the plot
    if i >= 50:
        positions = positions[1:]
        measurements = measurements[1:]
        filtered = filtered[1:]

plt.show()
