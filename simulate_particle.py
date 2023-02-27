import numpy as np
import matplotlib.pyplot as plt
from numpy.random import normal
from numpy.linalg import inv
from resampling import systematic_resampling

# Set up the initial state of the ball
pos = np.array([0, 10]) # Starting position
vel = np.array([5, -9.81]) # Starting velocity
acc = np.array([0, 9.81]) # Acceleration due to gravity
energy_loss = 0.9 # Energy loss on collision
dt = 0.01 # Time step

x_range = [0, 10] # x range
y_range = [0, 20] # y range
offset = 2 # Offset for plotting

std_x = 1 # Standard deviation of x measurement noise
std_y = 1 # Standard deviation of y measurement noise
n_iters = 1000 # Number of iterations
n_particles = 100 # Number of particles


def observe(pos):
    '''Simulates the observation function'''
    # Simulate measurement noise
    noise = [normal(scale=std_x), normal(scale=std_y)]
    return pos + noise


def transition(pos, vel, acc, dt):
    '''Simulates the transition function'''
    # Apply gravity
    vel = vel + acc * dt
    # Update position
    pos = pos + vel * dt
    # Bounce off the ground or ceiling
    if pos[1] < y_range[0] or pos[1] > y_range[1]:
        vel[1] = -vel[1] * energy_loss
        # Make sure the ball doesn't go through the ground or ceiling
        if pos[1] < y_range[0]:
            pos[1] = y_range[0]
        else:
            pos[1] = y_range[1]
    # Bounce off the walls
    if pos[0] < x_range[0] or pos[0] > x_range[1]:
        vel[0] = -vel[0] * energy_loss
        # Make sure the ball doesn't go through the walls
        if pos[0] < x_range[0]:
            pos[0] = x_range[0]
        else:
            pos[0] = x_range[1]
    return pos, vel


def mahalanobis_distance(p, m, cov_inv):
    '''Computes the Mahalanobis distance between two points (1-D arrays)'''
    delta = p - m
    mahal = np.dot(np.dot(delta, cov_inv), delta)
    return np.sqrt(mahal)


# Initialise particles randomly
particles = np.random.uniform(low=[x_range[0], y_range[0]], high=[x_range[1], y_range[1]], size=(n_particles, 2))
particle_vel = np.empty((n_particles, 2)) # Particle velocities
for i in range(n_particles):
    particle_vel[i] = vel

weights = np.ones(n_particles) / n_particles # Particle weights

error = [] # Error between the true position and the estimated position

# Run the simulation
for i in range(1, n_iters):
    # Move the ball
    pos, vel = transition(pos, vel, acc, dt)

    for j in range(n_particles):
        # Move the particles
        particles[j], particle_vel[j] = transition(particles[j], particle_vel[j], acc, dt)

    # Measure
    measured_pos = observe(pos)

    # Reweight
    cov_inv = inv(np.cov(particles.T))
    for j, p in enumerate(particles):
        mahal = mahalanobis_distance(p, measured_pos, cov_inv)
        weights[j] = np.exp(-0.5 * mahal) * weights[j]
    weights /= np.sum(weights)
    
    # Resample
    N_eff = 1 / np.sum(weights**2) # Effective sample size
    if N_eff < n_particles / 2:
        print(f'resampling at iteration {i}')
        particles, weights = systematic_resampling(particles, weights)

    # Estimate the position of the ball
    est_pos = np.average(particles, axis=0, weights=weights)

    # Error
    error.append(np.linalg.norm(pos - est_pos))

    # Plot the results
    plt.clf()
    plt.plot(pos[0], pos[1], 'ko', label='Ball') # True position of the ball
    plt.scatter(measured_pos[0], measured_pos[1], c='b', marker='x', label='Measurement') # Measured position
    plt.plot(est_pos[0], est_pos[1], 'go', label='Estimate') # Estimated position
    plt.scatter(particles[:, 0], particles[:, 1], marker='.', c='r', s=weights*10000, label='Particles') # Particles
    plt.xlim([x_range[0] - offset, x_range[1] + offset])
    plt.ylim([y_range[0] - offset, y_range[1] + offset])
    plt.gca().invert_yaxis()
    plt.legend(loc='upper right')
    plt.grid()
    plt.title('Bouncing Ball')
    plt.pause(0.01)

plt.show()

# Plot the error
error = (error - np.min(error)) / (np.max(error) - np.min(error)) # Normalise the error
plt.plot(error)
plt.title('Error')
plt.xlabel('Iteration')
plt.ylabel('Error')
plt.show()
