import numpy as np
from numpy.random import normal, random, uniform
from numpy.linalg import inv
import matplotlib.pyplot as plt



def mahalanobis_distance(p, m, cov_inv):
    '''Computes the Mahalanobis distance between two points (1-D arrays)'''
    delta = p - m
    mahal = np.dot(np.dot(delta, cov_inv), delta)
    return np.sqrt(mahal)


class Ball:

    def __init__(self, pos, vel, acc, energy_loss, dt, x_range, y_range, std_x, std_y):
        self.pos = pos
        self.vel = vel
        self.acc = acc
        self.energy_loss = energy_loss
        self.dt = dt
        self.x_range = x_range
        self.y_range = y_range
        self.std_x = std_x
        self.std_y = std_y

    def transition(self):
        # Apply gravity
        self.vel = self.vel + self.acc * self.dt
        # Update position
        self.pos = self.pos + self.vel * self.dt

        # Bounce off the ground or ceiling
        if self.pos[1] < self.y_range[0] or self.pos[1] > self.y_range[1]:
            self.vel[1] = -self.vel[1] * self.energy_loss
            # Make sure the ball doesn't go through the ground or ceiling
            if self.pos[1] < self.y_range[0]:
                self.pos[1] = self.y_range[0]
            else:
                self.pos[1] = self.y_range[1]

        # Bounce off the walls
        if self.pos[0] < self.x_range[0] or self.pos[0] > self.x_range[1]:
            self.vel[0] = -self.vel[0] * self.energy_loss
            # Make sure the ball doesn't go through the walls
            if self.pos[0] < self.x_range[0]:
                self.pos[0] = self.x_range[0]
            else:
                self.pos[0] = self.x_range[1]

        return self.pos

    def observation(self):
        # Simulate measurement noise
        noise = [normal(scale=self.std_x), normal(scale=self.std_y)]
        return self.pos + noise


class Cluster:

    def __init__(self, n_particles, vel, acc, energy_loss, dt, x_range, y_range, std_x, std_y):
        self.positions = np.empty((n_particles, 2))
        self.velocities = np.empty((n_particles, 2))
        self.weights = np.ones(n_particles) / n_particles

        for i in range(n_particles):
            self.positions[i] = uniform(low=[x_range[0], y_range[0]], high=[x_range[1], y_range[1]])
            self.velocities[i] = vel

        self.est = np.average(self.positions, axis=0, weights=self.weights)
        self.inv_cov = inv(np.cov(self.positions.T))

        self.n_particles = n_particles
        self.x_range = x_range
        self.y_range = y_range
        self.energy_loss = energy_loss
        self.dt = dt
        self.acc = acc
        self.std_x = std_x
        self.std_y = std_y

    def propagate(self):
        for i in range(self.n_particles):
            # Apply gravity
            self.velocities[i] = self.velocities[i] + self.acc * self.dt
            # Update position
            self.positions[i] = self.positions[i] + self.velocities[i] * self.dt
            # Bounce off the ground or ceiling
            if self.positions[i, 1] < self.y_range[0] or self.positions[i, 1] > self.y_range[1]:
                self.velocities[i, 1] = -self.velocities[i, 1] * self.energy_loss
                # Make sure the ball doesn't go through the ground or ceiling
                if self.positions[i, 1] < self.y_range[0]:
                    self.positions[i, 1] = self.y_range[0]
                else:
                    self.positions[i, 1] = self.y_range[1]
            # Bounce off the walls
            if self.positions[i, 0] < self.x_range[0] or self.positions[i, 0] > self.x_range[1]:
                self.velocities[i, 0] = -self.velocities[i, 0] * self.energy_loss
                # Make sure the ball doesn't go through the walls
                if self.positions[i, 0] < self.x_range[0]:
                    self.positions[i, 0] = self.x_range[0]
                else:
                    self.positions[0] = self.x_range[1]

    def update(self, measurement):
        self.inv_cov = inv(np.cov(self.positions.T))
        for i, p in enumerate(self.positions):
            mahal = mahalanobis_distance(p, measurement, self.inv_cov)
            self.weights[i] = np.exp(-0.5 * mahal) * self.weights[i]
        self.weights /= np.sum(self.weights)

    def resample(self):
        N_eff = 1 / np.sum(self.weights**2) # Effective sample size
        if N_eff < self.n_particles / 2:
            positions = (random() + np.arange(self.n_particles)) / self.n_particles
            indices = np.zeros(self.n_particles, 'i')
            cumulative_sum = np.cumsum(self.weights)

            i, j = 0, 0
            while i < self.n_particles:
                if positions[i] < cumulative_sum[j]:
                    indices[i] = j
                    i += 1
                else:
                    j += 1

            self.positions[:] = self.positions[indices]
            self.weights.resize(self.n_particles)
            self.weights.fill(1.0/self.n_particles)
            self.positions = self.positions + (random(self.positions.shape) - 0.5)

    def estimate(self):
        self.est = np.average(self.positions, axis=0, weights=self.weights)
        return self.est


class ParticleFilter:

    def __init__(self, n_particles, n_targets, vel, acc, energy_loss, dt, x_range, y_range, std_x, std_y):
        particles_per_cluster = int(n_particles/n_targets)
        self.clusters = [Cluster(particles_per_cluster, vel, acc, energy_loss, dt, x_range, y_range, std_x, std_y) for _ in range(n_targets)]

    def propagate(self):
        for cluster in self.clusters:
            cluster.propagate()

    def update(self, measurements):
        for cluster, measurement in zip(self.clusters, measurements):
            cluster.update(measurement)

    def resample(self):
        for cluster in self.clusters:
            cluster.resample()

    def estimate(self):
        return np.array([cluster.estimate() for cluster in self.clusters])
        
    


acc = np.array([0, 9.81]) # Acceleration due to gravity
vel = np.array([5, -9.81]) # Starting velocity
energy_loss = 0.9 # Energy loss on collision
dt = 0.01 # Time step

x_range = [0, 10] # x range
y_range = [0, 20] # y range
offset = 2 # Offset for plotting

std_x = 1 # Standard deviation of x measurement noise
std_y = 1 # Standard deviation of y measurement noise

n_iters = 1000 # Number of iterations
n_particles = 160 # Number of particles
n_balls = 4 # Number of balls

balls = np.empty(n_balls, dtype=Ball)
for i in range(n_balls):
    pos = uniform(low=[x_range[0], y_range[0]], high=[x_range[1], y_range[1]])
    ball = Ball(pos, vel, acc, energy_loss, dt, x_range, y_range, std_x, std_y)
    balls[i] = ball
ball_colors = ['k', 'c', 'm', 'y']

# Create the particle filter
pf = ParticleFilter(n_particles, n_balls, vel, acc, energy_loss, dt, x_range, y_range, std_x, std_y)

for i in range(n_iters):
    ball_positions = np.empty((n_balls, 2)) # True positions of the balls
    measurements = np.empty((n_balls, 2)) # Measurements

    for j, ball in enumerate(balls):
        # Simulate the motion of the balls
        ball_positions[j] = ball.transition()
        # Measure the ball
        measurements[j] = ball.observation()

    # Simulate the particle filter
    pf.propagate()
    pf.update(measurements)
    pf.resample()
    est_positions = pf.estimate()
    

    # Plot the results
    plt.clf()
    for j, cluster in enumerate(pf.clusters):
        plt.scatter(cluster.positions[:, 0], cluster.positions[:, 1], c='r', s=cluster.weights*1000, label=f'Particles')
    plt.scatter(ball_positions[:, 0], ball_positions[:, 1], c='k', marker='o', label=f'Target') # True position of the ball
    plt.scatter(measurements[:, 0], measurements[:, 1], c='b', marker='x', label='Measurement') # Measured position
    plt.scatter(est_positions[:, 0], est_positions[:, 1], c='g', marker='o', label='Estimate') # Estimated position
    plt.xlim([x_range[0] - offset, x_range[1] + offset])
    plt.ylim([y_range[0] - offset, y_range[1] + offset])
    plt.gca().invert_yaxis()
    plt.legend(loc='upper right')
    plt.grid()
    plt.title('Bouncing Ball')
    plt.pause(0.01)
