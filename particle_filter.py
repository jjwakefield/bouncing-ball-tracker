import numpy as np
from numpy.linalg import inv
from random import random


# def mahalanobis_distance(particles, p, m):
#     cov = np.cov(particles.T)
#     mahal = (p - m).T @ inv(cov) @ (p - m)
#     return mahal




class ParticleFilter:

    def __init__(self, n_particles, init_state, std_x, std_y):
        self.n_particles = n_particles
        self.i = 0
        self.std = std_x

        self.particles = np.zeros((n_particles, 2))
        self.weights = np.array([1/n_particles] * n_particles)

        for i in range(n_particles):
            self.particles[i, 0] = init_state[0] + np.random.normal(0, std_x)
            self.particles[i, 1] = init_state[1] + np.random.normal(0, std_y)


    def reweight(self, m):
        cov_inv = inv(np.cov(self.particles.T))
        for i, p in enumerate(self.particles):
            delta = np.abs(p - m)
            mahal = np.dot(np.dot(delta, cov_inv), delta)
            mahal = np.sqrt(mahal)
            self.weights[i] = np.exp(-0.5 * mahal) * self.weights[i]

        self.weights /= np.sum(self.weights)
        

    def resample(self):
        '''
        Systematic resampling.
        '''
        N_eff = 1 / np.sum(self.weights**2)

        print(np.sum(self.weights**2))
        # print(N_eff, self.n_particles/2)

        if N_eff < self.n_particles/2:
            cumulative_sum = np.cumsum(self.weights)
            cumulative_sum[-1] = 1      # avoid round-off error
            indices = np.searchsorted(cumulative_sum, np.random.rand(self.n_particles))
            # print(f'{self.i}: ok')
            # Resample according to indices
            self.particles[:] = self.particles[indices]
            self.weights.fill(1 / self.n_particles)
            self.i += 1
