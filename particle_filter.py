import numpy as np
from numpy.linalg import pinv
from resampling import *



def mahalanobis_distance(p, m, cov_inv):
    '''
    Computes the Mahalanobis distance between two points (1-D arrays)

    dist = sqrt( (u-v) V^{-1} (u-v)^T )
    '''
    delta = p - m
    mahal = np.dot(np.dot(delta, cov_inv), delta)
    return np.sqrt(mahal)



class ParticleFilter:

    def __init__(self, n_particles, init_state, std_x, std_y):
        self.n_particles = n_particles

        self.particles = np.zeros((n_particles, 2))
        self.weights = np.array([1/n_particles] * n_particles)

        for i in range(n_particles):
            self.particles[i, 0] = init_state[0] + np.random.normal(0, std_x)
            self.particles[i, 1] = init_state[1] + np.random.normal(0, std_y)

            
    def reweight(self, m, k):
        '''
        Reweighting based on mahalanobis distance between particles
        and measurement.

        Parameters
        ----------
        m : Measurement
        k : Current timestep
        '''
        cov_inv = pinv(np.cov(self.particles[:, k].T))
        for i, p in enumerate(self.particles[:, k]):
            mahal = mahalanobis_distance(p, m, cov_inv)
            self.weights[i] = np.exp(-0.5 * mahal) * self.weights[i]
        self.weights /= np.sum(self.weights)
        
        
    def resample(self):
        '''
        Systematic resampling to resolve degeneracy problem.
        '''
        N_eff = 1 / np.sum(self.weights**2)

        # Resample if too few effective particles
        print(f'{N_eff:.3f}')
        if N_eff < self.n_particles/2:
            print(f'resampling...')
            self.particles, self.weights = systematic_resampling(self.particles, self.weights)
