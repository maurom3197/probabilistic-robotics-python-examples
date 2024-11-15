import numpy as np
from math import atan2
from numpy import linalg as la

def residual(a, b, **kwargs):
    """
    Compute the residual between expected and sensor measurements, normalizing angles between [-pi, pi)
    If passed, angle_indx should indicate the positional index of the angle in the measurement arrays a and b

    Returns:
        y [np.array] : the residual between the two states
    """
    y = a - b

    if 'angle_idx' in kwargs:
        angle_idx = kwargs["angle_idx"]
        theta = y[angle_idx]
        y[angle_idx] = normalize_angle(theta)
        
    return y

def normalize_angle(theta):
    """
    Normalize angles between [-pi, pi)
    """
    theta = theta % (2 * np.pi)  # force in range [0, 2 pi)
    if theta > np.pi:  # move to [-pi, pi)
        theta -= 2 * np.pi
    
    return theta


def initial_gaussian_particles(N, dim_x, init_pose, std, angle_idx=None):
    """
    Initialize particles in case of known initial pose: use a Gaussian distribution
    """
    particles = np.empty((N, dim_x))
    for i in range(dim_x):
        particles[:, i] = np.random.normal(init_pose[i], std[i], N)

    if angle_idx is not None:
        particles[:, angle_idx] = normalize_angle(particles[:, angle_idx])
    return particles

def state_mean(particles, weights, **kwargs):
    dim_x = particles.shape[1]
    x = np.zeros(dim_x)
    idx_list = list(range(dim_x))

    if 'angle_idx' in kwargs:
        angle_idx = kwargs["angle_idx"]
        
        sum_sin = np.average(np.sin(particles[:, angle_idx]), axis=0, weights=weights)
        sum_cos = np.average(np.cos(particles[:, angle_idx]), axis=0, weights=weights)
        x[angle_idx] = atan2(sum_sin, sum_cos)
        idx_list.remove(angle_idx)

    for i in idx_list:
        x[i] = np.average(particles[:, i], axis=0, weights=weights)

    return x


def simple_resample(weights):
    N = len(weights)
    cumulative_sum = np.cumsum(weights)
    cumulative_sum[-1] = 1. # avoid round-off error
    indexes = np.searchsorted(cumulative_sum, np.random.random(N))
    return indexes


def residual_resample(weights):
    N = len(weights)
    indexes = np.zeros(N, 'i')

    # take int(N*w) copies of each weight
    num_copies = (N*np.asarray(weights)).astype(int)
    k = 0
    for i in range(N):
        for _ in range(num_copies[i]): # make n copies
            indexes[k] = i
            k += 1

    # use multinormial resample on the residual to fill up the rest.
    residual = w - num_copies     # get fractional part
    residual /= sum(residual)     # normalize
    cumulative_sum = np.cumsum(residual)
    cumulative_sum[-1] = 1. # ensures sum is exactly one
    indexes[k:N] = np.searchsorted(cumulative_sum, np.random.random(N-k))
    return indexes

def stratified_resample(weights):
    N = len(weights)
    # make N subdivisions, chose a random position within each one
    positions = (np.random.random(N) + range(N)) / N

    indexes = np.zeros(N, 'i')
    cumulative_sum = np.cumsum(weights)
    i, j = 0, 0
    while i < N:
        if positions[i] < cumulative_sum[j]:
            indexes[i] = j
            i += 1
        else:
            j += 1
    return indexes


def systematic_resample(weights):
    N = len(weights)

    # make N subdivisions, choose positions 
    # with a consistent random offset
    positions = (np.arange(N) + np.random.random()) / N

    indexes = np.zeros(N, 'i')
    cumulative_sum = np.cumsum(weights)
    i, j = 0, 0
    while i < N:
        if positions[i] < cumulative_sum[j]:
            indexes[i] = j
            i += 1
        else:
            j += 1
    return indexes