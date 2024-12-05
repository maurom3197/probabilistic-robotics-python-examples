import numpy as np
from math import atan2
from PIL import Image
import skimage
from plot_utils import plot_gridmap

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
    if np.isscalar(theta):
        if theta > np.pi:  # move to [-pi, pi)
            theta -= 2 * np.pi
    else:
        theta_ = theta.copy()
        theta_[theta>np.pi] -= 2 * np.pi
        return theta_
    
    return theta

def initial_uniform_particles_gridmap(N, dim_x, boundaries, occ_grid):
    """
    Initialize particles uniformly according to a given occupancy gridmap.
    Rejection sampling is used until N valid particles are generated.
    """
    particle = np.zeros((dim_x))  # particles
    particles = np.zeros((N, dim_x))  # particles
    n_particles = 0 # number of valid particles

    while n_particles != (N-1):
        
        for i in range(dim_x):
            particle[i] = np.random.uniform(boundaries[i][0], boundaries[i][1])
        # check that sampled particles lie on free space on the grid map
        xi, yi = int(particle[0]), int(particle[1])
        if occ_grid[xi, yi] == 0: # check that the corresponding map cell is free
            particles[n_particles] = particle
            n_particles += 1 

    return particles

def initial_uniform_particles_gridmap_from_free_spaces(N, dim_x, free_spaces):
    """
    Initialize particles uniformly according to a given occupancy gridmap.
    Rejection sampling is used until N valid particles are generated.
    """
    particles = np.zeros((N, dim_x))  # particles

    for i in range(N):
        idx = np.random.choice(np.arange(len(free_spaces), dtype=int))
        particles[i, 0:2] = np.random.uniform(0.05, 0.95) + free_spaces[idx]
        particles[i, 2] = np.random.uniform(-np.pi, np.pi)

    return particles

def initial_gaussian_particles(N, dim_x, init_pose, std, angle_idx=None, map=None):
    """
    Initialize particles in case of known initial pose: use a Gaussian distribution
    """
    particles = np.zeros((N, dim_x))
    particle = np.zeros((dim_x))  # particles
    n_particles = 0 # number of valid particles

    if map is None:
        for i in range(dim_x):
            particles[:, i] = np.random.normal(init_pose[i], std[i], N)
    else:
        while n_particles != (N-1):
        
            for i in range(dim_x):
                particle[i] = np.random.normal(init_pose[i], std[i])
            # check that sampled particles lie on free space on the grid map
            xi, yi = int(particle[0]), int(particle[1])

            if (xi>=0 and yi>=0 and xi<map.shape[0] and yi<map.shape[1] and map[xi, yi] == 0):
                particles[n_particles] = particle
                n_particles += 1 

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

    # use multinormal resample on the residual to fill up the rest.
    residual = weights - num_copies     # get fractional part
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


def get_map(map_path, xy_reso, plot_map=False):
    """"
    Load the image of the 2D map and convert into numpy ndarray with xy resolution
    """
    img = Image.open(map_path)
    npmap = np.asarray(img, dtype=int)
    print("Map size:", npmap.shape)

    if plot_map:
        plot_gridmap(npmap, 'Full Map')

    # reduce the resolution: from the original map to the grid map using a max pooling operation
    grid_map = skimage.measure.block_reduce(npmap, (xy_reso, xy_reso), np.max)
    print("Grid Map size:", grid_map.shape)

    return npmap, grid_map

def calc_grid_map_config(map_size, xyreso):
    minx = 0
    miny = 0
    maxx = map_size[0]
    maxy = map_size[1]
    xw = int(round((maxx - minx) / xyreso))
    yw = int(round((maxy - miny) / xyreso))

    return xw, yw

def compute_map_occ(map):
    """""
    Compute occupancy state for each cell of the gridmap 
    Possible states: 
      - occupied = 1 (obstacle present)
      - free = 0
      - unknown = not defined (usually -1)
    Returns two np arrays with poses of the obstacles in the map and all the map poses.
    It supports the pre-computation of likelihood field over the entire map
    """""
    n_o = np.count_nonzero(map)
    n_f = map.shape[0]*map.shape[1] - n_o
    occ_poses = np.zeros((n_o, 2), dtype=int)
    free_poses = np.zeros((n_f, 2), dtype=int)
    map_poses = np.zeros((map.shape[0]*map.shape[1], 2), dtype=int)

    i=0
    j=0
    for x in range(map.shape[0]):
        for y in range(map.shape[1]):
            if map[x, y] == 1:
                occ_poses[i,:] = x,y
                i+=1
            else:
                free_poses[j-i,:] = x,y
            
            map_poses[j,:] = x,y
            j+=1

    return occ_poses, free_poses, map_poses