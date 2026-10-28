import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

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

# Exponential Function
def exponential(x, _lambda):
    return _lambda * np.exp(-1 * _lambda * x)

# Gaussian Function
def gaussian(x, mu, sigma):
    return (1.0 / (np.sqrt(2*np.pi) * sigma)) * np.exp(-0.5 * ((x - mu) / sigma)**2)

def compute_p_hit_z(z, z_star, z_max, sigma):
    """
    Compute the hit probability p_hit for a given distance measurement using scipy.
    Args:
        z: observed distance measurement
        z_star: expected distance measurement
        z_max: maximum measurable distance
        sigma: standard deviation of the Gaussian noise
    Returns:
        p_hit: normalized hit probability
    """
    # Normalize the Gaussian PDF with the CDF probability computed over [0, z_max]
    # use scipy.stats.norm for the Gaussian PDF and CDF
    normalize_hit = norm.cdf(z_max, loc=z_star, scale=sigma) - norm.cdf(0, loc=z_star, scale=sigma)
    normalize_hit = 1. / normalize_hit

    p_hit = norm.pdf(z, loc=z_star, scale=sigma) * normalize_hit

    return p_hit

def compute_p_hit_dist(dist, max_dist, sigma):
    """
    Compute the hit probability p_hit for a given distance measurement using scipy.
    Args:
        dist: observed distance measurement
        max_dist: maximum measurable distance
        sigma: standard deviation of the Gaussian noise
    Returns:
        p_hit: normalized hit probability
    """

    # Normalize the Gaussian PDF with the CDF probability computed over [0, max_dist]
    # normalize_hit = norm.cdf(max_dist, loc=0, scale=sigma) - norm.cdf(0, loc=0, scale=sigma)
    # normalize_hit = 1. / normalize_hit

    p_hit = norm.pdf(dist, loc=0, scale=sigma) 
    # p_hit *= normalize_hit
    p_hit = np.where((dist > max_dist) | (dist < 0.0), 0.0, p_hit)  # Set p_hit to 0 for distances greater than max_dist or negative anomalies

    return p_hit


# Vectorized probabilistic likelihood field model
def precompute_p_hit_map(distances, max_dist=None, sigma=1.0):
    """
    Vectorized probabilistic model for computing gaussian hit probabilities of distances for likelihood fields over a grid map.
    Supports scalar or ndarray inputs for distances (must be broadcastable).
    
    Args:
        dist:     observed distances measurement(s), ndarray or float
        z_max:    max range (scalar)
        sigma:    std dev for hit Gaussian

    Returns:
        p_hit (hit prob)
    """
    dist_flat = np.asarray(distances.ravel())

    if max_dist is None:
        max_dist = np.max(dist_flat)
    
    # To compute the hit probability p_hit for a given distance measurement
    # we need to normalize the Gaussian PDF with the CDF probability computed over [0, max_dist]
    normalize_hit = norm.cdf(max_dist, loc=0, scale=sigma) - norm.cdf(0, loc=0, scale=sigma)
    normalize_hit = 1. / normalize_hit

    # Hit probability
    p_hit = norm.pdf(distances, loc=np.zeros_like(distances), scale=sigma) * normalize_hit
    p_hit = np.where((distances > max_dist) | (distances < 0.0), 0.0, p_hit)  # Set p_hit to 0 for distances greater than max_dist or negative anomalies
    return p_hit

# Vectorized probabilistic beam model
def evaluate_range_beam_dist_array(z, z_star, z_max, _mix_density, _sigma, _lamb_short):
    """
    Vectorized probabilistic beam model.
    Supports scalar or ndarray inputs for z and z_star (must be broadcastable).
    
    Args:
        z:        observed measurement(s), ndarray or float
        z_star:   expected measurement(s), ndarray or float
        z_max:    max range (scalar)
        _mix_density: mixture weights [z_hit, z_short, z_max, z_rand]
        _sigma:   std dev for hit Gaussian
        _lamb_short: lambda for short distribution

    Returns:
        p_hit, p_short, p_max, p_rand, p (stacked array), p_z (mixture prob)
        Shapes follow broadcast of z and z_star
    """

    z = np.asarray(z)
    z_star = np.asarray(z_star)

    # --- Hit mode normalization ---
    normalize_hit = norm.cdf(z_max, loc=z_star, scale=_sigma) - norm.cdf(0, loc=z_star, scale=_sigma)
    normalize_hit = 1. / normalize_hit

    # Hit probability
    p_hit = norm.pdf(z, loc=z_star, scale=_sigma) * normalize_hit

    # --- Short mode ---
    normalize_short = 1.0 - np.exp(-_lamb_short * z_star)
    p_short = np.where(
        z <= z_star,
        _lamb_short * np.exp(-_lamb_short * z) / (normalize_short + 1e-12),
        0.0
    )

    # --- Max mode ---
    p_max = np.where(z == z_max, 1.0, 0.0)

    # --- Random mode ---
    p_rand = np.full_like(z, 1.0 / z_max, dtype=float)

    # Stack all components: shape (..., 4)
    p = np.stack([p_hit, p_short, p_max, p_rand], axis=-1)

    # Weighted mixture probability
    p_z = np.tensordot(p, _mix_density, axes=([-1],[0]))

    return p_hit, p_short, p_max, p_rand, p_z

# Plot the distribution of z samples
def plot_sampling_dist(samples, title="Distribution of z samples", fig_name="z_star_hist.pdf"):
    '''
    Plot the distribution of z samples.
    Args:
        samples: array of z samples
        title: title of the plot
        fig_name: name of the file to save the plot
    '''
    
    n_bins = 100
    plt.hist(samples, n_bins)
    plt.title(title)
    plt.grid()
    plt.savefig(fig_name)
    plt.show()
    plt.close('all')


# Bresenham's line algorithm https://en.wikipedia.org/wiki/Bresenham%27s_line_algorithm
def bresenham_v0(x0, y0, x1, y1):
    ret = []

    dx =  abs(x1-x0)
    sx = 1 if (x0<x1) else -1
    dy = -abs(y1-y0)
    sy = 1 if (y0<y1) else -1
    err = dx+dy

    while (True):
        ret.append((x0, y0))
        if (x0==x1 and y0==y1):
            break
        e2 = 2*err
        if (e2 >= dy):
            err += dy
            x0 += sx
        if (e2 <= dx):
            err += dx
            y0 += sy

    return ret


def bresenham(x0, y0, x1, y1, map):
    """""
    x0, y0: coordinate of the starting point (robot position)
    x1, y1: map coordinate of the max range point
    map: 2D occupancy grid map
    return: coordinate of the obstacle point or the map boundary point
    """""

    dx =  abs(x1 - x0)
    sx = 1 if (x0 < x1) else -1
    dy = -abs(y1 - y0)
    sy = 1 if (y0 < y1) else -1
    err = dx + dy

    while (True):
        # check if obstacle encountered or ray reach end of map
        if x0 < 0.:
            obst = 0, y0
            break
        elif x0 >= map.shape[0]: # check if map border reached
            obst = x0, y0
            break
        elif y0 < 0.:
            obst = x0, 0
            break
        elif y0 >= map.shape[1]:
            obst = x0, y0
            break
        # elif map[int(x0-1), int(y0-1)]==1 or map[int(x0), int(y0)]==1 or map[int(x0-1), int(y0)]==1 or map[int(x0), int(y0-1)]==1 or ((x0==x1) and (y0==y1)):
        #     obst = [x0, y0]
        #     print("obst", obst)
        #     break
        elif map[int(x0), int(y0)]==1 or ((x0==x1) and (y0==y1)):
            obst = [x0, y0]
            break

        e2 = 2*err
        if (e2 >= dy):
            err += dy
            x0 += sx
        if (e2 <= dx):
            err += dx
            y0 += sy
        
    return obst
