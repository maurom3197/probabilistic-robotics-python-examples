import matplotlib.pyplot as plt
import numpy as np
import math
from matplotlib import colors

def plot_initial_particles(N, particles, ax=None):
    if ax is None:
        ax = plt.gca()
    alpha = .20
    if N > 5000:
        alpha *= np.sqrt(5000)/np.sqrt(N)           
    legend = ax.scatter(particles[:, 0], particles[:, 1], 
                alpha=alpha, color='g', label='Init Particles')
    return legend
    
def plot_particles(particles, robot_pose, mu, ax=None, color='k', alpha=1):
    if ax is None:
        ax = plt.gca()

    ax.scatter(particles[:, 0], particles[:, 1], 
                        color=color, marker=',', s=1, alpha=alpha)
    p1 = ax.scatter(robot_pose[0], robot_pose[1], marker='+',
                        color='k', s=180, lw=3, alpha=alpha, label='True robot pose')
    p2 = ax.scatter(mu[0], mu[1], marker='s', color='r', alpha=alpha, label='PF mean')

    return p1, p2

def plot_gridmap(map, ax=None):
    if ax is None:
        ax = plt.gca()

    cmap = colors.ListedColormap(['White','Black'])
    ax.pcolor(map[::-1],cmap=cmap, edgecolors='k', linewidths=0.8)

    if map.shape[0] < 30:
        ax.set_xticks(ticks=range(map.shape[1]+1), labels=range(0, map.shape[1]+1))
        ax.set_yticks(ticks=range(map.shape[0]), labels=range(map.shape[0], 0, -1))
    else:
        ax.axis('off')

def plot_initial_particles_gridmap(N, particles, map_shape, ax=None):
    if ax is None:
        ax = plt.gca()
    alpha = .20
    if N > 5000:
        alpha *= np.sqrt(5000)/np.sqrt(N)           
    legend = ax.scatter(particles[:, 1], map_shape[0]-particles[:, 0], 
                alpha=alpha, color='g', label='Init Particles')
    return legend
    
def plot_particles_gridmap(particles, robot_pose, mu, map_shape, ax=None, color='k', alpha=1):
    if ax is None:
        ax = plt.gca()

    ax.scatter(particles[:, 1], map_shape[0]-particles[:, 0], 
                        color=color, marker=',', s=1, alpha=alpha)
    p1 = ax.scatter(robot_pose[1], map_shape[0]-robot_pose[0], marker='+',
                        color='k', s=180, lw=3, alpha=alpha, label='True robot pose')
    p2 = ax.scatter(mu[1], map_shape[0]-mu[0], marker='s', color='r', alpha=alpha, label='PF mean')

    return p1, p2