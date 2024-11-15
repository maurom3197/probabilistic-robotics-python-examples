import matplotlib.pyplot as plt
import numpy as np

def plot_initial_particles(N, particles, ax=None):
    if ax is None:
        ax = plt.gca()
    alpha = .20
    if N > 5000:
        alpha *= np.sqrt(5000)/np.sqrt(N)           
    ax.scatter(particles[:, 0], particles[:, 1], 
                alpha=alpha, color='g')
    
def plot_particles(particles, robot_pose, mu, ax=None, color='k', alpha=1):
    if ax is None:
        ax = plt.gca()

    ax.scatter(particles[:, 0], particles[:, 1], 
                        color=color, marker=',', s=1, alpha=alpha)
    p1 = ax.scatter(robot_pose[0], robot_pose[1], marker='+',
                        color='k', s=180, lw=3, alpha=alpha, label='Actual robot pose')
    p2 = ax.scatter(mu[0], mu[1], marker='s', color='r', alpha=alpha, label='PF mean')

    return p1, p2