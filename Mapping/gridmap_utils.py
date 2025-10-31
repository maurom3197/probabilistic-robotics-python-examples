import math
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import skimage
from math import pi
from matplotlib import colors

def get_map(map_path, xy_reso):
    """"
    Load the image of the 2D map and convert into numpy ndarray with xy resolution
    """
    img = Image.open(map_path)
    # check that the image is loaded as 1 channel (grayscale)
    if img.mode != 'L':
        img = img.convert('L')  # convert to grayscale

    npmap = np.asarray(img, dtype=float)   
    print("Map size:", npmap.shape)

    # reduce the resolution: from the original map to the grid map using a max pooling operation
    grid_map = skimage.measure.block_reduce(npmap, (xy_reso, xy_reso), np.max)

    # convert to occupancy grid values (0: free, 1: occupied, -1: unknown)
    grid_map[grid_map==255] = 1  # occ space
    grid_map[grid_map==128] = -1 # unknown space
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
    n_o = np.sum(map==1)
    n_f = np.sum(map==0)
    n_u = map.shape[0]*map.shape[1] - n_o - n_f
    occ_poses = np.zeros((n_o, 2), dtype=int)
    free_poses = np.zeros((n_f, 2), dtype=int)
    unkn_poses = np.zeros((n_u, 2), dtype=int)
    map_poses = np.zeros((map.shape[0]*map.shape[1], 2), dtype=int)

    i=0 # index for occupied poses
    j=0 # index for all poses+
    k=0 # index for free poses

    for x in range(map.shape[0]):
        for y in range(map.shape[1]):
            if map[x, y] == 1: # occupied
                occ_poses[i,:] = x,y
                i+=1
            elif map[x, y] == 0: # free
                free_poses[k,:] = x,y
                k+=1
            else: # unknown
                unkn_poses[j-i-k,:] = x,y  

            map_poses[j,:] = x,y
            j+=1

    return occ_poses, free_poses, unkn_poses, map_poses

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

def plot_gridmap(map, robot_pose=None, ax=None):
    '''
    Plot the grid map
    Args:
        map: 2D numpy array representing the grid map
        robot_pose: (x, y, theta) robot pose to plot on the map
        ax: matplotlib axis to plot on (optional)
    Returns:
        pc: the pcolor object
    '''
    if ax is None:
        ax = plt.gca()

    gridmap_values = np.unique(map)
    if gridmap_values.size > 2:
        unknown_value = gridmap_values[(gridmap_values != 0) & (gridmap_values != 1)][0]
        # print("Map unknown_value:", unknown_value)
    else:
        unknown_value = 0.5  # just a value in [0, 1]

    cmap = {
        0.0:           'white',    # free space
        unknown_value: 'gray',     # unknown space
        1.0:           'black'     # occupied space
    }
    cmap = colors.ListedColormap([cmap[key] for key in sorted(cmap.keys())])
    # print("Colormap:", cmap.colors)

    pc = ax.pcolor(map[::-1], cmap=cmap, edgecolors='k', linewidths=0.8)

    if map.shape[0] < 20:
        ax.set_xticks(ticks=range(0, map.shape[1]+1), labels=range(0, map.shape[1]+1))
        ax.set_yticks(ticks=range(0, map.shape[0]+1), labels=range(map.shape[0], -1, -1))
    else:
        ax.set_xticks(ticks=range(0, map.shape[1]+1, 2), labels=range(0, map.shape[1]+1, 2))
        ax.set_yticks(ticks=range(0, map.shape[0]+1, 2), labels=range(map.shape[0], -1, -2))

    ax.set_aspect('equal')

    if robot_pose is not None:
        # unpack the first point
        x, y, theta = robot_pose[0], robot_pose[1], robot_pose[2]-math.pi/2
        print("Robot pose:", x, y, round(math.degrees(theta)))

        # find the end point
        endx = x - 1.0 * math.sin(theta)
        endy = y + 1.0 * math.cos(theta)

        ax.plot(robot_pose[1], map.shape[0]-robot_pose[0], 'or', ms=10)
        ax.plot([y, endy], [map.shape[0]-x, map.shape[0]-endx], linewidth = '2', color='r')
    
    return pc