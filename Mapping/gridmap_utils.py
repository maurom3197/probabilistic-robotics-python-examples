import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from PIL import Image
import skimage


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


def plot_gridmap(map, robot_pose=None, ax=None):
    if ax is None:
        ax = plt.gca()

    # cmap = colors.ListedColormap(['White', 'Gray','Black'])
    pc = ax.pcolor(map[::-1], cmap='Greys', edgecolors='k', linewidths=0.8)

    if map.shape[0] < 30:
        ax.set_xticks(ticks=range(0, map.shape[1]+1), labels=range(0, map.shape[1]+1, 1))
        ax.set_yticks(ticks=range(map.shape[0]), labels=range(map.shape[0], 0, -1))
    else:
        ax.axis('off')

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


def plot_gridmap_plt(map, title, robot_pose=None):
    cmap = colors.ListedColormap(['White','Gray','Black'])
    plt.figure(figsize=(6,6))
    plt.pcolor(map[::-1],cmap=cmap, edgecolors='k', linewidths=1)

    if map.shape[0] < 20:
        plt.xticks(ticks=range(map.shape[1]+1), labels=range(map.shape[1]+1))
        plt.yticks(ticks=range(map.shape[0]), labels=range(map.shape[0], 0, -1))
    plt.axis('equal')
    plt.title(title, fontsize = 14)

    if robot_pose is not None:
        # unpack the first point
        x, y = robot_pose[0], robot_pose[1]
        # print("Robot pose:", x, y, round(math.degrees(robot_pose[2]-math.pi/2)))

        # find the end point
        endx = x - 1.0 * math.sin(robot_pose[2]-math.pi/2)
        endy = y + 1.0 * math.cos(robot_pose[2]-math.pi/2)

        plt.plot(robot_pose[1], map.shape[0]-robot_pose[0], 'or', ms=10)
        plt.plot([y, endy], [map.shape[0]-x, map.shape[0]-endx], linewidth = '2', color='r')