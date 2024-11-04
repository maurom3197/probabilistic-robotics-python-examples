import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from PIL import Image
import skimage


def get_map(map_path, xy_reso, plot_map=False):
    # load the image and convert into
    # numpy array
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
    
    n_o = np.count_nonzero(map)
    obst_poses = np.zeros((n_o, 2), dtype=int)
    end_points = np.zeros((map.shape[0]*map.shape[1], 2), dtype=int)

    i=0
    j=0
    for x in range(map.shape[0]):
        for y in range(map.shape[1]):
            if map[x, y] == 1:
                obst_poses[i,:] = x,y
                i+=1
            
            end_points[j,:] = x,y
            j+=1

    return obst_poses, end_points


def plot_gridmap(map, title, robot_pose=None):
    cmap = colors.ListedColormap(['White','Black'])
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
        print("Robot pose:", x, y, round(math.degrees(robot_pose[2])))

        # find the end point
        endx = x - 1.0 * math.sin(robot_pose[2])
        endy = y + 1.0 * math.cos(robot_pose[2])

        plt.plot(robot_pose[1], map.shape[0]-robot_pose[0], 'or', ms=10)
        plt.plot([y, endy], [map.shape[0]-x, map.shape[0]-endx], linewidth = '2', color='r')

