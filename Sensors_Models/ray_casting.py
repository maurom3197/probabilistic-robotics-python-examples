import math
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from gridmap_utils import get_map, plot_gridmap

# TO CHECK: 
# - implement Bresenham ray casting
# - Test Bresenham() vs cast_rays()
 

# raycasting algorithm
def cast_rays(map, robot_pose, num_rays, fov, z_max):
    """""
    Check directly the presence of obstacles in Line of Sight in the FOV of the range sensor
    Compute the endpoint map coordinate and the associated distance z_star
    """""
    robot_x, robot_y, robot_angle = robot_pose[:]

    # define left most angle of FOV and step angle
    start_angle = robot_angle + fov/2
    step_angle = fov/num_rays
    
    end_points = np.zeros((num_rays, 2))
    z_star = np.zeros((num_rays))

    # loop over casted rays
    for i in range(num_rays):
        # cast ray step by step
        length = 0.25
        while(True):
            # get ray target coordinates
            target_x = robot_x - math.sin(start_angle) * length
            target_y = robot_y + math.cos(start_angle) * length
            
            # ray reach end of map
            if target_x < 0. or target_x > map.shape[0]: # check if map border reached
                end_points[i, :] = round(target_x), target_y
                z_star[i] = math.dist([round(target_x), target_y], [robot_x, robot_y])
                break
            elif target_y < 0. or target_y > map.shape[1]:
                end_points[i, :] = target_x, round(target_y)
                z_star[i] = math.dist([target_x, round(target_y)], [robot_x, robot_y])
                break
        
            # convert target X, Y coordinate to map col, row
            row = int(target_x)
            col = int(target_y)

            # ray does not hit any obstacle
            if math.dist([target_x, target_y], [robot_x, robot_y]) >= z_max:
                end_points[i, :] = target_x, target_y
                z_star[i] = z_max
                break
        
            # ray hits the condition
            elif map[row, col] == 1:
                end_points[i, :] = target_x, target_y
                z_star[i] = math.dist([target_x, target_y], [robot_x, robot_y])
                break

            length += 0.05
        # increment angle by a single step
        start_angle -= step_angle

    return end_points, z_star


def plot_ray_endpoints(map_size, end_points, robot_pose):
    robot_x, robot_y, _ = robot_pose[:]
    
    for i in range(end_points.shape[0]):
        ep_x = end_points[i, 0]
        ep_y = end_points[i, 1]

        # draw casted ray
        plt.plot([robot_y, ep_y], [map_size[0]-robot_x,  map_size[0]-ep_x], linewidth = '1.2', color='b')
        plt.plot(ep_y, map_size[0]-ep_x, 'ob', ms=6)


def plot_rays_on_gridmap(map, title, robot_pose, end_points):

    plot_gridmap(map, title, robot_pose)
    plot_ray_endpoints(map.shape, end_points, robot_pose)


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
    x1, y1: map coordinate of the obstacle end point
    """""

    dx =  abs(x1-x0)
    sx = 1 if (x0<x1) else -1
    dy = -abs(y1-y0)
    sy = 1 if (y0<y1) else -1
    err = dx+dy

    while (True):
        # ray reach end of map
        if x0 < 0.:
            obst = 0, y0
            break
        elif x0 > map.shape[0]: # check if map border reached
            obst = round(x0), y0
            break
        elif y0 < 0.:
            obst = x0, 0
            break
        elif y0 > map.shape[1]:
            obst = x0, round(y0)
            break
        elif map[round(x0), round(y0)]==1 or ((x0==x1) and (y0==y1)):
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


def cast_rays_bresenham(map, robot_pose, num_rays, fov, z_max):
    """""
    Check directly the presence of obstacles in Line of Sight in the FOV of the range sensor
    Compute the endpoint map coordinate and the associated distance z_star
    """""
    robot_x, robot_y, robot_angle = robot_pose[:]

    # define left most angle of FOV and step angle
    start_angle = robot_angle + fov/2
    step_angle = fov/num_rays
    
    end_points = np.zeros((num_rays, 2))
    z_star = np.zeros((num_rays))

    # loop over casted rays
    for i in range(num_rays):
        # cast ray step by step

        # get ray target coordinates
        target_x = int(robot_x - math.sin(start_angle) * z_max)
        target_y = int(robot_y + math.cos(start_angle) * z_max)

        target_x, target_y = bresenham(robot_x, robot_y, target_x, target_y, map)
        
        end_points[i, :] = target_x, target_y
        z_star[i] = math.dist([target_x, target_y], [robot_x, robot_y])

        # increment angle by a single step
        start_angle -= step_angle

    return end_points, z_star


def main():
    # global constants
    map_path = '../2D_maps/map0.png'

    xy_reso = 2
    map, grid_map = get_map(map_path, xy_reso)
    print(grid_map)
    fov = math.pi / 4
    num_rays = 16

    robot_pose = np.array([8, 12, -1*math.pi/3])
    z_max = 16.0

    # cast rays: compute end points and laser measurements
    end_points, z_star = cast_rays(grid_map, robot_pose, num_rays, fov, z_max)
    # print("Perceived obstacles end points:", end_points)
    # print("Laser measurements:", z_star)

    plot_rays_on_gridmap(grid_map, 'Ray Casted on Grid Map', robot_pose=robot_pose, end_points=end_points)
    plt.show()

    np.savez('ray_casting_z.npz', D=z_star)

    # cast rays: compute end points and laser measurements
    end_points, z_star = cast_rays_bresenham(grid_map, robot_pose, num_rays, fov, z_max)
    # print("Perceived obstacles end points:", end_points)
    # print("Laser measurements:", z_star)

    plot_rays_on_gridmap(grid_map, 'Ray Casted Bresenham', robot_pose=robot_pose, end_points=end_points)
    plt.show()

    plt.close('all')

if __name__ == "__main__":
    main()
