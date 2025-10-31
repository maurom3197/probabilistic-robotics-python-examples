import math
import numpy as np
import matplotlib.pyplot as plt

from Mapping.gridmap_utils import get_map, plot_gridmap
from Sensors_Models.utils import bresenham, normalize_angle

# ray-casting algorithm
def cast_rays(map, robot_pose, num_rays, fov, z_max):
    """""
    Check directly the presence of obstacles in Line of Sight in the FOV of the range sensor
    Compute the endpoint map coordinate and the associated distance z_star
    """""
    robot_x, robot_y, robot_angle = robot_pose[:]

    # define left most angle of FOV and step angle
    start_angle = robot_angle - fov/2
    step_angle = fov/num_rays
    
    end_points = np.zeros((num_rays, 2))
    z_star = np.zeros((num_rays))

    # loop over casted rays
    for i in range(num_rays):
        # cast ray step by step
        length = 0.25
        while(True):
            start_angle = normalize_angle(start_angle)
            # get ray target coordinates
            target_x = robot_x + math.cos(start_angle) * length
            target_y = robot_y + math.sin(start_angle) * length
            
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

            length += 0.1
        # increment angle by a single step
        start_angle += step_angle

    return end_points, z_star


def cast_rays_bresenham(map, robot_pose, num_rays, fov, z_max):
    """""
    Check directly the presence of obstacles in Line of Sight in the FOV of the range sensor
    Compute the endpoint map coordinate and the associated distance z_star
    """""
    robot_x, robot_y, robot_angle = robot_pose[:]

    # define left most angle of FOV and step angle
    start_angle = robot_angle - fov/2
    step_angle = fov/num_rays
    
    end_points = np.zeros((num_rays, 2))
    z_star = np.zeros((num_rays))

    # loop over casted rays
    for i in range(num_rays):
        # cast ray step by step
        start_angle = normalize_angle(start_angle)
        # get ray target coordinates
        target_x = robot_x + math.cos(start_angle) * z_max
        target_y = robot_y + math.sin(start_angle) * z_max
        print("targets: ", target_x, target_y)

        target_x, target_y = bresenham(robot_x, robot_y, target_x, target_y, map)
        
        end_points[i, :] = target_x, target_y
        z_star[i] = math.dist([target_x, target_y], [robot_x, robot_y])

        # increment angle by a single step
        start_angle += step_angle

    return end_points, z_star

def plot_ray_endpoints(map_size, end_points, robot_pose, ax=None):
    if ax is None:
        ax = plt.gca()
    
    robot_x, robot_y, _ = robot_pose[:]
    
    for i in range(end_points.shape[0]):
        ep_x = end_points[i, 0]
        ep_y = end_points[i, 1]

        # draw casted ray
        ax.plot([robot_y, ep_y], [map_size[0]-robot_x,  map_size[0]-ep_x], linewidth = '1.2', color='b')
        ax.plot(ep_y, map_size[0]-ep_x, 'ob', ms=6)

def plot_rays_on_gridmap(map, robot_pose, end_points, ax):

    pc = plot_gridmap(map, robot_pose, ax)
    plot_ray_endpoints(map.shape, end_points, robot_pose, ax)

    return pc


def main():
    ###########################################################

    # global constants
    map_path = '2D_maps/map3.png'

    xy_reso = 3
    map, grid_map = get_map(map_path, xy_reso)
    # print(grid_map)
    fov = 2*math.pi 
    num_rays = 12

    robot_pose = np.array([6, 13, math.pi/2])  # [x, y, theta] in map coordinates
    z_max = 10.0

    use_bresenham = False

    ###########################################################

    if not use_bresenham:

        # cast rays: compute end points and laser measurements
        end_points, z_star = cast_rays(grid_map, robot_pose, num_rays, fov, z_max)
        print("Perceived obstacles end points:", end_points)
        print("Laser measurements:", z_star)

        fig, ax = plt.subplots(figsize=(8,8))
        pc = plot_rays_on_gridmap(grid_map, robot_pose=robot_pose, end_points=end_points, ax=ax)
        fig.suptitle('Ray Casted on Grid Map', fontsize = 16)
        plt.show()

        # np.savez('ray_casting_z.npz', D=z_star)
    else:
        # cast rays: compute end points and laser measurements
        end_points, z_star = cast_rays_bresenham(grid_map, robot_pose, num_rays, fov, z_max)
        print("Perceived obstacles end points:", end_points)
        print("Laser measurements:", z_star)

        fig, ax = plt.subplots(figsize=(8,8))
        pc = plot_rays_on_gridmap(grid_map, robot_pose=robot_pose, end_points=end_points, ax=ax)
        fig.suptitle('Ray Casted Bresenham', fontsize = 16)
        plt.show()

    plt.close('all')

if __name__ == "__main__":
    main()
