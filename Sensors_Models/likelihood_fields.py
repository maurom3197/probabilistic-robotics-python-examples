import math
import numpy as np
import matplotlib.pyplot as plt

from gridmap_utils import get_map, plot_gridmap, compute_map_occ
from ray_casting import cast_rays

# Gaussian Function
def gaussian(x, mean, sigma):
    return (1 / np.sqrt(2 * np.pi * sigma ** 2)) * np.exp(- (x - mean) ** 2 / (2 * sigma ** 2))

# Normalized Gaussian pdf
def compute_p_hit(dist, max_dist, _sigma):
    normalize_hit = 0.0
    for j in range(round(max_dist)):
        normalize_hit += gaussian(j, 0., _sigma)
    normalize_hit = 1. / normalize_hit

    p_hit = gaussian(dist, 0., _sigma)*normalize_hit

    return p_hit

def evaluate_prob(dist, z, z_max, _mix_density, _sigma):

    max_dist = max(dist)
    p_z = np.zeros((z.shape[0]))
    for k, z_k in enumerate(z):
        # Calculate hit mode probability
        p_hit = compute_p_hit(dist[k], max_dist, _sigma)

        # Calculate max mode probability
        if z_k == z_max:
            p_max = 1.0
        else:
            p_max = 0.0

        # Calculate rand mode probability
        p_rand = 1.0 / z_max

        p = np.array([np.float_(p_hit), p_max, p_rand])
        p_z[k]= np.dot(_mix_density, p)  # (p_hit*z_hit) + (p_max*z_max) + (p_rand*z_rand)
    
    return p_z


def find_endpoints(robot_pose, z, num_rays, fov):
    """""
    Check directly the presence of obstacles in Line of Sight in the FOV of the range sensor
    Compute the endpoint map coordinate and the associated distance z_star
    """""
    robot_x, robot_y, robot_angle = robot_pose[:]

    # define left most angle of FOV and step angle
    start_angle = robot_angle + fov/2
    step_angle = fov/num_rays
    
    end_points = np.zeros((num_rays, 2))

    # loop over casted rays
    for i in range(num_rays):
        # get ray target coordinates
        target_x = robot_x - math.sin(start_angle) * z[i]
        target_y = robot_y + math.cos(start_angle) * z[i]
        
        end_points[i, :] = target_x, target_y
    
        # increment angle by a single step
        start_angle -= step_angle

    return end_points


def compute_distances(end_points, obst):

    distances = np.zeros((end_points.shape[0]))
    for k, ep in enumerate(end_points):
        # convert target X, Y coordinate to map col, row
        x_k = int(ep[0])
        y_k = int(ep[1])

        # Search minimum distance
        min_dis = float("inf")
        for i_obst in obst:
            if (ep == i_obst).all(0):
                min_dis = 0
                break
            iox = i_obst[0]
            ioy = i_obst[1]

            d = math.dist([iox, ioy], [x_k, y_k])
            if min_dis >= d:
                min_dis = d

        distances[k] = min_dis

    return distances

def plot_likelihood_fields(p_gridmap, robot_pose=None):
    plt.imshow(p_gridmap, cmap='gray')
    plt.title('Likelihood fields', fontsize = 14)
    plt.xticks(ticks=range(p_gridmap.shape[1]), labels=range(p_gridmap.shape[1]))
    plt.yticks(ticks=range(p_gridmap.shape[0]), labels=range(p_gridmap.shape[0]))
    
    if robot_pose is not None:
        # unpack the first point
        x, y = robot_pose[0], robot_pose[1]
        # find the end point
        endx = x - 1.5 * math.sin(robot_pose[2])
        endy = y + 1.5 * math.cos(robot_pose[2])

        plt.plot(robot_pose[1], robot_pose[0], 'or', ms=10)
        plt.plot([y, endy], [x, endx], linewidth = '2', color='r')


def plot_ray_prob(map_size, end_points, robot_pose, p_z):
    robot_x, robot_y, _ = robot_pose
    
    # draw casted ray
    for i in range(end_points.shape[0]):
        ep_x = end_points[i, 0]
        ep_y = end_points[i, 1]
        plt.plot([robot_y, ep_y], [map_size[0]-robot_x,  map_size[0]-ep_x], linewidth = '0.8', color='b')

    y = np.squeeze(end_points[:,1])
    x = np.squeeze(end_points[:,0])
    x = map_size[0]*np.ones_like(end_points[:,0]) - x
    col = p_z
    # draw endpoint with probability from Likelihood Fields
    plt.scatter(y, x, s=50, c=col, cmap='viridis')
    plt.colorbar()

    
def main():
    # global constants
    map_path = '../2D_maps/map3.png'

    xy_reso = 4
    _, grid_map = get_map(map_path, xy_reso)
    # print(grid_map)
    obst_arr, occ_state = compute_map_occ(grid_map)
    
    fov = math.pi / 4
    num_rays = 32

    robot_pose = np.array([12, 9, 2*math.pi/3])
    z_max = 8.0

    ###########################################################
    #### simulate Laser range with ray casting + some noise ###
    ###########################################################
    
    _, rng = cast_rays(grid_map, robot_pose, num_rays, fov, z_max)
    #simulate laser measurement adding noise to the obtained by casting rays in the map
    z = rng + np.random.normal(0, 0.1**2, size=1).item() + np.random.binomial(2, 0.001, 1).item() + 10*np.random.binomial(2, 0.001, 1).item()
    z = np.clip(z, 0., z_max)
    end_points_z = find_endpoints(robot_pose, z, num_rays, fov)
    distances_z = compute_distances(end_points_z, obst_arr)
    mix_density, sigma = [0.9, 0.05, 0.05], 0.75
    p_z_z = evaluate_prob(distances_z, z, z_max, mix_density, sigma)
    # print computed probabilities on the laser measurements
    print(p_z_z)

    plot_gridmap(grid_map, 'Grid Map - endpoints prob with LF', robot_pose)
    plot_ray_prob(grid_map.shape, end_points_z, robot_pose, p_z_z)
    plt.show()

    ###########################################################
    #### pre-compute likelihood fields on the entire map ######
    ###########################################################

    distances = compute_distances(occ_state, obst_arr)

    max_dist = max(distances)
    p_z = np.zeros((occ_state.shape[0]))
    for i, ep in enumerate(occ_state):
        p_zi = compute_p_hit(distances[i], max_dist, sigma)
        p_z[i] = p_zi
    
    p_gridmap = np.reshape(p_z, grid_map.shape)

    plot_likelihood_fields(p_gridmap, robot_pose=robot_pose)
    plt.show()

    plt.close('all')

if __name__ == "__main__":
    main()
