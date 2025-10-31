import math
import numpy as np
import matplotlib.pyplot as plt

from Mapping.gridmap_utils import get_map, plot_gridmap, compute_map_occ
from Sensors_Models.ray_casting import cast_rays
from Sensors_Models.utils import compute_p_hit_dist, precompute_p_hit_map


def evaluate_prob(dist, z, z_max, _mix_density, _sigma):
    """""
    Compute probability p(z|x) according to the range laser model, mixture of densities
    """""
    max_dist = max(dist)
    p_z = np.zeros((z.shape[0]))
    for k, z_k in enumerate(z):
        # Calculate hit mode probability
        p_hit = compute_p_hit_dist(dist[k], max_dist, _sigma)

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
    start_angle = robot_angle - fov/2
    step_angle = fov/num_rays
    
    end_points = np.zeros((num_rays, 2))

    # loop over casted rays
    for i in range(num_rays):
        # get ray target coordinates
        target_x = robot_x + math.cos(start_angle) * z[i]
        target_y = robot_y + math.sin(start_angle) * z[i]
        
        end_points[i, :] = target_x, target_y
    
        # increment angle by a single step
        start_angle += step_angle

    return end_points


def compute_distances(end_points, obst_cells):
    """""
    Compute the distance to the nearest obstacle in the map for a given endpoint (x,y)
    """""
    distances = np.zeros((end_points.shape[0]))
    for k, ep in enumerate(end_points):
        # convert target X, Y coordinate to map col, row
        x_k = int(ep[0])
        y_k = int(ep[1])

        # Search minimum distance
        min_dis = float("inf")
        for i_obst in obst_cells:
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

def precompute_likelihood_field(grid_map, sigma=None, max_dist=None):
    """""
    Pre-compute the likelihood field for the entire map
    """""
    occ_spaces, _, _, map_spaces = compute_map_occ(grid_map)
    distances = compute_distances(map_spaces, occ_spaces)
    distance_grid = distances.reshape(grid_map.shape)

    if not max_dist:
        max_dist = np.max(distances)
    if not sigma:
        sigma = np.std(distances)

    # handle unknown cells in the gridmap (if any)
    gridmap_values = np.unique(grid_map)
    if gridmap_values.size > 2:
        unknown_value = gridmap_values[(gridmap_values != 0) & (gridmap_values != 1)][0]
        print("unknown_value:", unknown_value)

        # unknown cells will have a distance of max_dist
        distance_grid[grid_map == unknown_value] = max_dist

    # the probability gridmap can be used as a look-up table during localization
    p_gridmap = precompute_p_hit_map(distance_grid, max_dist, sigma)

    return p_gridmap, distance_grid


def plot_likelihood_fields(p_gridmap, robot_pose=None, ax=None):
    if ax is None:
        ax = plt.gca()

    ax.imshow(p_gridmap, cmap='gray')
    ax.set_xticks(ticks=range(p_gridmap.shape[1]), labels=range(p_gridmap.shape[1]))
    ax.set_yticks(ticks=range(p_gridmap.shape[0]), labels=range(p_gridmap.shape[0]))
    ax.set_aspect('equal')
    ax.set_xlabel('Y [cells]', fontsize = 12)
    ax.set_ylabel('X [cells]', fontsize = 12)

    if robot_pose is not None:
        # unpack the first point
        x, y = robot_pose[0], robot_pose[1]
        # find the end point
        endx = x - 1.5 * math.sin(robot_pose[2]-math.pi/2)
        endy = y + 1.5 * math.cos(robot_pose[2]-math.pi/2)

        ax.plot(robot_pose[1], robot_pose[0], 'or', ms=10)
        ax.plot([y, endy], [x, endx], linewidth = '2', color='r')

def plot_ray_prob(map_size, end_points, robot_pose, p_z, ax=None):

    if ax is None:
        ax = plt.gca()
    
    robot_x, robot_y, _ = robot_pose
    # draw casted ray
    for i in range(end_points.shape[0]):
        ep_x = end_points[i, 0]
        ep_y = end_points[i, 1]
        ax.plot([robot_y, ep_y], [map_size[0]-robot_x,  map_size[0]-ep_x], linewidth = '0.8', color='b')

    y = np.squeeze(end_points[:,1])
    x = np.squeeze(end_points[:,0])
    x = map_size[0]*np.ones_like(end_points[:,0]) - x
    col = p_z
    # draw endpoint with probability from Likelihood Fields
    ax.scatter(y, x, s=50, c=col, cmap='viridis')

    
def main():
    # global constants
    map_path = '2D_maps/map31.bmp'

    xy_reso = 4
    _, grid_map = get_map(map_path, xy_reso)
    # print(grid_map)
    occ_spaces, free_spaces, unknown, map_spaces = compute_map_occ(grid_map)
    
    fov = math.pi / 4
    num_rays = 12

    robot_pose = np.array([12, 9, 2*math.pi/3])
    z_max = 8.0

    ###########################################################
    #### simulate Laser range with ray casting + some noise ###
    ###########################################################
    
    end_points_z, rng = cast_rays(grid_map, robot_pose, num_rays, fov, z_max)
    # print("Simulated laser ranges (no noise):", rng)
    # print("Perceived obstacles end points:", end_points_z)
    # simulate laser measurement adding noise to the obtained by casting rays in the map
    z = rng + np.random.normal(0, 0.1**2, size=1).item() + np.random.binomial(2, 0.001, 1).item() + 10*np.random.binomial(2, 0.001, 1).item()
    z = np.clip(z, 0., z_max)
    print("Noisy laser ranges:", z)
    end_points_z = find_endpoints(robot_pose, z, num_rays, fov)
    print("Noisy end_points_z:", end_points_z)

    # compute distances of the perceived end points to the nearest obstacle in the map
    distances_z = compute_distances(end_points_z, occ_spaces)
    # print("Distances to nearest obstacles:", distances_z)

    # Evaluate the likelihood field model
    mix_density, sigma = [0.9, 0.05, 0.05], 0.75
    p_z_z = evaluate_prob(distances_z, z, z_max, mix_density, sigma)
    print("Probabilities of the laser ranges:", p_z_z)

    # visualize the results
    fig, ax = plt.subplots()
    plot_gridmap(grid_map, robot_pose, ax)
    fig.suptitle('Probabilities of the laser ranges - LF', fontsize = 16)
    plot_ray_prob(grid_map.shape, end_points_z, robot_pose, p_z_z)
    
    fig.colorbar(plt.cm.ScalarMappable(cmap='viridis'), ax=ax, label='p(z|x)')
    # fig.savefig("ray_prob_likelihood_field.png")
    plt.show()


    ############################################################
    #### pre-compute likelihood fields on the entire map  ######
    ############################################################
    # using the provided utility function
    distances = compute_distances(map_spaces, occ_spaces)
    max_dist = np.max(distances)
    sigma_dist = np.std(distances)
    print("max_dist:", max_dist, "sigma_dist:", sigma_dist)
    distance_grid = distances.reshape(grid_map.shape)
    
    # the probability gridmap can be used as a look-up table during localization
    p_gridmap = precompute_p_hit_map(distance_grid, max_dist, sigma_dist) 

    fig, ax = plt.subplots()
    plot_likelihood_fields(p_gridmap, robot_pose=robot_pose, ax=ax)
    fig.colorbar(plt.cm.ScalarMappable(cmap='gray'), ax=ax, label='p(z|x)')
    ax.set_title('Pre-computed Likelihood Fields', fontsize = 14)
    # fig.savefig("likelihood_field_sigma_dist.png")
    plt.show()

    plt.close('all')

if __name__ == "__main__":
    main()
