import math
from math import log
import numpy as np
import matplotlib.pyplot as plt

from Mapping.gridmap_utils import get_map, plot_gridmap, normalize_angle
from Sensors_Models.utils import evaluate_range_beam_dist_array as sensor_model
from Sensors_Models.ray_casting import cast_rays
from Mapping.hill_climbing import hill_climb_binary, objective_MAP_occupancy_grid_mapping_hill_climb


def algorithm_MAP_occupancy_grid_mapping(map, robot_poses, ranges, z_max, num_rays, fov):
    """""
    Maximum Aposteriori Probability (MAP) estimation for occupancy grid mapping
    
    map: occupancy probability values of the map at time t
    robot_poses: robot poses [x, y, theta] at different time steps
    ranges: laser range measurements at different time steps
    z_max: max range of the laser sensor
    num_rays: number of rays of the laser sensor
    fov: field of view of the laser sensor
    """""

    mix_density, sigma, lamb_short = [0.7, 0.2, 0.05, 0.05], 0.75, 0.9

    l0 = log(0.5)  # prior log-odds value
    occ_map = np.copy(map)

    for i in range(map.shape[0]):
        for j in range(map.shape[1]):
            # print(f"Processing cell ({i}, {j})")
            m_i_values = []

            for k in [0,1]: # k = 0 (free), k = 1 (occupied)
                occ_map[i, j] = k
                total_log_likelihood = 0.0

                for t in range(len(robot_poses)):
                    x_t = robot_poses[t]
                    z = ranges[t]
                    _, z_star = cast_rays(occ_map, x_t, num_rays, fov, z_max)

                    m_i = np.array([i, j]) + 0.5  # cell center
                    # if m_i in the perceptuion field of the sensor:
                    r = np.round(np.sqrt((m_i[0] - x_t[0])**2 + (m_i[1] - x_t[1])**2), decimals=2)
                    phi = math.atan2(m_i[1] - x_t[1], m_i[0] - x_t[0]) - x_t[2]
                    phi = normalize_angle(phi)

                    if r > z_max or abs(phi) > fov / 2:
                        continue

                    # the ray interesects the cell m_i
                    idx = int(round((phi + fov / 2) / (fov / (num_rays - 1)))) 
                    z_t_k = z[idx]  # range measurement of the k-th ray
                    z_star = z_star[idx]  # expected range measurement of the k-th ray

                    # evaluate the measurement model p(z|x,m)
                    p_hit, p_short, p_max, p_rand, p_z = sensor_model(z_t_k, z_star, z_max, mix_density, sigma, lamb_short)
                    # print(f"t={t}, z={z}, z*={z_star}, p_hit={p_hit}, p_short={p_short}, p_max={p_max}, p_rand={p_rand}, p_z={p_z}")
                    total_log_likelihood += np.sum(np.log(p_z + 1e-9))
                total_log_likelihood += k * l0
                m_i_values.append(total_log_likelihood)
            
            m_i_values = np.array(m_i_values)
            # apply the MAP estimation
            occ_map[i, j] = 1 if m_i_values[1] > m_i_values[0] else 0
            # print(f"Updated occupancy probability for cell ({i}, {j}): {occ_map[i, j]}")
    return occ_map


def main():
    # Define gridmap
    map_path = '2D_maps/map3.png'
    xy_reso = 3
    _, grid_map = get_map(map_path, xy_reso)

    # load robot poses
    robot_poses = np.load('Mapping/robot_poses.npy')[::3]  # (x, y, theta) in pixel coordinates
    robot_pose0 = robot_poses[0]  # (x, y, theta) in pixel coordinates
    print("Robot initial pose:", robot_pose0[0], robot_pose0[1], math.degrees(robot_pose0[2]))

    # Range sensor parameters
    fov = math.pi # Sensor Field of View
    num_rays = 36 # Number of rays
    z_max = 10.0 # Max range

    # Solution method
    use_hill_climbing = True  # if False use the direct MAP estimation

    ######################################################################
    ### simulate Laser range with ray casting + Lidar ranges some noise ##
    ######################################################################
    
    ranges = []
    for x in robot_poses:
        _, rng = cast_rays(grid_map, x, num_rays, fov, z_max)
        # simulate laser measurement adding noise to the obtained by casting rays in the map
        z = rng + np.random.normal(0, 0.1**2, size=1).item() + np.random.binomial(2, 0.001, 1).item() + 10*np.random.binomial(2, 0.001, 1).item()
        z = np.round(np.clip(z, 0., z_max), decimals=2)
        ranges.append(z)
    ranges = np.array(ranges)

    ######################################################################
    #### Occupancy Grid Mapping with Maximum Aposteriori Probability #####
    ######################################################################

    if not use_hill_climbing:
        # initialize log-odds values of the map
        log_odds = np.zeros(grid_map.shape)  # log-odds values of the map
        occ_grid_map = 1 - 1 / (1 + np.exp(log_odds))  # convert log-odds to probability
        
        occ_grid_map = algorithm_MAP_occupancy_grid_mapping(occ_grid_map, robot_poses, ranges, z_max, num_rays, fov)

    #####################################################################
    #### Hill Climbing optimization Occupancy Grid Mapping with MAP #####
    #####################################################################

    else:
        # Use Hill Climbing optimization to find the best occupancy map
        best_occ_map, best_value = hill_climb_binary(
            objective_MAP_occupancy_grid_mapping_hill_climb, 
            n_bits=grid_map.size, 
            n_iterations=10000,
            objective_args=(robot_poses, ranges, z_max, num_rays, fov)
            )
        
        occ_grid_map = best_occ_map.reshape(grid_map.shape)
        print("Best occupancy map found with Hill Climbing optimization, objective value:", best_value)


    #############################################################
    ##### Final plot of the original and the obtained maps ######
    #############################################################
        
    fig, ax = plt.subplots(1, 2, figsize=(11, 5))
    plot_gridmap(grid_map, robot_pose0, ax=ax[0]) 
    ax[0].axis("equal")
    ax[0].set_title("Original Gridmap")
    pc = plot_gridmap(occ_grid_map, ax=ax[1])    
    ax[1].set_title('MAP Occupancy Grid Map')

    fig.suptitle("Occupancy Grid Mapping")
    fig.tight_layout()
    fig.colorbar(pc, ax=ax[1], label='Occupancy Probability', fraction = 0.12, pad=0.04)
    plt.show()
    
    plt.close('all')


if __name__ == "__main__":
    main()
