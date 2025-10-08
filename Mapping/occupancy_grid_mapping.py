import math
from math import log
import numpy as np
import matplotlib.pyplot as plt

from Mapping.gridmap_utils import get_map, plot_gridmap, normalize_angle
from Sensors_Models.ray_casting import cast_rays, plot_ray_endpoints


def algorithm_inverse_range_sensor_model(m_i, x_t, z_t, alpha, beta, z_max, fov, num_rays):
    '''
    Inverse sensor model for a laser range finder

    m_i: cell center coordinates
    Let (x,y) be the coordinates of the cell m_i
    Let (x_r,y_r) be the coordinates of the robot pose x_t
    Let theta_r be the orientation of the robot pose x_t
    Let z be the range measurement z_t
    Let alpha be the width of a cell
    Let beta be the angle of a ray
    Let z_max be the maximum range of the sensor
    Let k be the indek of the ray
    '''

    r = np.round(np.sqrt((m_i[0] - x_t[0])**2 + (m_i[1] - x_t[1])**2), decimals=2)
    phi = math.atan2(m_i[1] - x_t[1], m_i[0] - x_t[0]) - x_t[2]
    phi = normalize_angle(phi)
    # find the k-th ray that contain the m_i
    k = int(round((phi + fov / 2) / (fov / (num_rays - 1)))) 
    z_t_k = z_t[k]  # range measurement of the k-th ray
    phi_k = -fov / 2 + k * (fov / (num_rays - 1))  # angle of the k-th ray
    # print(f"Cell {m_i} is in the perception field of the sensor, r: {r}, phi: {round(math.degrees(phi))}, k: {k}, z_t_k: {z_t_k}, phi_k: {round(math.degrees(phi_k))}")
    
    # apply the inverse sensor model
    if r > min(z_max, z_t_k + alpha / 2) or abs(phi - phi_k) > beta / 2:
        # print(f"Cell {m_i} is unknown")
        return log(0.5)  # unknown cell
    if z_t_k < z_max  and abs(r - z_t_k) < (0.5 * alpha): 
        # print(f"Cell {m_i} is occupied")
        return log(0.7) # occupied cell
    if r <= z_t_k:
        # print(f"Cell {m_i} is free")
        return log(0.3) # free cell
    return log(0.5)  # unknown

def algorithm_occupancy_grid_mapping(l_t, x_t, z_t, alpha, beta, z_max, fov, num_rays):
    '''
    Occupancy Grid Mapping algorithm with Inverse Sensor Model

    l_t: log-odds values of the map at time t
    x_t: robot pose at time t
    z_t: range measurements at time t
    alpha: width of a cell
    beta: angle of a ray
    z_max: maximum range of the sensor
    fov: field of view of the sensor
    num_rays: number of rays
    '''

    l0 = log(0.5)  # prior log-odds value
    l_t1 = np.copy(l_t)
    for i in range(l_t.shape[0]):
        for j in range(l_t.shape[1]):
            m_i = np.array([i, j]) + 0.5  # cell center

            # if m_i in the perceptuion field of the sensor:
            r = np.round(np.sqrt((m_i[0] - x_t[0])**2 + (m_i[1] - x_t[1])**2), decimals=2)
            phi = math.atan2(m_i[1] - x_t[1], m_i[0] - x_t[0]) - x_t[2]
            phi = normalize_angle(phi)
            if r <= z_max and abs(phi) <= fov / 2:
                # update log-odds value
                l_t1[i, j] = l_t[i, j] + algorithm_inverse_range_sensor_model(m_i, x_t, z_t, alpha, beta, z_max, fov, num_rays) - l0
            else:
                l_t1[i, j] = l_t[i, j]
    
    return l_t1


def main():
    # Define gridmap
    map_path = '2D_maps/map3.png'
    xy_reso = 3
    _, grid_map = get_map(map_path, xy_reso)

    # load robot poses
    robot_poses = np.load('Mapping/robot_poses.npy')  # (x, y, theta) in pixel coordinates
    robot_pose0 = robot_poses[0]  # (x, y, theta) in pixel coordinates
    print("Robot initial pose:", robot_pose0[0], robot_pose0[1], math.degrees(robot_pose0[2]))

    # plot original map with robot initial pose
    PLOT_TIME_STEPS = True

    # Range sensor parameters
    fov = 2*math.pi # Sensor Field of View
    num_rays = 60 # Number of rays
    z_max = 12.0 # Max range

    # Inverse sensor model parameters for laser range finder
    alpha = 1.00 # width of a cell
    beta = fov / num_rays  # angle of a ray
    print(f"alpha: {alpha}, beta: {math.degrees(beta)}")

    ###########################################################
    #### simulate Laser range with ray casting + some noise ###
    ###########################################################
    
    ranges = []
    ray_end_points = []
    for x in robot_poses:
        # cast rays: compute end points and laser measurements
        end_point, rng = cast_rays(grid_map, x, num_rays, fov, z_max)
        # simulate laser measurement adding noise to the obtained by casting rays in the map
        z = rng + np.random.normal(0, 0.1**2, size=1).item() + np.random.binomial(2, 0.001, 1).item() + 10*np.random.binomial(2, 0.001, 1).item()
        z = np.round(np.clip(z, 0., z_max), decimals=2)
        ranges.append(z)
        ray_end_points.append(end_point)
    ranges = np.array(ranges)
    ray_end_points = np.array(ray_end_points)

    ###########################################################
    #### Occupancy Grid Mapping with Inverse Sensor Model #####
    ###########################################################

    # initialize log-odds values of the map
    log_odds = np.zeros(grid_map.shape)  # log-odds values of the map
    occ_grid_map = 1 - 1 / (1 + np.exp(log_odds))  # convert log-odds to probability
    
    for t in range(len(robot_poses)):
        log_odds = algorithm_occupancy_grid_mapping(log_odds, robot_poses[t], ranges[t], alpha, beta, z_max, fov, num_rays)
        occ_grid_map = 1 - 1 / (1 + np.exp(log_odds))  # convert log-odds to probability

        if PLOT_TIME_STEPS:
            if t % 60 == 0:
                print(f"Time step: {t}")
                fig1, ax1 = plt.subplots()
                plot_gridmap(occ_grid_map, robot_poses[t], ax1)  
                plot_ray_endpoints(occ_grid_map.shape, ray_end_points[t], robot_poses[t], ax1)
                plt.pause(0.05)
                plt.show()

    ###########################################################

    #### Final plot of the original and the obtained maps ######
    fig, ax = plt.subplots(1, 2, figsize=(11, 5))
    plot_gridmap(grid_map, robot_pose0, ax=ax[0]) 
    ax[0].set_title("Original Gridmap")
    pc = plot_gridmap(occ_grid_map, ax=ax[1])    
    ax[1].set_title('Inverse Range Model')

    fig.suptitle("Occupancy Grid Mapping")
    fig.tight_layout()
    fig.colorbar(pc, ax=ax[1], label='Occupancy Probability', fraction = 0.12, pad=0.04)
    plt.show()
    
    plt.close('all')

if __name__ == "__main__":
    main()
