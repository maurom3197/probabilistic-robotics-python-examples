import math
import numpy as np
from Sensors_Models.utils import compute_p_hit_dist
from Discrete_Filters.utils import normalize_angle

def sample_velocity_motion_model(x, u, a, dt):
    """Sample velocity motion model.
    Arguments:
    x -- pose of the robot before moving [x, y, theta]
    u -- velocity reading obtained from the robot [v, w]
    sigma -- noise parameters of the motion model [a1, a2, a3, a4, a5, a6] or [std_dev_v, std_dev_w]
    dt -- time interval of prediction
    """

    if x is list:
        x = np.array(x)

    if x.ndim == 1:  # manage the case of a single pose
        x = x.reshape(1, -1)

    if u is list:
        u = np.array(u)

    sigma = np.ones((3))
    if a.shape == u.shape:
        sigma[:-1] = a[:]
        sigma[-1] = a[1] * 0.5
    else:
        sigma[0] = a[0] * u[0] ** 2 + a[1] * u[1] ** 2
        sigma[1] = a[2] * u[0] ** 2 + a[3] * u[1] ** 2
        sigma[2] = a[4] * u[0] ** 2 + a[5] * u[1] ** 2

    v_hat = np.ones(x.shape[0]) * u[0] + np.random.normal(0, sigma[0], x.shape[0])
    w_hat = np.ones(x.shape[0]) * u[1] + np.random.normal(0, sigma[1], x.shape[0])
    gamma_hat = np.random.normal(0, sigma[2], x.shape[0])

    r = v_hat / w_hat

    x_prime = x[:, 0] - r * np.sin(x[:, 2]) + r * np.sin(x[:, 2] + w_hat * dt)
    y_prime = x[:, 1] + r * np.cos(x[:, 2]) - r * np.cos(x[:, 2] + w_hat * dt)
    theta_prime = x[:, 2] + w_hat * dt + gamma_hat * dt
    return np.squeeze(np.stack([x_prime, y_prime, theta_prime], axis=-1))


def get_odometry_command(odom_pose, odom_pose_prev):
    """Transform robot poses taken from odometry to u command
    Arguments:
    odom_pose -- last odometry pose of the robot [x, y, theta] at time t
    odom_pose_prev -- previous odometry pose of the robot [x, y, theta] at time t-1

    Output:
    u_odom : np.array [rot1, trasl, rot2]
    """

    x_odom, y_odom, theta_odom = odom_pose[:]
    x_odom_prev, y_odom_prev, theta_odom_prev = odom_pose_prev[:]

    rot1 = np.arctan2(y_odom - y_odom_prev, x_odom - x_odom_prev) - theta_odom_prev
    trasl = np.sqrt((x_odom - x_odom_prev) ** 2 + (y_odom - y_odom_prev) ** 2)
    rot2 = theta_odom - theta_odom_prev - rot1

    return np.array([rot1, trasl, rot2])


def sample_odometry_motion_model(x, u, a):
    """Sample odometry motion model.
    Arguments:
    x -- pose of the robot before moving [x, y, theta]
    u -- odometry reading obtained from the robot [rot1, trans, rot2]
    a -- noise parameters of the motion model [a1, a2, a3, a4] or [std_rot1, std_trans, std_rot2]
    """
    if x is list:
        x = np.array(x)

    if x.ndim == 1:  # manage the case of a single pose
        x = x.reshape(1, -1)

    if u is list:
        u = np.array(u)

    sigma = np.ones((3))
    if a.shape == u.shape:
        sigma = a
    else:
        sigma[0] = a[0] * abs(u[0]) + a[1] * abs(u[1])
        sigma[1] = a[2] * abs(u[1]) + a[3] * (abs(u[0]) + abs(u[2]))
        sigma[2] = a[0] * abs(u[2]) + a[1] * abs(u[1])

    # noisy odometric transformations: 1 translation and 2 rotations
    delta_hat_r1 = np.ones(x.shape[0]) * u[0] + np.random.normal(0, sigma[0], x.shape[0])
    delta_hat_t = np.ones(x.shape[1]) * u[1] + np.random.normal(0, sigma[1], x.shape[0])
    delta_hat_r2 = np.ones(x.shape[2]) * u[2] + np.random.normal(0, sigma[2], x.shape[0])

    x_prime = x[:, 0] + delta_hat_t * np.cos(x[:, 2] + delta_hat_r1)
    y_prime = x[:, 1] + delta_hat_t * np.sin(x[:, 2] + delta_hat_r1)
    theta_prime = x[:, 2] + delta_hat_r1 + delta_hat_r2

    return np.squeeze(np.stack([x_prime, y_prime, theta_prime], axis=-1))


def landmark_range_bearing_model(robot_pose, landmark, sigma):
    """""
    Sampling z from landmark model for range and bearing
    robot pose: can be the estimated robot pose or the particles
    """ ""
    if robot_pose is list:
        robot_pose = np.array(robot_pose)

    if robot_pose.ndim == 1:  # manage the case of a single pose
        robot_pose = robot_pose.reshape(1, -1)

    r_ = np.linalg.norm(robot_pose[:, 0:2] - landmark, axis=1) + np.random.normal(0.0, sigma[0], robot_pose.shape[0])
    phi_ = (
        np.arctan2(landmark[1] - robot_pose[:, 1], landmark[0] - robot_pose[:, 0])
        - robot_pose[:, 2]
        + np.random.normal(0.0, sigma[1], robot_pose.shape[0])
    )
    return np.squeeze(np.stack([r_, phi_], axis=-1))


def landmark_range_bearing_sensor(robot_pose, landmark, sigma, max_range=6.0, fov=math.pi / 2):
    """""
    Simulate the detection of a landmark with a virtual sensor able to estimate range and bearing
    """ ""
    z = landmark_range_bearing_model(robot_pose, landmark, sigma)

    # filter z for a more realistic sensor model (add a max range distance and a FOV)
    if z[0] > max_range or abs(z[1]) > fov / 2:
        return None

    return z

def likelihood_field_laser_model_pf(robot_pose, z_points, distances, p_hit_grid, sigma=1.0, num_rays=36, z_max=8.0, fov=math.pi, mix_density=[0.9, 0.0, 0.1]):
    """""
    Likelihood field probabilistic model function
    robot pose: the estimated robot pose
    z_points: the laser measurements
    distances: distances of nearest obstacles in the map, it can be precomputed and used as lookup table
    mix_density: weights for the different components of the model [hit, max, random]
    """ ""
    print("distances shape:", distances.shape)
    if robot_pose is list:
        robot_pose = np.array(robot_pose)

    if robot_pose.ndim == 1:  # manage the case of a single pose
        robot_pose = robot_pose.reshape(1, -1)

    # Goal: compute the prob associated to each particle based on the laser measurements
    
    # Precompute some constants
    max_dist = np.max(distances) # max distance in the distance map
    sigma_dist = np.std(distances) # use std of the distance map as sigma
    p_max_dist = compute_p_hit_dist(max_dist, max_dist, sigma_dist) # max distance prob

    p_rand = 1.0 / z_max # uniform random component
    step_angle = fov/num_rays # step angle between rays

    # print("max_dist, sigma_dist:", max_dist, sigma_dist)
    p_max_dist = compute_p_hit_dist(max_dist, max_dist, sigma_dist)
    # print("p_max_dist:", p_max_dist)
    # define array to store the prob associated to each particle
    probs = np.ones((robot_pose.shape[0], num_rays))
    
    # loop over particles
    for i in range(robot_pose.shape[0]):
        # define left most angle of FOV and step angle
        start_angle = robot_pose[i, 2] - fov/2
        
        # loop over laser rays
        for k, z_k in enumerate(z_points):
            # print(f"Particle {i}, ray {k}, z_k: {z_k}, start_angle: {start_angle}")
            # skip max range readings
            if z_k >= z_max:
                continue

            # get endpoint coordinates of the ray
            start_angle = normalize_angle(start_angle)
            target_x = robot_pose[i, 0] + np.cos(start_angle) * z_k
            target_y = robot_pose[i, 1] + np.sin(start_angle) * z_k

            # check if endpoint is inside the map limits
            x, y = int(target_x), int(target_y)
            if x>=0 and y>=0 and x<distances.shape[0] and y<distances.shape[1] and p_hit_grid[x, y]>p_max_dist:
                # Calculate Gaussian hit mode probability
                # p_hit_k = compute_p_hit_dist(distances[x, y], max_dist=max_dist, sigma=sigma_dist)
                p_hit_k = p_hit_grid[x, y]
                # print(f"p_hit for particle {i}, ray {k}: {p_hit_k}")
            else:
                # endpoint is out of map limits, assign max_dist probability
                p_hit_k = p_max_dist
                # print(f"Particle {i}, ray {k} is out of map limits, assigning p_hit = p_max_dist: {p_max_dist}")

            # Calculate the final mixed probability for the k-th ray
            p_k = mix_density[0] * p_hit_k + mix_density[2] * p_rand
            # store the prob associated to the k-th ray for the i-th particle
            probs[i, k] = p_k

            # increment angle by a single step
            start_angle += step_angle

    return probs


def likelihood_field_laser_model_pf_np(
    robot_pose,
    z_points,
    distances,
    p_hit_grid,
    sigma=1.0,
    num_rays=36,
    z_max=8.0,
    fov=math.pi,
    mix_density=(0.9, 0.0, 0.1)
):
    """
    Vectorized likelihood field probabilistic model for all particles at once.
    robot_pose: (N, 3) array of particle poses [x, y, theta]
    z_points: (num_rays,) array of laser measurements
    distances: 2D map of nearest obstacle distances
    p_hit_grid: precomputed likelihood field (same shape as distances)
    """
    robot_pose = np.asarray(robot_pose)
    if robot_pose.ndim == 1:
        robot_pose = robot_pose.reshape(1, -1)

    N = robot_pose.shape[0]

    # Precompute some constants
    max_dist = np.max(distances)
    sigma_dist = np.std(distances)
    p_max_dist = compute_p_hit_dist(max_dist, max_dist, sigma_dist)
    p_rand = 1.0 / z_max

    # Prepare angles for all rays
    rel_angles = np.linspace(-fov / 2, fov / 2, num_rays)
    x, y, theta = robot_pose[:, 0], robot_pose[:, 1], robot_pose[:, 2]

    # Compute all ray endpoint coordinates for all particles (N, num_rays)
    angles = theta[:, None] + rel_angles[None, :]
    angles = normalize_angle(angles)
    x_end = x[:, None] + np.cos(angles) * z_points[None, :]
    y_end = y[:, None] + np.sin(angles) * z_points[None, :]

    # Convert to integer map indices
    x_idx = np.floor(x_end).astype(int)
    y_idx = np.floor(y_end).astype(int)

    # Valid mask for endpoints inside map bounds
    valid_mask = (
        (x_idx >= 0)
        & (y_idx >= 0)
        & (x_idx < distances.shape[0])
        & (y_idx < distances.shape[1])
        & (z_points[None, :] < z_max)
        & (p_hit_grid[x_idx, y_idx]>p_max_dist)
    )

    # Initialize with p_max_dist (for invalid endpoints)
    p_hit = np.full((N, num_rays), p_max_dist)

    # Use advanced indexing to get hit probabilities where valid
    valid_x = x_idx[valid_mask]
    valid_y = y_idx[valid_mask]
    p_hit[valid_mask] = p_hit_grid[valid_x, valid_y]

    # Final mixed probability (hit + random components)
    probs = mix_density[0] * p_hit + mix_density[2] * p_rand
    return probs

