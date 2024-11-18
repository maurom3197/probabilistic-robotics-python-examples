import math
import numpy as np


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
