import math
import numpy as np


def sample_velocity_motion_model(x, u, a, dt):
    """ Sample velocity motion model.
    Arguments:
    x -- pose of the robot before moving [x, y, theta]
    u -- velocity reading obtained from the robot [v, w]
    sigma -- noise parameters of the motion model [a1, a2, a3, a4, a5, a6] or [std_dev_v, std_dev_w]
    dt -- time interval of prediction
    """

    sigma = np.ones((3))
    if a.shape == u.shape:
        sigma[:-1] = a[:]
        sigma[-1] = a[1]*0.5
    else:
        sigma[0] = a[0]*u[0]**2 + a[1]*u[1]**2
        sigma[1] = a[2]*u[0]**2 + a[3]*u[1]**2
        sigma[2] = a[4]*u[0]**2 + a[5]*u[1]**2

    # sample noisy velocity commands to consider actuation errors and dynamics not modeled 
    v_hat = u[0] + np.random.normal(0, sigma[0])
    w_hat = u[1] + np.random.normal(0, sigma[1])
    gamma_hat = np.random.normal(0, sigma[2])

    r = v_hat/w_hat

    if len(x.shape) == 1: # robot predict
        x_prime = x[0] - r*np.sin(x[2]) + r*np.sin(x[2]+w_hat*dt)
        y_prime = x[1] + r*np.cos(x[2]) - r*np.cos(x[2]+w_hat*dt)
        theta_prime = x[2] + w_hat*dt + gamma_hat*dt
        return np.array([x_prime, y_prime, theta_prime])
    
    else: # particles predict
        x_prime = x[:, 0] - r*np.sin(x[:, 2]) + r*np.sin(x[:, 2]+w_hat*dt)
        y_prime = x[:, 1] + r*np.cos(x[:, 2]) - r*np.cos(x[:, 2]+w_hat*dt)
        theta_prime = x[:, 2] + w_hat*dt + gamma_hat*dt
        return np.stack([x_prime, y_prime, theta_prime], axis=-1)


def get_odometry_command(odom_pose, odom_pose_prev):
    """ Transform robot poses taken from odometry to u command
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
    """ Sample odometry motion model.
    Arguments:
    x -- pose of the robot before moving [x, y, theta]
    u -- odometry reading obtained from the robot [rot1, trans, rot2]
    a -- noise parameters of the motion model [a1, a2, a3, a4] or [std_rot1, std_trans, std_rot2]
    """

    sigma = np.ones((3))
    if a.shape == u.shape:
        sigma = a
    else:
        sigma[0] = a[0]*abs(u[0]) + a[1]*abs(u[1])
        sigma[1] = a[2]*abs(u[1]) + a[3]*(abs(u[0])+abs(u[2]))
        sigma[2] = a[0]*abs(u[2]) + a[1]*abs(u[1])
    
    # noisy odometric transformations: 1 translation and 2 rotations
    delta_hat_r1 = u[0] + np.random.normal(0, sigma[0])
    delta_hat_t = u[1] + np.random.normal(0, sigma[1])
    delta_hat_r2 = u[2] + np.random.normal(0, sigma[2])

    if len(x.shape) == 1: # robot predict
        # new pose predicted for the robot
        x_prime = x[0] + delta_hat_t * np.cos(x[2] + delta_hat_r1)
        y_prime = x[1] + delta_hat_t * np.sin(x[2] + delta_hat_r1)
        theta_prime = x[2] + delta_hat_r1 + delta_hat_r2
        return np.array([x_prime, y_prime, theta_prime])
    
    else: # particles predict
        x_prime = x[:, 0] + delta_hat_t * np.cos(x[:, 2] + delta_hat_r1)
        y_prime = x[:, 1] + delta_hat_t * np.sin(x[:, 2] + delta_hat_r1)
        theta_prime = x[:, 2] + delta_hat_r1 + delta_hat_r2
        return np.stack([x_prime, y_prime, theta_prime], axis=-1)


def landmark_range_bearing_model(robot_pose, landmark, sigma):
    """""
    Sampling z from landmark model for range and bearing
    robot pose: can be the estimated robot pose or the particles
    """""

    if len(robot_pose.shape) == 1:
        r_ = np.linalg.norm(robot_pose[0:2] - landmark) + np.random.normal(0., sigma[0])
        phi_ = np.arctan2(landmark[1] - robot_pose[1], landmark[0] - robot_pose[0]) - robot_pose[2] + np.random.normal(0., sigma[1])
        return np.array([r_, phi_])
    else:
        r_ = np.linalg.norm(robot_pose[:, 0:2] - landmark, axis=1) + np.random.normal(0., sigma[0])
        phi_ = np.arctan2(landmark[1] - robot_pose[:,1], landmark[0] - robot_pose[:,0]) - robot_pose[:,2] + np.random.normal(0., sigma[1])
        return np.stack([r_, phi_], axis=-1)


def landmark_range_bearing_sensor(robot_pose, landmark, sigma, max_range=6.0, fov=math.pi/2):
    """""
    Simulate the detection of a landmark with a virtual sensor able to estimate range and bearing
    """""
    z = landmark_range_bearing_model(robot_pose, landmark, sigma)

    # filter z for a more realistic sensor model (add a max range distance and a FOV)
    if z[0] > max_range or abs(z[1]) > fov / 2:
        return None

    return z
