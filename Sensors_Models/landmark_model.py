import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from  matplotlib.patches import Arc
from Sensors_Models.utils import compute_p_hit_dist
arrow = u'$\u2191$'

def landmark_range_bearing_sensor(robot_pose, landmark, sigma, max_range=6.0, fov=math.pi/2):
    """""
    Simulate the detection of a landmark with a virtual sensor able to estimate range and bearing
    """""
    m_x, m_y = landmark[:]
    x, y, theta = robot_pose[:]

    r_ = math.dist([x, y], [m_x, m_y]) + np.random.normal(0., sigma[0])
    phi_ = math.atan2(m_y - y, m_x - x) - theta + np.random.normal(0., sigma[1])

    # filter z for a more realistic sensor simulation (add a max range distance and a FOV)
    if r_ > max_range or abs(phi_) > fov / 2:
        return None

    return [r_, phi_]

def landmark_model_prob(z, landmark, robot_pose, max_range, fov, sigma):
    """""
    Landmark sensor model algorithm:
    Inputs:
      - z: the measurements features (range and bearing of the landmark from the sensor) [r, phi]
      - landmark: the landmark position in the map [m_x, m_y]
      - x: the robot pose [x,y,theta]
    Outputs:
     - p: the probability p(z|x,m) to obtain the measurement z from the state x
        according to the estimated range and bearing
    """""
    m_x, m_y = landmark[:]
    x, y, theta = robot_pose[:]
    sigma_r, sigma_phi = sigma[:]

    r_hat = math.dist([x, y], [m_x, m_y])
    phi_hat = math.atan2(m_y - y, m_x - x) - theta
    p = compute_p_hit_dist(z[0] - r_hat, max_range, sigma_r) * compute_p_hit_dist(z[1] - phi_hat, fov/2, sigma_phi)

    return p

def landmark_model_sample_pose(z, landmark, sigma):
    """""
    Sample a robot pose from the landmark model
    Inputs:
        - z: the measurements features (range and bearing of the landmark from the sensor) [r, phi]
        - landmark: the landmark position in the map [m_x, m_y]
        - sigma: the standard deviation of the measurement noise [sigma_r, sigma_phi]
    Outputs:
        - x': the sampled robot pose [x', y', theta']
    """""
    m_x, m_y = landmark[:]
    sigma_r, sigma_phi = sigma[:]

    gamma_hat = np.random.uniform(0, 2*math.pi)
    r_hat = z[0] + np.random.normal(0, sigma_r)
    phi_hat = z[1] + np.random.normal(0, sigma_phi)

    x_ = m_x + r_hat * math.cos(gamma_hat)
    y_ = m_y + r_hat * math.sin(gamma_hat)
    theta_ = gamma_hat - math.pi - phi_hat

    return np.array([x_, y_, theta_])


def plot_sampled_poses(robot_pose, z, landmark, sigma):
    """""
    Plot sampled poses from the landmark model
    """""
    # plot samples poses
    for i in range(500):
        x_prime = landmark_model_sample_pose(z, landmark, sigma)
        # plot robot pose
        rotated_marker = mpl.markers.MarkerStyle(marker=arrow)
        rotated_marker._transform = rotated_marker.get_transform().rotate_deg(math.degrees(x_prime[2])-90)
        plt.scatter(x_prime[0], x_prime[1], marker=rotated_marker, s=80, facecolors='none', edgecolors='b')
    
    # plot real pose
    rotated_marker = mpl.markers.MarkerStyle(marker=arrow)
    rotated_marker._transform = rotated_marker.get_transform().rotate_deg(math.degrees(robot_pose[2])-90)
    plt.scatter(robot_pose[0], robot_pose[1], marker=rotated_marker, s=140, facecolors='none', edgecolors='r')

    plt.xlabel("x-position [m]")
    plt.ylabel("y-position [m]")
    plt.title("Landmark Model Pose Sampling")
    # plt.savefig("landmark_model_sampling.pdf")
    plt.show()


def plot_landmarks(landmarks, robot_pose, z, p_z, max_range=6.0, fov=math.pi/4):
    """""
    Plot landmarks, robot pose with sensor FOV, and detected landmarks with associated probability
    """""
    x, y, theta = robot_pose[:]

    start_angle = theta + fov/2
    end_angle = theta - fov/2

    plt.figure()
    ax = plt.gca()
    # plot robot pose
    # find the virtual end point for orientation
    endx = x + 0.5 * math.cos(theta)
    endy = y + 0.5 * math.sin(theta)
    plt.plot(x, y, 'or', ms=10)
    plt.plot([x, endx], [y, endy], linewidth = '2', color='r')

    # plot FOV
    # get ray target coordinates
    fov_x_left = x + math.cos(start_angle) * max_range
    fov_y_left = y + math.sin(start_angle) * max_range
    fov_x_right = x + math.cos(end_angle) * max_range
    fov_y_right = y + math.sin(end_angle) * max_range

    plt.plot([x, fov_x_left], [y, fov_y_left], linewidth = '1', color='b')
    plt.plot([x, fov_x_right], [y, fov_y_right], linewidth = '1', color='b')

    R = max_range
    a, b = 2*R, 2*R
    arc = Arc((x, y), a, b,
                 theta1=math.degrees(end_angle), theta2=math.degrees(start_angle), color='b', lw=1.2)
    ax.add_patch(arc)

    # plot landmarks
    for i, lm in enumerate(landmarks):
        plt.plot(lm[0], lm[1], "sk", ms=10, alpha=0.7)

    # plot perceived landmarks position and associated probability (color scale)
    lm_z = np.zeros((len(z), 2))
    for i in range(len(z)):
        # draw endpoint with probability from Likelihood Fields
        lx = x + z[i][0] * math.cos(z[i][1]+theta)
        ly = y + z[i][0] * math.sin(z[i][1]+theta)
        lm_z[i, :] = lx, ly
    
    col = np.array(p_z)
    plt.scatter(lm_z[:,0], lm_z[:,1], s=60, c=col, cmap='viridis')
    plt.colorbar()

    plt.show()
    plt.close('all')


def main():
    ##############################
    ### Landmark model example ###
    ##############################

    # robot pose
    robot_pose = np.array([0., 0., math.pi/4])
    # landmarks position in the map
    landmarks = [
                 np.array([5., 2.]),
                 np.array([-2.5, 3.]),
                 np.array([3., 1.5]),
                 np.array([4., -1.]),
                 np.array([-2., -2.])
                 ]
    # sensor parameters
    fov = math.pi/3
    max_range = 6.0
    sigma = np.array([0.3, math.pi/24])

    # compute measurements and associated probability
    z = []
    p = []
    for i in range(len(landmarks)):
        # read sensor measurements (range, bearing)
        z_i = landmark_range_bearing_sensor(robot_pose, landmarks[i], sigma=sigma, max_range=max_range, fov=fov)
         
        if z_i is not None: # if landmark is not detected, the measurement is None
            z.append(z_i)
            # compute the probability for each measurement according to the landmark model algorithm
            p_i = landmark_model_prob(z_i, landmarks[i], robot_pose, max_range, fov, sigma)
            p.append(p_i)

    print("Probability density value:", p)
    # Plot landmarks, robot pose with sensor FOV, and detected landmarks with associated probability
    plot_landmarks(landmarks, robot_pose, z, p, fov=fov)

    ##########################################
    ### Sampling poses from landmark model ###
    ##########################################
    if len(z) == 0:
        print("No landmarks detected!")
        return
    
    # consider only the first landmark detected
    landmark = landmarks[0]
    z = landmark_range_bearing_sensor(robot_pose, landmark, sigma)

    # plot landmark
    plt.plot(landmark[0], landmark[1], "sk", ms=10)
    plot_sampled_poses(robot_pose, z, landmark, sigma)
    
    plt.close('all')

if __name__ == "__main__":
    main()