import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

arrow = u'$\u2191$'

# Gaussian Function
def gaussian(x, mean, sigma):
    return (1 / np.sqrt(2 * np.pi * sigma ** 2)) * np.exp(- (x - mean) ** 2 / (2 * sigma ** 2))

def normalized_gaussian(x, max_x, _sigma):
    normalize_hit = 0.0
    for j in range(round(max_x)):
        normalize_hit += gaussian(j, 0., _sigma)
    normalize_hit = 1. / normalize_hit

    p = gaussian(x, 0., _sigma)*normalize_hit

    return p

def landmark_range_bearing_sensor(robot_pose, landmark, max_range=6.0, fov=math.pi/2):
    """""
    Simulate the detection of a landmark with a virtual sensor able to estimate range and bearing
    """""
    m_x, m_y = landmark[:]
    x, y, _ = robot_pose[:]

    r_ = math.dist([x, y], [m_x, m_y]) + np.random.uniform(-0.3, 0.3)
    phi_ = math.atan2(m_y - y, m_x - x) + np.random.uniform(-math.pi/24, math.pi/24)

    # filter z for a more realistic sensor simulation (add a max range distance and a FOV)
    if r_ > max_range or abs(phi_) > fov / 2:
        return None

    return [r_, phi_]

def landmark_model_prob(z, landmark, robot_pose, sigma):
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
    x, y, _ = robot_pose[:]
    sigma_r, sigma_phi = sigma[:]

    r_hat = math.dist([x, y], [m_x, m_y])
    phi_hat = math.atan2(m_y - y, m_x - x)
    p = normalized_gaussian(z[0] - r_hat, 6.0, sigma_r) * normalized_gaussian(z[1] - phi_hat, math.pi/4, sigma_phi)

    return p

def landmark_model_sample_pose(z, landmark, sigma):

    m_x, m_y = landmark[:]
    sigma_r, sigma_phi = sigma[:]

    gamma_hat = np.random.uniform(0, 2*math.pi)
    r_hat = z[0] + np.random.normal(0, sigma_r)
    phi_hat = z[1] + np.random.normal(0, sigma_phi)

    x_ = m_x + r_hat * math.cos(gamma_hat)
    y_ = m_y + r_hat * math.sin(gamma_hat)
    theta_ = gamma_hat - math.pi - phi_hat

    return np.array([x_, y_, theta_])


def plot_sampled_poses(x, x_prime):

    rotated_marker = mpl.markers.MarkerStyle(marker=arrow)
    rotated_marker._transform = rotated_marker.get_transform().rotate_deg(math.degrees(x[2])-90)
    plt.scatter(x[0], x[1], marker=rotated_marker, s=100, facecolors='none', edgecolors='b')

    for x_ in x_prime[:200]:
        rotated_marker = mpl.markers.MarkerStyle(marker=arrow)
        rotated_marker._transform = rotated_marker.get_transform().rotate_deg(math.degrees(x_[2])-90)
        plt.scatter(x_[0], x_[1], marker=rotated_marker, s=40, facecolors='none', edgecolors='r')

    plt.xlabel("x-position [m]")
    plt.ylabel("y-position [m]")
    plt.title("landmark model - pose sampling")
    plt.savefig("landmark_model_sampling.pdf")
    plt.show()


def plot_landmarks(landmarks, robot_pose, z, p_z, fov=math.pi/2):

    x, y, theta = robot_pose[:]

    start_angle = theta + fov/2
    end_angle = theta - fov/2

    plt.figure()
    # plot robot pose
    # find the virtual end point for orientation
    endx = x + 0.5 * math.sin(theta)
    endy = y + 0.5 * math.cos(theta)
    plt.plot(y, x, 'or', ms=10)
    plt.plot([y, endy], [x, endx], linewidth = '2', color='r')

    # plot FOV
    # get ray target coordinates
    fov_x_left = x + math.sin(start_angle) * 6.0
    fov_y_left = y + math.cos(start_angle) * 6.0
    fov_x_right = x + math.sin(end_angle) * 6.0
    fov_y_right = y + math.cos(end_angle) * 6.0

    plt.plot([y, fov_y_left], [x, fov_x_left], linewidth = '1', color='b')
    plt.plot([y, fov_y_right], [x, fov_x_right], linewidth = '1', color='b')

    # plot landmarks
    for i, lm in enumerate(landmarks):
        plt.plot(lm[1], lm[0], "sk", ms=10, alpha=0.7)

    # plot perceived landmarks position and associated probability (color scale)
    lm_z = np.zeros((len(z), 2))
    for i in range(len(z)):
        # draw endpoint with probability from Likelihood Fields
        lx = x + z[i][0] * math.cos(z[i][1])
        ly = y + z[i][0] * math.sin(z[i][1])
        lm_z[i, :] = lx, ly
    
    col = np.array(p_z)
    plt.scatter(lm_z[:,1], lm_z[:,0], s=60, c=col, cmap='viridis')
    plt.colorbar()

    plt.show()
    plt.close('all')


def main():

    robot_pose = np.array([1., 0., math.pi/2])
    landmarks = [
                 np.array([5., 2.]),
                 np.array([8., 3.]),
                 np.array([3., 1.5]),
                 np.array([4., -1.]),
                 np.array([2., -2.])
                 ]
    
    sigma = np.array([0.3, math.pi/12])

    z = []
    p = []
    for i in range(len(landmarks)):
        # read sensor measurements (range, bearing)
        z_i = landmark_range_bearing_sensor(robot_pose, landmarks[i])
         
        if z_i is not None: # if landmark is not detected, the measurement is None
            z.append(z_i)
            # compute the probability for each measurement according to the landmark model algorithm
            p_i = landmark_model_prob(z_i, landmarks[i], robot_pose, sigma)
            p.append(p_i)

    print("Measured landmarks:", z)
    print("Probability density value:", p)
    # Plot landmarks, robot pose with sensor FOV, and detected landmarks with associated probability
    plot_landmarks(landmarks, robot_pose, z, p)

    ##########################################
    ### Sampling poses from landmark model ###
    ##########################################

    landmark = landmarks[0]
    z = landmark_range_bearing_sensor(robot_pose, landmark)

    # plot landmark
    plt.plot(landmark[1], landmark[0], "sk", ms=10)

    # plot samples poses
    for i in range(300):
        x_prime = landmark_model_sample_pose(z, landmark, sigma)
        # plot robot pose
        rotated_marker = mpl.markers.MarkerStyle(marker=arrow)
        rotated_marker._transform = rotated_marker.get_transform().rotate_deg(math.degrees(x_prime[2])-90)
        plt.scatter(x_prime[1], x_prime[0], marker=rotated_marker, s=80, facecolors='none', edgecolors='b')
    
    # plot real pose
    rotated_marker = mpl.markers.MarkerStyle(marker=arrow)
    rotated_marker._transform = rotated_marker.get_transform().rotate_deg(math.degrees(robot_pose[2])-90)
    plt.scatter(robot_pose[1], robot_pose[0], marker=rotated_marker, s=140, facecolors='none', edgecolors='r')
    plt.show()
    
    plt.close('all')

if __name__ == "__main__":
    main()