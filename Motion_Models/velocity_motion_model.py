
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from math import cos, sin, degrees
import matplotlib as mpl

arrow = u'$\u2191$'

def sample_normal_distribution(sigma_sqrd):
    return 0.5 * np.sum(np.random.default_rng().uniform(-np.sqrt(sigma_sqrd), np.sqrt(sigma_sqrd), 12))

def evaluate_sampling_dist(mu, sigma, n_samples, sample_function):

    n_bins = 100
    samples = []

    for i in range(n_samples):
        samples.append(sample_function(mu, sigma))

    print("%30s : mean = %.3f, std_dev = %.3f" % ("Normal", np.mean(samples), np.std(samples)))

    count, bins, ignored = plt.hist(samples, n_bins)
    plt.plot(bins, norm(mu, sigma).pdf(bins), linewidth=2, color='r')
    plt.xlim([mu - 5*sigma, mu + 5*sigma])
    plt.title("Normal distribution of samples")
    plt.grid()
    plt.savefig("gaussian_dist.pdf")
    plt.show()

def sample_velocity_motion_model(x, u, a, dt):
    """ Sample velocity motion model.
    Arguments:
    x -- pose of the robot before moving [x, y, theta]
    u -- velocity reading obtained from the robot [v, w]
    a -- noise parameters of the motion model [a1, a2, a3, a4, a5, a6]
    dt -- time interval of prediction
    """
    v_hat = u[0] + np.random.normal(0, a[0]*u[0]**2 + a[1]*u[1]**2)
    w_hat = u[1] + np.random.normal(0, a[2]*u[0]**2 + a[3]*u[1]**2)
    gamma_hat = np.random.normal(0, a[4]*u[0]**2 + a[5]*u[1]**2)

    r = v_hat/w_hat
    x_prime = x[0] - r*sin(x[2]) + r*sin(x[2]+w_hat*dt)
    y_prime = x[1] + r*cos(x[2]) - r*cos(x[2]+w_hat*dt)
    theta_prime = x[2] + w_hat*dt + gamma_hat*dt

    return np.array([x_prime, y_prime, theta_prime])


def main():
    plt.close('all')
    n_samples = 5000
    n_bins = 100
    dt = 0.5

    x = [2, 4, 0]
    u = [0.8, 0.6]
    a = [0.001, 0.01, 0.1, 0.2, 0.05, 0.05] # noise variance

    x_prime = np.zeros([n_samples, 3])
    for i in range(n_samples):
        x_prime[i,:] = sample_velocity_motion_model(x, u, a, dt)

    ###################################
    ######### Plot x samples ##########
    ###################################
       
    mu = np.mean(x_prime, axis=0)
    sigma = np.std(x_prime, axis=0)
    evaluate_sampling_dist(mu[0], sigma[0], n_samples, np.random.normal)

    ###################################
    ### Sampling the velocity model ###
    ###################################

    rotated_marker = mpl.markers.MarkerStyle(marker=arrow)
    rotated_marker._transform = rotated_marker.get_transform().rotate_deg(degrees(x[2])-90)
    plt.scatter(x[0], x[1], marker=rotated_marker, s=100, facecolors='none', edgecolors='b')

    for x_ in x_prime[:200]:
        rotated_marker = mpl.markers.MarkerStyle(marker=arrow)
        rotated_marker._transform = rotated_marker.get_transform().rotate_deg(degrees(x_[2])-90)
        plt.scatter(x_[0], x_[1], marker=rotated_marker, s=40, facecolors='none', edgecolors='r')

    plt.xlabel("x-position [m]")
    plt.ylabel("y-position [m]")
    plt.title("velocity motion model sampling")
    plt.savefig("velocity_samples.pdf")
    plt.show()

    ###################################
    #### Multiple steps of sampling ###
    ###################################

    x = [2, 4, 0]
    a = [0.05, 0.1, 0.05, 0.1, 0.025, 0.025] # noise variance
    cmds = [
        [0.8, 0],
        [0.8, 0.0],
        [0.6, 0.5],
        [0.6, 0.5],
        [0.6, 1.5],
        [0.6, 0],
        [0.8, 0.0],
        [0.7, -0.5],
        [0.7, -0.5],
        [0.5, -1.5],
        [0.8, 0],
        [0.8, 0.0]
    ]

    x_prime = np.zeros([n_samples, 3])
    for t, u in enumerate(cmds):
        for i in range(0, n_samples):
            x_ = x_prime[i,:]
            if t ==0:
                x_prime[i,:] = sample_velocity_motion_model(x, u, a, dt)
            else:
                x_prime[i,:] = sample_velocity_motion_model(x_, u, a, dt)
        
        plt.plot(x_prime[:,0], x_prime[:,1], "r,")
        plt.plot(x[0], x[1], "bo")

        x = np.mean(x_prime, axis=0)
        sigma = np.std(x_prime, axis=0)
        print("mu: ", x, "sigma: ", sigma)
    
    plt.xlabel("x-position [m]")
    plt.ylabel("y-position [m]")
    plt.title("velocity multiple sampling")
    plt.savefig("multi_velocity_samples.pdf")
    plt.show()

    plt.close('all')

if __name__ == "__main__":
    main()
