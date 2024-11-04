
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import math
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


def sample_odometry_motion_model(x, u, a):
    """ Sample odometry motion model.
    Arguments:
    x -- pose of the robot before moving [x, y, theta]
    u -- odometry reading obtained from the robot [rot1, rot2, trans]
    a -- noise parameters of the motion model [a1, a2, a3, a4]
    """
    delta_hat_r1 = u[0] + np.random.normal(0, a[0]*abs(u[0]) + a[1]*abs(u[1]))
    delta_hat_t = u[1] + np.random.normal(0, a[2]*abs(u[1]) + a[3]*(abs(u[0])+abs(u[2])))
    delta_hat_r2 = u[2] + np.random.normal(0, a[0]*abs(u[2]) + a[1]*abs(u[1]))

    x_prime = x[0] + delta_hat_t * math.cos(x[2] + delta_hat_r1)
    y_prime = x[1] + delta_hat_t * math.sin(x[2] + delta_hat_r1)
    theta_prime = x[2] + delta_hat_r1 + delta_hat_r2

    return np.array([x_prime, y_prime, theta_prime])


def main():
    plt.close('all')
    n_samples = 5000
    n_bins = 100

    x = [2, 4, 0]
    u = [-np.pi/2, 1, 0]
    a = [0.1, 0.1, 0.01, 0.01] # noise variance: a0 (rot on rot), a1(t on rot), a2 (t on t), a3 (rot on t)

    x_prime = np.zeros([n_samples, 3])
    for i in range(n_samples):
        x_prime[i,:] = sample_odometry_motion_model(x,u,a)

    ###################################
    ######### Plot x samples ##########
    ###################################
       
    mu = np.mean(x_prime, axis=0)
    sigma = np.std(x_prime, axis=0)
    evaluate_sampling_dist(mu[0], sigma[0], n_samples, np.random.normal)

    ###################################
    ### Sampling the odometry model ###
    ###################################

    rotated_marker = mpl.markers.MarkerStyle(marker=arrow)
    rotated_marker._transform = rotated_marker.get_transform().rotate_deg(math.degrees(x[2])-90)
    plt.scatter(x[0], x[1], marker=rotated_marker, s=100, facecolors='none', edgecolors='b')

    for x_ in x_prime[:200]:
        rotated_marker = mpl.markers.MarkerStyle(marker=arrow)
        rotated_marker._transform = rotated_marker.get_transform().rotate_deg(math.degrees(x_[2])-90)
        plt.scatter(x_[0], x_[1], marker=rotated_marker, s=40, facecolors='none', edgecolors='r')

    plt.xlabel("x-position [m]")
    plt.ylabel("y-position [m]")
    plt.title("Odometry motion model sampling")
    plt.savefig("odometry_samples.pdf")
    plt.show()

    ###################################
    #### Multiple steps of sampling ###
    ###################################

    x = [2, 4, 0]
    a = [0.01, 0.01, 0.01, 0.01] # noise variance: a0 (rot on rot), a1(t on rot), a2 (t on t), a3 (rot on t)
    cmds = [
        [0, 1.5, 0],
        [0, 1.5, 0],
        [0, 1.5, 0],
        [0, 1.5, 0],
        [np.pi/2, 1.0, 0],
        [0, 1.0, 0],
        [0, 1.0, np.pi/2],
        [0, 1.5, 0],
        [0, 1.5, 0],
        [0, 1.5, 0],
        [0, 1.5, 0],
    ]

    x_prime = np.zeros([n_samples, 3])
    for t, u in enumerate(cmds):
        for i in range(0, n_samples):
            x_ = x_prime[i,:]
            if t ==0:
                x_prime[i,:] = sample_odometry_motion_model(x, u, a)
            else:
                x_prime[i,:] = sample_odometry_motion_model(x_, u, a)
        
        plt.plot(x_prime[:,0], x_prime[:,1], "r,")
        plt.plot(x[0], x[1], "bo")

        x = np.mean(x_prime, axis=0)
        sigma = np.std(x_prime, axis=0)
        print("mu: ", x, "sigma: ", sigma)
    
    plt.xlabel("x-position [m]")
    plt.ylabel("y-position [m]")
    plt.title("Odometry multiple sampling")
    plt.savefig("multi_odometry_samples.pdf")
    plt.show()

    plt.close('all')

if __name__ == "__main__":
    main()
