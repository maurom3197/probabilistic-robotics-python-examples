import numpy as np
import matplotlib.pyplot as plt
from generate_beam_data import plot_sampling_dist, sample_from_z_dist

# Gaussian Function
def gaussian(x, mean, sigma):
    return (1 / np.sqrt(2 * np.pi * sigma ** 2)) * np.exp(- (x - mean) ** 2 / (2 * sigma ** 2))

def evaluate_range_beam_distribution(z, z_star, z_max, _mix_density, _sigma, _lamb_short):
    # Calculate hit mode probability
    normalize_hit = 0.0
    for j in range(int(z_max)):
        normalize_hit += gaussian(j, z_star, _sigma)
    normalize_hit = 1. / normalize_hit

    p_hit = gaussian(z, z_star, _sigma) * normalize_hit

    # Calculate short mode probability
    if z <= z_star:
        normalize_short = 1 - np.exp(-1 * _lamb_short * z_star)
        p_short = _lamb_short * np.exp(-1 * _lamb_short * z) / normalize_short
    else:
        p_short = 0.0

    # Calculate max mode probability
    if z == z_max:
        p_max = 1.0
    else:
        p_max = 0.0

    # Calculate rand mode probability
    p_rand = 1.0 / z_max

    p = np.array([np.float_(p_hit), np.float_(p_short), p_max, p_rand])
    p_z = np.dot(_mix_density, p)  # (p_hit*z_hit) + (p_short*z_short) + (p_max*z_max) + (p_rand*z_rand)

    return p_hit, p_short, p_max, p_rand, p, p_z

def ML_params_estimator(inputData, sensor_max, data_num):
    """
    Expectation Maximation (EM) algorithm to estimate the set of parameters theta of the beam range sensor model

    Load sensor data:
    inputData[0,:] - z distance from ray casting
    inputData[1,:] - laser range z
    """

    # Initial guesses for intrinsic parameter
    _mix_density = np.random.dirichlet(np.ones(4), size=1)  # shape (1,4)
    print("Initial z values:", _mix_density)
    z_hit = _mix_density[:, 0].item()
    z_short = _mix_density[:, 1].item()
    z_max = _mix_density[:, 2].item()
    z_rand = _mix_density[:, 3].item()

    _sigma = float(np.random.randint(1, 10))
    _lamb_short = np.random.rand(1).item()

    convergence = False  # Check for convergence

    n_step = 50000
    steps = 0

    while not convergence:
        # Previous value
        prev = np.array([z_hit, z_short, z_max, z_rand, _sigma, _lamb_short], dtype=float)

        # Sum of e_i
        e_hit_sum = 0.0
        e_short_sum = 0.0
        e_max_sum = 0.0
        e_rand_sum = 0.0

        e_short_list = np.array([])  # For calculate _lamb_short
        _cal_sigma_tmp = 0.0  # For calculate _sigma
        _cal_sigma_tmp_sum = 0.0

        for i in range(data_num):
            z_star = inputData[0, i]  # true range from ray casting (in pseudo code : z_i^*)
            z = inputData[1, i]  # measurement (in pseudo code : z_i)

            p_hit, p_short, p_max, p_rand, p, p_z = evaluate_range_beam_distribution(z, z_star, 
                                                                                     sensor_max, _mix_density, 
                                                                                     _sigma, _lamb_short)

            # Expectation
            e_hit = np.float_(p_hit * z_hit / p_z)
            e_short = np.float_(p_short * z_short / p_z)
            e_max = np.float_(p_max * z_max / p_z)
            e_rand = np.float_(p_rand * z_rand / p_z)

            e_short_list = np.append(e_short_list, e_short)
            _cal_sigma_tmp = e_hit * (z - z_star) ** 2
            _cal_sigma_tmp_sum += _cal_sigma_tmp

            # Sum of expectation
            e_hit_sum += e_hit
            e_short_sum += e_short
            e_max_sum += e_max
            e_rand_sum += e_rand

        # Maximization
        z_hit = e_hit_sum / float(data_num)
        z_short = e_short_sum / float(data_num)
        z_max = e_max_sum / float(data_num)
        z_rand = e_rand_sum / float(data_num)
        _mix_density = [z_hit, z_short, z_max, z_rand]
        _sigma = (_cal_sigma_tmp_sum / e_hit_sum) ** (1 / 2)
        _lamb_short = np.float_(e_short_sum / (np.matmul(e_short_list.reshape(1, -1), inputData[1, :])))

        # Current value
        cur = np.array([z_hit, z_short, z_max, z_rand, _sigma, _lamb_short], dtype=float)

        # Check for convergence
        convergence = np.allclose(prev, cur, rtol=1e-06, atol=1e-03)
        if steps==n_step:
            convergence = True
        if convergence:
            print("n steps: ", steps)
        else:
            steps += 1

    # Result
    print("z_hit : ", z_hit)
    print("z_short : ", z_short)
    print("z_max : ", z_max)
    print("z_rand : ", z_rand)
    print("_sigma : ", _sigma)
    print("_lamb_short : ", _lamb_short)
    return cur

def main():
    # choose input range data
    #inputData = np.load('beam_range_data.npz')['D'] # more realistic
    inputData = np.load('noisy_beam_range_data.npz')['D'] # more noisy

    z_max = 8.0  # Sensor max range
    data_num = 5000

    theta = ML_params_estimator(inputData, z_max, data_num)
    np.savez('theta_ML.npz', D=theta)
    EM_samples = np.zeros((data_num))

    # use the beam range model distribution to sample
    for i in range(5000):
        z_star = inputData[0, i]
        z = sample_from_z_dist(z_star, z_max=z_max, mix_density=theta[:-2], sigma=theta[-2], lamb_short=theta[-1])
        EM_samples[i] = z

    plt.plot(EM_samples, 'o')
    plt.title("Estimated measurements z*")
    plt.savefig("z_estimate_EM.pdf")
    plt.show()

    plot_sampling_dist(EM_samples, fig_name="z_EM_hist.pdf")

if __name__ == "__main__":
    main()

