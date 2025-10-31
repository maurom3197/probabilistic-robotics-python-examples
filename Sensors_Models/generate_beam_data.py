import numpy as np
import matplotlib.pyplot as plt
from Sensors_Models.utils import plot_sampling_dist
    
def sample_noisy_range(rng, z_max):
    # simulate laser measurement adding noise to the obtained by casting rays in the map
    z = rng + np.random.normal(0, 0.1, size=1).item() + np.random.binomial(4, 0.001, 1).item() + 10*np.random.binomial(2, 0.001, 1).item()
    z = np.clip(z, 0., z_max)
    return z

def sample_from_z_dist(z, z_max, mix_density, sigma, lamb_short):
    # mode selection 
    mode = np.random.choice(len(mix_density), 1, p=list(mix_density)).item()

    # sample
    if mode == 0:       # hit mode
            while True:
                out = np.random.normal(z, sigma, size=1).item()
                if (out >= 0) and (out <=z_max):
                    break
            return out
    elif mode == 1:     #Short mode
        while True:
            out = np.random.exponential(scale=1/lamb_short, size=1).item()
            if (out >= 0 ) and (out <=z):
                break
        return out
    elif mode == 2:     # Max
        return z_max
    else:              #Rand
        out = np.random.uniform(0,z_max,size = 1).item()
        return out

def main():

    # z_star is the reference measurement computed with ray casting from the map
    z_star = 5.5*np.ones((6)) # in this case we only assume a fixed range value 
    #z_star = np.load('ray_casting_z.npz')['D']
    z_max = 8.0  # Sensor range
    data_num = 5000
    outputs = np.zeros((2,data_num))

    sampling_from_p_z = True

    ###############################################
    #### Generate pseudo-realistic laser range ####
    ###############################################
    if not sampling_from_p_z:
        for i in range(data_num):
            z = sample_noisy_range(z_star[0], z_max) # little noise
            outputs[:,i] = z_star[0], z


        plt.plot(outputs[1,:], 'o')
        plt.ylim([0, z_max+0.5])
        plt.title("Real measurements z")
        plt.savefig("z_real.pdf")
        plt.show()

        np.savez('beam_range_data.npz', D=outputs)
        plot_sampling_dist(outputs[1,:], fig_name="z_generated_hist.pdf")

        plt.close('all')    

    ##############################################
    ### Generate data sampling from p(z|x,m)  ####
    ##############################################
    else:
        mix_density, sigma, lamb_short = [0.8, 0.1, 0.05, 0.05], 1.0, 0.9

        for i in range(data_num):
            z = sample_from_z_dist(z_star[0], z_max, mix_density, sigma, lamb_short) 
            outputs[:,i] = z_star[0], z


        plt.plot(outputs[1,:], 'o')
        plt.ylim([0, z_max+0.5])
        plt.title("Real measurements z")
        plt.savefig("z_real.pdf")
        plt.show()

        np.savez('pz_beam_range_data.npz', D=outputs)

        plot_sampling_dist(outputs[1,:], fig_name="z_generated_hist_pz.pdf")
        plt.close('all')

if __name__ == "__main__":
    main()