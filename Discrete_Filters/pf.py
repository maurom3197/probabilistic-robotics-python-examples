import numpy as np
import scipy
import scipy.stats

class RobotPF:
    def __init__(
            self,
            dim_x=1,
            dim_u=1,
            eval_gux=None,
            resampling_fn=None,
            boundaries=[(.0, 20.), (.0, 20.), (-np.pi, np.pi)],
            N=1000,
    ):
        """
        Initializes the particle filter creating the necessary matrices

        ---
        dim_x : int
            dimension of the state

        dim_u : int
            dimension of the control input
        
        eval_gux : function
            function to update the motion of the particles
        
        resampling_fn : function
            function for particles resampling based on weights

        boundaries : list of tuples
            boundaries for the particles (size of the map)
            the last element is the limit for the orientation

        N : int
            number of particles
        """

        self.dim_x = dim_x
        self.dim_u = dim_u
        self.mu = np.zeros((dim_x))  # mean state estimate
        self.Sigma = np.eye(dim_x)  # covariance state estimate
        self.Mt = np.eye(dim_u)  # process noise

        self.eval_gux = eval_gux
        self.resampling_fn = resampling_fn
        self.N = N  # number of particles
        self.particles = np.zeros((N, dim_x))  # particles
        self.boundaries = boundaries
        

    def initialize_particles(self, initial_dist_fn=None, initial_dist_args=()):
        """""
        Initialize particles distribution over the map boundaries.
        Default: uniform distribution to solve a global localization problem without initial guess
        """""

        if initial_dist_fn is None or initial_dist_fn == np.random.uniform:
            for i in range(self.dim_x):
                self.particles[:, i] = np.random.uniform(self.boundaries[i][0], self.boundaries[i][1], self.N)
        else:
            self.particles = initial_dist_fn(self.N, *initial_dist_args)
        
        self.weights = np.ones(self.N) / self.N  # weights

    
    def predict(self, u, sigma_u, g_extra_args=()):
        """
        Update the state prediction of each particle using the control input u 
        Paramters
        ---------
        
        u : np.array
            command for this step.
        
        sigma_u : np.array
            std dev for each component of the command signal
        
        extra_args : tuple
            any additional required parameter: dt
            
        Modified variables:
            self.particles: the state prediction
        """
        # Update the state prediction evaluating the motion model
        self.particles = self.eval_gux(self.particles, u, sigma_u, *g_extra_args)


    def update(self, z, sigma_z, eval_hx, hx_args=()):
        """
        Performs the update innovation of the particle filter.
        
        Parameters
        ----------
        
        z : np.array
            measurement for this step.
        
        lm : [x, y] list-like
            landmark position
            
        residual : function (z, z2), optional
            Optional function that computes the residual (difference) between
            the two measurement vectors. If you do not provide this, then the
            built in minus operator will be used. You will normally want to use
            the built in unless your residual computation is nonlinear (for
            example, if they are angles)
        """

        # Convert the measurement to a vector if necessary. Needed for the residual computation
        if np.isscalar(z):
            z = np.asarray([z], float)

        sigma_z = sigma_z * 3.0
        # Evaluate the expected measurement and compute the residual, then update the state prediction
        z_hat = np.zeros((self.N, 2))

        z_hat = eval_hx(self.particles, *hx_args)
        # compute the probability of the measure according to the probabilistic sensor model
        if len(hx_args) > 2: # likelihood field model
            prob = z_hat
        else:                # landmark range bearing model
            prob = scipy.stats.norm(z_hat, sigma_z).pdf(z)

        self.weights *= np.prod(prob, axis=1)


    def normalize_weights(self):
        # particles far from a measurement will give us 0.0 for a probability
        # due to floating point limits. Once we hit zero we can never recover,
        # so add some small nonzero value to all points.
        self.weights += 1.e-10
        self.weights /= sum(self.weights) # normalize
        # print("Weights normalized: ", self.weights)


    def neff(self):
        """
        Compute the effective number of particles
        """
        return int(round(1 / np.sum(self.weights**2), 1))
    

    def estimate(self, mean_fn=np.average, residual_fn=np.subtract, **kwargs):
        """
        Estimate the state of the robot
        """
        kmax, n = self.particles.shape

        if mean_fn is None or mean_fn is np.average:
            # new mean is just the sum of particles * weight
            mu = np.average(self.particles, axis=0, weights=self.weights)

        else:
            # if the state include special cases as angles, use a custom mean function
            mu = mean_fn(self.particles, self.weights)

        if residual_fn is np.subtract or residual_fn is None:
            Sigma = np.average((self.particles - mu)**2, axis=0, weights=self.weights)

        else: 
            # if the state include special cases as angles, use a custom residual function to normalize in (-pi,pi)
            Sigma = np.zeros((n, n))
            for k in range(kmax):
                y = residual_fn(self.particles[k], mu, **kwargs)
                Sigma += self.weights[k] * np.outer(y, y)

        self.mu = mu
        self.Sigma = Sigma

    def resampling(self, resampling_fn, resampling_args=()):
        """
        Estimate the state of the robot
        """
        # get indexes of the resampled particles
        indexes = resampling_fn(*resampling_args)
        # resample according to indexes
        self.particles[:] = self.particles[indexes]
        self.weights.resize(len(self.particles))
        self.weights.fill(1.0 / len(self.weights))
