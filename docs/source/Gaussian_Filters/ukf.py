import numpy as np
from numpy.linalg import inv
from scipy.linalg import cholesky, sqrtm

class SigmaPoints:
    def __init__(
        self,
        n=1,
        alpha=0.3, # 0 < alpha < 1 suggested range
        kappa=1, # 3 - dim(x) is a suggested value
        beta=2, # beta=2 is the suggested value for Gaussian problems
        sqrt=cholesky, # cholesky or sqrtm (if working with triangular matrix)
        subtract=np.subtract,
    ):
        """
        Define sigma points methods for the scaled UKF
        """
        self.n = n
        self.num_sigma_points = 2*n + 1
        self.alpha = alpha
        self.beta = beta
        self.kappa = kappa
        self.sqrt = sqrt
        self.subtract = subtract

        self.Wm = np.zeros((self.num_sigma_points))
        self.Wc = np.zeros((self.num_sigma_points))
        self.compute_weights()


    def compute_sigma_points(self, mu, Sigma, **kwargs):
        """ 
        Computes the sigma points for an unscented Kalman filter
        given the mean (mu) and covariance(Sigma) of the filter.

        Parameters
        ----------
        mu : np.array with size equal to state dim x

        Sigma : np.array
           Covariance of the filter.

        Returns
        -------
        sigmas : np.array, of size (2n+1, n)
        """

        n = self.n
        lambda_ = self.alpha**2 * (n + self.kappa) - n
        U = (self.sqrt((lambda_ + n)*Sigma))

        sigmas = np.zeros((2*n+1, n))
        sigmas[0] = mu
        for k in range(n):
            sigmas[k+1]   = self.subtract(mu, -U[k], **kwargs)
            sigmas[n+k+1] = self.subtract(mu, U[k], **kwargs)

        return sigmas

    def compute_weights(self):
        """ 
        Computes the weights for the scaled Unscented Kalman filter.
        """
        n = self.n
        lambda_ = self.alpha**2 * (n +self.kappa) - n

        c = .5 / (n + lambda_)
        self.Wc = np.full(2*n + 1, c)
        self.Wm = np.full(2*n + 1, c)
        self.Wc[0] = lambda_ / (n + lambda_) + (1 - self.alpha**2 + self.beta)
        self.Wm[0] = lambda_ / (n + lambda_)

# TO DO: improve numerical stability
        
class RobotUKF:
    def __init__(
        self,
        dim_x=1,
        dim_u=1,
        dim_z=1,
        points=None,
        eval_gux=None,
    ):
        """
        Initializes the Unscented Kalman filter creating the necessary matrices, and functions
        """
        self.mu = np.zeros((dim_x))  # mean state estimate
        self.Sigma = np.diag([0.01, 0.01, 0.01])  # covariance state estimate
        self.R = np.eye(dim_x)  # covariance state estimate

        # motion model function
        self.eval_gux = eval_gux

        # sigma points
        self.sigmas = points
        self.num_sigma_points = points.num_sigma_points
        self.sigmas_bar_g = np.zeros((self.num_sigma_points, dim_x))
        self.sigmas_bar_h = np.zeros((self.num_sigma_points, dim_z))

        # weights for the means and covariances re-estimation w/ unscented transform
        self.Wm, self.Wc = points.Wm, points.Wc
    
    def predict(self, u, sigma_u, g_extra_args=(), mean_fn=np.dot, residual_fn=np.subtract, **kwargs):
        """
        Update the state prediction using the control input u and compute the relative uncertainty ellipse
        Parameters
        ----------

        u : np.array
            command for this step.

        sigma_u : np.array
            std dev for each component of the command signal

        extra_args : tuple
            any additional required parameter: dt

        Modified variables:
            self.mu: the state prediction
            self.Sigma: the covariance matrix of the state prediction
        """

        # calculate sigma points for given mean and covariance
        sigmas = self.sigmas.compute_sigma_points(self.mu, self.Sigma, **kwargs)

        # pass the sigma points through the non-linear motion model function
        for i in range(self.num_sigma_points):
            self.sigmas_bar_g[i] = self.eval_gux(sigmas[i], u, sigma_u, *g_extra_args)

        self.mu, self.Sigma = self.unscented_transform(self.sigmas_bar_g, self.R, mean_fn, residual_fn, **kwargs)


    def update(self, z, eval_hx, Qt, hx_args=(), mean_fn=np.dot, residual_fn=np.subtract, **kwargs):
        """Performs the update innovation of the extended Kalman filter.

        Parameters
        ----------

        z : np.array
            measurement for this step.

        lmark : [x, y] list-like
            Landmark location in cartesian coordinates.

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

        # transform sigma points into measurement space
        for i in range(self.num_sigma_points):
            self.sigmas_bar_h[i] = eval_hx(self.sigmas_bar_g[i], *hx_args)

        # mean and covariance of measurement model prediction passed through UT
        mu_z, Sigma_z = self.unscented_transform(self.sigmas_bar_h, Qt, mean_fn, residual_fn, **kwargs)

        # compute cross variance of the state and the measurements
        Sigma_xz = np.zeros((self.mu.shape[0], z.shape[0]))
        for i in range(self.num_sigma_points):
            Sigma_xz += self.Wc[i] * np.outer(residual_fn(self.sigmas_bar_g[i], self.mu),
                                        residual_fn(self.sigmas_bar_h[i], mu_z))

        K = np.dot(Sigma_xz, inv(Sigma_z)) # Kalman gain

        y = residual_fn(z, mu_z, **kwargs)
        self.mu = self.mu + np.dot(K, y)
        self.Sigma = self.Sigma - np.dot(K, Sigma_z).dot(K.T)


    def unscented_transform(self, sigma_points, Q, mean_fn=np.dot, residual_fn=np.subtract, **kwargs):
        """
        Computes unscented transform of a set of sigma points and weights.
        Returns the mean and covariance in a tuple.

        Parameters
        ----------

        mean_fn : callable (sigma_points, weights), optional
            Function that computes the mean of the provided sigma points
            and weights. Use this if your state variable contains nonlinear
            values such as angles which cannot be summed.

        residual_fn : callable (x, y), optional
            Function that computes the residual (difference) between x and y.
            You will have to supply this if your state variable cannot support
            subtraction, such as angles.

        Returns
        -------

        x : ndarray 
            Mean of the sigma points after passing through the transform.

        Sigma : ndarray
            covariance of the sigma points after passing through the transform.

        """

        kmax, n = sigma_points.shape

        # reconstructed mean as weighted average of sigma points and weights
        if mean_fn is None:
            mu = np.dot(self.Wm, sigma_points) 
        else:
            mu = mean_fn(sigma_points, self.Wm)

        # new covariance is the sum of the outer product of the residuals
        # times the weights

        if residual_fn is np.subtract or residual_fn is None:
            y = sigma_points - mu[np.newaxis, :]
            Sigma = np.dot(y.T, np.dot(np.diag(self.Wc), y))
        else:
            Sigma = np.zeros((n, n))
            for k in range(kmax):
                y = residual_fn(sigma_points[k], mu, **kwargs)
                Sigma += self.Wc[k] * np.outer(y, y)

        Sigma += Q

        return mu, Sigma