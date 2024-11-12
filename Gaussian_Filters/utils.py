import numpy as np
from math import atan2
from numpy import linalg as la


def residual(a, b, **kwargs):
    """
    Compute the residual between expected and sensor measurements, normalizing angles between [-pi, pi)
    If passed, angle_indx should indicate the positional index of the angle in the measurement arrays a and b

    Returns:
        y [np.array] : the residual between the two states
    """
    y = a - b

    if 'angle_idx' in kwargs:
        angle_idx = kwargs["angle_idx"]
        theta = y[angle_idx]
        y[angle_idx] = normalize_angle(theta)
        
    return y


def _error(actual: np.ndarray, predicted: np.ndarray):
    """ Simple error """
    return actual - predicted

def mse(actual: np.ndarray, predicted: np.ndarray):
    """ Mean Squared Error """
    if len(actual.shape)==1 and len(predicted.shape)==1:
        return np.mean(np.square(_error(actual, predicted)), axis=0)
    return np.mean(np.sum(np.square(_error(actual, predicted)), axis=1), axis=0)

def rmse(actual: np.ndarray, predicted: np.ndarray):
    """ Root Mean Squared Error """
    return np.sqrt(mse(actual, predicted))

def mae(error: np.ndarray):
    """ Mean Absolute Error """
    return np.mean(np.abs(error))

def normalize(arr: np.ndarray):
    """ normalize vector for plots """
    return (arr - np.min(arr)) / (np.max(arr) - np.min(arr))

def normalize_angle(theta):
    """
    Normalize angles between [-pi, pi)
    """
    theta = theta % (2 * np.pi)  # force in range [0, 2 pi)
    if theta > np.pi:  # move to [-pi, pi)
        theta -= 2 * np.pi
    
    return theta

def state_mean(sigmas, Wm):
    x = np.zeros(3)

    sum_sin = np.sum(np.dot(np.sin(sigmas[:, 2]), Wm))
    sum_cos = np.sum(np.dot(np.cos(sigmas[:, 2]), Wm))
    x[0] = np.sum(np.dot(sigmas[:, 0], Wm))
    x[1] = np.sum(np.dot(sigmas[:, 1], Wm))
    x[2] = atan2(sum_sin, sum_cos)
    return x

def z_mean(sigmas, Wm):
    dim_z = sigmas.shape[1]
    x = np.zeros(dim_z)

    sum_sin = np.sum(np.dot(np.sin(sigmas[:, 1]), Wm))
    sum_cos = np.sum(np.dot(np.cos(sigmas[:, 1]), Wm))

    x[0] = np.sum(np.dot(sigmas[:,0], Wm))
    x[1] = atan2(sum_sin, sum_cos)
    return x

def nearestPD(A):
    """Find the nearest positive-definite matrix to input
    A Python/Numpy port of John D'Errico's `nearestSPD` MATLAB code [1], which
    credits [2].
    [1] https://www.mathworks.com/matlabcentral/fileexchange/42885-nearestspd
    [2] N.J. Higham, "Computing a nearest symmetric positive semidefinite
    matrix" (1988): https://doi.org/10.1016/0024-3795(88)90223-6
    """

    B = (A + A.T) / 2
    _, s, V = la.svd(B)

    H = np.dot(V.T, np.dot(np.diag(s), V))

    A2 = (B + H) / 2

    A3 = (A2 + A2.T) / 2

    if isPD(A3):
        return A3

    spacing = np.spacing(la.norm(A))
    # The above is different from [1]. It appears that MATLAB's `chol` Cholesky
    # decomposition will accept matrixes with exactly 0-eigenvalue, whereas
    # Numpy's will not. So where [1] uses `eps(mineig)` (where `eps` is Matlab
    # for `np.spacing`), we use the above definition. CAVEAT: our `spacing`
    # will be much larger than [1]'s `eps(mineig)`, since `mineig` is usually on
    # the order of 1e-16, and `eps(1e-16)` is on the order of 1e-34, whereas
    # `spacing` will, for Gaussian random matrixes of small dimension, be on
    # othe order of 1e-16. In practice, both ways converge, as the unit test
    # below suggests.
    I = np.eye(A.shape[0])
    k = 1
    while not isPD(A3):
        mineig = np.min(np.real(la.eigvals(A3)))
        A3 += I * (-mineig * k**2 + spacing)
        k += 1

    return A3


def isPD(B):
    """Returns true when input is positive-definite, via Cholesky"""
    try:
        _ = la.cholesky(B)
        return True
    except la.LinAlgError:
        return False
