import numpy as np

def residual(a, b, angle_idx=None):
    """
    Compute the residual between expected and sensor measurements, normalizing angles between [-pi, pi)
    If passed, angle_indx should indicate the positional index of the angle in the measurement arrays a and b

    Returns:
        y [np.array] : the residual between the two states
    """

    y = a - b

    if angle_idx is not None:
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