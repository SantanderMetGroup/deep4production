import numpy as np

def standard(array):
    """
    Standard postprocessing function.
    Purpose: Returns the input array unchanged.
    Parameters:
        array (np.ndarray): Input array.
    Returns:
        np.ndarray: Unchanged input array.
    """
    return array

def from_bergamma_to_pred(array, threshold):
    """
    Postprocessing for Bernoulli-Gamma model output.
    Purpose: Samples occurrence and amount from Bernoulli and Gamma distributions.
    Parameters:
        array (np.ndarray): Model output array with p, log(shape), log(scale).
        threshold (float): Threshold to add to sampled amount.
    Returns:
        np.ndarray: Combined occurrence and amount array.
    """
    # Get the parameters of the Bernoulli and Gamma distributions.
    p = array[:, 0, ...]
    shape = np.exp(array[:, 1, ...])
    scale = np.exp(array[:, 2, ...])

    # Compute the occurrence
    p_random = np.random.uniform(0, 1, p.shape)
    ocurrence = (p >= p_random) * 1 

    # Compute the amount
    amount = np.random.gamma(shape=shape, scale=scale)
    amount = amount + threshold

    # Return: combine ocurrence and amount
    return ocurrence[None,:] * amount[None,:]


def from_gaussian_to_pred(array):
    """
    Postprocessing for Gaussian model output.
    Purpose: Samples from Gaussian distribution using mean and log variance.
    Parameters:
        array (np.ndarray): Model output array with mean and log_var.
    Returns:
        np.ndarray: Sampled array from Gaussian distribution.
    """
    # Get the parameters of the Gaussian distribution.
    mean = array[:, 0, ...]
    log_var = array[:, 1, ...]
    s_dev = np.exp(log_var) ** (1/2) # log_var --> var --> std

    # Return: sample from the gaussian distribution
    return np.random.normal(loc=mean, scale=s_dev)[None,:]