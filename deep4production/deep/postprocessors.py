import numpy as np

def standard(array):
    return array

def from_bergamma_to_pred(array, threshold):
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
    # Get the parameters of the Gaussian distribution.
    mean = array[:, 0, ...]
    log_var = array[:, 1, ...]
    s_dev = np.exp(log_var) ** (1/2) # log_var --> var --> std

    # Return: sample from the gaussian distribution
    return np.random.normal(loc=mean, scale=s_dev)[None,:]