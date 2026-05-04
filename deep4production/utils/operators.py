import numpy as np

def log1p(array, back=False):
    """Forward: log(1+x), Backward: exp(x)-1."""
    return np.expm1(array) if back else np.log1p(array)  


def cubic(array, back=False):
    """Forward: cube, Backward: cube root."""
    return array ** (1/3) if back else array ** (3)


def sqrt(array, back=False):
    """Forward: sqrt(x), Backward: x**2 (with negatives clipped to 0).

    Used e.g. for precipitation preprocessing in Addison et al. (2024):
    sqrt reduces skewness; the inverse clips negatives to 0 before squaring,
    since a diffusion sampler can produce small negative values that would
    otherwise become spurious positive precipitation after squaring.
    """
    if back:
        return np.clip(array, 0.0, None) ** 2
    return np.sqrt(array)


