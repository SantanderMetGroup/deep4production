import numpy as np

def log1p(array, back=False):
    """Forward: log(1+x), Backward: exp(x)-1."""
    return np.expm1(array) if back else np.log1p(array)  


def cubic(array, back=False):
    """Forward: cube, Backward: cube root."""
    return array ** (1/3) if back else array ** (3)


