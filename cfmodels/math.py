import numpy as np
import numba as nb

@nb.njit
def sigmoid(x):
    return 1./ (1. + np.exp(-x))