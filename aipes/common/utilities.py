"""Mathematical and I/O functions."""

import sys

import numpy as np


def calc_rmse(dx):
    """Calculate root-mean-square error from dx = x_predict - x_exact."""
    return np.sqrt(np.sum(dx**2) / dx.size)


def calc_maxresid(dx):
    """Calculate maximum residual from dx = x_predict - x_exact."""
    return np.max(np.fabs(dx))


def calc_mod(x):
    """Calculate modulus for vector x."""
    return np.sqrt(np.sum(x**2))


def echo(text="", rank=0, **kwargs):
    """Print text and flush on master node."""
    if rank == 0:
        print(text, **kwargs)
        sys.stdout.flush()
