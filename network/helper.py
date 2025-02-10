"""
@author: juschu

auxiliary functions needed for defining model
"""

##############################
#### imports and settings ####
##############################
import itertools
import numpy as np


def Gaussian2D(Amp, Shape, Sig):
    """
    Computes a 2D gaussian with given amplitude, shape and width (sigma)
    """
    Sig = np.array(Sig)
    # If one or both of the shape values are zero replace it by 6*sigma
    Shape += (Shape == np.zeros(2)) * np.ceil(3 * Sig).astype('int')
    # Decrease even shape values by 1
    Shape -= (Shape % 2 == 0)
    # Calculate the center of the shape
    G_Center = (Shape - 1) // 2
    # gauss = Amp * exp(-((x - cx)^2 / (2*sx^2) + (y - cy)^2 / (2*sy^2)))
    gauss = np.array(np.meshgrid(*list(map(np.arange, Shape))))
    gauss = 0.5 * (gauss - G_Center[:, None, None])**2 / Sig[:, None, None]**2
    gauss = Amp * np.exp(-np.sum(gauss, axis=0))
    return gauss.transpose()


def positive(Arr):
    """
    Removes values smaller than zero from the given Array
    """
    A = np.array(Arr)  # To avoid overwriting the input, a new array is created
    A[A < 0] = 0
    return A


def rangeX(iterations):
    """
    Multidimensional iterator using all possible combinations within the given
    boundaries e.g. rangeX((2,3)) => 00 01 02 10 11 12
    """
    if not isinstance(iterations, (tuple)):
        raise AttributeError
    return itertools.product(*list(map(range, iterations)))
