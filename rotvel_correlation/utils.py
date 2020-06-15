import sys
import os
import warnings
import itertools
import subprocess
import numpy as np
import pandas as pd
import scipy.stats as st


def bayesian_blocks(t):
    """Bayesian Blocks Implementation
    By Jake Vanderplas.  License: BSD
    Based on algorithm outlined in http://adsabs.harvard.edu/abs/2012arXiv1207.5578S
    Parameters
    ----------
    t : ndarray, length N
        data to be histogrammed
    Returns
    -------
    bins : ndarray
        array containing the (N+1) bin edges
    Notes
    -----
    This is an incomplete implementation: it may fail for some
    datasets.  Alternate fitness functions and prior forms can
    be found in the paper listed above.
    """
    # copy and sort the array
    t = np.sort(t)
    N = t.size
    # create length-(N + 1) array of cell edges
    edges = np.concatenate([t[:1],
                            0.5 * (t[1:] + t[:-1]),
                            t[-1:]])
    block_length = t[-1] - edges
    # arrays needed for the iteration
    nn_vec = np.ones(N)
    best = np.zeros(N, dtype=float)
    last = np.zeros(N, dtype=int)
    #-----------------------------------------------------------------
    # Start with first data cell; add one cell at each iteration
    #-----------------------------------------------------------------
    for K in range(N):
        # Compute the width and count of the final bin for all possible
        # locations of the K^th changepoint
        width = block_length[:K + 1] - block_length[K + 1]
        count_vec = np.cumsum(nn_vec[:K + 1][::-1])[::-1]
        # evaluate fitness function for these possibilities
        fit_vec = count_vec * (np.log(count_vec) - np.log(width))
        fit_vec -= 4  # 4 comes from the prior on the number of changepoints
        fit_vec[1:] += best[:K]
        # find the max of the fitness: this is the K^th changepoint
        i_max = np.argmax(fit_vec)
        last[K] = i_max
        best[K] = fit_vec[i_max]
    #-----------------------------------------------------------------
    # Recover changepoints by iteratively peeling off the last block
    #-----------------------------------------------------------------
    change_points =  np.zeros(N, dtype=int)
    i_cp = N
    ind = N
    while True:
        i_cp -= 1
        change_points[i_cp] = ind
        if ind == 0:
            break
        ind = last[ind - 1]
    change_points = change_points[i_cp:]
    return edges[change_points]

def freedman_diaconis(x: np.ndarray) -> np.ndarray:
    """
    The binwidth is proportional to the interquartile range (IQR) and inversely proportional to cube root of a.size.
    Can be too conservative for small datasets, but is quite good for large datasets. The IQR is very robust to
    outliers.

    :param x: np.ndarray
        The 1-dimensional x-data to bin.
    :return: np.ndarray
        The bins edges computed using the FD method.
    """
    return np.histogram_bin_edges(x, bins='fd')

def equal_number_FD(x: np.ndarray) -> np.ndarray:
    """
    Takes the number of bins computed using the FD method, but then selects the bin edges splitting
    the dataset in bins with equal number of data-points.

    :param x: np.ndarray
        The 1-dimensional x-data to bin.
    :return: np.ndarray
        The bins edges computed using the equal-N method.
    """
    nbin = len(np.histogram_bin_edges(x, bins='fd')) - 1
    npt = len(x)
    return np.interp(np.linspace(0, npt, nbin + 1),
                     np.arange(npt),
                     np.sort(x))