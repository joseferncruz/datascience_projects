# -*- coding: utf-8 -*-
"""
Created on Sun Dec 27 14:48:37 2020

@author: josec
"""

import numpy as np
from scipy import stats


def ecdf(
        data,
        ):
    """Returns x, y ready to plot an Empirical Cumulative Distribition Function.

    Parameters
    ----------
    data : list or int or floats
        List of values to be sorted.

    Returns
    -------
    x : np.array
        Sorted numpy array with data.
    y : np.array

    """

    # sort data
    x = np.sort(data)

    # create linear space between 0 and 1.
    y = np.linspace(0, 1, len(data))
    
    return x, y

###############################################################################

def draw_bs_reps(data, func, size=1):
    """Draw bootstrap replicates.

    Parameters
    ----------
    data : sequence of int/floats

    func : funct
        The test statistic to be used.

    size : opt, int
        The number of replicates to be drawn.

    returns
    bs_replicates : np.array
        Array with bootstrap replicates.
    """

    # Instantiate array to hold the results
    bs_replicates = np.empty(size)

    # iterate size times
    for i in range(size):

        # Get a new sample with replacement
        bs_sample = np.random.choice(data, size=len(data))

        # Perform a test statistic
        bs_replicates[i] = func(bs_sample)

    return bs_replicates

###############################################################################

### To improve documentation.

def draw_bs_pairs(x, y, func, size=1):
    """Perform pairs bootstrap for a single statistic."""

    # Set up array of indices to sample from: inds
    inds = np.arange(len(x))

    # Initialize replicates: bs_replicates
    bs_replicates = np.empty(size)

    # Generate replicates
    for i in range(size):
        bs_inds = np.random.choice(inds, size=len(inds))
        bs_x, bs_y = x[inds], y[inds]
        bs_replicates[i] = func(bs_x, bs_y)

    return bs_replicates


##############################################################################

def pearson_r(x, y):
    """Returns the pearson correlation coef"""
    
    return np.corrcoef(x, y)[0][1]

##############################################################################

def heritability(parents, offspring):
    """Compute the heritability from parent and offspring samples."""
    covariance_matrix = np.cov(parents, offspring)
    return covariance_matrix[0][1] / np.var(parents)