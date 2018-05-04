# -*- coding: utf-8 -*-
# pylint: disable-msg=E1101
"""
Created on Mon Jun  4 10:53:10 2012

This module contains the definition of several utility functions to perform
measurements over signals.

@author: T. Teijeiro
"""

from ..units_helper import phys2digital as ph2dg
import scipy.stats
import numpy as np

#Margin to set the baseline level when creating the histogram.
BL_MARGIN = ph2dg(0.02)


def mode(signal):
    """Obtains the mode of a signal fragment"""
    nbins = int((signal.max()-signal.min())/BL_MARGIN) or 1
    hist = np.histogram(signal, nbins)
    peak = hist[0].argmax()
    return hist[1][peak] + (hist[1][peak+1]-hist[1][peak])/2.0


def kurtosis(signal):
    """
    Obtains the kurtosis of a signal fragment. We use this value as an
    indicator of the signal quality.
    """
    return scipy.stats.kurtosis(signal)

def mvkurtosis(arr):
    """
    Obtains the kurtosis of a multivariate array of data. The first dimension
    of *arr* contains the variables, and the second the values.
    """
    n = np.size(arr, 1)
    #Mean vector and corrected covariance matrix
    med = np.mean(arr, 1)
    s = np.cov(arr)*(n-1)/n
    #Eigenvalue and eigenvector calculation
    lamb, v = np.linalg.eig(s)
    si12 = np.dot(np.dot(v,np.diag(1.0/np.sqrt(lamb))),np.transpose(v))
    #Multivariant standardization
    medrep = np.transpose(np.repeat(np.asmatrix(med), n, 0))
    xs = np.dot(np.transpose(arr-medrep),si12)
    #Similarities
    r = np.dot(xs,np.transpose(xs))
    return np.sum(np.diag(r)**2)/n

def mvskewness(arr):
    """
    Obtains the skewness of a multivariate array of data. The first dimension
    of *arr* contains the variables, and the second the values.
    """
    n = np.size(arr, 1)
    #Mean vector and corrected covariance matrix
    med = np.mean(arr, 1)
    s = np.cov(arr)*(n-1)/n
    #Eigenvalue and eigenvector calculation
    lamb, v = np.linalg.eig(s)
    si12 = np.dot(np.dot(v,np.diag(1.0/np.sqrt(lamb))),np.transpose(v))
    #Multivariant standardization
    medrep = np.transpose(np.repeat(np.asmatrix(med), n, 0))
    xs = np.dot(np.transpose(arr-medrep),si12)
    #Similarities
    r = np.array(np.dot(xs,np.transpose(xs)))
    return np.sum(r**3)/(n*n)

def get_peaks(arr):
    """
    Obtains the indices in an array where a peak is present, this is, where
    a change in the sign of the first derivative is found. The points with
    zero derivative are associated to the current trend.

    Parameters
    ----------
    arr:
        NumPy array

    Returns
    -------
    out:
        NumPy array containing the indices where there are peaks.
    """
    if len(arr) < 3:
        raise ValueError("The array needs to have at least three values")
    sdif = np.sign(np.diff(arr))
    #If all the series has zero derivative, there are no peaks.
    if not np.any(sdif):
        return np.array([])
    #If the sequence starts with a zero derivative, we associate it to the
    #first posterior trend.
    if sdif[0] == 0:
        i = 1
        while sdif[i] == 0:
            i += 1
        sdif[0] = sdif[i]
    for i in range(1, len(sdif)):
        if sdif[i] == 0:
            sdif[i] = sdif[i-1]
    return np.where(sdif[1:]!=sdif[:-1])[0] + 1