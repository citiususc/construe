# -*- coding: utf-8 -*-
# pylint: disable-msg=
"""
Created on Mon Sep 16 13:14:14 2013

This module provides several utility functions to perform transformations over
signal arrays.

@author: T. Teijeiro
"""


import numpy as np

def normalize_window(signal, beg=None, end=None):
    """
    Normalizes a window of a 1-D signal array, by substracting the mean of the
    fragment between two indices.

    Parameters
    ----------
    signal:
        Array containing the signal.
    beg:
        First index of the window to be normalized
    end:
        End of the slice that determines the window length. The last index
        included in the fragment will be (end-1).

    Returns
    -------
    out:
        Equivalent to signal[beg:end]-mean(signal[beg:end])
    """
    beg = beg or 0
    end = end or len(signal)
    frag = signal[beg:end]
    return frag-np.mean(frag)


def fft_filt(sig, bands, sampling_freq):
    """
    This function performs the same filtering than Zhang in the work: 'An
    algorithm for robust and efficient location of T-wave ends in
    electrocardiograms'.

    Parameters
    ----------
    sig:
        Signal array.
    bands:
        Tuple with the (low, high) cutoff frequencies of the filtering.
    sampling_freq:
        Sampling frequency of the signal.
    """
    bands = (max(bands[0], 0), min(bands[1], 0.5 * sampling_freq))
    N = len(sig)
    y = np.fft.fft(sig)
    lowicut = int(round(bands[0]*N/sampling_freq))
    lowmirror = N-lowicut+2
    highicut =  int(round(bands[1]*N/sampling_freq))
    highmirror = N-highicut+2
    y[:lowicut] = 0
    y[lowmirror:] = 0
    y[highicut:highmirror] = 0
    return np.fft.ifft(y).real