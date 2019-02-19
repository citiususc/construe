# -*- coding: utf-8 -*-
"""
Created on 28-nov-2011 10:06:40

This module provides the wavelet filter used for the segmentation of the ECG
signal, and permits to obtain the relevant observations on demand.
@author: T. Teijeiro
"""

import pywt
import numpy as np

def get_filter_energy(signal, levels=None, N=256):
    """
    Obtains the energy sigal of the wavelet filter used fot the segmentation.

    signal:
        Numpy array with the ECG signal.
    levels:
        Decomposition levels used for the reconstruction. By default, the
        5th and 7th levels are used.
    N:
        Window used for the reconstruction, in number of samples. A power of
        2 is recommended. By default, 256 is used.
    """
    levels = levels or [5, 7]
    if len(signal) < N:
        raise ValueError('Signal length must be >= N')
    filtered_signal = np.empty_like(signal)
    #Wavelet to use (Daubechies 1 o Haar)
    wavelet = pywt.Wavelet('haar')
    #Filtering in N-length fragments
    findex = 0
    while findex*N < len(signal):
        filtered_signal[findex*N : (findex+1) * N] = __apply_filter(
                                            signal[findex*N : (findex+1) * N],
                                                               wavelet, levels)
        findex = findex+1
    #The energy is the area
    return filtered_signal ** 2


def __apply_filter(signal, wavelet, levels):
    """
    Filters a signal, performing a multi-level wavelet transform and
    removing the levels not belonging the levels list.

    Parameters
    ----------
    signal:
        Signal fragment, as numpy array
    wavelet:
        Wavelet to use in the decomposition.
    levels:
        Levels used for the reconstruction of the signal.
    """
    coeffs = pywt.wavedec(signal, wavelet)
    for i in range(len(coeffs)):
        if not i in levels:
            coeffs[i] = np.zeros(len(coeffs[i]))
    return pywt.waverec(coeffs, wavelet)
