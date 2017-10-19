# -*- coding: utf-8 -*-
# pylint: disable-msg=
"""
Created on Thu Oct  9 09:59:19 2014

This module allows to compare two delineated QRS signals by the use of
cross-correlation.

@author: T. Teijeiro
"""

import numpy as np


def xcorr_valid(sig1, sig2, autosort=False):
    """
    Performs a normalized cross-correlation between two signals, assuming
    *sig2* is a subset or shorter sequence of *sig1*. Returns the maximum value
    and the delay with respect to the first signal to get that value. If
    *autosort* is True, then the length of *sig1* and *sig2* is checked to
    ensure that the second argument is shorter, and the position is changed if
    needed.
    """
    if autosort and len(sig2) > len(sig1):
        sig1, sig2 = sig2, sig1
    tr1 = sig1 - sig1[0] if sig1[0] != 0 else sig1
    tr2 = sig2 - sig2[0] if sig2[0] != 0 else sig2
    corr = np.correlate(tr1, tr2, mode='valid')
    corr /= np.sqrt(np.dot(tr1, tr1) * np.dot(tr2, tr2))
    idx = np.argmax(corr)
    return (corr[idx], idx)

def xcorr_full(sig1, sig2):
    """
    Performs a full normalized cross-correlation between two signals, returning
    the maximum value and the delay with respect to the first signal that
    achieves such value.

    Returns
    -------
    out:
        (corr, delay): Tuple with the maximum correlation factor (between -1
        and 1) and the delay that has to be applied to the second signal to get
        that correlation.
    """
    tr1 = sig1 - sig1[0] if sig1[0] != 0 else sig1
    tr2 = sig2 - sig2[0] if sig2[0] != 0 else sig2
    corr = np.correlate(tr1, tr2, mode='full')
    if np.any(tr1) and np.any(tr2):
        corr /= np.sqrt(np.dot(tr1, tr1) * np.dot(tr2, tr2))
    idx = np.argmax(corr)
    return (corr[idx], idx-len(tr2)+1)

def similarity(sig1, sig2):
    """
    Obtains a measure of the similarity between two multi-lead signals, as the
    mean of the cross-correlation value in every lead.
    """
    cleads = set(sig1.keys()).intersection(sig2.keys())
    corrs = []
    for lead in set(sig1.keys()).union(sig2.keys()):
        if lead not in cleads:
            corrs.append(0.0)
        else:
            arr1, arr2 = sig1[lead].sig, sig2[lead].sig
            if len(arr2) > len(arr1):
                arr1, arr2 = arr2, arr1
            corr, _ = xcorr_full(arr1, arr2)
            corrs.append(corr)
    return np.mean(corrs)

def signal_match(sig1, sig2):
    """
    Checks if two multi-lead signals are similar enough, using
    cross-correlation. The conditions to give a True result is that in one lead
    the correlation is higher than 0.9, or the minimum correlation in any lead
    is higher than 0.8. This function can be used to compare different ECG
    waveforms, such as QRS complexes, P and T-Waves, etc.

    Parameters
    ----------
    sig1, sig2:
        The information of the signals is assumed to be a dictionary in which
        the keys are the leads with available information, and the values have
        a property named **sig** that are numpy arrays containing the signal.
    """
    cleads = set(sig1.keys()).intersection(sig2.keys())
    lendiff = False
    corrs = []
    if not cleads:
        return False
    for lead in cleads:
        arr1, arr2 = sig1[lead].sig, sig2[lead].sig
        if len(arr2) > len(arr1):
            arr1, arr2 = arr2, arr1
        if len(arr1)/float(len(arr2)) > 1.5:
            _, idx = xcorr_valid(arr1, arr2)
            arr1 = arr1[idx:idx+len(arr2)]
            lendiff = True
        corr, _ = xcorr_full(arr1, arr2)
        corrs.append(corr)
    corrs = np.array(corrs)
    if lendiff:
        return np.mean(corrs) > 0.85
    else:
        return np.any(corrs > 0.9) or np.all(corrs > 0.8)

def identical(sig1, sig2):
    """
    Checks if two QRS are identical.
    """
    corrs = []
    if set(sig1.iterkeys()) != set(sig2.iterkeys()):
        return False
    for lead in sig1.keys():
        arr1, arr2 = sig1[lead].sig, sig2[lead].sig
        if len(arr2) > len(arr1):
            arr1, arr2 = arr2, arr1
        if len(arr1)/float(len(arr2)) > 1.25:
            return False
        ptp1, ptp2 = np.ptp(arr1), np.ptp(arr2)
        if (ptp1 == 0 or ptp2 == 0) and ptp1 != ptp2:
            return False
        if max(ptp1, ptp2)/min(ptp1, ptp2) > 1.25:
            return False
        corr, _ = xcorr_full(arr1, arr2)
        corrs.append(corr)
    return np.min(corrs) > 0.95


def signal_unmatch(sig1, sig2):
    """
    Checks if two multi-lead signals are different enough to be considered of
    different nature. This function is not simply the negation of the
    *signal_match* function, because it is possible to have two signals with
    enough similarity using cross-correlation, but with very different
    amplitudes.

    Parameters
    ----------
    sig1, sig2:
        The information of the signals is assumed to be a dictionary in which
        the keys are the leads with available information, and the values have
        two properties, one named **sig** that is a numpy array containing the
        signal, and the other one named **amplitude** with amplitude
        information.
    """
    cleads = set(sig1.keys()).intersection(sig2.keys())
    if not cleads:
        return True
    for lead in cleads:
        samp, qamp = sig1[lead].amplitude, sig2[lead].amplitude
        if min(samp, qamp)/max(samp, qamp) < 0.5:
            return True
    return not signal_match(sig1, sig2)

if __name__ == "__main__":
    pass