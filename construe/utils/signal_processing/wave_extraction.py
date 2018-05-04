# -*- coding: utf-8 -*-
# pylint: disable-msg=C0103
"""
Created on Thu Oct 17 13:15:52 2013

This module provides the functionality to obtain basic primitive structures,
called "peaks", from a signal fragment and its corresponding simplification
using the Douglas-Peucker algorithm. The process is based on the paper:

"Trahanias: Syntactic pattern recognition of the ECG, 1990".

@author: T. Teijeiro
"""
from .signal_measures import get_peaks
from ..units_helper import (msec2samples as ms2sp, phys2digital as ph2dg,
                            digital2phys as dg2ph, digital2mm as dg2mm,
                            samples2mm as sp2mm)
import numpy as np
import math

###############################################################################
# Amplitude and duration thresholds for the waves, extracted from the paper:  #
# European Heart Journal: Recommendations for measurement standards in        #
# quantitative electrocardiography. (1985)                                    #
###############################################################################
MIN_AMP = ph2dg(0.05)
MIN_DUR = ms2sp(10.)

#Custom threshold, taken as an intuitive reference
MIN_ANGLE = math.pi/4.0


class Wave(object):
    """
    This class provides the model of a Peak as is defined in the paper in which
    this module is based. We have added an amplitude attribute.
    """
    __slots__ = ('pts', 'e', 'amp')

    def __init__(self):
        #X coordinates for the left bound, peak, and right bound.
        self.pts = (0, 0, 0)
        #Wave energy
        self.e = 0.0
        #Wave amplitude
        self.amp = 0.0

    def __str__(self):
        return '{0} - {1} - {2}, e = {3}, amp = {4} mV'.format(self.l,
                                       self.m, self.r, self.e, dg2ph(self.amp))

    def __repr__(self):
        return str(self)

    def __eq__(self, other):
        return (type(self) is type(other) and self.e == other.e
                and self.amp == other.amp and  self.pts == other.pts)

    @property
    def sign(self):
        """
        Obtains whether this Wave is a positive or negative Wave.
        """
        return np.sign(self.amp)

    @property
    def l(self):
        """Returns the left boundary of the wave"""
        return self.pts[0]

    @property
    def r(self):
        """Returns the left boundary of the wave"""
        return self.pts[2]

    @property
    def m(self):
        """Returns the left boundary of the wave"""
        return self.pts[1]

    @property
    def dur(self):
        """Returns the duration of the wave"""
        return self.pts[2] - self.pts[0]

    def move(self, displacement):
        """
        Moves the wave a certain time, by adding the displacement value to
        each bound and peak.
        """
        self.pts = tuple(p+displacement for p in self.pts)


def extract_waves(signal, points, baseline= None):
    """
    Obtains the sequence of *Wave* objects present in a signal fragment, based
    on the shape simplification determined by points.

    Parameters
    ----------
    signal:
        Raw signal fragment.
    points:
        Indices of the relevant points in the signal, that will be used to
        determine the peaks.

    Returns
    -------
    out:
        Tuple of *Wave* objects.
    """
    if baseline is None or not np.min(signal) <= baseline <= np.max(signal):
        baseline = signal[0] - (signal[0]-signal[-1])/2.0
    result = []
    #Angle between two points
    angle = lambda a, b : math.atan(dg2mm(abs(signal[b]-signal[a])/sp2mm(b-a)))
    pks = points[get_peaks(signal[points])]
    #If there are no peaks, there are no waves.
    if len(pks) == 0:
        return tuple()
    #The limits of the waves will be the baseline level, or an angle decrease.
    for i in range(len(pks)):
        newpk = Wave()
        #The limits of each wave is the next and the prevoius peak.
        lb = 0 if i == 0 else pks[i-1]
        #Left slope
        idx = np.where(points==lb)[0][0]
        while (points[idx] < pks[i] and (angle(points[idx], pks[i]) < MIN_ANGLE
                            or angle(points[idx], points[idx+1]) < MIN_ANGLE)):
            idx += 1
        #If we stop in the peak, we discard a wave in that peak.
        if points[idx] == pks[i]:
            continue
        lb = points[idx]
        #Right slope
        rb = points[-1] if i == len(pks)-1 else pks[i+1]
        idx = np.where(points==rb)[0][0]
        while (points[idx] > pks[i] and (angle(pks[i], points[idx]) < MIN_ANGLE
                            or angle(points[idx-1], points[idx]) < MIN_ANGLE)):
            idx -= 1
        if points[idx] == pks[i]:
            continue
        rb = points[idx]
        #Now we have ensured to meet minimum angle requirements. We now check
        #duration and amplitude.
        newpk.pts = (lb, pks[i], rb)
        fref = min if signal[newpk.m] > signal[lb] else max
        newpk.amp = signal[newpk.m] - fref(signal[rb], signal[lb])
        #We remove peaks not satisfying basic constraints.
        if (newpk.dur >= MIN_DUR and abs(newpk.amp) >= MIN_AMP):
            result.append(newpk)
    #The limits of consecutive waves have to be refined.
    _refine_wave_limits(result, signal, baseline)
    return tuple(result)


def _refine_wave_limits(waves, signal, baseline):
    """
    This auxiliary function checks a sequence of wave objects, join two
    consecutive waves if they are very close, and establishing the proper
    join point if they overlap.
    """
    i = 0
    while i < len(waves):
        #First we check for overlaps with the precedent wave
        if i > 0 and waves[i].l < waves[i-1].r:
            #The join point is the point closer to the baseline.
            jp = waves[i].l + np.argmin(np.abs(
                                   signal[waves[i].l:waves[i-1].r+1]-baseline))
            waves[i].pts = (jp, waves[i].m, waves[i].r)
        #And then for overlaps with the next one
        if i < len(waves)-1 and waves[i].r > waves[i+1].l:
            jp = waves[i+1].l + np.argmin(np.abs(
                                   signal[waves[i+1].l:waves[i].r+1]-baseline))
            waves[i].pts = (waves[i].l, waves[i].m, jp)
        #Now we recompute amplitude.
        fref = min if signal[waves[i].m] > signal[waves[i].l] else max
        waves[i].amp = signal[waves[i].m] - fref(signal[waves[i].l],
                                                 signal[waves[i].r])
        if (abs(waves[i].amp) < MIN_AMP or waves[i].dur < MIN_DUR or
               waves[i].l == waves[i].m or waves[i].m == waves[i].r):
            waves.pop(i)
        else:
            waves[i].e = np.sum(np.diff(signal[waves[i].l:waves[i].r+1])**2)
            i += 1
    #Now we join waves that are very close
    for i in range(1, len(waves)):
        sep = waves[i].l - waves[i-1].r
        if 0 < sep < MIN_DUR:
            #We join the waves in the maximum deviation point from the
            #baseline.
            pk = waves[i-1].r + np.argmax(np.abs(
                                   signal[waves[i-1].r:waves[i].l+1]-baseline))
            waves[i-1].pts = (waves[i-1].l, waves[i-1].m, pk)
            waves[i].pts = (pk, waves[i].m, waves[i].r)



if __name__ == "__main__":
    import matplotlib.pyplot as plt
    #Small tests with a real delineated QRS example.
    #Example 1: Record 124, beat 0, lead MLII
    pts = np.array([ 0,  8, 14, 23, 27, 30, 42])
    sig = np.array([837,   841,   854,   874,   893,   910,   924,   931,
                    935,   925,   902,   874,   840,   821,   821,   842,
                    880,   929,   982,  1031,  1076,  1122,  1162,  1200,
                   1229,  1250,  1262,  1263,  1257,  1241,  1218,  1187,
                   1151,  1109,  1067,  1024,   981,   938,   895,   857,
                    828,   810,   799])
    #Example 2: Record 124, beat 0, lead V4
    pts = np.array([ 0,  7,  9, 12, 14, 22])
    sig = np.array([ 875.,  886.,  901.,  928.,  952.,  970.,  975.,  972.,
                     955.,  921.,  868.,  811.,  758.,  725.,  717.,  733.,
                     764.,  803.,  840.,  871.,  897.,  915.,  926.])
    #Example 2: Record 113, beat 0
    pts = np.array([ 0,  8, 10, 14, 17, 22, 28])
    sig = np.array([ 1042.,  1046.,  1053.,  1059.,  1066.,  1074.,  1079.,
                     1078.,  1082.,  1080.,  1069.,  1053.,  1031.,  1009.,
                      990.,   978.,   965.,   965.,   971.,   987.,  1011.,
                     1023.,  1032.,  1030.,  1025.,  1027.,  1034.,  1041.,
                     1045.])
    plt.figure()
    plt.plot(sig, '--')
    plt.plot(pts, sig[pts], 'bo')
    for p in extract_waves(sig, pts):
        x = np.array(p.pts)
        plt.plot(x, sig[x])
        print(str(p))

