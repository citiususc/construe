# -*- coding: utf-8 -*-
# pylint: disable-msg= E1002, E1101, C0103, E1103
"""
Created on Fri Jun  1 12:47:53 2012

This module contains the definition of the abstraction patterns related with
T and P waves.

@author: T. Teijeiro
"""

import numpy as np
import construe.knowledge.observables as o
import construe.utils.signal_processing.Douglas_Peucker as DP
import construe.knowledge.constants as C
import construe.acquisition.signal_buffer as sig_buf
from construe.utils.signal_processing import get_peaks, fft_filt
from construe.knowledge.abstraction_patterns.segmentation import (
                                                         characterize_baseline)
from construe.utils.units_helper import (samples2msec as sp2ms,
                                            phys2digital as ph2dg,
                                            SAMPLING_FREQ)
from construe.model import verify, Interval as Iv, InconsistencyError
from construe.model.automata import PatternAutomata, ABSTRACTED, ENVIRONMENT


######################
### T Wave Pattern ###
######################

#Auxiliary functions

def _zhang_tendpoint(signal, epts):
    """
    This function applies the method described in Zhang: 'An algorithm for
    robust and efficient location of T-wave ends in electrocardiograms' to a
    set of points selected from a signal fragment, and determines which one
    is the most probable endpoint of a T Wave

    Parameters
    ----------
    signal:
        Signal fragment.
    epts:
        Indices of selected points in the signal array. The endpoint is
        selected from this sequence
    """
    sig = fft_filt(signal-np.mean(signal), (0.5, 250), SAMPLING_FREQ)
    mthrld = 2.0
    fratio = SAMPLING_FREQ/250.0
    ptwin = int(np.ceil(4.0*fratio))
    swin = 32.0*fratio
    Tval = np.zeros_like(epts)
    for i in xrange(len(epts)):
        pt = epts[i]
        cutlevel = np.sum(sig[pt-ptwin:pt+ptwin+1])/(ptwin*2.0+1)
        corsig = sig[int(pt-swin):pt+1] - cutlevel
        Tval[i] = np.sum(corsig)
    maxind = np.argmax(Tval)
    dum = Tval[maxind]
    maxindinv = np.argmin(Tval)
    duminv = -Tval[maxindinv]
    if maxind < maxindinv:
        leftind = maxind
        rightind = maxindinv
        leftdum = dum
        rightdum = duminv
    else:
        leftind = maxindinv
        rightind = maxind
        leftdum = duminv
        rightdum = dum
    if leftdum > mthrld * rightdum:
        maxind = leftind
        dum = leftdum
    else:
        maxind = rightind
        dum = rightdum
    return epts[maxind], dum


def _delimit_t(signal, baseline, ls_lim, ee_lim, qrs_shape):
    """
    This function performs the delineation of a possible T Wave present
    in the fragment. To obtain the endpoint of the T Wave, it uses a method
    based on the work by Zhang: 'An algorithm for robust and efficient location
    of T-wave ends in electrocardiograms'. To get the beginning, it uses a
    probabilistic approach with some basic morphology constraints. All the
    processing is made to a simplification of the signal fragment with at most
    7 points.
    """
    try:
        #We exclude the areas in which the slope of the signal exceeds limit.
        maxtslope = qrs_shape.maxslope * C.TQRS_MAX_DIFFR
        lidx, uidx = 0, len(signal)
        if ls_lim > 0:
            idx = np.where(np.max(np.abs(np.diff(signal[:ls_lim+1])))
                                                              > maxtslope)[0]+1
            lidx = max(idx) if len(idx) > 0 else 0
        if ee_lim < len(signal)-1:
            idx = np.where(np.max(np.abs(np.diff(signal[ee_lim:])))
                                                         > maxtslope)[0]+ee_lim
            uidx = min(idx) if len(idx) > 0 else len(signal)-1
            if (uidx > 1 and
                       abs(signal[uidx]-baseline) > C.TWEND_BASELINE_MAX_DIFF):
                dfsign = np.sign(np.diff(signal[:uidx+1]))
                signchange = ((np.roll(dfsign, 1) - dfsign) != 0).astype(int)
                if np.any(signchange):
                    uidx = np.where(signchange)[0][-1]
        verify(uidx >= lidx)
        signal = signal[lidx:uidx+1]
        ls_lim -= lidx
        ee_lim -= lidx
        #Any T waveform should be representable with at most 7 points.
        points = DP.arrayRDP(signal, max(ph2dg(0.02),
                                                  qrs_shape.amplitude/20.0), 7)
        n = len(points)
        verify(n >= 3)
        #1. Endpoint estimation
        epts = points[points >= ee_lim]
        verify(len(epts) > 0)
        Tend, dum = _zhang_tendpoint(signal, epts)
        #2. Onset point estimation.
        bpts = points[np.logical_and(points < Tend, points <= ls_lim)]
        score = {}
        #Range to normalize differences in the signal values
        rang = max(baseline, signal.max()) - min(signal.min(), baseline)
        #There must be between one and 3 peaks in the T Wave.
        for i in xrange(len(bpts)):
            sigpt = signal[points[i:np.where(points == Tend)[0][0]+1]]
            npks = len(get_peaks(sigpt)) if len(sigpt) >= 3 else 0
            if (npks < 1 or npks > 2 or np.ptp(sigpt) <= ph2dg(0.05)):
                continue
            bl_dist = 1.0 - np.abs(signal[bpts[i]]-baseline)/rang
            tdur = sp2ms(Tend-bpts[i])
            score[bpts[i]] = bl_dist * _check_histogram(_TDUR_HIST, tdur)
        verify(score)
        Tbeg = max(score, key=score.get)
        verify(score[Tbeg] > 0)
        verify(np.max(np.abs(np.diff(signal[Tbeg:Tend+1]))) <= maxtslope)
        return (Iv(Tbeg+lidx, Tend+lidx), dum)
    except InconsistencyError:
        return None

def _t_qrs_tconst(pattern, qrs):
    """
    Temporal constraints wrt the leading QRS complex.
    """
    twave = pattern.hypothesis
    tc = pattern.tnet
    tc.add_constraint(qrs.end, twave.start, C.ST_INTERVAL)
    tc.add_constraint(qrs.start, twave.end, C.QT_INTERVAL)
    tc.add_constraint(qrs.end, twave.end, C.SQT_INTERVAL)
    tc.add_constraint(twave.start, twave.end, C.TW_DURATION)

def _t_defl_tconst(pattern, defl):
    """
    Temporal constraints wrt the abstracted energy interval.
    """
    qrs = pattern.evidence[o.QRS][0] if pattern.evidence[o.QRS] else None
    twave = pattern.hypothesis
    tc = pattern.tnet
    tc.add_constraint(defl.start, defl.end, Iv(0, C.TW_DURATION.end))
    tc.add_constraint(twave.start, defl.start, Iv(-C.TW_DEF_OVER_MAX,
                                                           C.TW_DEF_OVER_MIN))
    tc.add_constraint(twave.end, defl.end, Iv(-C.TW_DEF_OVER_MIN,
                                                           C.TW_DEF_ENDIFF))
    tc.set_before(defl.start, twave.end)
    tc.set_before(twave.start, defl.end)
    if qrs is not None:
        qrsdur = qrs.earlyend - qrs.latestart
        if qrsdur - C.TMARGIN <= C.TW_DURATION.end:
            tc.add_constraint(twave.start, twave.end,
                                                Iv(qrsdur - C.TMARGIN, np.inf))
        tc.add_constraint(qrs.start, defl.end, Iv(0, C.QT_INTERVAL.end))
        tc.set_before(qrs.end, defl.start)

def _t_gconst(pattern, defl):
    """
    T Wave abstraction pattern general constraints, checked when all the
    evidence has been observed.
    """
    twave = pattern.hypothesis
    if defl.earlystart != defl.latestart or not pattern.evidence[o.QRS]:
        return
    qrs = pattern.evidence[o.QRS][0]
    #Wave limits
    beg = int(twave.earlystart)
    end = int(twave.lateend)
    ls_lim = int(twave.latestart - beg)
    ee_lim = int(twave.earlyend - beg)
    #Start and end estimation.
    endpoints = {}
    for lead in sorted(qrs.shape, key=lambda l: qrs.shape[l].amplitude,
                                                             reverse=True):
        baseline, _ = characterize_baseline(lead, beg, end)
        sig = sig_buf.get_signal_fragment(beg, end, lead=lead)[0]
        verify(len(sig) == end-beg+1)
        ep = _delimit_t(sig, baseline, ls_lim, ee_lim, qrs.shape[lead])
        if ep is not None:
            endpoints[lead] = ep
    verify(endpoints)
    limits = max(endpoints.iteritems(), key=lambda ep: ep[1][1])[1][0]
    #We verify that in all leads the maximum slope of the T wave fragment does
    #not exceed the threshold.
    for lead in endpoints:
        sig = sig_buf.get_signal_fragment(beg+limits.start, beg+limits.end,
                                                                  lead=lead)[0]
        verify(np.max(np.abs(np.diff(sig))) <=
                                   qrs.shape[lead].maxslope * C.TQRS_MAX_DIFFR)
        #Amplitude measure
        if lead in endpoints:
            mx, mn = np.amax(sig), np.amin(sig)
            pol = (1.0 if max(mx-sig[0], mx-sig[-1])
                                       >= -min(mn-sig[0], mn-sig[1]) else -1.0)
            twave.amplitude[lead] = pol*np.ptp(sig)
    twave.start.set(beg+limits.start, beg+limits.start)
    twave.end.set(beg+limits.end, beg+limits.end)
    #The duration of the T Wave must be greater than the QRS
    #(with a security margin)
    verify(twave.earlyend-twave.latestart >
                                          qrs.earlyend-qrs.latestart-C.TMARGIN)
    #The overlapping between the energy interval and the T Wave must be at
    #least the half of the duration of the energy interval.
    verify(Iv(twave.earlystart, twave.lateend).intersection(
                    Iv(defl.earlystart, defl.lateend)).length >=
                                            (defl.lateend-defl.earlystart)/2.0)
    #If the Deflection is a R-Deflection, we require a margin before
    #the end of the twave.
    if isinstance(defl, o.RDeflection):
        verify(twave.lateend - defl.time.end > C.TW_RDEF_MIN_DIST)

###########################
### Automata definition ###
###########################

TWAVE_PATTERN = PatternAutomata()
TWAVE_PATTERN.name = 'T Wave'
TWAVE_PATTERN.Hypothesis = o.TWave
TWAVE_PATTERN.add_transition(0, 1, o.QRS, ENVIRONMENT, _t_qrs_tconst)
TWAVE_PATTERN.add_transition(1, 2, o.Deflection, ABSTRACTED, _t_defl_tconst,
                                                                     _t_gconst)
TWAVE_PATTERN.final_states.add(2)
TWAVE_PATTERN.freeze()

##################################################
### Statistical knowledge stored as histograms ###
##################################################

def _check_histogram(hist, value):
    """
    Obtains a score of a value according to an histogram, between 0.0 and 1.0
    """
    i = 0
    while i < len(hist[1]) and value > hist[1][i]:
        i += 1
    return 0.0 if i == 0 or i == len(hist[1]) else hist[0][i-1]


#####################################
#### Static variables definition ####
#####################################

#Static definition of the QT duration histogram
_QT_HIST = (np.array([1.73499075e-05,   1.95186459e-04,   1.50944195e-03,
                      2.57646126e-03,   2.18608834e-03,   5.75149433e-03,
                      5.98571808e-03,   6.45416558e-03,   6.41512829e-03,
                      4.97074849e-03,   4.50230099e-03,   1.93885216e-03,
                      1.23618091e-03,   6.11584238e-04,   4.68447502e-04,
                      2.86273473e-04,   9.10870142e-05,   1.17111875e-04,
                      1.30124306e-05,   2.60248612e-05,   9.10870142e-05,
                      1.56149167e-04,   2.21211320e-04,   1.04099445e-04,
                      2.60248612e-05,   6.50621530e-05,   5.20497224e-05,
                      3.90372918e-05,   2.60248612e-05,   0.90248612e-05]),
        np.array([250.        ,  271.69666667,  293.39333333,  315.09      ,
                  336.78666667,  358.48333333,  380.18      ,  401.87666667,
                  423.57333333,  445.27      ,  466.96666667,  488.66333333,
                  510.36      ,  532.05666667,  553.75333333,  575.45      ,
                  597.14666667,  618.84333333,  640.54      ,  662.23666667,
                  683.93333333,  705.63      ,  727.32666667,  749.02333333,
                  770.72      ,  792.41666667,  814.11333333,  835.81      ,
                  857.50666667,  879.20333333,  900.9       ]))

#Static definition of the histogram from the end of the QRS to the T wave end
_STOFF_HIST = (np.array([5.64652739e-05,   3.95256917e-04,   1.35516657e-03,
                         2.39977414e-03,   3.31733484e-03,   5.74534161e-03,
                         6.38057595e-03,   5.84415584e-03,   6.47939018e-03,
                         6.33822699e-03,   4.44664032e-03,   2.96442688e-03,
                         1.42574816e-03,   8.89328063e-04,   4.79954828e-04,
                         3.95256917e-04,   1.27046866e-04,   9.88142292e-05,
                         1.41163185e-05,   1.12930548e-04,   1.27046866e-04,
                         1.55279503e-04,   1.83512140e-04,   4.23489554e-05,
                         5.64652739e-05,   2.82326369e-05,   5.64652739e-05,
                         5.64652739e-05,   2.82326369e-05,   1.34326369e-05]),
        np.array([150.,  170.,  190.,  210.,  230.,  250.,  270.,  290.,  310.,
                  330.,  350.,  370.,  390.,  410.,  430.,  450.,  470.,  490.,
                  510.,  530.,  550.,  570.,  590.,  610.,  630.,  650.,  670.,
                  690.,  710.,  730.,  750.]))

#Static definition of the T wave duration histogram
_TDUR_HIST = (np.array([0.00011493,  0.00034478,  0.0006321 ,  0.00218362,
                        0.00385006,  0.00453962,  0.00367767,  0.00465455,
                        0.00626353,  0.0067807 ,  0.00764265,  0.0067807 ,
                        0.00838968,  0.00747026,  0.00396499,  0.00304557,
                        0.00229854,  0.00201122,  0.00143659,  0.00091942,
                        0.00074703,  0.00074703,  0.00057464,  0.00057464,
                        0.00051717,  0.00028732,  0.00022985,  0.00011493,
                        0.00028732,  0.00009423]),
              np.array([80.        ,   92.33333333,  104.66666667,
                        117.       ,   129.33333333,  141.66666667,
                        154.        ,  166.33333333,  178.66666667,
                        191.        ,  203.33333333,  215.66666667,
                        228.        ,  240.33333333,  252.66666667,
                        265.        ,  277.33333333,  289.66666667,
                        302.        ,  314.33333333,  326.66666667,
                        339.        ,  351.33333333,  363.66666667,
                        376.        ,  388.33333333,  400.66666667,
                        413.        ,  425.33333333,  437.66666667,  450.]))


if __name__ == "__main__":
    # pylint: disable-msg=C0103
    pass
