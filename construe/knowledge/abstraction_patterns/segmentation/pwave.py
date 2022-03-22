# -*- coding: utf-8 -*-
# pylint: disable=C0326
"""
Created on Tue Sep 30 12:31:04 2014

This module contains the definition of the P wave abstraction pattern.

@author: T. Teijeiro
"""


import pickle
import numpy as np
from pathlib import Path
import sklearn.preprocessing as preproc
import construe.knowledge.observables as o
import construe.utils.signal_processing.Douglas_Peucker as DP
import construe.knowledge.constants as C
import construe.acquisition.signal_buffer as sig_buf
from construe.utils.units_helper import (samples2msec as sp2ms,
                                            phys2digital as ph2dg,
                                            digital2phys as dg2ph)
from construe.model import verify, Interval as Iv
from construe.model.automata import PatternAutomata, ABSTRACTED, ENVIRONMENT


####################################################
### Definition of the P Wave abstraction pattern ###
####################################################

#Auxiliary functions
def _delimit_p(signal, lead, es_lim, ls_lim, ee_lim):
    """
    Performs the delimitation of a P wave in a signal fragment. If a waveform
    compatible with a P wave cannot be found, returns None, else return an
    Interval within signal length.
    """
    #shape simplification (ignoring the environment signal)
    delta = ph2dg(0.02)
    points = DP.arrayRDP(signal[int(es_lim):], delta, 6) + int(es_lim)
    #If no relevant disturbances are detected, there is no a P wave
    if len(points) == 2:
        return None
    #Now we look for the shorter limits that satisfy the P-Wave classifier.
    cand = None
    i = next(k for k in range(len(points)-1, -1, -1) if points[k] <= ls_lim)
    while i >= 0:
        j = next(k for k in range(i+1, len(points)) if points[k] >= ee_lim)
        while j < len(points):
            sigfr = signal[points[i]:points[j]+1]
            #We consider a good P wave environment if the signal has no
            #amplitude variations
            beg = int(max(0, points[i]-C.PWAVE_ENV))
            plainenv = not np.any(signal[beg:points[i]+1]-signal[beg])
            #The minimum threshold varies with the environment quality
            ampthres = C.PWAVE_MIN_AMP if not plainenv else delta
            if (seems_pwave(sigfr, lead) and np.ptp(sigfr) >= ampthres):
                cand = (points[i], points[j])
                break
            j += 1
        if cand is not None:
            break
        i -= 1
    return None if cand is None else Iv(int(cand[0]-es_lim),
                                        int(cand[1]-es_lim))

def delineate_pwave(es_lim, ls_lim, ee_lim, le_lim, pwave):
    """
    Performs the delineation of a possible P-wave contained in the given limits.

    Parameters
    ----------
    es_lim:
        earliest possible time for the beginning of the P-wave.
    ls_lim:
        latest possible time for the beginning of the P-wave.
    ee_lim:
        earliest possible time for the ending of the P-wave.
    le_lim:
        latest possible time for the ending of the P-wave.
    pwave:
        PWave instance, **which is modified** to establish the amplitude in all
        those leads in which the identification was correct.

    Returns
    -------
    out:
        Interval with the delineation of the p-wave, relative to *es_lim*. If
        a p-wave cannot be delineated, returns None.
    """
    start = finish = None
    for lead in (l for l in C.PWAVE_LEADS if sig_buf.is_available(l)):
        #We take some environment signal for the P wave.
        beg = int(es_lim-C.PWAVE_ENV)
        beg = 0 if beg < 0 else beg
        sig = sig_buf.get_signal_fragment(beg, le_lim, lead=lead)[0]
        endpoints = _delimit_p(sig, lead, es_lim-beg, ls_lim-beg, ee_lim-beg)
        if endpoints is None:
            continue
        elif start is None:
            start, finish = endpoints.start, endpoints.end
            if finish > start:
                pwave.amplitude[lead] = np.ptp(sig[start:finish+1])
        else:
            if abs(start - endpoints.start) < C.TMARGIN:
                start = min(start, endpoints.start)
            if abs(finish - endpoints.end) < C.TMARGIN:
                finish = max(finish, endpoints.end)
            if finish > start:
                pwave.amplitude[lead] = np.ptp(sig[start:finish+1])
    return None if start is None else Iv(start, finish)


def _p_qrs_tconst(pattern, qrs):
    """
    Adds the temporal constraints wrt the qrs environment observation.
    """
    pwave = pattern.hypothesis
    #Temporal constraints
    tnet = pattern.tnet
    #P wave duration constraint
    tnet.add_constraint(pwave.start, pwave.end, C.PW_DURATION)
    #Relations between P wave and QRS complex
    tnet.add_constraint(pwave.end, qrs.start, C.PQ_INTERVAL)
    tnet.add_constraint(pwave.start, qrs.start, C.PR_INTERVAL)

def _p_defl_tconst(pattern, defl):
    """
    Temporal constraints definition wrt the abstracted energy interval.
    """
    pwave = pattern.hypothesis
    qrs = None
    if pattern.evidence[o.QRS]:
        qrs = pattern.evidence[o.QRS][0]
    #Temporal constraints
    tnet = pattern.tnet
    #P wave duration constraint
    tnet.add_constraint(pwave.start, pwave.end, C.PW_DURATION)
    #Constraints with the energy interval
    tnet.add_constraint(defl.start, defl.end, C.PW_DEF_DUR)
    tnet.add_constraint(pwave.start, defl.start, Iv(-C.PW_DEF_OVER,
                                                    C.PW_DEF_OVER))
    tnet.add_constraint(pwave.end, defl.end,
                        Iv(-C.PW_DEF_OVER, C.PW_DEF_OVER))
    tnet.set_before(defl.start, pwave.end)
    tnet.set_before(pwave.start, defl.end)
    if qrs is not None:
        tnet.add_constraint(defl.start, qrs.start, C.PR_DEF_SEP)
        tnet.add_constraint(defl.end, qrs.start, C.PQ_DEF_SEP)
        tnet.set_before(defl.end, qrs.start)

def _p_gconst(pattern, defl):
    """
    General constraints of the P Wave abstraction pattern, once all the
    evidence has been observed.
    """
    pwave = pattern.hypothesis
    if ((defl is not None and defl.earlystart != defl.latestart)
                                               or not pattern.evidence[o.QRS]):
        return
    qrs = pattern.evidence[o.QRS][0]
    beg = pwave.earlystart
    if beg < 0:
        beg = 0
    #We try the delineation of the P-Wave
    endpoints = delineate_pwave(beg, int(pwave.latestart),
                                int(pwave.earlyend), int(pwave.lateend), pwave)
    verify(endpoints is not None)
    #Now we obtain the amplitudes, and we ensure the QRS amplitude is at
    #least twice the P Wave amplitude in each lead
    pwave.start.set(beg + endpoints.start, beg + endpoints.start)
    pwave.end.set(beg + endpoints.end, beg + endpoints.end)
    for lead in pwave.amplitude:
        verify(pwave.amplitude[lead] <= C.PWAVE_AMP[lead])
        verify(lead not in qrs.shape or
               pwave.amplitude[lead] < qrs.shape[lead].amplitude)


#########################
## Automata definition ##
#########################

PWAVE_PATTERN = PatternAutomata()
PWAVE_PATTERN.name = 'P Wave'
PWAVE_PATTERN.Hypothesis = o.PWave
PWAVE_PATTERN.add_transition(0, 1, o.QRS, ENVIRONMENT, _p_qrs_tconst)
PWAVE_PATTERN.add_transition(1, 2, o.Deflection, ABSTRACTED, _p_defl_tconst,
                             _p_gconst)
#PWAVE_PATTERN.add_transition(1, 2, gconst=_p_gconst)
PWAVE_PATTERN.final_states.add(2)
PWAVE_PATTERN.freeze()


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


#Static definition of the PR histogram
_PR_HIST = (np.array(
             [0.20300752,  0.22932331,  0.32330827,  0.57142857,  0.66541353,
              0.92857143,  0.84962406,  0.84210526,  0.87969925,  0.93984963,
              1.        ,  0.77443609,  0.63533835,  0.51879699,  0.43609023,
              0.48496241,  0.28947368,  0.29699248,  0.29323308,  0.27443609,
              0.2556391 ,  0.19172932,  0.20676692,  0.16165414,  0.17293233,
              0.17293233,  0.10150376,  0.07518797,  0.07142857,  0.04323308,
              0.01503759,  0.0112782 ,  0.01503759,  0.02255639,  0.0075188 ,
              0.0112782 ,  0.0112782 ,  0.01503759,  0.01503759,  0.01691729,
              0.01879699,  0.0037594 ,  0.03383459,  0.0075188 ,  0.02631579,
              0.03759398,  0.01879699,  0.04887218,  0.0112782 ,  0.04511278]),
            np.array(
             [20. ,   23.6,   27.2,   30.8,   34.4,   38. ,   41.6,   45.2,
              48.8,   52.4,   56. ,   59.6,   63.2,   66.8,   70.4,   74. ,
              77.6,   81.2,   84.8,   88.4,   92. ,   95.6,   99.2,  102.8,
              106.4,  110. ,  113.6,  117.2,  120.8,  124.4,  128. ,  131.6,
              135.2,  138.8,  142.4,  146. ,  149.6,  153.2,  156.8,  160.4,
              164. ,  167.6,  171.2,  174.8,  178.4,  182. ,  185.6,  189.2,
              192.8,  196.4,  200. ]))

#Static definition of the PQ histogram
_PQ_HIST = (np.array(
             [0.07231405,  0.15495868,  0.4338843 ,  0.58677686,  0.92768595,
              1.        ,  0.94214876,  0.6714876 ,  0.52066116,  0.39669421,
              0.23760331,  0.1446281 ,  0.14876033,  0.10330579,  0.08057851,
              0.03512397,  0.02479339,  0.01239669,  0.01859504,  0.0268595 ,
              0.01239669,  0.03099174,  0.00826446,  0.00413223,  0.00413223]),
        np.array([ 80.,  112.,  124.,  136.,  148.,  160.,  172.,  184.,  196.,
                  208.,  220.,  232.,  244.,  256.,  268.,  280.,  292.,  304.,
                  316.,  328.,  340.,  352.,  364.,  376.,  388.,  400.]))

#Static definition of the P wave duration histogram
_PDUR_HIST = (np.array(
             [0.00410678,  0.0164271 ,  0.05749487,  0.13552361,  0.24640657,
              0.40246407,  0.6899384 ,  0.78234086,  1.        ,  0.87474333,
              0.69609856,  0.58316222,  0.33880903,  0.23819302,  0.17453799,
              0.12936345,  0.07597536,  0.05338809,  0.01848049,  0.01232033,
              0.0164271 ,  0.00410678,  0.00410678,  0.        ,  0.00410678]),
       np.array([44.  ,   51.84,   59.68,   67.52,   75.36,   83.2 ,   91.04,
                 98.88,  106.72,  114.56,  122.4 ,  130.24,  138.08,  145.92,
                 153.76,  161.6 ,  169.44,  177.28,  185.12,  192.96,  200.8 ,
                 208.64,  216.48,  224.32,  232.16,  240.  ]))


#################################
### P Wave signal classifiers ###
#################################

#We have one classifier for the limb leads, and other for precordial leads.
#The classifiers are one-class SVM trained with 73 clean records from the QT
#database. The features used are the 4 points in coordinates (X,Y) obtained
#from a RDP simplification of the signal delimiting the P wave, using 5 points
#for the simplification and assuming the first point is always (0,0). The
#units of the coordinates are in msec and mV.

def _scale_sample(signal, lead):
    if not lead in _CL_MAP:
        raise ValueError('Invalid lead.')
    scaler = _SCALERS[_CL_MAP[lead]]
    #The signal is converted to physical units, and the first point is (0,0)
    sig = dg2ph(signal-signal[0])
    #The features are the 4 points better representing the signal shape.
    points = DP.arrayRDP(sig, 0.001, 5)[1:]
    if len(points) < 4:
        raise ValueError('Not enough points after path simplification')
    sample = np.concatenate((sp2ms(points), sig[points])).reshape(1, -1)
    return scaler.transform(sample)

def seems_pwave(signal, lead):
    """
    Checks if a signal fragment looks like a P wave. It is assumed the signal
    is in raw signal units at the record sampling frequency.

    Parameters
    ----------
    signal:
        Raw signal array with the delimited P wave.
    lead:
        Lead where the signal has been obtained. It must be a limb
        or a precordial lead.
    """
    if not lead in _CL_MAP:
        raise ValueError('Invalid lead.')
    classifier = _CLASSIFIERS[_CL_MAP[lead]]
    try:
        sample = _scale_sample(signal, lead)
    except ValueError:
        return False
    return classifier.predict(sample)[0] == 1

def pwave_distance(signal, lead):
    """
    Obtains a distance measure of a signal fragment to a recognized P wave
    morphology.
    """
    if not lead in _CL_MAP:
        raise ValueError('Invalid lead.')
    classifier = _CLASSIFIERS[_CL_MAP[lead]]
    try:
        sample = _scale_sample(signal, lead)
    except ValueError:
        #Unknown values are located in the decision boundary
        return 0.0
    return classifier.decision_function(sample)[0][0]

#Mapping for each lead with the corresponding classifier.
_CL_MAP = {sig_buf.Leads.UNIQUE: 0, sig_buf.Leads.MLI:   0,
           sig_buf.Leads.MLII:   0, sig_buf.Leads.MLIII: 0,
           sig_buf.Leads.V1:     1, sig_buf.Leads.V2:    1,
           sig_buf.Leads.V3:     1, sig_buf.Leads.V4:    1,
           sig_buf.Leads.V5:     1, sig_buf.Leads.V6:    1}

#Each classifier has a scaler to preprocess the feature set.

_SCALERS = [preproc.StandardScaler(), preproc.StandardScaler()]
_SCALERS[0].mean_ = np.array(
        [3.64100119e+01,   6.27794994e+01,   8.74159714e+01,   1.18181168e+02,
         6.97318236e-02,   1.17550656e-01,   8.52205006e-02,  -1.42669845e-02])
_SCALERS[1].mean_ = np.array(
        [3.52004505e+01,   6.13603604e+01,   8.69031532e+01,   1.17867117e+02,
         5.26295045e-02,   8.14245495e-02,   3.52759009e-02,  -1.85416667e-02])
_std0 = np.array([18.28456548,  21.80939775,  25.19837448,  27.46293336,
                  0.06650826,   0.09788993,   0.09393059,   0.04712977])
_std1 = np.array([16.8934232 ,  19.45625391,  23.22422215,  26.39513332,
                  0.05968984,   0.07814591,   0.08662095,   0.05071906])
#Fix for scikit-learn > 0.16
try:
    _SCALERS[0].std_ = _std0
    _SCALERS[1].std_ = _std1
    _SCALERS[0].scale_ = _std0
    _SCALERS[1].scale_ = _std1
except AttributeError:
    _SCALERS[0].scale_ = _std0
    _SCALERS[1].scale_ = _std1

#Fix for scikit-learn > 0.22
try:
    _SCALERS[0].n_features_in_ = 8
    _SCALERS[1].n_features_in_ = 8
except AttributeError:
    pass

# Trained classifiers. These classifiers were serialized using the pickle
# module. They are instances of sklearn.svm.OneClassSVM, and have been
# successfully tested with sklearn versions from 0.15 to 0.21.3

_localdir = Path(__file__).resolve().parent

with _localdir.joinpath('limb_pw_classifier.pickle').open('rb') as f:
    _LIMB_CLS = pickle.load(f, encoding='latin1')
    #Fix for scikit-learn >= 0.22
    _LIMB_CLS._n_support = _LIMB_CLS.__dict__['n_support_']

with _localdir.joinpath('precordial_pw_classifier.pickle').open('rb') as f:
    _PREC_CLS = pickle.load(f, encoding='latin1')
    #Fix for scikit-learn >= 0.22
    _PREC_CLS._n_support = _PREC_CLS.__dict__['n_support_']

_CLASSIFIERS = [_LIMB_CLS, _PREC_CLS]
