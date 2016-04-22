# -*- coding: utf-8 -*-
# pylint: disable-msg=
"""
Created on Thu May  8 09:39:15 2014

This module defines the abstraction pattern for ventricular flutter.

@author: T. Teijeiro
"""

import construe.knowledge.observables as o
import construe.acquisition.signal_buffer as sig_buf
import construe.knowledge.constants as C
from construe.utils.signal_processing import fft_filt
from construe.utils.units_helper import SAMPLING_FREQ
from construe.model import ConstraintNetwork, verify, Interval as Iv
from construe.model.automata import (PatternAutomata, ABSTRACTED,
                                        ENVIRONMENT, BASIC_TCONST)
from construe.utils.signal_processing.xcorr_similarity import (xcorr_valid,
                                                           signal_unmatch)
import numpy as np
import math
from scipy.signal.signaltools import resample


def _contains_qrs(pattern):
    """
    Checks if inside the flutter fragment there is a waveform "identical" to
    the first environment QRS complex.
    """
    qrs = pattern.evidence[o.QRS][0]
    #We limit the duration of the QRS to check this condition.
    if qrs.lateend-qrs.earlystart not in C.NQRS_DUR:
        return False
    eints = pattern.evidence[o.Energ_Int]
    if len(eints) > 1:
        limit = (eints[-3].lateend if len(eints) > 2 else qrs.lateend)
        sig  = {}
        #We take the signal fragment with maximum correlation with the QRS
        #signal in each lead, and we check if the two fragments can be
        #clustered as equal QRS complexes.
        qshape = {}
        corr = -np.Inf
        delay = 0
        leads = sig_buf.get_available_leads()
        for lead in leads:
            qshape[lead] = o.QRSShape()
            sigfr = sig_buf.get_signal_fragment(qrs.earlystart,
                                                qrs.lateend+1, lead=lead)[0]
            qshape[lead].sig = sigfr-sigfr[0]
            qshape[lead].amplitude = np.ptp(qshape[lead].sig)
            sig[lead] = sig_buf.get_signal_fragment(limit,
                                            eints[-1].earlystart, lead=lead)[0]
            if len(sig[lead]) > 0 and len(qshape[lead].sig) > 0:
                lcorr, ldelay = xcorr_valid(sig[lead], qshape[lead].sig)
                if lcorr > corr:
                    corr, delay = lcorr, ldelay
        if 0 <= delay < len(sig[lead]):
            sshape = {}
            for lead in leads:
                sshape[lead] = o.QRSShape()
                sshape[lead].sig = (sig[lead][delay:delay+len(qshape[lead].sig)]
                                    - sig[lead][delay])
                sshape[lead].amplitude = np.ptp(sshape[lead].sig)
            return not signal_unmatch(sshape, qshape)
    return False


############################
### Temporal constraints ###
############################

def _prev_rhythm_tconst(pattern, rhythm):
    """Temporal constraints of the flutter with the precedent rhythm"""
    BASIC_TCONST(pattern, rhythm)
    tnet = pattern.last_tnet
    tnet.set_equal(pattern.hypothesis.start, rhythm.end)
    tnet.add_constraint(pattern.hypothesis.start, pattern.hypothesis.end,
                                               Iv(C.VFLUT_MIN_DUR, np.inf))

def _eint0_tconst(pattern, eint):
    """Temporal constraints of the first energy interval"""
    BASIC_TCONST(pattern, eint)
    tnet = pattern.last_tnet
    tnet.add_constraint(pattern.hypothesis.start, eint.time, C.VFLUT_LIM_INTERV)
    tnet.set_before(eint.time, pattern.hypothesis.end)

def _eint_tconst(pattern, eint):
    """Temporal constraints of the posterior energy intervals"""
    eints = pattern.evidence[o.Energ_Int]
    idx = eints.index(eint)
    hyp = pattern.hypothesis
    tnet = pattern.last_tnet
    prev = eints[idx-1]
    tnet.remove_constraint(hyp.end, prev.time)
    #We create a new temporal network for the cyclic observations
    tnet = ConstraintNetwork()
    tnet.add_constraint(prev.time, eint.time, C.VFLUT_WW_INTERVAL)
    pattern.temporal_constraints.append(tnet)
    BASIC_TCONST(pattern, eint)
    tnet.add_constraint(eint.start, eint.end, Iv(0, C.VFLUT_WW_INTERVAL.end))
    tnet.set_before(eint.time, hyp.end)

def _qrs0_tconst(pattern, qrs):
    """
    Temporal constraints of the QRS complex that must be at the beginning of
    the flutter.
    """
    BASIC_TCONST(pattern, qrs)
    tnet = pattern.last_tnet
    tnet.set_equal(pattern.hypothesis.start, qrs.time)
    tnet.add_constraint(pattern.hypothesis.start, pattern.hypothesis.end,
                                                   Iv(C.VFLUT_MIN_DUR, np.inf))

def _qrs_tconst(pattern, qrs):
    """Temporal constraints of the QRS complex that determines the end of the
    flutter"""
    BASIC_TCONST(pattern, qrs)
    eint = pattern.evidence[o.Energ_Int][-1]
    tnet = pattern.last_tnet
    tnet.add_constraint(eint.time, qrs.time, C.VFLUT_LIM_INTERV)
    tnet.set_equal(pattern.hypothesis.end, qrs.time)
    tnet.add_constraint(qrs.start, qrs.end, C.QRS_DUR)


###########################
### General constraints ###
###########################

def _is_VF(signal):
    """
    This function checks if a signal array meets the criteria to be
    considered a Ventricular Fibrillation following the method explained in:
    'Amann: Detecting Ventricular Fibrillation by Time-Delay Methods. 2007'.
    In our case, the measured variable is the Pearson's goodness of fit
    statistic instead of the recurrence rate of the original paper.
    """
    dfreq = 50
    window = 8*dfreq
    #The threshold is the 0.99 quantile of the chi square distribution.
    thr = 6.64
    tau = int(0.5*dfreq)
    #The signal is filtered and resampled to 50 Hz
    signal = fft_filt(signal, (0.5, 25), SAMPLING_FREQ)
    signal = resample(signal, int(len(signal) * dfreq / SAMPLING_FREQ))
    n = int(math.ceil(len(signal)/float(window)))
    isvf = True
    #The conditions are validated in fragments of *window* size
    for i in xrange(n):
        if i == n-1 and n > 1:
            frag = signal[-window-tau:-tau]
            dfrag = signal[-min(len(frag), window):]
        else:
            frag = signal[i*window:(i+1)*window]
            dfrag = signal[i*window+tau:(i+1)*window+tau]
            frag = frag[:len(dfrag)] if len(frag) > len(dfrag) else frag
        #Constant arrays cannot be flutters.
        if len(frag) > 1 and not np.allclose(frag, frag[0]):
            #The size of the histogram scales with the size of the signal.
            nbins = int(math.ceil(math.sqrt(len(frag)*40.0**2/window)))
            H, _, _ = np.histogram2d(frag, dfrag, nbins)
            H /= len(frag)
            Ei = 1.0/(nbins*nbins)
            #The measured variable for the dispersion is the statistic of the
            #Pearson's goodnes of fit test (agains the uniform distribution).
            det = np.sum((H-Ei)**2/Ei)
            isvf = isvf and det < thr
        else:
            isvf = False
        if not isvf:
            break
    return isvf

def _vflut_gconst(pattern, _):
    """
    General constraints of the pattern, checked every time a new observation
    is added to the evidence. These constraints simply state that the majority
    of the leads must show a positive detection of a ventricular flutter.
    """
    if not pattern.evidence[o.Cardiac_Rhythm]:
        return
    hyp = pattern.hypothesis
    ##################
    beg = int(hyp.earlystart)
    if beg < 0:
        beg = 0
    end = int(hyp.earlyend)
    verify(not _contains_qrs(pattern), 'QRS detected during flutter')
    lpos = 0.
    ltot = 0.
    for lead in sig_buf.get_available_leads():
        if _is_VF(sig_buf.get_signal_fragment(beg, end, lead= lead)[0]):
            lpos += 1
        ltot += 1
    verify(lpos/ltot > 0.5)
    eints = pattern.evidence[o.Energ_Int]
    if len(eints) > 1:
        rrs = np.diff([eint.earlystart for eint in eints])
        hyp.meas = o.CycleMeasurements((np.mean(rrs), np.std(rrs)),
                                                                  (0,0), (0,0))


VFLUTTER_PATTERN = PatternAutomata()
VFLUTTER_PATTERN.name = "Ventricular Flutter"
VFLUTTER_PATTERN.Hypothesis = o.Ventricular_Flutter
VFLUTTER_PATTERN.add_transition(0, 1, o.Cardiac_Rhythm, ENVIRONMENT,
                                                           _prev_rhythm_tconst)
VFLUTTER_PATTERN.add_transition(1, 2, o.QRS, ENVIRONMENT, _qrs0_tconst)
VFLUTTER_PATTERN.add_transition(2, 3, o.Energ_Int, ABSTRACTED, _eint0_tconst,
                                                                 _vflut_gconst)
VFLUTTER_PATTERN.add_transition(3, 3, o.Energ_Int, ABSTRACTED, _eint_tconst,
                                                                 _vflut_gconst)
VFLUTTER_PATTERN.add_transition(3, 4, o.QRS, ABSTRACTED, _qrs_tconst,
                                                                 _vflut_gconst)
VFLUTTER_PATTERN.final_states.add(4)
VFLUTTER_PATTERN.abstractions[o.Energ_Int] = (VFLUTTER_PATTERN.transitions[2],)
VFLUTTER_PATTERN.freeze()


if __name__ == "__main__":
    pass

