# -*- coding: utf-8 -*-
# pylint: disable-msg=W0621
"""
Created on Tue Aug  5 12:16:27 2014

This module contains the definition of the abstraction pattern describing
atrial fibrillation.

@author: T. Teijeiro
"""

import copy
import collections
import numpy as np
import scipy.interpolate
import construe.knowledge.observables as o
import construe.knowledge.constants as C
import construe.acquisition.record_acquisition as IN
import construe.acquisition.signal_buffer as sig_buf
from construe.model import verify, Interval as Iv
from construe.model.automata import PatternAutomata, ABSTRACTED, ENVIRONMENT
from construe.knowledge.abstraction_patterns.segmentation.pwave import (
                                                               delineate_pwave)
from construe.knowledge.abstraction_patterns.rhythm.regular import (
                                                           _check_missed_beats)
from construe.knowledge.abstraction_patterns.segmentation import (
                                                        characterize_baseline)
from construe.knowledge.abstraction_patterns.rhythm.vflutter import _is_VF
from construe.utils.units_helper import (msec2samples as ms2sp,
                                         samples2sec as sp2sc)
from construe.utils.signal_processing.xcorr_similarity import signal_match

#Global structure to cache atrial activity delineation information. It will be
#cleared on every reset of the input system.
PCACHE = {}
IN.on_reset += lambda: PCACHE.clear()

######################
### Help functions ###
######################

#This matrix contains the score matrix to calculate the probability of rhythm
#transition as described in "Moody: A new method for detecting atrial
#fibrillation using RR intervals".

Sij = np.array([[-0.075, -1.460,  0.346],
                [-0.806,  0.256, -0.304],
                [ 0.828, -1.926,  0.426]])

def is_afib_rhythm_moody(rrs):
    """
    Checks if an RR series matches the AF classification criteria explained in
    the G. Moody 1983 paper.
    """
    if len(rrs) < 2:
        return True
    score = 0.0
    pstate = 1
    rrmean = rrs[0]
    i = 1
    while i < len(rrs):
        rrmean = 0.75*rrmean + 0.25*rrs[i]
        if rrs[i] < 0.85*rrmean:
            state = 0
        elif rrs[i] > 1.15*rrmean:
            state = 2
        else:
            state = 1
        score += Sij[state, pstate]
        pstate = state
        i += 1
    return score < 0

def is_afib_rhythm_lian(rrs):
    """
    Checks if an RR series matches the AF classification criteria explained in
    the Lian 2011 paper.
    """
    if len(rrs) < 3:
        return True
    elif len(rrs) > 128:
        i = len(rrs)-128
        isafib = True
        while isafib and i > 0:
            isafib = isafib and is_afib_rhythm_lian(rrs[i:i+128])
            i = max(0, i-128)
        return isafib
    drr = np.diff(rrs)
    xbins = np.arange(int(np.min(rrs)-ms2sp(50)),
                                int(np.max(rrs)+ms2sp(50)), int(ms2sp(25)))
    ybins = np.arange(int(np.min(drr)-ms2sp(50)),
                                int(np.max(drr)+ms2sp(50)), int(ms2sp(25)))
    hist2d, _, _ = np.histogram2d(rrs[1:], drr, [xbins, ybins])
    thres = min(len(drr), round(_NEC(len(drr))))
    return np.count_nonzero(hist2d) >= thres


#############################
### Observation procedure ###
#############################

def _rhythm_obs_proc(pattern):
    """Observation procedure executed once the rhythm pattern has finished"""
    #We asign the endpoint of the hypothesis.
    pattern.hypothesis.end.cpy(pattern.evidence[o.QRS][-1].time)

############################
### Temporal constraints ###
############################

def _prev_afib_tconst(pattern, afib):
    """
    Temporal constraints of the fibrillation wrt a previous atrial
    fibrillation that will helps us to reduce the necessary evidence
    """
    pattern.tnet.add_constraint(afib.end, pattern.hypothesis.start,
                                                       Iv(0, C.AFIB_MAX_DELAY))

def _prev_multrhythm_tconst(pattern, rhythm):
    """
    Temporal constraints of the fibrillation with the cardiac rhythms between
    the last atrial fibrillation and the precedent one.
    """
    const = Iv(1, C.AFIB_MAX_DELAY-1)
    pattern.tnet.add_constraint(rhythm.start, rhythm.end, const)
    pattern.tnet.add_constraint(rhythm.start, pattern.hypothesis.start, const)
    pattern.tnet.add_constraint(rhythm.end, pattern.hypothesis.start, const)


def _prev_rhythm_tconst(pattern, rhythm):
    """Temporal constraints of the fibrillation with the precedent rhythm"""
    pattern.tnet.set_equal(pattern.hypothesis.start, rhythm.end)
    #An atrial fibrillation needs at least 7 QRS complexes.
    pattern.tnet.add_constraint(pattern.hypothesis.start,
                        pattern.hypothesis.end, Iv(7*C.TACHY_RR.start, np.inf))

def _qrs0_tconst(pattern, qrs):
    """
    Temporal constraints of the QRS complex that must be at the beginning of
    the flutter.
    """
    pattern.tnet.set_equal(pattern.hypothesis.start, qrs.time)
    pattern.tnet.add_constraint(pattern.hypothesis.start, pattern.hypothesis.end,
                                                Iv(5*C.TACHY_RR.start, np.inf))

def get_t_tconst(qrsidx):
    """
    Obtains the temporal constraints function for the T wave associated to
    the QRS at given position.
    """
    def _t_tconst(pattern, twave):
        """
        Temporal constraints of the T wave.
        """
        beats = pattern.evidence[o.QRS]
        tnet = pattern.tnet
        qidx = qrsidx+len(beats) if qrsidx < 0 else qrsidx
        qrs = beats[qidx]
        if qidx < len(beats) - 1:
            tnet.set_before(twave.end, beats[qidx+1].start)
        if qidx > 0:
            refrr = qrs.time.end - pattern.evidence[o.QRS][qidx-1].time.start
            refrr = max(min(refrr, C.QTC_RR_LIMITS.end), C.QTC_RR_LIMITS.start)
            rtc, rtstd = pattern.hypothesis.meas.rt
            if rtc > 0:
                #Expected QT value from the QT corrected value
                rtmean = ms2sp(1000.0*sp2sc(rtc)*np.cbrt(sp2sc(refrr)))
                tnet.add_constraint(qrs.time, twave.end, Iv(rtmean-2.5*rtstd,
                                                            rtmean+2.5*rtstd))
            try:
                tnet.add_constraint(qrs.time, twave.end,
                                              Iv(0, refrr - C.TQ_INTERVAL_MIN))
            except ValueError:
                pass
        tnet.add_constraint(qrs.start, twave.end, C.N_QT_INTERVAL)
        #ST interval
        tnet.add_constraint(qrs.end, twave.start, C.ST_INTERVAL)
    return _t_tconst

def _qrs_tconst(pattern, qrs):
    """
    Temporal constraints for the QRS complexes.
    """
    beats = pattern.evidence[o.QRS]
    idx = beats.index(qrs)
    hyp = pattern.hypothesis
    tnet = pattern.tnet
    obseq = pattern.obs_seq
    oidx = pattern.get_step(qrs)
    if idx > 0:
        prev = beats[idx-1]
        rr_bounds = Iv(C.TACHY_RR.start, C.BRADY_RR.end)
        tnet.add_constraint(prev.time, qrs.time, rr_bounds)
        tnet.add_constraint(prev.start, qrs.start, rr_bounds)
        tnet.add_constraint(prev.end, qrs.end, rr_bounds)
        tnet.set_before(prev.end, qrs.start)
        #If there is a prior T Wave, it must finish before the start
        #of the QRS complex.
        if isinstance(obseq[oidx-1], o.TWave):
            prevt = obseq[oidx-1]
            tnet.set_before(prevt.end, qrs.start)
    tnet.add_constraint(qrs.start, qrs.end, C.QRS_DUR)
    tnet.set_before(qrs.time, hyp.end)
    #We can introduce constraints on the morphology of the new QRS complex.
    if hyp.morph and not qrs.frozen:
        qrs.shape = hyp.morph

###########################
### General constraints ###
###########################

PW_SIG = collections.namedtuple('PwSig', ['pr', 'sig'])

def _prev_afib_gconst(pattern, afib):
    """
    Verification of the existence of a previous atrial fibrillation episode.
    """
    verify(isinstance(afib, o.Atrial_Fibrillation))
    pattern.hypothesis.morph = copy.deepcopy(afib.morph)

def _prev_rhythm_gconst(_, rhythm):
    """General constraints of a cardiac rhythm with the precedent one."""
    #An atrial fibrillation cannot be immediately preceded by another afib.
    verify(not isinstance(rhythm, o.Atrial_Fibrillation))

def _prev_afib_exists_gconst(pattern, _):
    """
    Verification of the existence of a previous atrial fibrillation episode.
    """
    prhythms = pattern.evidence[o.Cardiac_Rhythm]
    verify(prhythms and isinstance(prhythms[0], o.Atrial_Fibrillation))
    pattern.hypothesis.morph = copy.deepcopy(prhythms[0].morph)

def _update_measures(pattern):
    """
    Updates the cycle time measures of the pattern.
    """
    #Maximum number of observations considered for the measures (to avoid
    #excessive influence of old observations)
    nobs = 30
    beats = pattern.evidence[o.QRS][-nobs:]
    #RR
    rrs = np.diff([b.time.start for b in beats])
    obseq = pattern.obs_seq
    #The RT (QT) measure is updated by a Kalman Filter strategy.
    #Belief values
    rtmean, rtstd = pattern.hypothesis.meas.rt
    if (len(obseq) > 1 and isinstance(obseq[-2], o.TWave)
                                         and obseq[-2] is not pattern.finding):
        twave = obseq[-2]
        #Current RR measure (bounded)
        qrs = next((q for q in reversed(beats)
                                       if q.lateend <= twave.earlystart), None)
        rr = qrs.time.start - beats[beats.index(qrs)-1].time.start
        rr = max(min(rr, C.QTC_RR_LIMITS.end), C.QTC_RR_LIMITS.start)
        #Kalman filter algorithm, as explained in "Probabilistic Robotics"
        sigma_tbar = rtstd**2 + C.KF_Q**2
        #rt and corrected rt measure in the current iteration
        rt = twave.earlyend - qrs.time.start
        rtc = ms2sp(1000.0*sp2sc(rt)/np.cbrt(sp2sc(rr)))
        meas_err = rtc-rtmean
        #Abnormally QT intervals have associated higher uncertainty
        qt = twave.earlyend - qrs.earlystart
        qt_lims = C.QT_FROM_RR(Iv(rr, rr))
        #Measure uncertainty, represented by the R matrix in the Kalman filter
        KF_R = meas_err if qt in qt_lims else ms2sp(120)
        k_t = sigma_tbar/(sigma_tbar + max(KF_R, C.MIN_QT_STD)**2)
        if rtmean == 0:
            rtmean = meas_err
            rtstd = C.QT_ERR_STD
        else:
            rtmean = rtmean + k_t*meas_err
            rtstd = np.sqrt((1.0-k_t)*sigma_tbar)
    pattern.hypothesis.meas = o.CycleMeasurements((np.mean(rrs), np.std(rrs)),
                                                  (rtmean, rtstd), (0.0, 0.0))
def _get_pwave_sig(beg, end):
    """
    Checks if before a QRS complex there is a waveform similar to a P Wave. In
    an atrial fibrillation context, there cannot be any recognizable atrial
    activity.

    Parameters:
    ----------
    beg:
        Earliest point for the starting of the P Wave. This limit may be
        further constrained if the distance between the two parameters is
        excessive.
    end:
        Latest point for the ending of the P Wave. **It is assumed to be the
        starting point of the QRS complex associated to the P Wave**.

    Returns
    -------
    out:
        Dictionary with a tuple for each lead in which a P-Wave can be
        recognized. The tuple contains the distance in samples from *end* to
        the beginning of the P-Wave, and the signal fragment containing the
        P-Wave.
    """
    #If the result is cached, we use it
    result = PCACHE.get((beg, end), None)
    if result is not None:
        return result.copy()
    est = end - ms2sp(250) if end-beg > ms2sp(250) else beg
    lst = end - ms2sp(80)
    eend = est + ms2sp(40)
    ltnd = end - ms2sp(20)
    if est > lst or eend > end or eend > ltnd:
        #Inconsistency
        return None
    pwave = o.PWave()
    limits = delineate_pwave(est, lst, eend, ltnd, pwave)
    if limits is None:
        return None
    result = {}
    for lead in pwave.amplitude:
        sig = sig_buf.get_signal_fragment(est+limits.start, est+limits.end+1,
                                                                  lead=lead)[0]
        result[lead] = PW_SIG(end-(est+limits.start), sig)
    #Result is cached
    PCACHE[(beg, end)] = result
    return result.copy()

def _verify_afib_rhythm(rrs):
    """
    Checks the rhythm constraints of the atrial fibrillation patterns. If these
    constraints are not satisfied, an InconsistentStateError is raised.
    Otherwise, the procedure finishes and returns nothing.

    Parameters
    ----------
    rrs:
        RR sequence.
    """
    if len(rrs) > 3:
        verify(np.std(rrs)/np.median(rrs) > 0.08)
    if len(rrs) < C.AFIB_MIN_NQRS-1:
        #We ensure that there are not consecutive constant RRs. We usually
        #consider the last 3 RRs, but if we have only two, they cannot be very
        #similar.
        lastrr = rrs[-3:]
        if len(lastrr) == 2:
            verify(abs(lastrr[1]-lastrr[0]) > C.TMARGIN)
        else:
            verify(not np.all(np.abs(lastrr-np.mean(lastrr)) < C.TMARGIN))
    #Lian 2011 afib classification method.
    verify(is_afib_rhythm_lian(rrs))

def _verify_atrial_activity(pattern):
    """
    Checks if the atrial activity is consistent with the definition of atrial
    fibrillation (that is, absence of constant P Waves or flutter-like
    baseline activity.)
    """
    beats = pattern.evidence[o.QRS][-5:]
    obseq = pattern.obs_seq
    atr_sig = {lead : [] for lead in sig_buf.get_available_leads()}
    pw_lims = []
    idx = pattern.get_step(beats[0])
    #First we get all the signal fragments between ventricular observations,
    #which are the only recognized by this pattern. In these fragments is where
    #atrial activity may be recognized.
    for i in xrange(idx+1, len(obseq)):
        if isinstance(obseq[i], o.QRS):
            beg = next(obs for obs in reversed(obseq[:i])
                                                    if obs is not None).lateend
            end = obseq[i].earlystart
            if end-beg > ms2sp(200):
                beg = end - ms2sp(200)
            pw_lims.append((beg, end))
    for i in xrange(len(beats)-1):
        beg, end = beats[i].lateend, beats[i+1].earlystart
        for lead in atr_sig:
            atr_sig[lead].append(sig_buf.get_signal_fragment(beg, end,
                        lead=lead)[0]-characterize_baseline(lead, beg, end)[0])
    #Flutter check (only for atrial activity)
    aflut = set()
    for lead in atr_sig:
        sigfr = np.concatenate(atr_sig[lead])
        if len(sigfr) > 15 and _is_VF(sigfr):
            aflut.add(lead)
    #FIXME improve flutter check, now is quite poor.
    #aflut = frozenset()
    #P waveform check (only for leads where flutters were not found.)
    pwaves = []
    for beg, end in pw_lims:
        pwsig = _get_pwave_sig(beg, end)
        if pwsig is not None:
            for lead in aflut:
                pwsig.pop(lead, None)
            if not pwsig:
                continue
            for wave in pwaves:
                verify(abs(wave.values()[0].pr -
                                        pwsig.values()[0].pr) > C.TMARGIN or
                                                 not signal_match(wave, pwsig))
            pwaves.append(pwsig)

def _qrs_gconst(pattern, _):
    """
    General constraints to be added when a new cycle is observed, which
    currently coincides with the observation of the T waves or a QRS complex
    not followed by an observed T wave.
    """
    #We update the measurements of the rhythm.
    _update_measures(pattern)
    #And check that there are no missed beat forms.
    _check_missed_beats(pattern)
    beats = pattern.evidence[o.QRS]
    #Morphology check. We require the rhythm morphology to be matched
    #by the new beat in the sequence.
    ref = pattern.hypothesis.morph
    #We initialize the morphology with the first beat.
    if not ref:
        ref = copy.deepcopy(beats[0].shape)
        pattern.hypothesis.morph = ref
    verify(signal_match(ref, beats[-1].shape))
    #TODO improve morphology updating.
    #_update_morphology(pattern)
    envrhythms = pattern.evidence[o.Cardiac_Rhythm]
    prevafib = envrhythms and isinstance(envrhythms[0], o.Atrial_Fibrillation)
    if len(beats) == 2 and envrhythms:
        refrr, stdrr = envrhythms[-1].meas.rr
        #The first RR cannot be within the mean +- 2*std of the previous RR.
        #There must be a rhythm change.
        verify(not refrr - 2*stdrr <= beats[1].time.start - beats[0].time.start
                                                            <= refrr + 2*stdrr)
    if len(beats) >= 3:
        verify(not beats[-1].paced)
        rpks = np.array([b.time.start for b in beats])
        rrs = np.diff(rpks)
        _verify_afib_rhythm(rrs)
        #With this check, we avoid false positives with bigeminies, checking
        #the RR constraints with even and odd rrs.
        if len(beats) >= 6:
            _verify_afib_rhythm(rrs[0::2])
            _verify_afib_rhythm(rrs[1::2])
            _verify_afib_rhythm(np.diff(rpks[0::2]))
    #Atrial activity is only checked at the beginning of the pattern and if
    #there are not previous atrial fibrillation episodes.
    if 1 < len(beats) < 32 and not prevafib:
        _verify_atrial_activity(pattern)


AFIB_PATTERN = PatternAutomata()
AFIB_PATTERN.name = "Atrial Fibrillation"
AFIB_PATTERN.Hypothesis = o.Atrial_Fibrillation
AFIB_PATTERN.add_transition(0, 1, o.Cardiac_Rhythm, ENVIRONMENT,
                                          _prev_afib_tconst, _prev_afib_gconst)
AFIB_PATTERN.add_transition(1, 1, o.Cardiac_Rhythm, ENVIRONMENT,
                                  _prev_multrhythm_tconst, _prev_rhythm_gconst)
AFIB_PATTERN.add_transition(1, 2, o.Cardiac_Rhythm, ENVIRONMENT,
                                      _prev_rhythm_tconst, _prev_rhythm_gconst)
AFIB_PATTERN.add_transition(0, 2, o.Cardiac_Rhythm, ENVIRONMENT,
                                      _prev_rhythm_tconst, _prev_rhythm_gconst)
AFIB_PATTERN.add_transition(2, 3, o.QRS, ENVIRONMENT, _qrs0_tconst)
AFIB_PATTERN.add_transition(3, 4, o.QRS, ABSTRACTED, _qrs_tconst, _qrs_gconst)
AFIB_PATTERN.add_transition(4, 5, o.QRS, ABSTRACTED, _qrs_tconst, _qrs_gconst)
#We have two different ways to get a sufficient set of evidence. If there is
#a previous Afib episode, then we only need 3 QRS to have a firm hypothesis of
#a new episode.
AFIB_PATTERN.add_transition(5, 6, gconst=_prev_afib_exists_gconst)
#Optional T waves are searched now
AFIB_PATTERN.add_transition(6, 7, o.TWave, ABSTRACTED, get_t_tconst(1))
AFIB_PATTERN.add_transition(6, 7)
AFIB_PATTERN.add_transition(7, 8, o.TWave, ABSTRACTED, get_t_tconst(-1))
AFIB_PATTERN.add_transition(7, 8)
AFIB_PATTERN.add_transition(8, 7, o.QRS, ABSTRACTED, _qrs_tconst, _qrs_gconst)
#Else, we need at least AFIB_MIN_NQRS QRS complexes.
AFIB_PATTERN.add_transition(5, 9, o.QRS, ABSTRACTED, _qrs_tconst, _qrs_gconst)
for i in xrange(C.AFIB_MIN_NQRS-4):
    AFIB_PATTERN.add_transition(9+i, 10+i, o.QRS, ABSTRACTED,
                                                      _qrs_tconst, _qrs_gconst)
#T waves are searched after the necessary QRS have been observed.
ST = 9 + C.AFIB_MIN_NQRS - 4
for i in xrange(C.AFIB_MIN_NQRS-2):
    AFIB_PATTERN.add_transition(ST, ST+1, o.TWave, ABSTRACTED,
                                                             get_t_tconst(i+1))
    AFIB_PATTERN.add_transition(ST, ST+1)
    ST += 1
#T wave for the last observed QRS, and cycle QRS observation.
AFIB_PATTERN.add_transition(ST, ST+1, o.TWave, ABSTRACTED, get_t_tconst(-1))
AFIB_PATTERN.add_transition(ST, ST+1)
AFIB_PATTERN.add_transition(ST+1, ST, o.QRS, ABSTRACTED, _qrs_tconst,
                                                                   _qrs_gconst)
AFIB_PATTERN.final_states.add(8)
AFIB_PATTERN.final_states.add(ST+1)
AFIB_PATTERN.abstractions[o.QRS] = (AFIB_PATTERN.transitions[5],)
AFIB_PATTERN.obs_proc = _rhythm_obs_proc
AFIB_PATTERN.freeze()

#########################
### Helper structures ###
#########################

# The following polynomial gives us the minimum number of Non-Empty Cells in
# the RdR plot to positively detect atrial fibrillation, as described in
# "Lian: A simple method to detect Atrial Fibrillation using RR intervals"

_NEC = scipy.interpolate.interpolate.lagrange([32, 64, 128], [23, 40, 65])


if __name__ == "__main__":
    pass
