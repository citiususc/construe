# -*- coding: utf-8 -*-
# pylint: disable-msg=
"""
Created on Mon Jul 28 18:18:20 2014

This module defines the abstraction pattern to describe "Ventricular Couplets",
a rhythm pattern consisting of two consecutive ventricular extrasystoles
between two normal complexes.

@author: T. Teijeiro
"""

from construe.model.automata import (PatternAutomata, ABSTRACTED as ABS,
                                        ENVIRONMENT as ENV, BASIC_TCONST)
from construe.model import verify, Interval as Iv
import construe.knowledge.observables as o
import construe.knowledge.constants as C
from construe.utils.signal_processing.xcorr_similarity import signal_unmatch
from construe.knowledge.abstraction_patterns.rhythm.regular import (
                                                           _check_missed_beats)
import copy
import numpy as np

###################################
### Previous rhythm constraints ###
###################################

def _prev_rhythm_tconst(pattern, rhythm):
    """Temporal constraints of a cardiac rhythm with the precedent one."""
    BASIC_TCONST(pattern, rhythm)
    tnet = pattern.last_tnet
    tnet.set_equal(pattern.hypothesis.start, rhythm.end)


#######################
### QRS constraints ###
#######################

def _qrs_after_twave(pattern, qrs):
    """
    Sets the appropriate temporal constraints to force a QRS complex to begin
    after the prior T Wave finishes.
    """
    obseq = pattern.obs_seq
    oidx = pattern.get_step(qrs)
    tnet = pattern.last_tnet
    #If there is a prior T Wave, it must finish before the start
    #of the QRS complex.
    if oidx > 0 and isinstance(obseq[oidx-1], o.TWave):
        prevt = obseq[oidx-1]
        tnet.set_before(prevt.end, qrs.start)

def _common_qrs_constraints(pattern, qrs):
    """Temporal constraints affecting all QRS complex."""
    tnet = pattern.last_tnet
    hyp = pattern.hypothesis
    BASIC_TCONST(pattern, qrs)
    tnet.add_constraint(qrs.start, qrs.end, C.QRS_DUR)
    tnet.set_before(qrs.time, hyp.end)

def _n0_tconst(pattern, qrs):
    """Temporal constraints of the first environment QRS complex"""
    tnet = pattern.last_tnet
    hyp = pattern.hypothesis
    _common_qrs_constraints(pattern, qrs)
    #The environment QRS complex determines the beginning of the couplet.
    tnet.set_equal(hyp.start, qrs.time)

def _v0_tconst(pattern, qrs):
    """Temporal constraints of the first extrasystole in the couplet"""
    beats = pattern.evidence[o.QRS]
    idx = beats.index(qrs)
    tnet = pattern.last_tnet
    if idx > 0:
        prev = beats[idx-1]
        #The reference RR varies from an upper limit to the last measurement,
        #through the contextual previous rhythm.
        refrr = C.BRADY_RR.end
        stdrr = 0.1*refrr
        if pattern.evidence[o.Cardiac_Rhythm] and idx == 1:
            mrr, srr = pattern.evidence[o.Cardiac_Rhythm][0].meas.rr
            if mrr > 0:
                refrr, stdrr = mrr, srr
        adv = max(stdrr, C.TMARGIN)
        #The first ectopic beat must be advanced wrt the reference RR
        tnet.add_constraint(prev.time, qrs.time,
                        Iv(C.TACHY_RR.start, max(C.TACHY_RR.start, refrr-adv)))
        tnet.set_before(prev.end, qrs.start)
    _common_qrs_constraints(pattern, qrs)

def _v1_tconst(pattern, qrs):
    """Temporal constraints of the second extrasystole in the couplet"""
    beats = pattern.evidence[o.QRS]
    idx = beats.index(qrs)
    tnet = pattern.last_tnet
    prev = beats[idx-1]
    #The second extrasystole must be shorter or approximately equal to the
    #first one, so we give the standard margin for the RR increasing.
    refrr = prev.time.end - beats[idx-2].time.start
    const = Iv(C.TACHY_RR.start, refrr+C.ICOUPLET_MAX_DIFF)
    tnet.add_constraint(prev.time, qrs.time, const)
    tnet.add_constraint(prev.end, qrs.end, const)
    tnet.add_constraint(prev.end, qrs.start, Iv(C.TQ_INTERVAL_MIN, np.Inf))
    #The second extrasystole should include also the same RR shortening
    #constraints of the first one.
    _v0_tconst(pattern, qrs)
    _qrs_after_twave(pattern, qrs)

def _n1_tconst(pattern, qrs):
    """Temporal constraints of the normal beat determining the couplet end"""
    beats = pattern.evidence[o.QRS]
    idx = beats.index(qrs)
    tnet = pattern.last_tnet
    hyp = pattern.hypothesis
    _common_qrs_constraints(pattern, qrs)
    _qrs_after_twave(pattern, qrs)
    #Compensatory pause RR
    minrr = min(beats[idx-1].time.start - beats[idx-2].time.start,
                beats[idx-2].time.start - beats[idx-3].time.start)
    maxrr = max(beats[idx-1].time.end - beats[idx-2].time.start,
                beats[idx-2].time.end - beats[idx-3].time.start)
    mincompause = max(C.COMPAUSE_MIN_DUR, min(minrr*C.ICOUPLET_MIN_RREXT_F,
                                              minrr+C.ICOUPLET_MIN_RREXT))
    tnet.add_constraint(beats[idx-1].time, qrs.time,
                                 Iv(mincompause, maxrr*C.COMPAUSE_RREXT_MAX_F))
    tnet.set_equal(qrs.time, hyp.end)
    #The morphology of the first and last QRS complexes should be similar
    qrs.shape = beats[idx-3].shape
    qrs.paced = beats[idx-3].paced

def _v0_gconst(pattern, qrs):
    """
    General constraints to be satisfied by the first ectopic QRS complex of
    the couplet, which is the abduction point of the pattern. These constraints
    ensure that the ectopic beat is really advanced. This is needed because the
    first ectopic beat is the abduction point, and therefore the reference RR
    cannot be calculated when establishing the temporal constraints for the
    first time.
    """
    if pattern.evidence[o.Cardiac_Rhythm]:
        mrr, stdrr = pattern.evidence[o.Cardiac_Rhythm][0].meas[0]
        if mrr > 0:
            short = max(C.TMARGIN, stdrr)
            ectrr = qrs.time.start - pattern.evidence[o.QRS][0].time.start
            verify(ectrr < 0.9 * mrr)
            verify(ectrr < mrr-short)

def _couplet_gconst(pattern, _):
    """
    General constraints to be checked when the couplet finishes.
    """
    _check_missed_beats(pattern)
    #The second extrasystole cannot be in rhythm with contextual beats, or in
    #such case it must have a different shape.
    beats = pattern.evidence[o.QRS]
    mpt = beats[0].time.start + (beats[-1].time.start - beats[0].time.start)/2.
    verify(abs(mpt-beats[2].time.start) >= C.ICOUPLET_RCHANGE
           or signal_unmatch(beats[2].shape, beats[-1].shape))
    pattern.hypothesis.meas = copy.copy(
                                    pattern.evidence[o.Cardiac_Rhythm][0].meas)

#################################
### P and T waves constraints ###
#################################

def _p_tconst(pattern, pwave):
    """
    Temporal constraints of the P Waves wrt the corresponding QRS complex
    """
    BASIC_TCONST(pattern, pwave)
    tnet = pattern.last_tnet
    tnet.add_constraint(pwave.start, pwave.end, C.PW_DURATION)
    #We find the QRS observed just before that P wave.
    idx = pattern.get_step(pwave)
    if idx > 0 and isinstance(pattern.trseq[idx-1][1], o.QRS):
        qrs = pattern.trseq[idx-1][1]
        #PR interval
        tnet.add_constraint(pwave.start, qrs.start, C.N_PR_INTERVAL)
        tnet.set_before(pwave.end, qrs.start)

def _t_tconst(pattern, twave):
    """
    Temporal constraints of the T Waves wrt the corresponding QRS complex.
    """
    BASIC_TCONST(pattern, twave)
    obseq = pattern.obs_seq
    idx = pattern.get_step(twave)
    try:
        tnet = pattern.last_tnet
        #We find the qrs observation precedent to this T wave.
        qrs = next(obseq[i] for i in xrange(idx-1, -1, -1)
                                                if isinstance(obseq[i], o.QRS))
        #If we have more than one QRS, it is possible to constrain even more
        #the location of the T-Wave, based on rhythm information.
        qidx = pattern.evidence[o.QRS].index(qrs)
        if qidx > 1:
            refrr = ((qrs.time.end - pattern.evidence[o.QRS][qidx-2].time.start)
                                                                          /2.0)
            tnet.add_constraint(qrs.time, twave.end,
                                              Iv(0, refrr - C.TQ_INTERVAL_MIN))
        if idx > 0 and isinstance(obseq[idx-1], o.PWave):
            pwave = obseq[idx-1]
            tnet.add_constraint(pwave.end, twave.start, Iv(C.ST_INTERVAL.start,
                                            C.PQ_INTERVAL.end + C.QRS_DUR.end))
        #ST interval
        tnet.add_constraint(qrs.end, twave.start, C.ST_INTERVAL)
        #QT duration
        tnet.add_constraint(qrs.start, twave.end, C.N_QT_INTERVAL)
    except StopIteration:
        pass

def _tv0_tconst(pattern, twave):
    """
    Temporal constraints for the T Wave of the first extrasystole, that is
    only observed after the second extrasystole QRS has been properly observed.
    """
    BASIC_TCONST(pattern, twave)
    tnet = pattern.last_tnet
    beats = pattern.evidence[o.QRS]
    #The T wave must be in the hole between the two QRS complexes, and must
    #finish at least 1mm before the next QRS starts.
    tnet.set_before(beats[1].end, twave.start)
    tnet.set_before(twave.end, beats[2].start)
    tnet.add_constraint(twave.end, beats[2].start, Iv(C.TMARGIN, np.Inf))



COUPLET_PATTERN = PatternAutomata()
COUPLET_PATTERN.name = "Couplet"
COUPLET_PATTERN.Hypothesis = o.Couplet
COUPLET_PATTERN.add_transition(0, 1, o.Cardiac_Rhythm, ENV, _prev_rhythm_tconst)
#N
COUPLET_PATTERN.add_transition(1, 2, o.QRS, ENV, _n0_tconst)
#V
COUPLET_PATTERN.add_transition(2, 3, o.QRS, ABS, _v0_tconst, _v0_gconst)
#V
COUPLET_PATTERN.add_transition(3, 4, o.QRS, ABS, _v1_tconst)
#The T wave of the first extrasystole is only observed after the second
#extrasystole QRS complex.
COUPLET_PATTERN.add_transition(4, 5, o.TWave, ABS, _tv0_tconst)
COUPLET_PATTERN.add_transition(4, 6, o.TWave, ABS, _t_tconst)
#N
COUPLET_PATTERN.add_transition(4, 7, o.QRS, ABS, _n1_tconst, _couplet_gconst)
COUPLET_PATTERN.add_transition(5, 6, o.TWave, ABS, _t_tconst)
COUPLET_PATTERN.add_transition(5, 6)
COUPLET_PATTERN.add_transition(6, 7, o.QRS, ABS, _n1_tconst, _couplet_gconst)
COUPLET_PATTERN.add_transition(7, 8, o.PWave, ABS, _p_tconst)
COUPLET_PATTERN.add_transition(7, 9, o.TWave, ABS, _t_tconst)
COUPLET_PATTERN.add_transition(7, 9)
COUPLET_PATTERN.add_transition(8, 9, o.TWave, ABS, _t_tconst)
COUPLET_PATTERN.final_states.add(9)
COUPLET_PATTERN.abstractions[o.QRS] = (COUPLET_PATTERN.transitions[2], )
COUPLET_PATTERN.freeze()