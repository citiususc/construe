# -*- coding: utf-8 -*-
# pylint: disable-msg=
"""
Created on Wed Feb  5 09:21:11 2014

This module contains the definition of the extrasystole abstraction pattern,
that allows the recognition of all types of extrasystoles.

@author: T. Teijeiro
"""

import copy
from construe.model.automata import PatternAutomata, ABSTRACTED, ENVIRONMENT
from construe.model.constraint_network import verify
from construe.model import Interval as Iv
from construe.knowledge.abstraction_patterns.rhythm.regular import (
                                                           _check_missed_beats)
from construe.utils.signal_processing.xcorr_similarity import (signal_match,
                                                           signal_unmatch)
import construe.knowledge.constants as C
import construe.knowledge.observables as o

def _prev_rhythm_tconst(pattern, rhythm):
    """Temporal constraints of a cardiac rhythm with the precedent one."""
    pattern.tnet.set_equal(pattern.hypothesis.start, rhythm.end)
    pattern.tnet.add_constraint(pattern.hypothesis.start,
                                pattern.hypothesis.end,
                                Iv(2*C.TACHY_RR.start, 3*C.BRADY_RR.end))

def _prev_rhythm_gconst(pattern, rhythm):
    """General constraints of a cardiac rhythm with the preceden one."""
    #We only accept the concatenation of the same rhythm for asystoles.
    verify(type(pattern.hypothesis) is o.Asystole
                                   or type(pattern.hypothesis) != type(rhythm))
    verify(rhythm.earlystart != pattern.hypothesis.earlystart)
    #An extrasystole does not modify the reference RR.
    pattern.hypothesis.meas = copy.copy(rhythm.meas)

def _prev_rhythm_nreg_gconst(pattern, rhythm):
    _prev_rhythm_gconst(pattern, rhythm)
    verify(not isinstance(rhythm, o.RegularCardiacRhythm))

def _qrs_rref_tconst(pattern, qrs):
    """Temporal constraints of the first environment QRS complex"""
    if pattern.evidence[o.Cardiac_Rhythm]:
        pattern.tnet.set_before(qrs.end,
                                     pattern.evidence[o.Cardiac_Rhythm][0].end)

def _qrs_env_tconst(pattern, qrs):
    """Temporal constraints of the second environment QRS complex"""
    pattern.tnet.set_equal(pattern.hypothesis.start, qrs.time)
    if pattern.evidence[o.QRS].index(qrs) == 1:
        prev = pattern.evidence[o.QRS][0]
        pattern.tnet.add_constraint(prev.time, qrs.time, Iv(C.TACHY_RR.start,
                                                               C.BRADY_RR.end))

def _qrs_ext_tconst(ventricular):
    """
    Returns the temporal constraints function of the extrasystole beat,
    requiring an anticipated beat. It accepts a parameter to determine if the
    anticipated beat must be ventricular.
    """
    def tconst(pattern, qrs):
        """
        Defines the temporal constraints function for the ectopic beat in an
        extrasystole, depending on its ventricular nature or not.
        """
        tnet = pattern.tnet
        tnet.set_before(qrs.end, pattern.hypothesis.end)
        if ventricular:
            tnet.add_constraint(qrs.start, qrs.end, C.VQRS_DUR)
        #It must be the third beat.
        beats = pattern.evidence[o.QRS]
        idx = beats.index(qrs)
        #If there is a previous beat
        if idx > 0:
            tnet.add_constraint(beats[idx-1].time, qrs.time,
                                      Iv(C.TACHY_RR.start, 0.9*C.BRADY_RR.end))
        #If all the previous evidence has been observed
        if pattern.istate == 0:
            #Anticipation of at least the 10% of the reference RR, or 1mm.
            if idx == 2:
                refrr = beats[1].time.end - beats[0].time.start
            elif pattern.evidence[o.Cardiac_Rhythm][0] is not pattern.finding:
                refrr = pattern.hypothesis.meas.rr[0]
            else:
                refrr = None
            if refrr is not None:
                short = min(0.1*refrr, C.TMARGIN)
                tnet.add_constraint(beats[idx-1].time, qrs.time,
                      Iv(C.TACHY_RR.start, max(C.TACHY_RR.start, refrr-short)))
    return tconst


def _qrs_fin_pause_tconst(pattern, qrs):
    """
    Temporal constraints for the compensatory pause of the last QRS of the
    extrasystole.
    """
    tnet = pattern.tnet
    tnet.set_equal(pattern.hypothesis.end, qrs.time)
    beats = pattern.evidence[o.QRS]
    idx = beats.index(qrs)
    #We need all the previous evidence.
    if pattern.istate == 0:
        step = pattern.get_step(qrs)
        if isinstance(pattern.trseq[step-1][1], o.TWave):
            twave = pattern.trseq[step-1][1]
            tnet.set_before(twave.end, qrs.start)
        #Reference RR
        minrr = (beats[1].time.start - beats[0].time.end if len(beats) == 4
                                            else pattern.hypothesis.meas.rr[0])
        maxrr = (beats[1].time.end - beats[0].time.start if len(beats) == 4
                                            else pattern.hypothesis.meas.rr[0])
        #Compensatory pause
        tnet.add_constraint(beats[-3].time, qrs.time,
                      Iv(min(minrr+C.COMPAUSE_MIN_DUR, minrr*C.COMPAUSE_MIN_F),
                                                       maxrr*C.COMPAUSE_MAX_F))
        #Advanced beat RR
        minrr = beats[-2].time.start - beats[-3].time.end
        maxrr = beats[-2].time.end - beats[-3].time.start
        tnet.add_constraint(beats[-2].time, qrs.time,
              Iv(min(minrr*C.COMPAUSE_RREXT_MIN_F, minrr+C.COMPAUSE_RREXT_MIN),
                                                 maxrr*C.COMPAUSE_RREXT_MAX_F))
        #The last QRS should have the same morphology than the one before the
        #extrasystole.
        if not qrs.frozen:
            qrs.shape = beats[-3].shape
            qrs.paced = beats[-3].paced
    elif idx > 0:
        #We constraint the previous beat location.
        tnet.add_constraint(beats[idx-1].time, qrs.time,
                            Iv(C.TACHY_RR.start*C.COMPAUSE_MIN_F,
                                      C.BRADY_RR.start*C.COMPAUSE_RREXT_MAX_F))

def _qrs_fin_npause_tconst(pattern, qrs):
    """
    Temporal constraints of the fourth beat in an extrasystole without
    compensatory pause.
    """
    tnet = pattern.tnet
    tnet.set_equal(pattern.hypothesis.end, qrs.time)
    beats = pattern.evidence[o.QRS]
    #We need all previous evidence
    if pattern.istate == 0:
        step = pattern.get_step(qrs)
        twave = pattern.trseq[step-1][1]
        if isinstance(twave, o.TWave):
            tnet.set_before(twave.end, qrs.start)
        #Reference RR
        minrr = (beats[1].time.start - beats[0].time.end if len(beats) == 4
                                            else pattern.hypothesis.meas.rr[0])
        maxrr = (beats[1].time.end - beats[0].time.start if len(beats) == 4
                                            else pattern.hypothesis.meas.rr[0])
        tnet.add_constraint(beats[-3].time, qrs.time,
                              Iv(minrr - C.RR_MAX_DIFF, maxrr + C.RR_MAX_DIFF))
        #The last QRS should have the same morphology than the one before the
        #extrasystole.
        qrs.shape = beats[-3].shape

def _p_qrs_tconst(pattern, pwave):
    """
    Temporal constraints of the P waves with the corresponding QRS complex.
    """
    obseq = pattern.obs_seq
    idx = pattern.get_step(pwave)
    #We find the qrs observation precedent to this P wave.
    if idx == 0 or not isinstance(obseq[idx-1], o.QRS):
        return
    qrs = obseq[idx-1]
    pattern.tnet.add_constraint(pwave.start, pwave.end, C.PW_DURATION)
    #PR interval
    pattern.tnet.add_constraint(pwave.start, qrs.start, C.N_PR_INTERVAL)
    pattern.tnet.set_before(pwave.end, qrs.start)

def _t_qrs_tconst(pattern, twave):
    """
    Temporal constraints of the T waves with the corresponding QRS complex
    """
    obseq = pattern.obs_seq
    idx = pattern.get_step(twave)
    #We find the qrs observation precedent to this T wave.
    try:
        qrs = next(obseq[i] for i in xrange(idx-1, -1, -1)
                                                if isinstance(obseq[i], o.QRS))
        tnet = pattern.tnet
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

def _qrs_ext_gconst(pattern, qrs):
    """
    General constraints to verify that the extrasystole qrs is advanced wrt
    the environment rhythm.
    """
    if pattern.evidence[o.Cardiac_Rhythm]:
        mrr, stdrr = pattern.evidence[o.Cardiac_Rhythm][0].meas.rr
        if mrr > 0:
            beats = pattern.evidence[o.QRS]
            idx = beats.index(qrs)
            rr = qrs.time.start - beats[idx-1].time.start
            mshort = min(stdrr, 0.1*mrr, C.TMARGIN)
            verify(rr <= mrr-mshort)

def _qrs_ext_gconst_npause(pattern, qrs):
    """
    General constraints to verify that the extrasystole QRS is advanced wrt the
    environment rhythm. In addition, if there is no compensatory pause, the
    advanced QRS must have different origin than the surrounding complexes.
    """
    _qrs_ext_gconst(pattern, qrs)
    beats = pattern.evidence[o.QRS]
    if len(beats) > 1:
        verify(signal_unmatch(beats[-2].shape, qrs.shape))


def _extrasyst_gconst(pattern, _):
    """
    General constraints of the pattern that are checked when the last T wave
    has been observed.
    """
    beats = pattern.evidence[o.QRS]
    if pattern.istate == 0:
        #We ensure that there are no missed beats.
        _check_missed_beats(pattern)
        #We must ensure that the first two beats and the last one have the
        #same shape.
        verify((beats[-3].paced and beats[-1].paced) or
                                signal_match(beats[-3].shape, beats[-1].shape))

EXTRASYSTOLE_PATTERN = PatternAutomata()
EXTRASYSTOLE_PATTERN.name = 'Extrasystole'
EXTRASYSTOLE_PATTERN.Hypothesis = o.Extrasystole
EXTRASYSTOLE_PATTERN.add_transition(0, 1, o.Cardiac_Rhythm, ENVIRONMENT,
                                      _prev_rhythm_tconst, _prev_rhythm_gconst)
EXTRASYSTOLE_PATTERN.add_transition(1, 2, o.QRS, ENVIRONMENT, _qrs_rref_tconst)
EXTRASYSTOLE_PATTERN.add_transition(0, 2, o.Cardiac_Rhythm, ENVIRONMENT,
                                 _prev_rhythm_tconst, _prev_rhythm_nreg_gconst)
EXTRASYSTOLE_PATTERN.add_transition(2, 3, o.QRS, ENVIRONMENT, _qrs_env_tconst)
#Now there are two ways in the automata, one in which we look for the
#compensatory pause, and another in which a ventricular extrasystole may not
#involve a compensatory pause.
##First way: Compensatory pause.
EXTRASYSTOLE_PATTERN.add_transition(3, 4, o.QRS, ABSTRACTED,
                                       _qrs_ext_tconst(False), _qrs_ext_gconst)
EXTRASYSTOLE_PATTERN.add_transition(4, 5, o.PWave, ABSTRACTED, _p_qrs_tconst)
EXTRASYSTOLE_PATTERN.add_transition(5, 6, o.TWave, ABSTRACTED, _t_qrs_tconst)
EXTRASYSTOLE_PATTERN.add_transition(4, 6, o.TWave, ABSTRACTED, _t_qrs_tconst)
EXTRASYSTOLE_PATTERN.add_transition(4, 7, o.QRS, ABSTRACTED,
                                                         _qrs_fin_pause_tconst)
EXTRASYSTOLE_PATTERN.add_transition(6, 7, o.QRS, ABSTRACTED,
                                                         _qrs_fin_pause_tconst)
EXTRASYSTOLE_PATTERN.add_transition(7, 8, o.PWave, ABSTRACTED, _p_qrs_tconst)
EXTRASYSTOLE_PATTERN.add_transition(8, 9, o.TWave, ABSTRACTED, _t_qrs_tconst,
                                                             _extrasyst_gconst)
EXTRASYSTOLE_PATTERN.add_transition(7, 9, o.TWave, ABSTRACTED, _t_qrs_tconst,
                                                             _extrasyst_gconst)
EXTRASYSTOLE_PATTERN.add_transition(7, 9, gconst=_extrasyst_gconst)
##Second way: Ventricular extrasystole without compensatory pause
EXTRASYSTOLE_PATTERN.add_transition(3, 10, o.QRS, ABSTRACTED,
                                 _qrs_ext_tconst(True), _qrs_ext_gconst_npause)
EXTRASYSTOLE_PATTERN.add_transition(10, 11, o.TWave, ABSTRACTED, _t_qrs_tconst)
EXTRASYSTOLE_PATTERN.add_transition(10, 12, o.QRS, ABSTRACTED,
                                                        _qrs_fin_npause_tconst)
EXTRASYSTOLE_PATTERN.add_transition(11, 12, o.QRS, ABSTRACTED,
                                                        _qrs_fin_npause_tconst)
EXTRASYSTOLE_PATTERN.add_transition(12, 13, o.PWave, ABSTRACTED, _p_qrs_tconst)
EXTRASYSTOLE_PATTERN.add_transition(13, 14, o.TWave, ABSTRACTED, _t_qrs_tconst,
                                                             _extrasyst_gconst)
EXTRASYSTOLE_PATTERN.add_transition(12, 14, o.TWave, ABSTRACTED, _t_qrs_tconst,
                                                             _extrasyst_gconst)
EXTRASYSTOLE_PATTERN.add_transition(12, 14, gconst=_extrasyst_gconst)
EXTRASYSTOLE_PATTERN.final_states.add(9)
EXTRASYSTOLE_PATTERN.final_states.add(14)
EXTRASYSTOLE_PATTERN.abstractions[o.QRS] = (EXTRASYSTOLE_PATTERN.transitions[4],
                                            EXTRASYSTOLE_PATTERN.transitions[8],
                                           EXTRASYSTOLE_PATTERN.transitions[14])
EXTRASYSTOLE_PATTERN.freeze()
