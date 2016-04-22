# -*- coding: utf-8 -*-
# pylint: disable-msg=
"""
Created on Thu May 15 13:34:38 2014

This module contains the definition of the abstraction pattern representing
a rhythm block.

@author: T. Teijeiro
"""

from construe.model.automata import (PatternAutomata, ABSTRACTED,
                                                     ENVIRONMENT, BASIC_TCONST)
from construe.knowledge.abstraction_patterns.rhythm.regular import (
                                                           _check_missed_beats)
from construe.model import ConstraintNetwork, verify, Interval as Iv
from construe.utils.signal_processing.xcorr_similarity import signal_match
import construe.knowledge.constants as C
import construe.knowledge.observables as o
import copy

def _prev_rhythm_tconst(pattern, rhythm):
    """Temporal constraints of a cardiac rhythm with the precedent one."""
    BASIC_TCONST(pattern, rhythm)
    tnet = pattern.last_tnet
    tnet.set_equal(pattern.hypothesis.start, rhythm.end)

def _prev_rhythm_gconst(pattern, rhythm):
    """General constraints of a cardiac rhythm with the preceden one."""
    #We only accept the concatenation of the same rhythm for asystoles.
    verify(not isinstance(rhythm, o.Asystole) and
                                      type(pattern.hypothesis) != type(rhythm))
    #A block keeps the reference measures.
    pattern.hypothesis.meas = copy.copy(rhythm.meas)

def _prev_asyst_gconst(pattern, asyst):
    """
    Verification of the existence of a previous asystole episode.
    """
    verify(isinstance(asyst, o.Asystole))
    #A block keeps the reference measures.
    pattern.hypothesis.meas = copy.copy(asyst.meas)

def _qrs1_tconst(pattern, qrs):
    """Temporal constraints of the first QRS complex"""
    BASIC_TCONST(pattern, qrs)
    pattern.last_tnet.add_constraint(qrs.start, qrs.end, C.QRS_DUR)
    if pattern.evidence[o.Cardiac_Rhythm]:
        pattern.last_tnet.set_before(qrs.end,
                                     pattern.evidence[o.Cardiac_Rhythm][0].end)

def _qrs2_tconst(pattern, qrs):
    """Temporal constraints of the second QRS complex"""
    BASIC_TCONST(pattern, qrs)
    tnet = pattern.last_tnet
    tnet.set_equal(pattern.hypothesis.start, qrs.time)
    tnet.add_constraint(qrs.start, qrs.end, C.QRS_DUR)
    if pattern.evidence[o.QRS].index(qrs) == 1:
        prev = pattern.evidence[o.QRS][0]
        tnet.add_constraint(prev.time, qrs.time, Iv(C.TACHY_RR.start,
                                                               C.BRADY_RR.end))
    elif (pattern.evidence[o.Cardiac_Rhythm] and
                isinstance(pattern.evidence[o.Cardiac_Rhythm][0], o.Asystole)):
        pattern.last_tnet.set_equal(pattern.hypothesis.start, qrs.time)


def _qrs3_tconst(pattern, qrs):
    """Temporal constraints of the third QRS complex"""
    BASIC_TCONST(pattern, qrs)
    tnet = pattern.last_tnet
    tnet.set_before(qrs.time, pattern.hypothesis.end)
    tnet.add_constraint(qrs.start, qrs.end, C.QRS_DUR)
    beats = pattern.evidence[o.QRS]
    #If there is a previous QRS
    if beats.index(qrs) == 1:
        tnet.add_constraint(beats[0].time, qrs.time,
                          Iv(C.TACHY_RR.start + C.RR_MAX_DIFF, C.BRADY_RR.end))
    #If we have reached an initial state.
    if pattern.istate == 0:
        idx = beats.index(qrs)
        meanrr, stdrr = pattern.hypothesis.meas.rr
        minrr = (beats[1].time.start - beats[0].time.end if idx == 2
                                                             else meanrr-stdrr)
        tnet.add_constraint(beats[idx-1].time, qrs.time,
                              Iv(min(C.ASYSTOLE_RR.start, minrr+C.RR_MAX_DIFF),
                                                          C.ASYSTOLE_RR.start))
        #The block time has to be higher than the mean RR plus the standard
        #deviation.
        if meanrr > 0:
            tnet.add_constraint(beats[idx-1].time, qrs.time,
                      Iv(meanrr+stdrr, max(meanrr+stdrr, C.ASYSTOLE_RR.start)))

def _qrsn_tconst(pattern, qrs):
    """
    Temporal constraints for the QRS complexes.
    """
    beats = pattern.evidence[o.QRS]
    idx = beats.index(qrs)
    hyp = pattern.hypothesis
    tnet = pattern.last_tnet
    obseq = pattern.obs_seq
    oidx = pattern.get_step(qrs)
    prev = beats[idx-1]
    #In cyclic observations, we have to introduce more networks to simplify
    #the minimization operation.
    tnet.remove_constraint(hyp.end, prev.time)
    tnet = ConstraintNetwork()
    pattern.temporal_constraints.append(tnet)
    meanrr, stdrr = pattern.hypothesis.meas.rr
    rr_bounds = Iv(min(C.ASYSTOLE_RR.start, meanrr-stdrr+C.RR_MAX_DIFF),
                                                           C.ASYSTOLE_RR.start)
    tnet.add_constraint(prev.time, qrs.time, rr_bounds)
    tnet.add_constraint(prev.start, qrs.start, rr_bounds)
    tnet.add_constraint(prev.end, qrs.end, rr_bounds)
    tnet.set_before(prev.end, qrs.start)
    #If there is a prior T Wave, it must finish before the start
    #of the QRS complex.
    if isinstance(obseq[oidx-1], o.TWave):
        prevt = obseq[oidx-1]
        tnet.set_before(prevt.end, qrs.start)
    BASIC_TCONST(pattern, qrs)
    tnet.add_constraint(qrs.start, qrs.end, C.QRS_DUR)
    tnet.set_before(qrs.time, hyp.end)
    #We can introduce constraints on the morphology of the new QRS complex.
    if hyp.morph and not qrs.frozen:
        qrs.shape = hyp.morph


def _p_qrs_tconst(pattern, pwave):
    """
    Temporal constraints of the P waves with the corresponding QRS complex.
    """
    BASIC_TCONST(pattern, pwave)
    obseq = pattern.obs_seq
    idx = pattern.get_step(pwave)
    #We find the qrs observation precedent to this P wave.
    if idx == 0 or not isinstance(obseq[idx-1], o.QRS):
        return
    qrs = obseq[idx-1]
    tnet = pattern.last_tnet
    tnet.add_constraint(pwave.start, pwave.end, C.PW_DURATION)
    #PR interval
    tnet.add_constraint(pwave.start, qrs.start, C.N_PR_INTERVAL)
    tnet.set_before(pwave.end, qrs.start)

def _t_qrs_tconst(pattern, twave):
    """
    Temporal constraints of the T waves with the corresponding QRS complex
    """
    BASIC_TCONST(pattern, twave)
    obseq = pattern.obs_seq
    idx = pattern.get_step(twave)
    #We find the qrs observation precedent to this T wave.
    try:
        qrs = next(obseq[i] for i in xrange(idx-1, -1, -1)
                                                if isinstance(obseq[i], o.QRS))
        tnet = pattern.last_tnet
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

def _qrs_gconst(pattern, _):
    """
    General constraints to be added when a new cycle is observed, which
    currently coincides with the observation of the T waves or a QRS complex
    not followed by an observed T wave.
    """
    #We check that there are no missed beat forms.
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
    #This comparison avoids positive matchings with extrasystoles,
    #but we only check it if the beat before the first block is advanced.
    if len(beats) == 3:
        refrr, stdrr = pattern.hypothesis.meas.rr
        if (beats[1].time.start - beats[0].time.start <
                                                min(0.9 * refrr, refrr-stdrr)):
            verify(beats[2].time.start - beats[0].time.start >
                                                      refrr * C.COMPAUSE_MAX_F)
            verify(beats[2].time.start - beats[1].time.start >
                                                    refrr + C.COMPAUSE_MIN_DUR)
    #We require a significant change in consecutive RR intervals.
    if len(beats) >= 3:
        rr = beats[-1].time.start - beats[-2].time.start
        prevrr = beats[-2].time.start - beats[-3].time.start
        verify(abs(prevrr-rr) >= C.RR_MAX_DIFF)

#############################
### Observation procedure ###
#############################
def _rhythmblock_obs_proc(pattern):
    """Observation procedure executed once the rhythm pattern has finished"""
    #We asign the endpoint of the hypothesis.
    pattern.hypothesis.end.value = pattern.evidence[o.QRS][-1].time.value


RHYTHMBLOCK_PATTERN = PatternAutomata()
RHYTHMBLOCK_PATTERN.name = 'Rhythm Block'
RHYTHMBLOCK_PATTERN.Hypothesis = o.RhythmBlock
RHYTHMBLOCK_PATTERN.add_transition(0, 1, o.Cardiac_Rhythm, ENVIRONMENT,
                                      _prev_rhythm_tconst, _prev_rhythm_gconst)
RHYTHMBLOCK_PATTERN.add_transition(0, 2, o.Cardiac_Rhythm, ENVIRONMENT,
                                       _prev_rhythm_tconst, _prev_asyst_gconst)
RHYTHMBLOCK_PATTERN.add_transition(1, 2, o.QRS, ENVIRONMENT, _qrs1_tconst)
RHYTHMBLOCK_PATTERN.add_transition(2, 3, o.QRS, ENVIRONMENT, _qrs2_tconst)
RHYTHMBLOCK_PATTERN.add_transition(3, 4, o.QRS, ABSTRACTED, _qrs3_tconst,
                                                                   _qrs_gconst)
RHYTHMBLOCK_PATTERN.add_transition(4, 5, o.PWave, ABSTRACTED, _p_qrs_tconst)
RHYTHMBLOCK_PATTERN.add_transition(4, 6, o.TWave, ABSTRACTED, _t_qrs_tconst)
RHYTHMBLOCK_PATTERN.add_transition(4, 6)
RHYTHMBLOCK_PATTERN.add_transition(5, 6, o.TWave, ABSTRACTED, _t_qrs_tconst)
RHYTHMBLOCK_PATTERN.add_transition(6, 4, o.QRS, ABSTRACTED, _qrsn_tconst,
                                                                  _qrs_gconst)
RHYTHMBLOCK_PATTERN.final_states.add(6)
RHYTHMBLOCK_PATTERN.abstractions[o.QRS] = (RHYTHMBLOCK_PATTERN.transitions[4],)
RHYTHMBLOCK_PATTERN.obs_proc = _rhythmblock_obs_proc
RHYTHMBLOCK_PATTERN.freeze()

