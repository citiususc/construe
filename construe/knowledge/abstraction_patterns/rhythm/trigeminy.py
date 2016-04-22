# -*- coding: utf-8 -*-
# pylint: disable-msg=
"""
Created on Tue Jul  8 13:09:54 2014

This module contains the definition of the abstraction pattern representing
a trigeminy rhythm.

@author: T. Teijeiro
"""

from construe.model.automata import (PatternAutomata, ABSTRACTED as ABS,
                                        ENVIRONMENT as ENV, BASIC_TCONST)
from construe.model import ConstraintNetwork, verify, Interval as Iv
from construe.knowledge.abstraction_patterns.rhythm.regular import (
                                                           _check_missed_beats)
import construe.knowledge.observables as o
import construe.knowledge.constants as C
import numpy as np

###############################
### Miscellaneous functions ###
###############################

def _is_ectopic(qidx):
    """
    Checks if a QRS index is an ectopic beat in a trigeminy, according to its
    index.

    Parameters
    ----------
    qidx:
        Index of the QRS complex within the QRS observations of this pattern.

    Returns
    -------
    out:
        True if the index corresponds to an ectopic beat, False otherwise.
    """
    return (qidx-1) % 3 == 0

def _get_measures(pattern, ectopic= False):
    """
    Obtains the characteristic measures of the cardiac rhythms (RR, PQ and RT
    intervals), allowing to filter by the beat type, by using the *ectopic*
    flag. The output tuple (rrs, pqs, rts) contains the series of these
    parameters.
    """
    beats = [b for b in pattern.evidence[o.QRS] if b not in pattern.findings]
    #RR
    rrs = np.diff([beats[i].time.start for i in xrange(len(beats))
                                                 if _is_ectopic(i) == ectopic])
    if not ectopic:
        #For normal beats, the RR intervals has to be reduced in the ectopic
        #positions.
        for i in xrange(0, len(rrs), 2):
            rrs[i] = rrs[i]/2.0
    #RT
    rts = []
    for twave in pattern.evidence[o.TWave]:
        if twave not in pattern.findings:
            qrs = next(q for q in reversed(beats)
                                               if q.lateend < twave.earlystart)
            if _is_ectopic(beats.index(qrs)) == ectopic:
                rts.append(twave.lateend - qrs.time.start)
    #PQ
    pqs = []
    for pwave in pattern.evidence[o.PWave]:
        if pwave not in pattern.findings:
            qrs = next(q for q in beats if q.earlystart > pwave.lateend)
            if _is_ectopic(beats.index(qrs)) == ectopic:
                pqs.append(qrs.earlystart - pwave.earlystart)
    return (rrs, pqs, rts)

#############################
### Observation procedure ###
#############################

def _rhythm_obs_proc(pattern):
    """Observation procedure executed once the rhythm pattern has finished"""
    #We asign the endpoint of the hypothesis.
    pattern.hypothesis.end.value = pattern.evidence[o.QRS][-1].time.value

###########################################
### Previous cardiac rhythm constraints ###
###########################################

def _prev_rhythm_tconst(pattern, rhythm):
    """Temporal constraints of a cardiac rhythm with the precedent one."""
    BASIC_TCONST(pattern, rhythm)
    tnet = pattern.last_tnet
    tnet.set_equal(pattern.hypothesis.start, rhythm.end)


#################################
### QRS complexes constraints ###
#################################

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

def _env_qrs_tconst(pattern, qrs):
    """Temporal constraints for the environment QRS observation, the first QRS
    of the pattern"""
    tnet = pattern.last_tnet
    BASIC_TCONST(pattern, qrs)
    tnet.set_equal(pattern.hypothesis.start, qrs.time)
    tnet.set_before(qrs.time, pattern.hypothesis.end)
    tnet.add_constraint(qrs.start, qrs.end, C.QRS_DUR)


def _reg_ae_tconst(pattern, qrs):
    """
    Temporal constraints for regular beats coming after ectopic beats.
    """
    beats = pattern.evidence[o.QRS]
    idx = beats.index(qrs)
    assert _is_ectopic(idx-1)
    tnet = pattern.last_tnet
    hyp = pattern.hypothesis
    BASIC_TCONST(pattern, qrs)
    tnet.add_constraint(qrs.start, qrs.end, C.NQRS_DUR)
    tnet.set_before(qrs.time, hyp.end)
    #Constraints with the precedent T Wave
    _qrs_after_twave(pattern, qrs)
    #The first regular beat takes the reference RR from the previous rhythm
    #and the subsequent take the reference from the proper trigeminy.
    if idx == 2:
        refrr = pattern.evidence[o.Cardiac_Rhythm][0].meas[0][0]
    else:
        refrr = beats[idx-2].time.end - beats[idx-3].time.start
    const = Iv(min(2*refrr-C.RR_MAX_DIFF, refrr*C.COMPAUSE_MIN_F),
               max(2*refrr+C.RR_MAX_DIFF, refrr*C.COMPAUSE_MAX_F))
    tnet.add_constraint(beats[idx-2].time, qrs.time, const)
    tnet.add_constraint(beats[idx-2].start, qrs.start, const)
    tnet.add_constraint(beats[idx-2].end, qrs.end, const)
    #Compensatory pause RR constraints
    minrr = beats[idx-1].time.start - beats[idx-2].time.start
    maxrr = beats[idx-1].time.end - beats[idx-2].time.start
    mincompause = max(C.COMPAUSE_MIN_DUR, min(minrr*C.COMPAUSE_RREXT_MIN_F,
                                                   minrr+C.COMPAUSE_RREXT_MIN))
    tnet.add_constraint(beats[idx-1].time, qrs.time,
                                 Iv(mincompause, maxrr*C.COMPAUSE_RREXT_MAX_F))
    #The morphology should be similar to the previous non-ectopic QRS
    qrs.shape = beats[idx-2].shape
    qrs.paced = beats[idx-2].paced

def _reg_nae_tconst(pattern, qrs):
    """
    Temporal constraints for regular beats not coming after ectopic beats.
    """
    beats = pattern.evidence[o.QRS]
    idx = beats.index(qrs)
    assert not _is_ectopic(idx)
    hyp = pattern.hypothesis
    tnet = pattern.last_tnet
    prev = beats[idx-1]
    if idx > 3:
        #We create a new temporal network for the new trigeminy cycle.
        tnet.remove_constraint(hyp.end, prev.time)
        tnet = ConstraintNetwork()
        pattern.temporal_constraints.append(tnet)
        rrev = beats[idx-3].time.start - beats[idx-4].time.start
    ##RR evolution constraint.
    else:
        rrev = pattern.evidence[o.Cardiac_Rhythm][0].meas.rr[0]
    tnet.add_constraint(prev.time, qrs.time,
                                Iv(rrev - C.RR_MAX_DIFF, rrev + C.RR_MAX_DIFF))
    BASIC_TCONST(pattern, qrs)
    tnet.add_constraint(qrs.start, qrs.end, C.NQRS_DUR)
    tnet.set_before(qrs.time, hyp.end)
    #Constraints with the precedent T Wave
    _qrs_after_twave(pattern, qrs)
    #Morphology should be similar to the previous QRS, since both are normal
    qrs.shape = prev.shape
    qrs.paced = prev.paced

def _ect_qrs_tconst(pattern, qrs):
    """
    Temporal constraints for ectopic beats, which appear after every pair of
    regular beats.
    """
    beats = pattern.evidence[o.QRS]
    idx = beats.index(qrs)
    tnet = pattern.last_tnet
    hyp = pattern.hypothesis
    BASIC_TCONST(pattern, qrs)
    tnet.add_constraint(qrs.start, qrs.end, C.QRS_DUR)
    tnet.set_before(qrs.time, hyp.end)
    #Constraints with the precedent T Wave
    _qrs_after_twave(pattern, qrs)
    #This check is needed because there is an abduction point invoking this
    #function.
    if idx > 0:
        assert _is_ectopic(idx)
        prev = beats[idx-1]
        #The interval between ectopic beats should also be stable.
        if idx > 6:
            refrr = beats[idx-3].time.end - beats[idx-4].time.start
            tnet.add_constraint(prev.time, qrs.time,
                                  Iv(refrr-C.RR_MAX_DIFF, refrr+C.RR_MAX_DIFF))
        #The reference RR varies from an upper limit to the last measurement,
        #through the contextual previous rhythm.
        refrr = C.BRADY_RR.end
        stdrr = 0.1*refrr
        if pattern.evidence[o.Cardiac_Rhythm] and idx == 1:
            mrr, srr = pattern.evidence[o.Cardiac_Rhythm][0].meas.rr
            if mrr > 0:
                refrr, stdrr = mrr, srr
        elif idx > 1:
            refrr, stdrr = hyp.meas.rr
            #There must be an instantaneous shortening of the RR.
            prevrr = prev.time.end - beats[idx-2].time.start
            tnet.add_constraint(prev.time, qrs.time, Iv(C.TACHY_RR.start,
                                      max(C.TACHY_RR.start, prevrr-C.TMARGIN)))
        #Ectopic beats must be advanced wrt the reference RR.
        tnet.add_constraint(prev.time, qrs.time,
                      Iv(C.TACHY_RR.start, max(C.TACHY_RR.start, refrr-stdrr)))
        tnet.set_before(prev.end, qrs.start)

def _ect0_gconst(pattern, qrs):
    """
    General constraints to be satisfied by the first ectopic QRS complex of
    the trigeminy, which is the abduction point of the pattern. These
    constraints ensure that the ectopic beat is really advanced. This is needed
    because the first ectopic beat is the abduction point, and therefore the
    reference RR cannot be calculated when establishing the temporal
    constraints for the first time.
    """
    if pattern.evidence[o.Cardiac_Rhythm]:
        mrr, stdrr = pattern.evidence[o.Cardiac_Rhythm][0].meas[0]
        if mrr > 0:
            ectrr = qrs.time.start - pattern.evidence[o.QRS][0].time.start
            verify(ectrr < 0.9 * mrr)
            verify(ectrr < mrr-stdrr)

#################################
### P and T waves constraints ###
#################################

def get_p_tconst(qrsidx):
    """
    Obtains the temporal constraints function for the P wave associated to
    the QRS at given position.
    """
    def _p_tconst(pattern, pwave):
        """P waves temporal constraints"""
        BASIC_TCONST(pattern, pwave)
        tnet = pattern.last_tnet
        tnet.add_constraint(pwave.start, pwave.end, C.PW_DURATION)
        #We find the associated QRS.
        beats = pattern.evidence[o.QRS]
        qidx = qrsidx+len(beats) if qrsidx < 0 else qrsidx
        qrs = beats[qidx]
        if qidx > 0:
            tnet.set_before(beats[qidx-1].end, pwave.start)
        tnet.add_constraint(pwave.start, qrs.start, C.N_PR_INTERVAL)
        tnet.set_before(pwave.end, qrs.start)
        if not _is_ectopic(qidx):
            pqmean, pqstd = pattern.hypothesis.meas.pq
        else:
            pqs = _get_measures(pattern, True)[2]
            pqmean, pqstd = np.mean(pqs), np.std(pqs)
        if pqmean > 0:
            #The mean and standard deviation of the PQ measurements will
            #influence the following observations.
            maxdiff = (C.TMARGIN if len(pattern.evidence[o.PWave]) < 10
                                                                  else 2*pqstd)
            interv = Iv(int(pqmean-maxdiff), int(pqmean+maxdiff))
            if interv.overlap(C.N_PR_INTERVAL):
                tnet.add_constraint(pwave.start, qrs.start, interv)
    return _p_tconst

def get_t_tconst(qrsidx):
    """
    Obtains the temporal constraints function for the T wave associated to
    the QRS at given position.
    """
    def _t_tconst(pattern, twave):
        """
        Temporal constraints of the T Waves wrt the corresponding QRS complex.
        """
        BASIC_TCONST(pattern, twave)
        tnet = pattern.last_tnet
        #We find the associated QRS.
        beats = pattern.evidence[o.QRS]
        qidx = qrsidx+len(beats) if qrsidx < 0 else qrsidx
        qrs = beats[qidx]
        if qidx < len(beats) - 1:
            tnet.set_before(twave.end, beats[qidx+1].start)
        #ST interval
        tnet.add_constraint(qrs.end, twave.start, C.ST_INTERVAL)
        #QT duration
        tnet.add_constraint(qrs.start, twave.end, C.N_QT_INTERVAL)
        #RT variation
        if not _is_ectopic(qidx):
            rtmean, rtstd = pattern.hypothesis.meas.rt
        else:
            rts = _get_measures(pattern, True)[2]
            rtmean, rtstd = np.mean(rts), np.std(rts)
        if rtmean > 0:
            #The mean and standard deviation of the PQ measurements will
            #influence the following observations.
            maxdiff = (C.TMARGIN if len(pattern.evidence[o.TWave]) < 10
                                                                 else  2*rtstd)
            interv = Iv(int(rtmean-maxdiff), int(rtmean+maxdiff))
            #We avoid possible inconsistencies with constraint introduced by
            #the rhythm information.
            try:
                existing = tnet.get_constraint(qrs.time, twave.end).constraint
            except KeyError:
                existing = Iv(-np.inf, np.inf)
            if interv.overlap(existing):
                tnet.add_constraint(qrs.time, twave.end, interv)
    return _t_tconst

###########################
### General Constraints ###
###########################

def _cycle_finished_gconst(pattern, _):
    """
    General constraints to be added when a trigeminy cycle is finished, this is,
    with the normal beat following an ectopy.
    """
    #We check that there are no missed beats.
    _check_missed_beats(pattern)
    #We update the measurements of the rhythm taking the measures of the
    #regular cycles.
    rrs, pqs, rts = _get_measures(pattern, False)
    pattern.hypothesis.meas = o.CycleMeasurements((np.mean(rrs), np.std(rrs)),
                                                  (np.mean(rts), np.std(rts)),
                                                  (np.mean(pqs), np.std(pqs)))


TRIGEMINY_PATTERN = PatternAutomata()
TRIGEMINY_PATTERN.name = "Trigeminy"
TRIGEMINY_PATTERN.Hypothesis = o.Trigeminy
TRIGEMINY_PATTERN.add_transition(0, 1, o.Cardiac_Rhythm, ENV,
                                                           _prev_rhythm_tconst)
#N
TRIGEMINY_PATTERN.add_transition(1, 2, o.QRS, ENV, _env_qrs_tconst)
#V
TRIGEMINY_PATTERN.add_transition(2, 3, o.QRS, ABS, _ect_qrs_tconst,
                                                                  _ect0_gconst)
#N
TRIGEMINY_PATTERN.add_transition(3, 4, o.QRS, ABS, _reg_ae_tconst,
                                                        _cycle_finished_gconst)
#N
TRIGEMINY_PATTERN.add_transition(4, 5, o.QRS, ABS, _reg_nae_tconst)
#V
TRIGEMINY_PATTERN.add_transition(5, 6, o.QRS, ABS, _ect_qrs_tconst)
#N
TRIGEMINY_PATTERN.add_transition(6, 7, o.QRS, ABS, _reg_ae_tconst,
                                                        _cycle_finished_gconst)
#P and T waves for the first 5 abstracted QRS
TRIGEMINY_PATTERN.add_transition(7, 8, o.PWave, ABS, get_p_tconst(1))
TRIGEMINY_PATTERN.add_transition(7, 9, o.TWave, ABS, get_t_tconst(1))
TRIGEMINY_PATTERN.add_transition(7, 9)
TRIGEMINY_PATTERN.add_transition(8, 9, o.TWave, ABS, get_t_tconst(1))
#
TRIGEMINY_PATTERN.add_transition(9, 10, o.PWave, ABS, get_p_tconst(2))
TRIGEMINY_PATTERN.add_transition(9, 11, o.TWave, ABS, get_t_tconst(2))
TRIGEMINY_PATTERN.add_transition(9, 11)
TRIGEMINY_PATTERN.add_transition(10, 11, o.TWave, ABS, get_t_tconst(2))
#
TRIGEMINY_PATTERN.add_transition(11, 12, o.PWave, ABS, get_p_tconst(3))
TRIGEMINY_PATTERN.add_transition(11, 13, o.TWave, ABS, get_t_tconst(3))
TRIGEMINY_PATTERN.add_transition(11, 13)
TRIGEMINY_PATTERN.add_transition(12, 13, o.TWave, ABS, get_t_tconst(-3))
#
TRIGEMINY_PATTERN.add_transition(13, 14, o.PWave, ABS, get_p_tconst(-2))
TRIGEMINY_PATTERN.add_transition(13, 15, o.TWave, ABS, get_t_tconst(-2))
TRIGEMINY_PATTERN.add_transition(13, 15)
TRIGEMINY_PATTERN.add_transition(14, 15, o.TWave, ABS, get_t_tconst(-2))
#
TRIGEMINY_PATTERN.add_transition(15, 16, o.PWave, ABS, get_p_tconst(-1))
TRIGEMINY_PATTERN.add_transition(15, 17, o.TWave, ABS, get_t_tconst(-1))
TRIGEMINY_PATTERN.add_transition(15, 17)
TRIGEMINY_PATTERN.add_transition(16, 17, o.TWave, ABS, get_t_tconst(-1))
#Each new cycle adds three more QRS complexes.
#N
TRIGEMINY_PATTERN.add_transition(17, 18, o.QRS, ABS, _reg_nae_tconst)
#V
TRIGEMINY_PATTERN.add_transition(18, 19, o.QRS, ABS, _ect_qrs_tconst)
#N
TRIGEMINY_PATTERN.add_transition(19, 20, o.QRS, ABS, _reg_ae_tconst,
                                                        _cycle_finished_gconst)
#And the corresponding P and T waves.
TRIGEMINY_PATTERN.add_transition(20, 12, o.PWave, ABS, get_p_tconst(-3))
TRIGEMINY_PATTERN.add_transition(20, 13, o.TWave, ABS, get_t_tconst(-3))
TRIGEMINY_PATTERN.add_transition(20, 13)
#
TRIGEMINY_PATTERN.final_states.add(17)
TRIGEMINY_PATTERN.abstractions[o.QRS] = (TRIGEMINY_PATTERN.transitions[2],)
TRIGEMINY_PATTERN.obs_proc = _rhythm_obs_proc
TRIGEMINY_PATTERN.freeze()