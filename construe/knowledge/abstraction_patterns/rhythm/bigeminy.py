# -*- coding: utf-8 -*-
# pylint: disable-msg=
"""
Created on Tue Jul  8 13:09:54 2014

This module contains the definition of the abstraction pattern representing
a bigeminy rhythm.

@author: T. Teijeiro
"""

from construe.model.automata import (PatternAutomata, ABSTRACTED as ABS,
                                        ENVIRONMENT as ENV, BASIC_TCONST)
from construe.model import ConstraintNetwork, verify, Interval as Iv
import construe.knowledge.observables as o
import construe.knowledge.constants as C
from construe.knowledge.abstraction_patterns.rhythm.regular import (
                                                           _check_missed_beats)
import numpy as np

def _rhythm_obs_proc(pattern):
    """Observation procedure executed once the rhythm pattern has finished"""
    #We asign the endpoint of the hypothesis.
    pattern.hypothesis.end.value = pattern.evidence[o.QRS][-1].time.value

###########################################
### Previous cardiac rhythm constraints ###
###########################################

def _prev_rhythm_gconst(_, rhythm):
    """General constraints of a cardiac rhythm with the preceden one."""
    #A bigeminy cannot be preceded by another bigeminy or an extrasystole.
    verify(not isinstance(rhythm, (o.Bigeminy, o.Extrasystole)))

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

def _reg_qrs_tconst(pattern, qrs):
    """
    Temporal constraints for regular beats, which appear after every ectopic
    beat.
    """
    beats = pattern.evidence[o.QRS]
    idx = beats.index(qrs)
    tnet = pattern.last_tnet
    hyp = pattern.hypothesis
    BASIC_TCONST(pattern, qrs)
    tnet.add_constraint(qrs.start, qrs.end, C.NQRS_DUR)
    tnet.set_before(qrs.time, hyp.end)
    #Constraints with the precedent T Wave
    _qrs_after_twave(pattern, qrs)
    #The environment QRS complex determines the beginning of the bigeminy.
    if pattern.get_evidence_type(qrs)[1] is ENV:
        tnet.set_equal(hyp.start, qrs.time)
    else:
        #The first regular beat takes the reference RR from the previous rhythm
        #and the subsequent take the reference from the proper bigeminy.
        if idx == 2:
            refrr, stdrr = pattern.evidence[o.Cardiac_Rhythm][0].meas[0]
            max_var = max(2*C.RR_MAX_DIFF, 4*stdrr)
            tnet.add_constraint(beats[0].time, qrs.time, Iv(
                               min(2*refrr-max_var, refrr*C.COMPAUSE_MIN_F),
                               max(2*refrr+max_var, refrr*C.COMPAUSE_MAX_F)))
        else:
            ref2rr = beats[idx-2].time.end - beats[idx-4].time.start
            mrr, srr = hyp.meas.rr
            const = Iv(min(ref2rr-2*C.RR_MAX_DIFF, 2*mrr-4*srr),
                       max(ref2rr+2*C.RR_MAX_DIFF, 2*mrr+4*srr))
            tnet.add_constraint(beats[idx-2].time, qrs.time, const)
            tnet.add_constraint(beats[idx-2].start, qrs.start, const)
            tnet.add_constraint(beats[idx-2].end, qrs.end, const)
        #We guide the morphology search to be similar to the previous regular
        #QRS complex.
        qrs.shape = beats[idx-2].shape
        qrs.paced = beats[idx-2].paced
        #Compensatory pause RR
        minrr = beats[idx-1].time.start - beats[idx-2].time.end
        maxrr = beats[idx-1].time.end - beats[idx-2].time.start
        refcompause = (beats[idx-2].time.start - beats[idx-3].time.start
                                  if idx > 2 else maxrr*C.COMPAUSE_RREXT_MAX_F)
        mincompause = max(C.COMPAUSE_MIN_DUR, maxrr,
                                              min(minrr*C.COMPAUSE_RREXT_MIN_F,
                                                  refcompause-C.TMARGIN,
                                                  minrr+C.COMPAUSE_RREXT_MIN))
        tnet.add_constraint(beats[idx-1].time, qrs.time,
                                 Iv(mincompause, maxrr*C.COMPAUSE_RREXT_MAX_F))
        #Beats cannot overlap
        tnet.add_constraint(beats[idx-1].end, qrs.start,
                                                 Iv(C.TQ_INTERVAL_MIN, np.Inf))

def _ect_qrs_tconst(pattern, qrs):
    """
    Temporal constraints for ectopic beats, which appear after every regular
    beat.
    """
    beats = pattern.evidence[o.QRS]
    idx = beats.index(qrs)
    tnet = pattern.last_tnet
    hyp = pattern.hypothesis
    if idx > 0:
        prev = beats[idx - 1]
        #After the second couplet, every ectopic beat introduces a new temporal
        #network in the pattern to make it easier the minimization.
        if idx > 3:
            tnet.remove_constraint(hyp.end, prev.time)
            #We create a new temporal network for the cyclic observations
            tnet = ConstraintNetwork()
            pattern.temporal_constraints.append(tnet)
            #The duration of each couplet should not have high instantaneous
            #variations.
            refrr = beats[idx-2].time.end - beats[idx-3].time.start
            tnet.add_constraint(prev.time, qrs.time,
                                  Iv(refrr-C.RR_MAX_DIFF, refrr+C.RR_MAX_DIFF))
            #We guide the morphology search to be similar to the previous
            #ectopic QRS complex.
            qrs.shape = beats[idx-2].shape
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
        #Ectopic beats must be advanced wrt the reference RR
        tnet.add_constraint(prev.time, qrs.time,
                      Iv(C.TACHY_RR.start, max(C.TACHY_RR.start, refrr-stdrr)))
        #Beats cannot overlap
        tnet.add_constraint(prev.end, qrs.start, Iv(C.TQ_INTERVAL_MIN, np.Inf))
    BASIC_TCONST(pattern, qrs)
    tnet.add_constraint(qrs.start, qrs.end, C.QRS_DUR)
    tnet.set_before(qrs.time, hyp.end)
    #Constraints with the precedent T Wave
    _qrs_after_twave(pattern, qrs)

def _ect0_gconst(pattern, qrs):
    """
    General constraints to be satisfied by the first ectopic QRS complex of
    the bigeminy, which is the abduction point of the pattern. These
    constraints ensure that the ectopic beat is really advanced. This is needed
    because the first ectopic beat is the abduction point, and therefore the
    reference RR cannot be calculated when establishing the temporal
    constraints for the first time.
    """
    if pattern.evidence[o.Cardiac_Rhythm]:
        mrr, stdrr = pattern.evidence[o.Cardiac_Rhythm][0].meas[0]
        if mrr > 0:
            ectrr = qrs.time.start - pattern.evidence[o.QRS][0].time.start
            mshort = min(0.1*mrr, stdrr)
            verify(ectrr < mrr-mshort)

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
        if len(pattern.evidence[o.PWave]) > 10:
            #The mean and standard deviation of the PQ measurements will
            #influence the following observations.
            if qidx % 2 == 0:
                pqmean, pqstd = pattern.hypothesis.meas.pq
            else:
                pqs = _get_measures(pattern, True)[2]
                pqmean, pqstd = np.mean(pqs), np.std(pqs)
            if not np.isnan(pqmean) and not np.isnan(pqstd):
                interv = Iv(int(pqmean-2*pqstd), int(pqmean+2*pqstd))
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
        obseq = pattern.obs_seq
        idx = pattern.get_step(twave)
        beats = pattern.evidence[o.QRS]
        qidx = qrsidx+len(beats) if qrsidx < 0 else qrsidx
        qrs = beats[qidx]
        if qidx > 1:
            refsq = beats[qidx-1].earlystart - beats[qidx-2].lateend
            tnet.add_constraint(qrs.time, twave.end,
                                      Iv(0, max(0, refsq - C.TQ_INTERVAL_MIN)))
        if idx > 0 and isinstance(obseq[idx-1], o.PWave):
            pwave = obseq[idx-1]
            tnet.add_constraint(pwave.end, twave.start, Iv(C.ST_INTERVAL.start,
                                            C.PQ_INTERVAL.end + C.QRS_DUR.end))
        if qidx < len(beats) - 1:
            tnet.set_before(twave.end, beats[qidx+1].start)
        #ST interval
        tnet.add_constraint(qrs.end, twave.start, C.ST_INTERVAL)
        #QT duration
        tnet.add_constraint(qrs.start, twave.end, C.N_QT_INTERVAL)
        #RT variation
        if qidx % 2 == 0:
            rtmean, rtstd = pattern.hypothesis.meas.rt
            #We also define a constraint on T wave end based on the last
            #distance between normal and ectopic QRS.
            if qidx > 0:
                tnet.add_constraint(qrs.end, twave.end,
                       Iv(0, beats[qidx-1].earlystart - beats[qidx-2].lateend))
        else:
            rts = _get_measures(pattern, 1)[2]
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

def _get_measures(pattern, even = 0):
    """
    Obtains the characteristic measures of the cardiac rhythms (RR, PQ and RT
    intervals), allowing to filter by the beat type. If even==0, then the
    regular beats are taken, and if even==1, the ectopic beats are used for the
    measurement. The output tuple (rrs, pqs, rts) contains the series of these
    parameters.
    """
    n = 20  #Number of observations to get the statistical measures.
    beats = [q for q in pattern.evidence[o.QRS] if q is not pattern.finding]
    #RR
    rrs = np.diff([beats[i].time.start for i in xrange(len(beats))
                                   if i % 2 == even and len(beats)-i <= n])/2.0
    #RT
    rts = []
    for twave in pattern.evidence[o.TWave][-n:]:
        if twave is not pattern.finding:
            qrs = next(q for q in reversed(beats)
                                               if q.lateend < twave.earlystart)
            if beats.index(qrs) % 2 == even:
                rts.append(twave.lateend - qrs.time.start)
    #PQ
    pqs = []
    for pwave in pattern.evidence[o.PWave][-n:]:
        if pwave is not pattern.finding:
            qrs = next(q for q in beats if q.earlystart > pwave.lateend)
            if beats.index(qrs) % 2 == even:
                pqs.append(qrs.earlystart - pwave.earlystart)
    return (rrs, pqs, rts)

def _cycle_finished_gconst(pattern, _):
    """
    General constraints to be added when a bigeminy cycle is observed, this is,
    with every normal beat.
    """
    _check_missed_beats(pattern)
    #We update the measurements of the rhythm taking the measures of the
    #regular cycles.
    rrs, pqs, rts = _get_measures(pattern, 0)
    pattern.hypothesis.meas = o.CycleMeasurements((np.mean(rrs), np.std(rrs)),
                                                  (np.mean(rts), np.std(rts)),
                                                  (np.mean(pqs), np.std(pqs)))


BIGEMINY_PATTERN = PatternAutomata()
BIGEMINY_PATTERN.name = "Bigeminy"
BIGEMINY_PATTERN.Hypothesis = o.Bigeminy
BIGEMINY_PATTERN.add_transition(0, 1, o.Cardiac_Rhythm, ENV,
                                      _prev_rhythm_tconst, _prev_rhythm_gconst)
#Necessary evidence are QRS complexes.
#N
BIGEMINY_PATTERN.add_transition(1, 2, o.QRS, ENV, _reg_qrs_tconst)
#V
BIGEMINY_PATTERN.add_transition(2, 3, o.QRS, ABS, _ect_qrs_tconst,
                                                                  _ect0_gconst)
#N
BIGEMINY_PATTERN.add_transition(3, 4, o.QRS, ABS, _reg_qrs_tconst,
                                                        _cycle_finished_gconst)
#V
BIGEMINY_PATTERN.add_transition(4, 5, o.QRS, ABS, _ect_qrs_tconst)
#N
BIGEMINY_PATTERN.add_transition(5, 6, o.QRS, ABS, _reg_qrs_tconst,
                                                        _cycle_finished_gconst)
#Optional P and T waves for the necessary evidence
BIGEMINY_PATTERN.add_transition(6, 7, o.PWave, ABS, get_p_tconst(1))
BIGEMINY_PATTERN.add_transition(6, 8, o.TWave, ABS, get_t_tconst(1))
BIGEMINY_PATTERN.add_transition(6, 8)
BIGEMINY_PATTERN.add_transition(7, 8, o.TWave, ABS, get_t_tconst(1))
#
BIGEMINY_PATTERN.add_transition(8, 9, o.PWave, ABS, get_p_tconst(2))
BIGEMINY_PATTERN.add_transition(8, 10, o.TWave, ABS, get_t_tconst(2))
BIGEMINY_PATTERN.add_transition(8, 10)
BIGEMINY_PATTERN.add_transition(9, 10, o.TWave, ABS, get_t_tconst(2))
#
BIGEMINY_PATTERN.add_transition(10, 11, o.PWave, ABS, get_p_tconst(3))
BIGEMINY_PATTERN.add_transition(10, 12, o.TWave, ABS, get_t_tconst(3))
BIGEMINY_PATTERN.add_transition(10, 12)
BIGEMINY_PATTERN.add_transition(11, 12, o.TWave, ABS, get_t_tconst(-2))
#
BIGEMINY_PATTERN.add_transition(12, 13, o.PWave, ABS, get_p_tconst(-1))
BIGEMINY_PATTERN.add_transition(12, 14, o.TWave, ABS, get_t_tconst(-1))
BIGEMINY_PATTERN.add_transition(12, 14)
BIGEMINY_PATTERN.add_transition(13, 14, o.TWave, ABS, get_t_tconst(-1))
#Each new cycle adds a new extrasystole and a new regular QRS
#V
BIGEMINY_PATTERN.add_transition(14, 15, o.QRS, ABS, _ect_qrs_tconst)
#N
BIGEMINY_PATTERN.add_transition(15, 16, o.QRS, ABS, _reg_qrs_tconst,
                                                        _cycle_finished_gconst)
#And the corresponding optional P and T waves.
BIGEMINY_PATTERN.add_transition(16, 11, o.PWave, ABS, get_p_tconst(-2))
BIGEMINY_PATTERN.add_transition(16, 12, o.TWave, ABS, get_t_tconst(-2))
BIGEMINY_PATTERN.add_transition(16, 12)
#Pattern finishing
BIGEMINY_PATTERN.final_states.add(14)
BIGEMINY_PATTERN.abstractions[o.QRS] = (BIGEMINY_PATTERN.transitions[2], )
BIGEMINY_PATTERN.obs_proc = _rhythm_obs_proc
BIGEMINY_PATTERN.freeze()