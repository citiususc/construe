# -*- coding: utf-8 -*-
# pylint: disable-msg=
"""
Created on Mon Apr 21 08:26:45 2014

This module contains the definition of beat-related abstraction patterns.

@author: T. Teijeiro
"""

from construe.model.automata import (PatternAutomata, ABSTRACTED, ENVIRONMENT,
                                                                  BASIC_TCONST)
from construe.model import Interval as Iv
from construe.utils.units_helper import msec2samples, samples2msec
import construe.knowledge.constants as C
import construe.knowledge.observables as o
import numpy as np

#Constant values to initialize the Kalman Filter for QT (actually RT) measure
QT_ERR_STD = msec2samples(58) #Standard deviation of the QT error (R matrix)
MIN_QT_STD = msec2samples(20) #Minimum standard deviation of the QT error
KF_Q = msec2samples(40) #Dynamic noise of the Kalman filter (Q matrix)
#Upper and lower limit of the RR intervals
RR_LIMITS = Iv(msec2samples(300), msec2samples(1200))

def sp2sg(samples):
    """Converts a measure in samples to secods"""
    return samples2msec(samples)/1000.0

def _envbeat_tconst(pattern, obs):
    """
    Temporal constraints for the environment QRS and CardiacCycle observation
    """
    BASIC_TCONST(pattern, obs)
    pattern.last_tnet.add_constraint(obs.end, pattern.hypothesis.time,
                                     Iv(msec2samples(20), np.inf))
    if isinstance(obs, o.QRS) and isinstance(pattern.obs_seq[0], o.CardiacCycle):
        pattern.last_tnet.set_equal(pattern.obs_seq[0].time, obs.time)

def _btime_tconst(pattern, qrs):
    """
    Temporal constraints for the abstracted QRS observation
    """
    BASIC_TCONST(pattern, qrs)
    pattern.last_tnet.set_equal(qrs.time, pattern.hypothesis.time)

def _qrs_time_tconst(pattern, _):
    """
    Temporal constraints for beat ending without T wave observation
    """
    qrs = pattern.obs_seq[-2]
    pattern.last_tnet.set_equal(qrs.start, pattern.hypothesis.start)
    pattern.last_tnet.set_equal(qrs.end, pattern.hypothesis.end)

def _p_qrs_tconst(pattern, pwave):
    """
    Temporal constraints of the P Wave wrt the corresponding QRS complex
    """
    BASIC_TCONST(pattern, pwave)
    obseq = pattern.obs_seq
    idx = pattern.get_step(pwave)
    tnet = pattern.last_tnet
    #Beat start
    tnet.set_equal(pwave.start, pattern.hypothesis.start)
    tnet.add_constraint(pwave.start, pwave.end, C.PW_DURATION)
    if idx == 0 or not isinstance(obseq[idx-1], o.QRS):
        return
    qrs = obseq[idx-1]
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
    tnet = pattern.last_tnet
    #Beat end
    tnet.set_equal(twave.end, pattern.hypothesis.end)
    #We find the qrs observation precedent to this T wave.
    try:
        qrs = next(obseq[i] for i in xrange(idx-1, -1, -1)
                                                if isinstance(obseq[i], o.QRS))
        #If there is no P Wave, the beat start is the QRS start.
        if pattern.trseq[idx][0].istate in (1, 3):
            tnet.set_equal(qrs.start, pattern.hypothesis.start)
        if idx > 0 and isinstance(obseq[idx-1], o.PWave):
            pwave = obseq[idx-1]
            tnet.add_constraint(pwave.end, twave.start, Iv(C.ST_INTERVAL.start,
                                            C.PQ_INTERVAL.end + C.QRS_DUR.end))
        #ST interval
        tnet.add_constraint(qrs.end, twave.start, C.ST_INTERVAL)
        #QT duration
        tnet.add_constraint(qrs.start, twave.end, C.N_QT_INTERVAL)
        #If we observed a previous QRS complex, we set a limit for the QT
        #interval according to the RR and to the previous measures.
        if isinstance(obseq[1], o.QRS):
            rr = qrs.time.start-obseq[1].time.start
            rr = max(min(rr, RR_LIMITS.end), RR_LIMITS.start)
            rtc, rtstd = obseq[0].meas.rt
            if rtc > 0:
                #Expected QT value from the QT corrected value
                rtmean = msec2samples(1000.0*sp2sg(rtc)*np.cbrt(sp2sg(rr)))
                tnet.add_constraint(qrs.time, twave.end, Iv(rtmean-2.5*rtstd,
                                                            rtmean+2.5*rtstd))
            #The PR interval is also included to limit the T wave duration
            pr = obseq[0].meas.pq[0]
            try:
                tnet.add_constraint(qrs.time, twave.end, Iv(0, rr-pr))
            except ValueError:
                pass
    except StopIteration:
        pass

def _firstbeat_gconst(pattern, twave):
    """General constraints for the first beat, to obtain the first measure"""
    if twave is None:
        rt = 0.0
    else:
        rt = twave.earlyend - pattern.obs_seq[0].time.start
    pattern.hypothesis.meas = o.CycleMeasurements((0.0, 0.0), (rt, QT_ERR_STD),
                                                  (0.0, 0.0))

def _cycle_gconst(pattern, twave):
    """
    General constraints applied after all the evidence of a heartbeat has
    been observed. The Kalman filter update for the QT limits is performed in
    this function.
    """
    #Belief values
    rtmean, rtstd = pattern.obs_seq[0].meas.rt
    #Current RR measure (bounded)
    qrs = pattern.obs_seq[2]
    rr = qrs.time.start - pattern.obs_seq[1].time.start
    rr = max(min(rr, RR_LIMITS.end), RR_LIMITS.start)
    #Kalman filter algorithm, as explained in "Probabilistic Robotics"
    sigma_tbar = rtstd**2 + KF_Q**2
    if twave is not None:
        #rt and corrected rt measure in the current iteration
        rt = twave.earlyend - qrs.time.start
        rtc = msec2samples(1000.0*sp2sg(rt)/np.cbrt(sp2sg(rr)))
        meas_err = rtc-rtmean
        #Abnormally QT intervals have associated higher uncertainty
        qt = twave.earlyend - qrs.earlystart
        qt_lims = C.QT_FROM_RR(Iv(rr, rr))
        #Measure uncertainty, represented by the R matrix in the Kalman filter
        KF_R = meas_err if qt in qt_lims else msec2samples(120)
        k_t = sigma_tbar/(sigma_tbar + max(KF_R, MIN_QT_STD)**2)
    else:
        #No measure - 0 Kalman gain
        meas_err = QT_ERR_STD
        k_t = 0
    mu_t = rtmean + k_t*meas_err
    sigma_t = (1.0-k_t)*sigma_tbar
    #PR interval
    pr = 0.0
    if isinstance(pattern.obs_seq[3], o.PWave):
        pr = qrs.time.start - pattern.obs_seq[3].earlystart
        prmean = pattern.obs_seq[0].meas.pq[0]
        pr = (pr+prmean)/2 if prmean > 0 else pr
    pattern.hypothesis.meas = o.CycleMeasurements((rr, 0.0),
                                          (mu_t, np.sqrt(sigma_t)), (pr, 0.0))


FIRST_BEAT_PATTERN = PatternAutomata()
FIRST_BEAT_PATTERN.name = "First Beat"
FIRST_BEAT_PATTERN.Hypothesis = o.FirstBeat
FIRST_BEAT_PATTERN.add_transition(0, 1, o.QRS, ABSTRACTED, _btime_tconst)
FIRST_BEAT_PATTERN.add_transition(1, 2, o.PWave, ABSTRACTED, _p_qrs_tconst)
FIRST_BEAT_PATTERN.add_transition(2, 3, o.TWave, ABSTRACTED, _t_qrs_tconst,
                                  _firstbeat_gconst)
FIRST_BEAT_PATTERN.add_transition(1, 3, o.TWave, ABSTRACTED, _t_qrs_tconst,
                                  _firstbeat_gconst)
FIRST_BEAT_PATTERN.add_transition(1, 3, tconst=_qrs_time_tconst,
                                  gconst=_firstbeat_gconst)
FIRST_BEAT_PATTERN.final_states.add(3)
FIRST_BEAT_PATTERN.abstractions[o.QRS] = (FIRST_BEAT_PATTERN.transitions[0],)
FIRST_BEAT_PATTERN.freeze()

CARDIAC_CYCLE_PATTERN = PatternAutomata()
CARDIAC_CYCLE_PATTERN.name = "Cardiac Cycle"
CARDIAC_CYCLE_PATTERN.Hypothesis = o.Normal_Cycle
CARDIAC_CYCLE_PATTERN.add_transition(0, 1, o.CardiacCycle, ENVIRONMENT,
                                     _envbeat_tconst)
CARDIAC_CYCLE_PATTERN.add_transition(1, 2, o.QRS, ENVIRONMENT, _envbeat_tconst)
CARDIAC_CYCLE_PATTERN.add_transition(2, 3, o.QRS, ABSTRACTED, _btime_tconst)
CARDIAC_CYCLE_PATTERN.add_transition(3, 4, o.PWave, ABSTRACTED, _p_qrs_tconst)
CARDIAC_CYCLE_PATTERN.add_transition(4, 5, o.TWave, ABSTRACTED, _t_qrs_tconst,
                                     _cycle_gconst)
CARDIAC_CYCLE_PATTERN.add_transition(3, 5, o.TWave, ABSTRACTED, _t_qrs_tconst,
                                     _cycle_gconst)
CARDIAC_CYCLE_PATTERN.add_transition(3, 5, tconst=_qrs_time_tconst,
                                     gconst=_cycle_gconst)
CARDIAC_CYCLE_PATTERN.final_states.add(5)
CARDIAC_CYCLE_PATTERN.abstractions[o.QRS] = (CARDIAC_CYCLE_PATTERN.transitions[2],)
CARDIAC_CYCLE_PATTERN.freeze()
