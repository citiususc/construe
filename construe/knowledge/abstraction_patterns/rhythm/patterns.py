# -*- coding: utf-8 -*-
# pylint: disable-msg= E1002, E1101
"""
Created on Wed Nov 21 09:04:17 2012

This file contains the definition of a set of very simple abstraction patterns
in order to perform rhythm interpretation on an ECG signal.

@author: T. Teijeiro
"""

import construe.knowledge.observables as o
from construe.knowledge.constants import (PW_DURATION, ST_INTERVAL,
                                             N_PR_INTERVAL, N_QT_INTERVAL,
                                             ASYSTOLE_RR, PQ_INTERVAL, QRS_DUR)
from construe.model import Interval as Iv
from construe.model.automata import (PatternAutomata, ABSTRACTED,
                                        ENVIRONMENT, BASIC_TCONST)
from construe.utils.units_helper import msec2samples as ms2sp
import copy

def _rstart_tconst(pattern, qrs):
    """
    Temporal constraints for the Rhythm Start abstraction pattern.
    """
    BASIC_TCONST(pattern, qrs)
    pattern.tnet.set_equal(qrs.time, pattern.hypothesis.time)


def _p_qrs_tconst(pattern, pwave):
    """
    Temporal constraints of the P Wave wrt the corresponding QRS complex
    """
    BASIC_TCONST(pattern, pwave)
    obseq = pattern.obs_seq
    idx = pattern.get_step(pwave)
    if idx == 0 or not isinstance(obseq[idx-1], o.QRS):
        return
    qrs = obseq[idx-1]
    pattern.tnet.add_constraint(pwave.start, pwave.end, PW_DURATION)
    #PR interval
    pattern.tnet.add_constraint(pwave.start, qrs.start, N_PR_INTERVAL)
    pattern.tnet.set_before(pwave.end, qrs.start)

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
        tnet = pattern.tnet
        if idx > 0 and isinstance(obseq[idx-1], o.PWave):
            pwave = obseq[idx-1]
            tnet.add_constraint(pwave.end, twave.start, Iv(ST_INTERVAL.start,
                                            PQ_INTERVAL.end + QRS_DUR.end))
        #ST interval
        tnet.add_constraint(qrs.end, twave.start, ST_INTERVAL)
        #QT duration
        tnet.add_constraint(qrs.start, twave.end, N_QT_INTERVAL)
    except StopIteration:
        pass

def _prev_rhythm_tconst(pattern, rhythm):
    """Temporal constraints of a cardiac rhythm with the precedent one."""
    BASIC_TCONST(pattern, rhythm)
    pattern.tnet.set_equal(pattern.hypothesis.start, rhythm.end)

def _asyst_prev_rhythm_tconst(pattern, rhythm):
    """Temporal constraints of an asystole with the precedent rhythm."""
    BASIC_TCONST(pattern, rhythm)
    pattern.tnet.set_equal(pattern.hypothesis.start, rhythm.end)
    pattern.tnet.add_constraint(pattern.hypothesis.start,
                                           pattern.hypothesis.end, ASYSTOLE_RR)

def _qrs1_tconst(pattern, qrs):
    """Temporal constraints of the first QRS in the asystole."""
    BASIC_TCONST(pattern, qrs)
    pattern.tnet.set_equal(pattern.hypothesis.start, qrs.time)
    pattern.tnet.set_before(qrs.end, pattern.hypothesis.end)

def _qrs2_tconst(pattern, qrs):
    """Temporal constraints of the delayed QRS in the asystole."""
    BASIC_TCONST(pattern, qrs)
    pattern.tnet.set_equal(qrs.time, pattern.hypothesis.end)
    if len(pattern.evidence[o.QRS]) > 1:
        prev = pattern.evidence[o.QRS][0]
        pattern.tnet.add_constraint(prev.time, qrs.time, ASYSTOLE_RR)

def _rhythmstart_gconst(pattern, _):
    """General constraints of the rhythm start pattern."""
    #We assume an starting mean rhythm of 75ppm, but the range allows from 65
    #to 85bpm
    pattern.hypothesis.meas = o.CycleMeasurements((ms2sp(800), ms2sp(200)),
                                                                  (0,0), (0,0))

def _asystole_gconst(pattern, _):
    """General constraints of the asystole pattern."""
    #The rhythm information is copied from the precedent rhythm.
    if pattern.evidence[o.Cardiac_Rhythm]:
        rhythm = pattern.evidence[o.Cardiac_Rhythm][0]
        pattern.hypothesis.meas = copy.copy(rhythm.meas)


RHYTHMSTART_PATTERN = PatternAutomata()
RHYTHMSTART_PATTERN.name = "Rhythm Start"
RHYTHMSTART_PATTERN.Hypothesis = o.RhythmStart
RHYTHMSTART_PATTERN.add_transition(0, 1, o.QRS, ABSTRACTED, _rstart_tconst,
                                                           _rhythmstart_gconst)
RHYTHMSTART_PATTERN.add_transition(1, 2, o.PWave, ABSTRACTED, _p_qrs_tconst)
RHYTHMSTART_PATTERN.add_transition(2, 3, o.TWave, ABSTRACTED, _t_qrs_tconst)
RHYTHMSTART_PATTERN.add_transition(1, 3, o.TWave, ABSTRACTED, _t_qrs_tconst)
RHYTHMSTART_PATTERN.add_transition(1, 3)
RHYTHMSTART_PATTERN.final_states.add(3)
RHYTHMSTART_PATTERN.abstractions[o.QRS] = (RHYTHMSTART_PATTERN.transitions[0],)
RHYTHMSTART_PATTERN.freeze()



ASYSTOLE_PATTERN = PatternAutomata()
ASYSTOLE_PATTERN.name = "Asystole"
ASYSTOLE_PATTERN.Hypothesis = o.Asystole
ASYSTOLE_PATTERN.add_transition(0, 1, o.Cardiac_Rhythm, ENVIRONMENT,
                                                     _asyst_prev_rhythm_tconst)
ASYSTOLE_PATTERN.add_transition(1, 2, o.QRS, ENVIRONMENT, _qrs1_tconst)
ASYSTOLE_PATTERN.add_transition(2, 3, o.QRS, ABSTRACTED, _qrs2_tconst,
                                                              _asystole_gconst)
ASYSTOLE_PATTERN.add_transition(3, 4, o.PWave, ABSTRACTED, _p_qrs_tconst)
ASYSTOLE_PATTERN.add_transition(4, 5, o.TWave, ABSTRACTED, _t_qrs_tconst)
ASYSTOLE_PATTERN.add_transition(3, 5, o.TWave, ABSTRACTED, _t_qrs_tconst)
ASYSTOLE_PATTERN.add_transition(3, 5)
ASYSTOLE_PATTERN.final_states.add(5)
ASYSTOLE_PATTERN.abstractions[o.QRS] = (ASYSTOLE_PATTERN.transitions[2],)
ASYSTOLE_PATTERN.freeze()


if __name__ == "__main__":
    pass

