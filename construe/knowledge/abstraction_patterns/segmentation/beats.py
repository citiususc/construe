# -*- coding: utf-8 -*-
# pylint: disable-msg=
"""
Created on Mon Apr 21 08:26:45 2014

This module contains the definition of beat-related abstraction patterns.

@author: T. Teijeiro
"""

from construe.model.automata import (PatternAutomata, ABSTRACTED,
                                                                  BASIC_TCONST)
from construe.model import Interval as Iv
import construe.knowledge.constants as C
import construe.knowledge.observables as o

def _btime_tconst(pattern, qrs):
    """
    Temporal constraints for the Rhythm Start abstraction pattern.
    """
    BASIC_TCONST(pattern, qrs)
    pattern.last_tnet.set_equal(qrs.time, pattern.hypothesis.time)

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
        if pattern.trseq[idx][0].istate == 1:
            tnet.set_equal(qrs.start, pattern.hypothesis.start)
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


SINUS_BEAT_PATTERN = PatternAutomata()
SINUS_BEAT_PATTERN.name = "Sinus Beat"
SINUS_BEAT_PATTERN.Hypothesis = o.Normal_Cycle
SINUS_BEAT_PATTERN.add_transition(0, 1, o.QRS, ABSTRACTED, _btime_tconst)
SINUS_BEAT_PATTERN.add_transition(1, 2, o.PWave, ABSTRACTED, _p_qrs_tconst)
SINUS_BEAT_PATTERN.add_transition(2, 3, o.TWave, ABSTRACTED, _t_qrs_tconst)
SINUS_BEAT_PATTERN.add_transition(1, 3, o.TWave, ABSTRACTED, _t_qrs_tconst)
SINUS_BEAT_PATTERN.final_states.add(1)
SINUS_BEAT_PATTERN.final_states.add(3)
SINUS_BEAT_PATTERN.abstractions[o.QRS] = (SINUS_BEAT_PATTERN.transitions[0],)
SINUS_BEAT_PATTERN.freeze()