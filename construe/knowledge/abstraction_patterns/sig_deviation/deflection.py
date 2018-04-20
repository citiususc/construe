# -*- coding: utf-8 -*-
# pylint: disable-msg=E1103
"""
Created on Mon Jul 15 12:41:57 2013

This module describes an abstraction pattern for the deductive discover of
energy intervals over a signal fragment.

@author: T. Teijeiro
"""

import construe.knowledge.observables as o
import construe.acquisition.signal_buffer as sig_buf
import construe.acquisition.obs_buffer as obs_buf
from construe.knowledge.base_evidence.energy import (get_energy_intervals,
                                                                       TWINDOW)
from construe.knowledge.constants import DEF_DUR, TMARGIN
from construe.model.constraint_network import verify
from construe.model.automata import PatternAutomata
from construe.model import Interval as Iv
import numpy as np
from sortedcontainers import SortedList


def generate_Deflection_Patterns(npats):
    """
    This function creates a set of *PatternAutomata* patterns, each one
    responsible for obtaining the i-th relevant deflection in a given
    scope. It allows to overcome the limitation of having a single hypothesis
    for a pattern with the same base evidence (in the case of this pattern,
    this base evidence is None, since the only transition of the pattern
    automata does not include any observable, only general constraints).

    Parameters
    ----------
    npats:
        Integer specifying the number of the generated patterns.

    Returns:
    out:
        Ordered list with *n* abstraction patterns, each one responsible for
        the deduction of the *i-th* interesting energy interval in a specific
        area delimited by the hypothesis.
    """
    pats = []
    for i in xrange(npats):
        pat = PatternAutomata()
        pat.name = "Deflection"
        pat.Hypothesis = o.Deflection
        pat.add_transition(0, 1, tconst= _def_tconst, gconst= get_gconst(i))
        pat.final_states.add(1)
        pat.freeze()
        pats.append(pat)
    return pats

def _def_tconst(pattern, _):
    """Temporal constraints for the energy interval abstraction pattern"""
    deflection = pattern.hypothesis
    pattern.tnet.add_constraint(deflection.start, deflection.end, DEF_DUR)

def get_gconst(int_idx):
    """
    Obtains the general constraints function for a specific level.
    """
    def _def_gconst(pattern, _):
        """General constraints for the energy interval abstraction pattern"""
        verify(pattern.hypothesis.lateend < np.inf)
        #The margin to group consecutive fragments is 1 mm
        #Limits for the detection.
        beg = int(pattern.hypothesis.earlystart)
        end = int(pattern.hypothesis.lateend)
        #Now we get the energy accumulated in all leads.
        energy = None
        for lead in sig_buf.get_available_leads():
            lenerg, fbeg, fend = sig_buf.get_energy_fragment(beg, end,
                                                                 TWINDOW, lead)
            energy = lenerg if energy is None else energy + lenerg
        if energy is None:
            return 0.0
        #We get the already published fragments affecting our temporal support.
        conflictive = []
        published = SortedList(obs_buf.get_observations(o.Deflection))
        idx = published.bisect_left(pattern.hypothesis)
        if idx > 0 and published[idx-1].lateend > beg:
            idx -= 1
        while (idx < len(published) and Iv(beg, end).overlap(
                       Iv(published[idx].earlystart, published[idx].lateend))):
            conflictive.append(Iv(published[idx].earlystart - beg + fbeg,
                                  published[idx].lateend - beg + fbeg))
            idx += 1
        #We obtain the relative limits of the energy interval wrt the fragment
        iv_start = Iv(fbeg, fbeg + int(pattern.hypothesis.latestart - beg))
        iv_end = Iv(fend - int(end - pattern.hypothesis.earlyend), fend)
        #We look for the highest-level interval satisfying the limits.
        interval = None
        lev = 0
        while interval is None and lev <= 20:
            areas = [iv for iv in get_energy_intervals(energy, lev,
                                                                 group=TMARGIN)
                            if iv.start in iv_start and iv.end in iv_end and
                              all(not iv.overlapm(ein) for ein in conflictive)]
            #We sort the areas by energy, with the highest energy first.
            areas.sort(key = lambda interv :
                                     np.sum(energy[interv.start:interv.end+1]),
                                                                reverse = True)
            #Now we take the element indicated by the index.
            if len(areas) > int_idx:
                interval = areas[int_idx]
            else:
                lev += 1
        verify(interval is not None)
        pattern.hypothesis.start.set(interval.start + beg - fbeg,
                                     interval.start + beg - fbeg)
        pattern.hypothesis.end.set(interval.end + beg - fbeg,
                                   interval.end + beg - fbeg)
        for lead in sig_buf.get_available_leads():
            pattern.hypothesis.level[lead] = lev
    return _def_gconst
