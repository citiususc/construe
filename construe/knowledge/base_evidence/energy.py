# -*- coding: utf-8 -*-
# pylint: disable-msg=
"""
Created on Thu Nov  7 17:38:39 2013

This module contains the functionality to obtain the base evidence for the
interpretation process consisting of energy intervals observations.

@author: T. Teijeiro
"""

import construe.acquisition.signal_buffer as sig_buf
import construe.knowledge.observables as o
from construe.model import Interval as Iv
from construe.utils.units_helper import msec2samples as ms2sp, changeTime
from scipy.stats.mstats import mquantiles
import numpy as np
import itertools as it
import bisect
from blist import sortedlist

############################
### Constants definition ###
############################
LPUB = 0        # Maximum level of the observations considered as base evidence
TWINDOW = 1280  # Temporal window for obtaining the relevan intervals


def get_energy_intervals(energy, level = 0, percentile = 0.95, group = 1):
    """
    Obtains the relevant energy intervals for a specific level, using the
    default wavelet filter. It starts in the maximum energy of the signal, and
    for each level it descends to the 95 percentile.

    Parameters
    ----------
    energy:
        Energy signal of the wavelet filter. It can be obtained as the output
        of the *get_filter_energy()* function
    level:
        Level of interest, starting in 0.
    percentile:
        Descending percentile for each level, in the [0-1] interval. By default
        is 0.95.
    group:
        Groups the observations that are separated by a distance <= group in a
        unique observation.
    Returns
    -------
    List of Intervals, detected according the parameters.
    """
    #First, we set the threshold over which we consider relevant a wave.
    thres = mquantiles(energy, prob = percentile ** (level + 1))[0]
    #Filter
    indices = np.nonzero(energy > thres)[0]
    #Integration of consecutive indices
    observations = []
    i = 0
    while i < len(indices):
        start = indices[i]
        while (i+1 < len(indices) and (indices[i+1] - indices[i] <= group)):
            i = i+1
        end = indices[i]
        observations.append(Iv(start, end))
        i = i+1
    return observations

def get_deflection_observations(start, end, lead, max_level=0, group=ms2sp(20)):
    """
    Obtains deflection observations present in a signal fragment,
    specified by their limits. The returned intervals are separated by levels,
    and grouped by a closeness parameter.

    Parameters
    ----------
    start:
        Start index of the signal fragment.
    end:
        End index of the fragment.
    lead:
        Lead used for the
    signal:
        Signal fragment, as a one-dimensional array.
    max_level:
        Energy level we want to reach in the search.
    group:
        Distance parameter. Observations with differences less than this
        value are grouped in a single observation.

    Result
    ------
    out: dict
        Dict with one list of observations by level. The temporal variables
        of the intervals are set according to the start index.

    See Also
    --------
    wavelet_filter.get_energy_intervals
    """
    energ = sig_buf.get_energy_fragment(start, end, lead=lead)[0]
    obs = {}
    for i in xrange(max_level + 1):
        obs[i] = []
        for interv in get_energy_intervals(energ, level = i, group = group):
            defl = o.Deflection()
            defl.start.value = Iv(interv.start, interv.start)
            defl.end.value = Iv(interv.end, interv.end)
            defl.level[lead] = i
            obs[i].append(defl)
        #We update the time of the intervals
        changeTime(obs[i], start)
    #Now we need to remove redundant observations of upper levels
    for i in xrange(max_level, 0, -1):
        j = 0
        while j < len(obs[i]):
            obj = obs[i][j]
            found = False
            for upper in obs[i-1]:
                #First posterior observation
                if upper.earlystart >= obj.earlystart:
                    #If is contained, then remove the lower level obs.
                    if (upper.earlystart >= obj.earlystart and
                                                 upper.lateend <= obj.lateend):
                        found = True
                        obs[i].pop(j)
                    break
            if not found:
                j += 1
    return obs


def combine_energy_intervals(dicts, margin = ms2sp(20)):
    """
    Combines the overlapping observations in several dicts in the result format
    of the get_deflection_observations() function.

    Parameters
    ----------
    dicts:
        List of dictionaries. The combination is always performed to the
        first dictionary.
    score:
        Dictionary that stores the score for each observation. For overlapping
        observations, the result score is the sum of the overlapped
        observations.
    margin:
        Group margin. Intervals separated by less than this margin are removed.
    """
    chain = it.chain.from_iterable
    dict1 = dicts[0]
    for wint in chain(dict1.itervalues()):
        for i in xrange(1, len(dicts)):
            conflictive = []
            for lst in dicts[i].itervalues():
                if not lst:
                    continue
                idx = bisect.bisect_left(lst, wint)
                #We go to the first real index
                while (idx > 0 and
                          lst[idx-1].lateend+margin >= wint.earlystart-margin):
                    idx -= 1
                #Now we search for overlapping intervals
                while (idx < len(lst) and
                            lst[idx].earlystart-margin <= wint.lateend+margin):
                    w = lst[idx]
                    if Iv(w.earlystart-margin, w.lateend+margin).overlap(
                              Iv(wint.earlystart-margin, wint.lateend+margin)):
                        conflictive.append(w)
                    idx += 1
            if conflictive:
                alleads = set.union(*(set(w.level.iterkeys())
                            for w in conflictive)) - set(wint.level.iterkeys())
                for lead in alleads:
                    wint.level[lead] = min(w.level.get(lead, np.Inf)
                                                          for w in conflictive)
                for wconf in conflictive:
                    dicts[i][wconf.level.values()[0]].remove(wconf)


def get_combined_energy(start, end, max_level, group=ms2sp(80)):
    """
    This function obtains the energy intervals between two time points combined
    in a multilead fashion. And grouping by a distance criteria.

    Parameters
    ----------
    start:
        Start time point to get the observations with respect to the signal
        buffer.
    end:
        Finish time point to get the observations wrt the signal buffer.
    max_level:
        Maximum level to search for energy intervals. See the description of
        the level in the *get_energy_intervals* function.
    group:
        Distance used to group close observations.

    Returns
    -------
    out:
        Sorte list of *EnergyInterval* observations.
    """
    #Dictionaries to store the energy intervals for each lead
    dicts = {}
    for lead in sig_buf.get_available_leads():
        dicts[lead] = {}
        for i in xrange(max_level + 1):
            dicts[lead][i] = []
    #Energy intervals detection and combination
    idx = start
    while idx < end:
        wfs = {}
        for lead in dicts:
            wfs[lead] = get_deflection_observations(start + idx,
                                                start + idx + TWINDOW,
                                                lead = lead,
                                                max_level = max_level,
                                                group = group)
            for i in xrange(max_level + 1):
                if dicts[lead][i] and wfs[lead][i]:
                    if (wfs[lead][i][0].earlystart - dicts[lead][i][-1].lateend
                                                                     <= group):
                        dicts[lead][i][-1].end.value = (
                                                   wfs[lead][i][0].start.value)
                        wfs[lead][i].pop(0)
                dicts[lead][i].extend(wfs[lead][i])
        idx += TWINDOW
    #Remove overlapping intervals
    combine_energy_intervals(dicts.values())
    #Now we flatten the dictionaries, putting all the intervals in a sequence
    #sorted by the earlystart value.
    return sortedlist(w for w in it.chain.from_iterable(it.chain.from_iterable(
                                      dic.values() for dic in dicts.values())))


if __name__ == "__main__":
    pass
