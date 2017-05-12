# -*- coding: utf-8 -*-
# pylint: disable-msg=
"""
Created on Mon May 26 12:00:37 2014

This module provides a global dynamic observations buffer for the
interpretation process, where all base evidence is published and made available
to all interpretations.

@author: T. Teijeiro
"""

from ..model import Observable, EventObservable, Interval as Iv
from ..model.observable import overlap, end_cmp_key
import sortedcontainers
import numpy as np

class Status(object):
    """
    This enum-like class defines the possible status of the observations
    buffer.
    """
    STOPPED = 0
    ACQUIRING = 1

_OBS = sortedcontainers.SortedList(key=end_cmp_key)
_STATUS = Status.STOPPED

def reset():
    """Resets the observations buffer."""
    global _STATUS
    del _OBS[:]
    _STATUS = Status.STOPPED


def publish_observation(observation):
    """
    Adds a new piece of evidence to this global observations buffer. This
    observation must have a temporal location after any already published
    observation.
    """
    if _OBS.bisect(observation) < len(_OBS):
        raise ValueError(
              'The global observations buffer only accepts ordered insertions')
    _OBS.add(observation)


def get_observations(clazz=Observable, start=0, end=np.inf,
                                         filt=lambda obs: True, reverse=False):
    """
    Obtains a list of observations matching the search criteria, ordered
    by the earliest time of the observation.

    Parameters
    ----------
    clazz:
        Only instances of the *clazz* class (or any subclass) are returned.
    start:
        Only observations whose earlystart attribute is after or equal this
        parameter are returned.
    end:
        Only observations whose lateend attribute is lower or equal this
        parameter are returned.
    filt:
        General filter provided as a boolean function that accepts an
        observation as a parameter. Only the observations satisfying this
        filter are returned.
    reverse:
        Boolean parameter. If True, observations are returned in reversed
        order, from last to first.
    """
    dummy = EventObservable()
    if start == 0:
        idx = 0
    else:
        dummy.time.value = Iv(start, start)
        idx = _OBS.bisect_left(dummy)
    if end ==np.inf:
        udx = len(_OBS)
    else:
        dummy.time.value = Iv(end, end)
        udx = _OBS.bisect_right(dummy)
    return (obs for obs in _OBS.islice(idx, udx, reverse)
            if obs.earlystart >= start and isinstance(obs, clazz) and filt(obs))

def contains_observation(observation):
    """Checks if an observation is in the observations buffer"""
    return observation in _OBS

def nobs_before(time):
    """
    Obtains the number of observations in the observation buffer before a
    given time.
    """
    dummy = EventObservable()
    dummy.time.value = Iv(time, time)
    return _OBS.bisect_right(dummy)

def find_overlapping(observation, clazz=Observable):
    """
    Utility function used by the interpretation module to check the
    satisfaction of the exclusion relation. This function makes the following
    assumptions to speed-up the process: 1) 'observation' is not in the
    observation buffer, and 2) in the observations buffer, if
    obs1.start < obs2.start, then obs1.end < obs2.end.
    """
    dummy = EventObservable()
    dummy.time.value = Iv(observation.latestart, observation.latestart)
    idx = _OBS.bisect_right(dummy)
    while idx < len(_OBS):
        other = _OBS[idx]
        if isinstance(other, clazz) and overlap(other, observation):
            return other
        elif other.latestart > observation.earlyend:
            return None
        idx += 1
    return None

def get_status():
    """Obtains the status of the observations buffer"""
    return _STATUS

def set_status(status):
    """Changes the status of the observations buffer"""
    assert status in (Status.STOPPED, Status.ACQUIRING), 'Unknown status'
    global _STATUS
    _STATUS = status
