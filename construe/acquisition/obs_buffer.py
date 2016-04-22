# -*- coding: utf-8 -*-
# pylint: disable-msg=
"""
Created on Mon May 26 12:00:37 2014

This module provides a global dynamic observations buffer for the
interpretation process, where all base evidence is published and made available
to all interpretations.

@author: T. Teijeiro
"""

from ..model import Observable, Interval as Iv
import blist
import numpy as np
import itertools as it

class Status:
    """
    This enum-like class defines the possible status of the observations
    buffer.
    """
    STOPPED = 0
    ACQUIRING = 1

_OBS = blist.sortedlist()
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


def get_observations(clazz = Observable, start = 0, end = np.inf,
                                                      filt = lambda obs: True):
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
        """
        dummy = Observable()
        dummy.start.value = Iv(start, start)
        idx = _OBS.bisect_left(dummy)
        if end == np.inf:
            udx = len(_OBS)
        else:
            dummy.start.value = Iv(end, end)
            udx = _OBS.bisect_right(dummy)
        return (obs for obs in it.islice(_OBS, idx, udx)
                if obs.lateend <= end and isinstance(obs, clazz) and filt(obs))

def contains_observation(observation):
    return observation in _OBS

def get_status():
    """Obtains the status of the observations buffer"""
    return _STATUS

def set_status(status):
    """Changes the status of the observations buffer"""
    assert status in (Status.STOPPED, Status.ACQUIRING), 'Unknown status'
    global _STATUS
    _STATUS = status