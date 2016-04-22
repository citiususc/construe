# -*- coding: utf-8 -*-
# pylint: disable-msg=E1101, E0102, E0202, C0103
"""
Created on Fri Jun  1 10:44:33 2012

This module contains the definition of the base class for all the domain
observables.

@author: T. Teijeiro
"""
from .FreezableObject import FreezableObject
from .constraint_network import Variable
from .interval import Interval
from numpy import inf
import itertools as it


def overlap(obs1, obs2):
    """
    Determine if two observations overlap. Returns True only if we are 100%
    sure that the observations overlap.
    """
    return (not obs1.earlyend <= obs2.latestart and
                                           not obs2.earlyend <= obs1.latestart)

def between(o1, o2, o3):
    """
    Determines if an observation *o2* is between *o1* and *o3*. It only returns
    True if it is sure that the *o2* has an instant between *o1* and *o3*. It
    is assumed than *o1* is before *o3*.
    """
    #We create a "dummy" observation between o1 and o3, and we check if it
    #overlaps with the checked observation.
    hole = Observable()
    hole.start.value = o1.end.value
    hole.end.value = o3.start.value
    return overlap(hole, o2)

def overlap_any(obs, obs_lst):
    """
    Checks if an observation overlaps any of the observations in a list. It
    only returns True if it is sure that is impossible for *obs* not overlap
    an observation in *obs_lst*. It is assumed the list has at least one
    element.
    """
    #We join the observations sharing their limits to avoid the side effect
    #of considering possible the presence of *obs* in a zero-size interval
    #between two consecutive observations in the list.
    dummy = Observable()
    dummy.start.value = obs_lst[0].start.value
    dummy.end.value = obs_lst[0].end.value
    i = 1
    while i < len(obs_lst):
        if (dummy.earlyend != dummy.lateend or
                                    dummy.end.value != obs_lst[i].start.value):
            dummy.start.value = obs_lst[i].start.value
        dummy.end.value = obs_lst[i].end.value
        if overlap(obs, dummy):
            return True
        i+=1
    return False


#TODO maybe this is not the best place for this function
def non_consecutive(obs1, obs2, observations):
    """
    Checks if two observations are non consecutive in a list, this is, there is
    another observation between *obs1* and *obs2* in the *observations* list.

    Parameters
    ----------
    obs1:
        First observation
    obs2:
        Second observation
    observations:
        Iterable of observations (maybe containing obs1 and obs2).

    Returns
    -------
    out:
        True if it is guaranteed that there is a third observation in
        *observations* (different from *obs1* and *obs2*) that is between
        *obs1* and *obs2*.
    """
    tocheck = it.ifilter(lambda obs : obs.earlystart >= obs1.earlystart and
                                    obs.lateend <= obs2.lateend, observations)
    return any(obs is not obs1 and obs is not obs2 and between(obs1, obs, obs2)
                                                            for obs in tocheck)


def singleton_observable(observable):
    """
    This function defines a decorator to declare some observable classes as
    *Singleton observables*. This means that in a specific interpretation, only
    one observation of that observable can be present.
    This characteristic is implemented as a decorator because it is more
    flexible and cleaner than a inherit-based method.

    Usage
    -----
    Simply put the annotation @singletonobservable before the class
    declaration.
    """
    observable.__singletonobservable__ = True
    return observable


def is_singleton(observable):
    """
    Checks if an observable type (or an observation) is a singleton observable.
    See the documentation of *singletonobservable* for details.

    Arguments
    ---------
    observable:
        Class to check if it is a singleton observable.
    """
    return getattr(observable, '__singletonobservable__', False)


class Observable(FreezableObject):
    """
    Base class for all the observables. It has three attributes:
    *start*:
        The start temporal variable, of type *Variable*.
    *time*:
        Temporal variable used to represent the observable as a single event.
    *end*:
        The finish temporal variable, of type *Variable*.
    """
    def __init__(self):
        super(Observable, self).__init__()
        self.start = Variable(value = Interval(0, inf))
        self.time = Variable(value = Interval(0, inf))
        self.end = Variable(value = Interval(0, inf))

    def __str__(self):
        """
        Obtains the representation of the observable as a character string.
        """
        time = str(self.start)
        if self.time is not self.start:
            time = ''.join((time, ', ', str(self.time)))
        if self.end is not self.time:
            time = ''.join((time, ', ', str(self.end)))
        return ''.join((type(self).__name__, ' - <', time, '>'))

    def __repr__(self):
        return self.__str__()

    def __lt__(self, other):
        """
        Implements the operation 'less than' based on the start interval of
        the observable. This allows for the sorting of the set of observations
        in the event system.
        """
        return ((self.earlystart, self.latestart,
                   self.earlyend, self.lateend,
                   type(self).__name__)  <
                                           (other.earlystart, other.latestart,
                                              other.earlyend, other.lateend,
                                              type(other).__name__))

    @property
    def earlystart(self):
        """Returns the earliest start time of the observation"""
        return self.start.value._start

    @property
    def latestart(self):
        """Returns the latest start time of the observation"""
        return self.start.value._end

    @property
    def earlyend(self):
        """Returns the earliest end time of the observation"""
        return self.end.value._start

    @property
    def lateend(self):
        """Returns the latest end time of the observation"""
        return self.end.value._end


class EventObservable(Observable):
    """
    This is the base class for all the observables that reference to an
    specific time instant instead of an interval. It is also an observable.
    """

    def __init__(self):
        """
        This type of observables have only one temporal variable, to which the
        two temporal variables of full observables are referenced.
        """
        self.time = Variable(value = Interval(0, inf))
        self.start = self.end = self.time



if __name__ == "__main__":
    o = Observable()
    o2 = Observable()
    o3 = Observable()
    o4 = Observable()
    e = EventObservable()
    o.start.value = Interval(2, 4)
    o.end.value = Interval(6, 6)
    o2.start.value = Interval(8, 12)
    o2.end.value = Interval(9, 14)
    o3.start.value = Interval(3, 4)
    o3.end.value = Interval(4, 5)
    o4.start.value = Interval(4, 5)
    o4.end.value = Interval(7, 9)
    e.time.value = Interval(10, 10)
    assert o < o2
    assert o < o3
    assert o < o4
    assert between(o, o4, o2)
    assert o2 < e