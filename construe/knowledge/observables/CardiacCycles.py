# -*- coding: utf-8 -*-
# pylint: disable-msg=E1101, E0102, E0202
"""
Created on Fri Jun  1 12:07:47 2012

This module contains the definition of those observables representing complete
cardiac cycles.

@author: T. Teijeiro
"""
from construe.model import Observable, singleton_observable
from construe.knowledge.observables.Rhythm import CycleMeasurements


class CardiacCycle(Observable):
    """This is the base class to represent cardiac cycles."""

    __slots__ = ('meas',)

    def __init__(self):
        super().__init__()
        #All rhythms have some representative measurements of their constituent
        #cycles.
        self.meas = CycleMeasurements((0, 0), (0, 0), (0, 0))

@singleton_observable
class FirstBeat(CardiacCycle):
    """
    Class to represent the first heartbeat in a interpretation, used to
    break the recursion in the search for the previous beat.
    """
    pass


class Normal_Cycle(CardiacCycle):
    """This class represents a normal cardiac cycle, with all its components"""
    pass

