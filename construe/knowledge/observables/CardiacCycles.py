# -*- coding: utf-8 -*-
# pylint: disable-msg=E1101, E0102, E0202
"""
Created on Fri Jun  1 12:07:47 2012

This module contains the definition of those observables representing complete
cardiac cycles.

@author: T. Teijeiro
"""
from construe.model import Observable


class CardiacCycle(Observable):
    """This is the base class to represent cardiac cycles."""
    pass


class Normal_Cycle(CardiacCycle):
    """This class represents a normal cardiac cycle, with all its components"""
    pass

