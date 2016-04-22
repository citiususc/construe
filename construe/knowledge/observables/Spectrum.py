# -*- coding: utf-8 -*-
# pylint: disable-msg=E1101, E0102, E0202
"""
Created on Fri Jun  1 11:00:06 2012

This module contains the domain observables related with the spectrum
analysis of the ECG signal.

@author: T. Teijeiro
"""
from construe.model import Observable


class Energ_Int(Observable):
    """
    This class represents a signal interval with a certain energy level,
    derived from the wavelet transformation of the signal.
    """
    def __init__(self):
        """Creates a new energy interval instance, at level 0"""
        super(Energ_Int, self).__init__()
        #The single reference will correspond to the start variable
        self.time = self.start
        self.level = {}

    def __str__(self):
        """
        Obtains the representation of the observable as a character string.
        """
        level = '-' if not self.level else min(self.level.itervalues())
        lead = '-' if not self.level else min(self.level, key= self.level.get)
        return '{0} ({1}, {2})'.format(super(Energ_Int, self).__str__(),
                                                                  level, lead)

class BeatAnn(Energ_Int):
    """
    This class specifically represents a beat annotation obtained by an
    external algorithm.
    """

    def __init__(self):
        """Creates a new instance of a beat annotation, that is instantaneous"""
        super(BeatAnn, self).__init__()
        #Beat annotations are instantaneous observables.
        self.end = self.start
