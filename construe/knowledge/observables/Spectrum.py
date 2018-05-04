# -*- coding: utf-8 -*-
# pylint: disable-msg=E1101, E0102, E0202
"""
Created on Fri Jun  1 11:00:06 2012

This module contains the domain observables related with the spectrum
analysis of the ECG signal.

@author: T. Teijeiro
"""
from construe.model import Observable


class Deflection(Observable):
    """
    This class represents a signal deviation consistent with the electrical
    activity of the cardiac muscle fibers. It is associated with a certain
    energy level derived from the wavelet decomposition/reconstruction of the
    signal.
    """

    __slots__ = ('level', )

    def __init__(self):
        """Creates a new Deflection instance, at level 0"""
        super(Deflection, self).__init__()
        #The single reference will correspond to the start variable
        self.time = self.start
        self.level = {}

    def __str__(self):
        """
        Obtains the representation of the observable as a character string.
        """
        level = '-' if not self.level else min(self.level.values())
        lead = '-' if not self.level else min(self.level, key= self.level.get)
        return '{0} ({1}, {2})'.format(super(Deflection, self).__str__(),
                                                                  level, lead)

class RDeflection(Deflection):
    """
    This class represents a signal deviation consistent with the electrical
    activity generated in the ventricular activation. It can be obtained by
    any external QRS detection algorithm
    """

    __slots__ = ('tag', )

    def __init__(self):
        """Creates a new instance of a R-Deflection, that is instantaneous"""
        super(RDeflection, self).__init__()
        #Beat annotations are instantaneous observables.
        self.end = self.start
