# -*- coding: utf-8 -*-
# pylint: disable-msg=
"""
Created on Fri Jun  1 11:55:41 2012

This module contains the definition of all the domain observables related with
the segmentation of the ECG signal in components.

@author: T. Teijeiro
"""

from construe.model import Observable, EventObservable, FreezableObject
import numpy as np

class PWave(Observable):
    """Observable that represents a P Wave"""
    def __init__(self):
        super(PWave, self).__init__()
        #The single reference will correspond to the start time
        self.time = self.start
        self.amplitude = {}


class TWave(Observable):
    """Observable that represents a T Wave"""
    def __init__(self):
        super(TWave, self).__init__()
        #The single reference will correspond to the start time
        self.time = self.start
        self.amplitude = {}


class QRS(Observable):
    """
    Observable that represents a QRS complex

    Attributes
    ----------
    shape:
        Dictionary with the shape of the QRS complex in each lead, indexed
        by lead.
    """
    def __init__(self):
        super(QRS, self).__init__()
        self.shape = {}
        #By default, all QRS are tagged as normal.
        self.tag = 1
        self.clustered = False

    @property
    def paced(self):
        """Checks if this QRS complex is paced."""
        #WFDB code for paced beats is 12. We cannot reference  it here due to
        #cyclic references.
        return self.tag == 12

    @paced.setter
    def paced(self, paced):
        """
        Sets this QRS complex as paced/non paced.

        Parameters
        ----------
        paced:
            Boolan value indicating wether this is a paced QRS or not. If True,
            the **tag** atribute of this complex is assigned to the **PACE**
            code. Else, the attribute is not modified unless its current value
            is **PACE**. In that case, the attribute is assigned to **NORMAL**.
        """
        if paced:
            self.tag = 12
        elif self.tag == 12:
            self.tag = 1

class QRSShape(FreezableObject):
    """
    Class that represents the shape of a QRS complex in a specific leads. It
    consists in a sequence of waves, a string tag abstracting those waves,
    an amplitude and energy and maximum slope measures, and a numpy array
    representing the signal.
    """
    def __init__(self):
        super(QRSShape, self).__init__()
        self.waves = ()
        self.amplitude = 0.0
        self.energy = 0.0
        self.maxslope = 0.0
        self.tag = ''
        self.sig = np.array([])

    def __repr__(self):
        return self.tag

    def __eq__(self, other):
        if type(self) is type(other):
            return (self.waves == other.waves and
                    self.amplitude == other.amplitude and
                    self.energy == other.energy and
                    self.maxslope == other.maxslope and
                    self.tag == other.tag and np.all(self.sig == other.sig))
        return False

    def move(self, displacement):
        """Moves the temporal references of the waves forming this shape"""
        for wave in self.waves:
            wave.move(displacement)


class Noise(Observable):
    """
    Observable that represents a noisy signal fragment.
    """
    def __init__(self):
        super(Noise, self).__init__()
        #The single reference will correspond to the start time.
        self.time = self.start


class RPeak(EventObservable):
    """
    Observable that represents a R wave peak.

    Attributes
    ----------
    amplitude:
        Amplitude of the R Peak.
    """
    def __init__(self):
        super(RPeak, self).__init__()
        self.amplitude = 0.0

class Baseline(Observable):
    """
    Observable that represents a baseline observation.
    """
    def __init__(self):
        super(Baseline, self).__init__()
        #The single reference will correspond to the start time
        self.time = self.start

