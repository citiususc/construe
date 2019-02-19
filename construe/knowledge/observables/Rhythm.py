# -*- coding: utf-8 -*-
# pylint: disable-msg=E1101, E0102, E0202
"""
Created on Fri Jun  1 12:12:38 2012

This module contains the definition of those observables related with the
cardiac rhythm analysis.

@author: T. Teijeiro
"""
from construe.model import Observable, singleton_observable
from collections import namedtuple


class RR(Observable):
    """
    Class that represents the RR observable, this is, the distance between
    two consecutive ventricular depolarizations.
    """
    def __init__(self):
        super().__init__()
        #The single reference will correspond to the start time
        self.time = self.start

class Cardiac_Rhythm(Observable):
    """Class that represents a general and unspecified cardiac rhythm"""

    __slots__ = ('meas',)

    def __init__(self):
        super().__init__()
        #The single reference will correspond to the start time
        self.time = self.start
        #All rhythms have some representative measurements of their constituent
        #cycles.
        self.meas = CycleMeasurements((0, 0), (0, 0), (0, 0))

CycleMeasurements = namedtuple('CycleMeasurements', ['rr', 'rt', 'pq'])


@singleton_observable
class RhythmStart(Cardiac_Rhythm):
    """Class that represents the start of the first detected rhythm."""
    def __init__(self):
        super().__init__()
        #The rhythm start, in addition to be a rhythm, is an eventual and
        #unique observation.
        self.time = self.end = self.start


class RegularCardiacRhythm(Cardiac_Rhythm):
    """Class that represents a regular rhythm."""

    __slots__ = ('morph', )

    def __init__(self):
        super().__init__()
        #A regular cardiac rhythm has a single beat morphology. To evaluate
        #the morphology we use a shape similarity strategy.
        self.morph = {}

class Sinus_Rhythm(RegularCardiacRhythm):
    """Class that represents sinus rhythm"""
    pass

class Tachycardia(RegularCardiacRhythm):
    """Class that represents tachycardia rhythm"""
    pass

class Bradycardia(RegularCardiacRhythm):
    """Class that represents bradycardia rhythm"""
    pass

class Extrasystole(Cardiac_Rhythm):
    """This class represents an extrasystole"""
    pass

class Bigeminy(Cardiac_Rhythm):
    """This class represents a bigeminy rhythm"""
    pass

class Trigeminy(Cardiac_Rhythm):
    """This class represents a trigeminy rhythm"""
    pass

class Asystole(Cardiac_Rhythm):
    """Class that represents an asystole (absence of cardiac activity)"""
    pass

class Ventricular_Flutter(Cardiac_Rhythm):
    """Class that represents a ventricular flutter rhythm"""
    pass

class Couplet(Cardiac_Rhythm):
    """Class that represents a ventricular couplet rhythm"""
    pass

class RhythmBlock(Cardiac_Rhythm):
    """Class that represents a rhythm block"""

    __slots__ = ('morph',)

    def __init__(self):
        super().__init__()
        #Rhythm block must keep the same beat morphology throughout its
        #duration. This mophology consists of a QRSShape struct for each lead
        #in the record.
        self.morph = {}

class Atrial_Fibrillation(Cardiac_Rhythm):
    """Class that represents atrial fibrillation"""

    __slots__ = ('morph',)

    def __init__(self):
        super().__init__()
        #Atrial fibrillation must keep the same beat morphology throughout its
        #duration. This mophology consists of a QRSShape struct for each lead
        #in the record.
        self.morph = {}