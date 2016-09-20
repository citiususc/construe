# -*- coding: utf-8 -*-
"""
This module provides access to the ECG signal, and to the wavelet-transform of
the signal used for feature detection.
"""

__author__ = "T. Teijeiro"
__date__ = "$01-dic-2011 13:59:16$"

import numpy
import segmentation.wavelets.wavelet_filter as wf
from collections import OrderedDict

class Leads:
    """This enum-like class defines the standard leads of the ECG"""
    UNIQUE = "UNIQUE"
    MLI = "MLI"
    MLII = "MLII"
    MLIII = "MLIII"
    aVR = "aVR"
    aVL = "aVL"
    aVF = "aVF"
    V1 = "V1"
    V2 = "V2"
    V3 = "V3"
    V4 = "V4"
    V5 = "V5"
    V6 = "V6"

#This dictionary contains different names that may be given to the standard
#leads, associating each one with its unique key.
VALID_LEAD_NAMES = {
    'UNIQUE' : Leads.UNIQUE,
    'MLI' : Leads.MLI,
    'MLII' : Leads.MLII,
    'MLIII' : Leads.MLIII,
    'aVR' : Leads.aVR,
    'aVL' : Leads.aVL,
    'aVF' : Leads.aVF,
    'V1' : Leads.V1,
    'V2' : Leads.V2,
    'V3' : Leads.V3,
    'V4' : Leads.V4,
    'V5' : Leads.V5,
    'V6' : Leads.V6,
    'I'   : Leads.MLI,
    'II'  : Leads.MLII,
    'III' : Leads.MLIII,
    'V'   : Leads.V1,
    'V1-V2' : Leads.V2,
    'V2-V3' : Leads.V2,
    'V4-V5' : Leads.V4,
    'ECG1': Leads.MLII,
    'ECG2': Leads.V1,
    'D3'  : Leads.MLIII,
    'D4'  : Leads.V4,
    'CM2' : Leads.V2,
    'CM4' : Leads.V4,
    'CM5' : Leads.V5,
    'CC5' : Leads.MLI,
    'ML5' : Leads.MLIII,
    'mod.V1' : Leads.V1
}


_SIGNAL = OrderedDict()
_SIGNAL[Leads.MLI] = numpy.array([])
_SIGNAL[Leads.MLII] = numpy.array([])
_SIGNAL[Leads.MLIII] = numpy.array([])
_SIGNAL[Leads.aVR] = numpy.array([])
_SIGNAL[Leads.aVL] = numpy.array([])
_SIGNAL[Leads.aVF] = numpy.array([])
_SIGNAL[Leads.V1] = numpy.array([])
_SIGNAL[Leads.V2] = numpy.array([])
_SIGNAL[Leads.V3] = numpy.array([])
_SIGNAL[Leads.V4] = numpy.array([])
_SIGNAL[Leads.V5] = numpy.array([])
_SIGNAL[Leads.V6] = numpy.array([])

_ENERG = OrderedDict()
_ENERG[Leads.MLI] = numpy.array([])
_ENERG[Leads.MLII] = numpy.array([])
_ENERG[Leads.MLIII] = numpy.array([])
_ENERG[Leads.aVR] = numpy.array([])
_ENERG[Leads.aVL] = numpy.array([])
_ENERG[Leads.aVF] = numpy.array([])
_ENERG[Leads.V1] = numpy.array([])
_ENERG[Leads.V2] = numpy.array([])
_ENERG[Leads.V3] = numpy.array([])
_ENERG[Leads.V4] = numpy.array([])
_ENERG[Leads.V5] = numpy.array([])
_ENERG[Leads.V6] = numpy.array([])


def reset():
    """
    Resets the signal buffer
    """
    for lead in _SIGNAL:
        _SIGNAL[lead] = numpy.array([])
        _ENERG[lead] = numpy.array([])


def _get_block(array, start, end, blocksize):
    """
    Obtains a fragment of an array adjusted to a block size.
    """
    #Adjusting to a multiple of the blocksize
    window_size = blocksize * int(numpy.ceil((end - start) / float(blocksize)))
    real_start = start - ((window_size - (end - start)) / 2)
    #If we cannot center the block, we put it at start
    real_start = 0 if real_start < 0 else real_start
    block = array[real_start:real_start + window_size + 1]
    return (block, start - real_start, min(end - real_start, len(block)))

def get_signal_fragment(start, end, blocksize = None, lead = Leads.MLII):
    """
    Obtains the signal fragment in the specified interval, allowing the
    specification of a block size, of which the fragment length will be
    multiplo. It also returns the relative indices corresponding to the
    start and end parameters inside the fragment
    """
    assert lead in VALID_LEAD_NAMES, 'Unrecognized lead {0}'.format(lead)
    lead = VALID_LEAD_NAMES[lead]
    #TODO remove this!!!!
    start = 0 if start < 0 else start
    end = len(_SIGNAL[lead] - 1) if end >= len(_SIGNAL[lead]) else end
    #If blocksize not specified, return the requested fragment
    array = _SIGNAL[lead]
    return ((array[start:end+1], 0, end-start) if blocksize is None
                                 else _get_block(array, start, end, blocksize))

def get_energy_fragment(start, end, blocksize = None, lead = Leads.MLII):
    """
    Obtains the energy transform of the ECG signal in the specified lead
    within two limits.
    """
    assert lead in VALID_LEAD_NAMES, 'Unrecognized lead {0}'.format(lead)
    lead = VALID_LEAD_NAMES[lead]
    array = _ENERG[lead]
    return ((array[start:end+1], 0, end-start) if blocksize is None
                                 else _get_block(array, start, end, blocksize))

def get_signal_limits(lead = Leads.MLII):
    """Obtains a tuple(min,max) of the signal limits"""
    assert lead in VALID_LEAD_NAMES, 'Unrecognized lead {0}'.format(lead)
    lead = VALID_LEAD_NAMES[lead]
    return (numpy.amin(_SIGNAL[lead]), numpy.amax(_SIGNAL[lead]))

def get_signal(lead = Leads.MLII):
    """
    Obtains the whole signal in this buffer
    """
    assert lead in VALID_LEAD_NAMES, 'Unrecognized lead {0}'.format(lead)
    lead = VALID_LEAD_NAMES[lead]
    return _SIGNAL[lead]

def get_fake_signal(lead = Leads.MLII):
    """Obtains a null signal fragment of the same length than real signal"""
    assert lead in VALID_LEAD_NAMES, 'Unrecognized lead {0}'.format(lead)
    lead = VALID_LEAD_NAMES[lead]
    arr = numpy.zeros(len(_SIGNAL[lead]))
    arr[0] = 100
    return arr

def get_signal_length():
    """Obtains the number of availabe signal samples in this buffer"""
    return max([len(_SIGNAL[lead]) for lead in _SIGNAL])

def get_available_leads():
    """Obtains a list with the leads having signal in this buffer"""
    available = []
    for lead in _SIGNAL:
        if len(_SIGNAL[lead]) > 0:
            available.append(lead)
    return available

def is_available(lead):
    """Checks if a specific lead is available, that is has data"""
    assert lead in VALID_LEAD_NAMES, 'Unrecognized lead {0}'.format(lead)
    lead = VALID_LEAD_NAMES[lead]
    return len(_SIGNAL.get(lead, [])) > 0


def add_signal_fragment(fragment, lead = Leads.MLII):
    """Appends a new signal fragment to the buffer"""
    assert lead in VALID_LEAD_NAMES, 'Unrecognized lead {0}'.format(lead)
    lead = VALID_LEAD_NAMES[lead]
    _SIGNAL[lead] = numpy.append(_SIGNAL[lead], fragment)
    _ENERG[lead] = numpy.append(_ENERG[lead], wf.get_filter_energy(fragment))


if __name__ == "__main__":
    pass
