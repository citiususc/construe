# -*- coding: utf-8 -*-
# pylint: disable-msg=W0603
"""
Created on Tue May 27 10:20:03 2014

This module contains the main functionality of the acquisition system to work
with signal records simulating the real-time input.

@author: T. Teijeiro
"""

from ..utils.MIT import load_MIT_record, read_annotations, is_qrs_annotation
from ..utils.units_helper import (samples2msec as sp2ms, msec2samples as ms2sp,
                                                        SAMPLING_FREQ, ADCGain)
from ..model import Interval as Iv
from ..utils.axel import Event
import construe.utils.MIT.ECGCodes as ECGCodes
import construe.acquisition.obs_buffer as BUF
import construe.acquisition.signal_buffer as SIG
import construe.knowledge.observables as o
import time as T
import numpy as np


_T0 = 0
_REC = None
_RECNAME = ''
_OFFSET = 0
_DURATION = 0
_TFACTOR = 1.0                         #Time factor to control the input speed.
_STEP = 256          #Size in signal samples of each evidence acquisition step.
_LAST_POS = 0
_ANNOTS = []

def set_record(record, annotator = None, physical_units= False):
    """
    Sets the record used for input and the initial evidence.

    Parameters
    ----------
    record:
        Name/Path of the MIT-BIH record containing the input signal.
    annotator:
        It can be a string with the name of the annotator, or a list of
        annotations with the initial evidence to be interpreted.
    physical_units:
        Flag to set whether the signal is in digital or physical units.
    """
    global _REC
    global _ANNOTS
    global _RECNAME
    _REC = load_MIT_record(record, physical_units)
    _RECNAME = record
    assert SAMPLING_FREQ == _REC.frequency, ('Incorrect sampling frequency: '
                                             'expected {0}, got {1}'.format(
                                                 SAMPLING_FREQ, _REC.frequency))
    assert ADCGain == _REC.gain, ('Incorrect ADC Gain: expected {0}, '
                                  'got {1}'.format(ADCGain, _REC.gain))
    if annotator is not None:
        if isinstance(annotator, str):
            _ANNOTS = read_annotations(record + '.' + annotator)
        else:
            _ANNOTS = annotator[:]

def set_offset(offset):
    """Sets an offset in the record from where the acquisition should start"""
    global _OFFSET
    _OFFSET = offset

def set_duration(duration):
    """Sets the duration of the record to be interpreted"""
    global _DURATION
    assert int(duration) % _STEP == 0
    _DURATION = int(duration)

def set_tfactor(tfactor):
    """Sets the time factor to control the input speed."""
    global _TFACTOR
    _TFACTOR = tfactor

def get_tfactor(tfactor):
    """Obtains the temporal factor controlling the current input speed."""
    return _TFACTOR

#Public event that will be fired on acquisition system reset.
on_reset = Event()

def reset():
    """Resets the acquisition system"""
    global _T0
    _T0 = 0
    BUF.reset()
    SIG.reset()
    #Fire reset event.
    on_reset()

def start():
    """Starts the acquisition of signal and evidence"""
    global _T0
    _T0 = T.time()
    BUF.set_status(BUF.Status.ACQUIRING)

def stop():
    """Stops the acquisition of signal and evidence"""
    BUF.set_status(BUF.Status.STOPPED)

def get_more_evidence():
    """
    Obtains a new piece of evidence and introduces it in the appropriate
    structures.
    """
    if BUF.get_status() is BUF.Status.STOPPED:
        return
    dtime = _TFACTOR * (T.time() - _T0) * 1000.0
    cursize = SIG.get_signal_length()
    if dtime - sp2ms(cursize) > sp2ms(_STEP):
        nchunks = int((min(ms2sp(dtime), _DURATION) - cursize)/_STEP)
        init = _OFFSET + cursize
        for i in xrange(len(_REC.leads)):
            fragment = _REC.signal[i, init:init+nchunks*_STEP]
            if len(fragment) < nchunks*_STEP:
                fragment = np.concatenate((fragment,
                            fragment[-1]*np.ones(_STEP-len(fragment)%_STEP)))
            SIG.add_signal_fragment(fragment, _REC.leads[i])
        for ann in (a for a in _ANNOTS if ((is_qrs_annotation(a) or
                                            a.code is ECGCodes.FLWAV) and
                                         init <= a.time < init+nchunks*_STEP)):
            rdef = o.RDeflection()
            atime = ann.time - _OFFSET
            rdef.time.value = Iv(atime, atime)
            #The level is established according to the annotation information.
            rdef.level = {SIG.VALID_LEAD_NAMES[lead] :
                                                    127 for lead in _REC.leads}
            rdef.level[SIG.VALID_LEAD_NAMES[_REC.leads[ann.chan]]] = (127 -
                                                                       ann.num)
            rdef.freeze()
            BUF.publish_observation(rdef)
        newsize = SIG.get_signal_length()
        if newsize >= _DURATION or newsize >= len(_REC.signal[0])-_OFFSET:
            BUF.set_status(BUF.Status.STOPPED)

def get_acquisition_point():
    """
    Obtains the time point, in signal samples, where the acquisition process
    is.
    """
    return SIG.get_signal_length()

def get_record_length():
    """Obtains the length of the record"""
    return len(_REC.signal[0])

def get_record_name():
    """Obtains the name of the input reecord"""
    return _RECNAME

if __name__ == "__main__":
    REC = '/databases/mit/100'
    set_record(REC, 'qrs')
    set_duration(10240)
    set_offset(18550)
    start()
    while True:
        get_more_evidence()
        print(SIG.get_signal_length(), BUF.get_status())
        if BUF.get_status() == BUF.Status.STOPPED:
            break
        T.sleep(0.1)
    print('Total length of acquired signal: {0} (in {1} s)'.format(
                                         SIG.get_signal_length(),T.time()-_T0))
    reset()
    #We test higher temporal factors
    _TFACTOR = 30.0
    set_duration(650240)
    set_offset(0)
    start()
    while True:
        get_more_evidence()
        print(SIG.get_signal_length(), BUF.get_status())
        if BUF.get_status() == BUF.Status.STOPPED:
            break
        T.sleep(0.1)
    print('Total length of acquired signal: {0} (in {1} s)'.format(
                                         SIG.get_signal_length(),T.time()-_T0))

