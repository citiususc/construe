# -*- coding: utf-8 -*-
# pylint: disable-msg=
"""
Created on Fri Jul 18 19:48:47 2014

This module modifies a set of .atr annotators, that are assumed to select the
peak of each QRS complex, and produces a set of .batr annotators that are
identical to the original one, but the QRS annotations are now set to the
beginning of the QRS complex.

@author: T. Teijeiro
"""
from construe.utils.MIT import (get_leads, read_annotations,
                                   is_qrs_annotation, save_annotations,
                                                               load_MIT_record)
from construe.utils.units_helper import (set_sampling_freq, msec2samples,
                                                                  samples2msec)
import numpy as np
import matplotlib.pyplot as plt

def move_to_qrs_onset(ann, mitr):
    """
    Changes the time point of a QRS annotation to the beginning of the complex,
    estimated as the point with the maximum decrease in the slope of the signal
    """
    assert is_qrs_annotation(ann)
    beg, end = int(ann.time-msec2samples(80)), int(ann.time-msec2samples(10))+1
    if beg<0:
        beg=0
    pts = []
    for lead in xrange(len(mitr.signal)):
        sigfr = mitr.signal[lead][beg:end]
        sg = np.sign(np.diff(sigfr))
        idx = len(sg)-1
        while idx >= 0 and sg[idx] == sg[-1]:
            idx -= 1
        pts.append(idx+1)
    ann.time = beg+min(pts)


DB_DIR = '/tmp/mit/'
ORIG_ANNOTATOR = 'atr'
OUT_ANNOTATOR = 'batr'

RECORDS = [100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 111, 112, 113,
           114, 115, 116, 117, 118, 119, 121, 122, 123, 124, 200, 201, 202,
           203, 205, 207, 208, 209, 210, 212, 213, 214, 215, 217, 219, 220,
           221, 222, 223, 228, 230, 231, 232, 233, 234]
#RECORDS = [101]

for record in RECORDS:
    rec = DB_DIR + str(record)
    mitr = load_MIT_record(rec)
    set_sampling_freq(mitr.frequency)
    slen = len(mitr.signal[0])
    annots = read_annotations(rec + '.' + ORIG_ANNOTATOR)
    for ann in (a for a in annots if is_qrs_annotation(a)):
        move_to_qrs_onset(ann, mitr)
    save_annotations(annots, rec + '.' + OUT_ANNOTATOR)
    print('Record {0} processed.'.format(record))
