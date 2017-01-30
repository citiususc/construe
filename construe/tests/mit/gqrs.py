# -*- coding: utf-8 -*-
# pylint: disable-msg=
"""
Created on Wed Jun 25 09:06:27 2014

This module tries to obtain an improved set of inital annotations for the
records of a database, by combining the output of the *gqrs* algorithm in all
the available leads of the record.

@author: T. Teijeiro
"""

from construe.utils.MIT import (get_leads, read_annotations,
                                             save_annotations, load_MIT_record)
from construe.utils.MIT.MITAnnotation import MITAnnotation
from construe.utils.signal_processing import kurtosis
from construe.utils.units_helper import (set_sampling_freq,
                                            msec2samples as ms2sp,
                                            samples2msec as sp2ms)
from scipy.stats import entropy
import construe.knowledge.constants as C
import subprocess
import os
import copy
import bisect
import numpy as np
import sortedcontainers

def H(data):
    """Entropy calculation for amplitudes"""
    hist = np.histogram(data, 128, density = True)
    return entropy(hist[0])

def mismatch(data):
    """
    Mismatch calculation between consecutive measures as described in:
    'Wang: A new method for evaluating ECG signal quality for multi-lead
    arrhythmia analysis'
    """
    hist = np.histogram(np.abs(data[1:]-data[:-1])/
                                       (data[1:]+data[:-1]), 128, density=True)
    csum = np.cumsum(hist[0])
    return csum/np.max(csum)

def improve_annotations(orig, annotators):
    """
    Improves a list of annotations by taking other annotations detected in
    in other channels and that have better amplitude measure.

    Parameters
    ----------
    orig:
        List of annotations to be improved.
    annotators:
        Iterable of *sortedlist* of annotations.
    """
    for i in xrange(len(orig)):
        dummy = copy.copy(orig[i])
        dummy.time = dummy.time - C.BEATANN_MIN_DIST
        impr = set([orig[i]])
        for annotator in annotators:
            lidx = bisect.bisect_left(annotator, dummy)
            dummy.time += 2 * C.BEATANN_MIN_DIST
            uidx = bisect.bisect_right(annotator, dummy)
            impr = impr.union(set(annotator[lidx:uidx]))
        #We select the annotation with the maximum value for each point.
        orig[i] = min(impr, key=lambda ann:(-ann.num, ann.time))
    return orig

def best_quality_lead(rec):
    """
    Obtains the index of the lead with best quality in a given record.
    """
    wlen = int(ms2sp(1000))
    n = len(rec.leads)
    siglen = len(rec.signal[0])
    quality = np.zeros(n)
    for i in xrange(n):
        quality[i] = np.median(np.array([kurtosis(rec.signal[i][j:j+wlen])
                                            for j in xrange(0, siglen, wlen)]))
    print record, quality
    return quality.argmax()

def best_quality_annotator(annotators, base_time= 0):
    """
    Obtains the lead with best quality, according to the indicator selected in
    acunote task #983.

    Parameters
    ----------
    annotators:
        List of annotators, from which one will be selected.
    base_time:
        Time (in samples) where the annotators started.
    """
    #First we discard empty annotators
    annotators = [ann for ann in annotators if len(ann) > 0]
    if not annotators:
        return []
    #Now we filter with a basic global frequency rule.
    #FIXME we assume annotators starting at 0, be careful!
    freqs = [len(ann)/(sp2ms(ann[-1].time - base_time)/60000.0)
                                                         for ann in annotators]
    gfreqann = [annotators[i] for i in xrange(len(freqs))
                                                      if 30 <= freqs[i] <= 140]
    if gfreqann:
        annotators = gfreqann
    amplitudes = [np.array([a.num for a in ann]) for ann in annotators]
    amplitudes = [amp/np.mean(amp) for amp in amplitudes]
    #Maximum amplitude
#    return annotators[np.argmax([np.mean(amp) if len(amp) > 1 else -np.inf
#                                                       for amp in amplitudes])]
    #Minimum amplitude variation
#    return annotators[np.argmin([np.std(amp) if len(amp) > 1 else np.inf
#                                                       for amp in amplitudes])]
    #Minimum amplitude variation with std windowing
    return annotators[np.argmin([np.std(
                          [np.std(amp[i:i+20]) for i in xrange(0,len(amp),20)])
                                                       for amp in amplitudes])]
    #Minimum amplitude variation with mean windowing
#    return annotators[np.argmin([np.std(
#                         [np.mean(amp[i:i+20]) for i in xrange(0,len(amp),20)])
#                                                       for amp in amplitudes])]
    #Entropy
#    return annotators[np.argmin([H(amp) if len(amp) > 1 else np.inf
#                                                       for amp in amplitudes])]
    #Mismatch calculation
    mismatchs = [mismatch(amp) if len(amp) > 1 else [0] for amp in amplitudes]
    bst = np.argmin([min(np.where(mm>0.95)[0]) for mm in mismatchs])
    return annotators[bst]

def truncate_annotators(annotators, ltime, utime):
    """
    Truncates a list of annotators, by taking from each one only the
    annotations between two specific times.
    """
    wannots = []
    lann = MITAnnotation()
    lann.time = ltime
    uann = MITAnnotation()
    uann.time = utime
    for annot in annotators:
        lidx = bisect.bisect_left(annot, lann)
        uidx = bisect.bisect_right(annot, uann)
        wannots.append(annot[lidx:uidx])
    return wannots

DB_DIR = '/datos/tomas.teijeiro/cinc_challenge15/training/'

ANNOTATOR = 'gqrs'

RECORDS = [100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 111, 112, 113,
           114, 115, 116, 117, 118, 119, 121, 122, 123, 124, 200, 201, 202,
           203, 205, 207, 208, 209, 210, 212, 213, 214, 215, 217, 219, 220,
           221, 222, 223, 228, 230, 231, 232, 233, 234]
#RECORDS = [101]
RECORDS = [l.strip() for l in open(DB_DIR + 'RECORDS')]

series = {}
for record in RECORDS:
    rec = DB_DIR + str(record)
    mitr = load_MIT_record(rec)
    set_sampling_freq(mitr.frequency)
    slen = len(mitr.signal[0])
    wlen = int(ms2sp(30*60*1000))
    leads = get_leads(rec)
    tmpann = 'tmp'
    annotators = []
    for lead in leads:
        command = ['gqrs', '-r', rec, '-outputName', tmpann, '-s', lead]
        subprocess.check_call(command)
        annpath = rec + '.' + tmpann
        annotators.append(read_annotations(annpath)[1:])
        os.remove(annpath)
    series[record] = np.array([[a.num for a in ann] for ann in annotators])
    for i in xrange(len(leads)):
        series[record][i] = series[record][i]/np.mean(series[record][i])
    #bestann = annotators[best_quality_lead(mitr)]
    annots = []
    i = 0
    while i < slen:
        wannots = truncate_annotators(annotators, i, i+wlen)
        bestann = best_quality_annotator(wannots, i)
        annots.extend(improve_annotations(bestann,
                               [ann for ann in wannots if ann is not bestann]))
        i += wlen
    #TODO only for CinC challenge 2015
    ###########################################################################
    if record.startswith('a'):
        annots = sortedcontainers.SortedList(annots)
        #We look for all annotations in the last 20 seconds of the record
        for lead in leads:
            command = ['gqrs', '-r', rec, '-outputName', tmpann,
                                                       '-m', '0.1', '-s', lead]
            subprocess.check_call(command)
            annpath = rec + '.' + tmpann
            tthres = (4*60+40)*250.0
            lowanns = read_annotations(annpath)
            for ann in lowanns:
                if ann.time > tthres:
                    idx = annots.bisect(ann)
                    btime = -np.inf if idx == 0 else annots[idx-1].time
                    atime = np.inf if idx == len(annots) else annots[idx].time
                    if (ann.time-btime > C.BEATANN_MIN_DIST and
                                          atime-ann.time > C.BEATANN_MIN_DIST):
                        annots.add(ann)
    ###########################################################################
    save_annotations(annots, rec + '.' + ANNOTATOR)
    print('Record {0} processed.'.format(record))
