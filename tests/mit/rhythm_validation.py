# -*- coding: utf-8 -*-
# pylint: disable-msg=
"""
Created on Thu Nov  6 13:46:05 2014

This module performs the validation of rhythm characterization in the MIT-BIH
database, by comparing the temporal overlapping of the different rhythms
identified by two different annotators.

@author: T. Teijeiro
"""

import construe.utils.MIT as MIT
import construe.utils.MIT.ECGCodes as ECGCodes
import numpy as np
from construe.utils.units_helper import samples2msec as s2m, samples2hour
from construe.model import Interval as Iv
from collections import defaultdict

#This dictionary contains the the tags accepted for the classification of those
#rhythms we understand there is an inconsistency in the original database. For
#those rhythms not present here, we only accept exactly the same tag.
VALID_TAGS = {
    '(N'   : ('(N', '(SBR', '(SVTA'),
    '(SBR' : ('(SBR', '(BK', 'P'),
    '(AFL' : ('(AFL', '(AFIB'),
    '(BII' : ('(BII', '(SBR'),
    '(IVR' : ('(IVR', '(SBR'),
    '(PREX': ('(PREX', '(N', '(SBR', '(SVTA'),
    '(VT'  : ('(VT', '(SVTA'),
    '(AB'  : ('(AB', '(B',)
    }
assert all([k in VALID_TAGS[k] for k in VALID_TAGS])

#This dictionary contains the rhythm tags excluded in the evaluation of another
#rhythm tag. Check the epicmp documentation for details.
EXCLUSION_TAGS = {
    '(AFIB' : ('(AFL', ),
    '(B'    : ('(AB', ),
    '(N'    : tuple(),
    '(P'    : tuple(),
    '(SBR'  : ('(N', ),
    '(SVTA' : ('(N', '(VT'),
    '(T'    : tuple(),
    '(VFL'  : tuple()
}

class Rhythm(object):
    """Just a simple wrapper for rhythm observations."""
    def __init__(self):
        self.code = '(N'
        self.start = 0
        self.end = 0

    def __lt__(self, other):
        """
        Implements the operation 'less than' based on the start interval of
        the observable. This allows for the sorting of the set of observations
        in the event system.
        """
        return (self.start, self.end) < (other.start, other.end)

    def __str__(self):
        """String representation of the rhythm"""
        return '{0} {1} - {2}'.format(self.code, self.start, self.end)

    def __repr__(self):
        return str(self)

    @property
    def iv(self):
        """Return an interval representing the duration of the rhythm"""
        return Iv(self.start, self.end)

def hitrate(results):
    """
    Obtains the hit rate of the classification (in %). A hit is considered when
    the reference and the test tag are the same.
    """
    hit = sum(results[k] for k in results
                                      if k[1] in VALID_TAGS.get(k[0], (k[0],)))
    total = sum(results.values())
    return 100.0*hit/total

def print_results(results):
    """
    Prints the rhythm classification results as a confusion matrix.
    """
    rtags = set(lab[0] for lab in results)
    ttags = set(lab[1] for lab in results)
    reflabels = sorted(rtags.intersection(ttags)) + sorted(rtags - ttags)
    testlabels = sorted(rtags.intersection(ttags)) + sorted(ttags - rtags)
    #Confusion matrix
    mat = np.zeros((len(reflabels), len(testlabels)))
    for i in range(len(reflabels)):
        for j in range(len(testlabels)):
            mat[i, j] = results[(reflabels[i], testlabels[j])]
    #Normalized results
    #HINT change by mat to get absolute results.
    norm = mat#(mat/mat.sum(axis=1)[:, np.newaxis])*100.0
    #Results display
    out = np.zeros((len(reflabels)+1, len(testlabels)+2), dtype='object')
    out[0, 0] = ''
    #Column headers
    out[0, 1:-1] = testlabels
    out[0, -1] = 'Total(s)'
    #Row headers
    out[1:, 0] = reflabels
    #Results
    out[1:, 1:-1] = np.around(norm, 2)
    #Aggregated time
    out[1:, -1] = np.around(np.sum(mat, axis = 1), 2)
    n, m = out.shape
    #String representation of the confusion matrix
    outstr = ''
    for i in range(n):
        for j in range(m):
            if j == m-1:
                outstr += '|'
            outstr += str(out[i, j]).rjust(9)
            outstr += '\t'
        outstr += '\n'
    print(outstr)

def epicmp(ref, test, rtag):
    """
    Compares the episode number and duration sensitivity and specificity for a
    rhythm tag. The implementation is equivalent to that found in the epicmp
    program of the WFDB software distribution, but it does not support
    ischaemia episodes comparison.

    Parameters
    ----------
    ref:
        Sorted list of reference rhythm episodes.
    test:
        Sorted list of test rhythm episodes.
    rtag:
        String representing the tag code for the rhythm type to evaluate.
    """
    stp = fn = ref_duration = ref_overlap = 0
    pp = fp = test_duration = test_overlap = 0
    for a, b in ((ref, test), (test, ref)):
        total_duration = total_overlap = match = mismatch = 0
        for ep in (rh for rh in a if rh.code == rtag):
            ep_dur = ep.iv.length
            #Duration reduction by rhythms excluded from the comparison.
            if a is test:
                for excl in (rh.iv.intersection(ep.iv) for rh in b
                                    if rh.code in EXCLUSION_TAGS[ep.code]
                                                     and rh.iv.overlap(ep.iv)):
                    ep_dur -= excl.length
            if ep_dur <= 0:
                continue
            total_duration += ep_dur
            ep_overl = sum(rh.iv.intersection(ep.iv).length
                    for rh in b if rh.code == ep.code and rh.iv.overlap(ep.iv))
            if ep_overl > 0:
                match += 1
                total_overlap += ep_overl
            else:
                mismatch += 1
        if a is ref:
            stp, fn, ref_duration, ref_overlap = (match, mismatch,
                                                 total_duration, total_overlap)
        else:
            pp, fp, test_duration, test_overlap = (match, mismatch,
                                                 total_duration, total_overlap)
    return ((stp, fn, ref_duration, ref_overlap),
            (pp, fp, test_duration, test_overlap))

def print_epicmp_results(etag, rec, meas, lineformat):
    """
    Returns a string compatible with the epicmp  output format.

    Parameters
    ----------
    etag:
        Tag representing the evaluated rhythm ('(AFIB', '(N', ...)
    rec:
        Evaluated record name.
    meas:
        Array with 8 numerical values: Sensitivity TP,
        false negatives, reference duration, reference positive overlap,
        specificity TP, false positives, test duration, test positive overlap.
    lineformat:
        If true, result is created in line format, else in matrix format.
    """
    eSe = ('-' if meas[0]+meas[1] == 0
                         else int(round(100.0*meas[0]/float(meas[0]+meas[1]))))
    ePp = ('-' if meas[4]+meas[5] == 0
                         else int(round(100.0*meas[4]/float(meas[4]+meas[5]))))
    dSe = ('-' if meas[2] == 0 else int(round(100.0*meas[3]/float(meas[2]))))
    dPp = ('-' if meas[6] == 0 else int(round(100.0*meas[7]/float(meas[6]))))
    if lineformat:
        return ('{0:>6} {1:>4d} {2:>4d} {3:>4d} {4:>4d} {5:>3} {6:>3} {7:>3} '
                '{8:>3} {9:>14} {10:>14}'.format(
                str(rec), int(meas[0]), int(meas[1]), int(meas[4]),
                int(meas[5]), str(eSe), str(ePp), str(dSe), str(dPp),
                samples2hour(meas[2]), samples2hour(meas[6])))
    else:
        outstr = ''
        outstr += (' {0} episode-by-episode comparison results for '
                   'record {1}\n'.format(etag, rec))
        outstr += '            Episode sensitivity: {0}% ({1}/{2})\n'.format(
                                  str(eSe), int(meas[0]), int(meas[0]+meas[1]))
        outstr += '  Episode positive predictivity: {0}% ({1}/{2})\n'.format(
                                  str(ePp), int(meas[4]), int(meas[4]+meas[5]))
        outstr += '           Duration sensitivity: {0}% ({1}/{2})\n'.format(
                                  str(dSe), samples2hour(meas[3]),
                                                         samples2hour(meas[2]))
        outstr += ' Duration positive predictivity: {0}% ({1}/{2})\n'.format(
                                  str(dPp), samples2hour(meas[7]),
                                                         samples2hour(meas[6]))
        return outstr


if __name__ == "__main__":
    DB = '/tmp/mit/'
    RECORDS = [100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 111, 112, 113,
               114, 115, 116, 117, 118, 119, 121, 122, 123, 124, 200, 201, 202,
               203, 205, 207, 208, 209, 210, 212, 213, 214, 215, 217, 219, 220,
               221, 222, 223, 228, 230, 231, 232, 233, 234]
    REF_ANN = '.atr'
    TEST_ANN = '.rhy'
    epicmpres = {tag : np.zeros(8) for tag in EXCLUSION_TAGS}
    results = {}
    for rec in RECORDS:
        results[rec] = defaultdict(int)
        ref = MIT.read_annotations(DB + str(rec) + REF_ANN)
        test = MIT.read_annotations(DB + str(rec) + TEST_ANN)
        reclen = ref[-1].time
        #Rhythm loading in form of intervals
        ref_rhythms = []
        test_rhythms = []
        for alst in (ref, test):
            rlst = ref_rhythms if alst is ref else test_rhythms
            for ann in (a for a in alst if a.code is ECGCodes.RHYTHM):
                if rlst:
                    rlst[-1].end = ann.time
                newrhythm = Rhythm()
                newrhythm.code = ann.aux.strip('\x00')
                newrhythm.start = ann.time
                rlst.append(newrhythm)
            rlst[-1].end = alst[-1].time
        #Interval overlapping measures
        for rhythm in ref_rhythms:
            overl = [rh for rh in test_rhythms if rhythm.iv.overlap(rh.iv)]
            vtags = VALID_TAGS.get(rhythm.code, (rhythm.code, ))
            hits = [rh for rh in overl if rh.code in vtags]
            for rh in overl:
                results[rec][(rhythm.code, rh.code)] += s2m(
                                   rhythm.iv.intersection(rh.iv).length)/1000.0
        #Output
        print('==== {0} ({1:2.2f}%) ====\n'.format(rec, hitrate(results[rec])))
        print_results(results[rec])
        for rhtag in sorted(set(epicmpres.keys()).intersection(
                          set(rh.code for rh in test_rhythms).union(
                                         set(rh.code for rh in ref_rhythms)))):
            sem, ppm = epicmp(ref_rhythms, test_rhythms, rhtag)
            meas = sem+ppm
            print(print_epicmp_results(rhtag, rec, meas, False))
            for i in range(len(meas)):
                epicmpres[rhtag][i] += meas[i]
    #Global results
    gresults = defaultdict(int)
    allkeys = set.union(*(set(v.keys()) for v in results.values()))
    for k in allkeys:
        gresults[k] = sum(v[k] for v in results.values())
    print('====Global results ({0:2.2f}%):===='.format(hitrate(gresults)))
    print_results(gresults)
    #Episode evaluation through epicmp
    print('==== Episode comparison results ====')
    for rhtag in sorted(epicmpres):
        res = epicmpres[rhtag]
        print(print_epicmp_results(rhtag, 'Gross', res, False))