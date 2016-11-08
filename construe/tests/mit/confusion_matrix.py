# -*- coding: utf-8 -*-
# pylint: disable-msg=
"""
Created on Fri Oct 23 08:50:02 2015

This script obtains the confusion matrix for the beat annotations of a set
of records.

@author: T. Teijeiro
"""
import sys
import numpy as np
import construe.utils.MIT as MIT
import construe.utils.MIT.ECGCodes as C
from construe.utils.units_helper import msec2samples as ms2sp

def pprint_matrix(tags, matrix):
    """
    Prints a confusion matrix in a pretty format.
    """
    #Matrix header
    sys.stdout.write('{:6}'.format(''))
    for tag in tags:
        c = C.ICHARMAP(tag) if tag > -1 else 'O'
        sys.stdout.write('{:6}'.format(c))
    sys.stdout.write('\n')
    #Matrix rows
    for i in xrange(len(tags)):
        sys.stdout.write(C.ICHARMAP(tags[i]) if i > 0 else 'O')
        for j in xrange(len(tags)):
            sys.stdout.write('{:>6d}'.format(int(matrix[i,j])))
        sys.stdout.write('|{:>6d}'.format(int(np.sum(matrix[i]))))
        sys.stdout.write('\n')
    #Column sum
    sys.stdout.write(' ')
    for i in xrange(len(tags)):
        sys.stdout.write('{:>6d}'.format(int(np.sum(matrix[:,i]))))
    sys.stdout.write('\n')


if __name__ == "__main__":
    #Config variables
    PATH = '/tmp/mit/'
    REF = '.atr'
    TEST = '.cls'
    MWIN = ms2sp(150.0)
    #Records to be interpreted can be selected from command line
    SLC_STR = '0:48' if len(sys.argv) < 2 else sys.argv[1]
    #We get a slice from the input string
    SLC = slice(*[{True: lambda n: None, False: int}[x == ''](x)
                             for x in (SLC_STR.split(':') + ['', '', ''])[:3]])
    CMATS = {}
    for REC in [l.strip() for l in open(PATH + 'RECORDS')][SLC]:
        print 'Record {}'.format(REC)
        tp = fn = fp = 0
        ref = [a for a in MIT.read_annotations(PATH + REC + REF)
                                                   if MIT.is_qrs_annotation(a)]
        for a in ref:
            a.code = C.aami_to_mit(C.mit_to_aami(a.code))
        test = [a for a in MIT.read_annotations(PATH + REC + TEST)
                                                   if MIT.is_qrs_annotation(a)]
        for a in test:
            a.code = C.aami_to_mit(C.mit_to_aami(a.code))
        tags = sorted(set(a.code for a in test).union(a.code for a in ref))
        #The -1 tag is used for false positives and false negatives.
        tags.insert(0, -1)
        cmat = np.zeros((len(tags), len(tags)))
        i = j = 0
        while i < len(ref) and j < len(test):
            if abs(ref[i].time - test[j].time) <= MWIN:
                #True positive, introduced in the corresponding matrix cell
                cmat[tags.index(ref[i].code), tags.index(test[j].code)] += 1
                i += 1
                j += 1
            elif ref[i].time < test[j].time:
                #False negative
                cmat[tags.index(ref[i].code), 0] += 1
                fn += 1
                i += 1
            else:
                #False positive
                cmat[0, tags.index(test[j].code)] += 1
                fp += 1
                j += 1
        pprint_matrix(tags, cmat)
        print ''
        CMATS[REC] = (tags, cmat)
    #Global confusion matrix
    GTAGS = sorted(set.union(*(set(tags) for tags, _ in CMATS.itervalues())))
    GMAT = np.zeros((len(GTAGS), len(GTAGS)))
    for tags, mat in CMATS.itervalues():
        for i in xrange(len(tags)):
            for j in xrange(len(tags)):
                GMAT[GTAGS.index(tags[i]), GTAGS.index(tags[j])] += mat[i,j]
    print 'Global confusion matrix:'
    pprint_matrix(GTAGS, GMAT)
