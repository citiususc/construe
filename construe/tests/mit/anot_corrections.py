# -*- coding: utf-8 -*-
# pylint: disable-msg=
"""
Created on Thu Jan 30 08:50:39 2014

This script performs the correction of a set of annotation files resulting
from the application of the *bxb* utility.

@author: T. Teijeiro
"""

import construe.utils.MIT.MITAnnotation as MIT
import construe.utils.MIT.ECGCodes as CODES
import construe.utils.units_helper as UNITS
from blist import sortedlist

def correct_bxb():
    ANNOTS_DIR = ('/home/remoto/tomas.teijeiro/Escritorio/anots_dani/')
    RECORDS = [100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 111, 112, 113,
               114, 115, 116, 117, 118, 119, 121, 122, 123, 124, 200, 201, 202,
               203, 205, 207, 208, 209, 210, 212, 213, 214, 215, 217, 219, 220,
               221, 222, 223, 228, 230, 231, 232, 233, 234]
    for rec in RECORDS:
        IN_FILE = ANNOTS_DIR + str(rec) + '.bxb'
        OUT_FILE = ANNOTS_DIR + str(rec) + '.bxd'
        anots = MIT.read_annotations(IN_FILE)
        out = []
        #Corrections according to the -o flag of the bxb utility.
        for ann in anots:
            if MIT.is_qrs_annotation(ann):
                out.append(ann)
            #Missed beats
            elif ann.code == CODES.NOTE and ann.aux[0] not in ('O', 'X'):
                new = MIT.MITAnnotation()
                new.code = CODES.CHARMAP[ann.aux[0]]
                new.time = ann.time
                out.append(new)
        MIT.save_annotations(out, OUT_FILE)
        print('Record {0} processed'.format(rec))
    print('The full database was successfully processed')


def correct_bxc_bxd():
    UNITS.set_sampling_freq(360.0)
    ANNOTS_DIR = ('/home/remoto/tomas.teijeiro/Escritorio/anots_dani/')
    RECORDS = [100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 111, 112, 113,
               114, 115, 116, 117, 118, 119, 121, 122, 123, 124, 200, 201, 202,
               203, 205, 207, 208, 209, 210, 212, 213, 214, 215, 217, 219, 220,
               221, 222, 223, 228, 230, 231, 232, 233, 234]
    for rec in RECORDS:
        REF = ANNOTS_DIR + str(rec) + '.atr'
        TEST = ANNOTS_DIR + str(rec) + '.bxd'
        OUT = ANNOTS_DIR + str(rec) + '.bxD'
        ref = sortedlist(MIT.read_annotations(REF))
        test = MIT.read_annotations(TEST)
        for tann in test:
            dummy = MIT.MITAnnotation()
            dummy.time = int(tann.time - UNITS.msec2samples(150))
            idx = ref.bisect_left(dummy)
            try:
                rann = next(a for a in ref[idx:] if MIT.is_qrs_annotation(a)
                          and abs(a.time-tann.time) <= UNITS.msec2samples(150))
                tann.code = rann.code
            except StopIteration:
                pass
        MIT.save_annotations(test, OUT)
        print('Record {0} processed'.format(rec))
    print('The full database was successfully processed')

