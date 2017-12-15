# -*- coding: utf-8 -*-
# pylint: disable-msg=E1101, E0102, E0202
"""
Created on Mon Nov 19 16:20:51 2012

This script performs certain comparisons between two different annotators in
the database.

@author: T. Teijeiro
"""
import numpy as np
import construe.utils.MIT.MITAnnotation as MIT
from construe.utils.units_helper import (samples2msec as sp2ms,
                                            msec2samples as ms2sp,
                                            set_sampling_freq)

if __name__ == "__main__":
    ANNOTS_DIR = ('/tmp/mit/')
    RECORDS = [100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 111, 112, 113,
               114, 115, 116, 117, 118, 119, 121, 122, 123, 124, 200, 201, 202,
               203, 205, 207, 208, 209, 210, 212, 213, 214, 215, 217, 219, 220,
               221, 222, 223, 228, 230, 231, 232, 233, 234]
    set_sampling_freq(360.0)
    #Dictionary to save the discrepancy at record-level
    dist = {}
    miss = 0
    for rec in RECORDS:
        dist[rec] = []
        REF_FILE = ANNOTS_DIR + str(rec) + '.atr'
        TEST_FILE = ANNOTS_DIR + str(rec) + '.wbr'
        reference = np.array(
                        [anot.time for anot in MIT.read_annotations(REF_FILE)
                                              if MIT.is_qrs_annotation(anot)])
        test = np.array([anot.time for anot in MIT.read_annotations(TEST_FILE)
                                               if MIT.is_qrs_annotation(anot)])
        #Missing beat search
        for b in reference:
            err = np.Inf
            for t in test:
                bdist = t-b
                if abs(bdist) > abs(err):
                    break
                else:
                    err = bdist
            if abs(err) > ms2sp(150.0):
                print('{0}: {1}'.format(rec, b))
                miss += 1
            else:
                dist[rec].append(sp2ms(err))
        dist[rec] = np.array(dist[rec])
        print('Record {0} processed'.format(rec))
    alldist = np.concatenate(tuple(dist[r] for r in RECORDS))


