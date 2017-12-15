# -*- coding: utf-8 -*-
# pylint: disable-msg=E1101, E0102, E0202
"""
Created on Wed May 06 12:54:17 2015

This module performs the full interpretation of every ECG record in the dataset
from the BMI lab.

@author: T. Teijeiro
"""
#Path setting
import sys
import os
sys.path.append(os.getcwd())
from construe.utils.units_helper import (set_sampling_freq,
                                            msec2samples as ms2sp)
set_sampling_freq(250.0)
import construe.utils.MIT.MITAnnotation as MITAnnotation
import construe.knowledge.base_evidence.energy as energy
import construe.inference.reasoning as reasoning
from construe.tests.record_processing import process_record
from construe.model.interpretation import Interpretation
import time
import pprint



#Searching settings
KFACTOR = 12
#Length of the fragments where we will perform the interpretation.
FR_LEN = 23040
#Storage management
ANNOTATOR = 'qrs'      #Standard annotator used as base evidence.
DATABASE_DIR = ('/home/tomas/Escritorio/data/'
                                        if len(sys.argv) < 3 else sys.argv[2])
OUTPUT_DIR = (DATABASE_DIR if len(sys.argv) < 4 else  sys.argv[3])
#Energy intervals detection and publishing
LMAX = energy.LPUB

RECORDS = [l.strip() for l in open(DATABASE_DIR + 'RECORDS')]

#Records to be interpreted can be selected from command line
SLC_STR = '0:{0}'.format(len(RECORDS)) if len(sys.argv) < 2 else sys.argv[1]
#We get a slice from the input string
SLC = slice(*[{True: lambda n: None, False: int}[x == ''](x)
                             for x in (SLC_STR.split(':') + ['', '', ''])[:3]])

for rec in RECORDS[SLC]:
    fname = OUTPUT_DIR + str(rec) + '.i' + ANNOTATOR
    if os.path.isfile(fname):
        print('Output file "{0}" already exists. Skipping record {1}'.format(
                                                                   fname, rec))
        continue
    #Time check
    T0 = time.time()
    TFACTOR = 5.0
    FR_OVERLAP = int(ms2sp(3000))
    MIN_DELAY = 1750
    MAX_DELAY = 20.0
    print 'Processing record {0} at 250.0 Hz '.format(rec)
    ANNOTS = process_record(DATABASE_DIR + str(rec), ANNOTATOR, TFACTOR, FR_LEN,
                                     FR_OVERLAP, MIN_DELAY, MAX_DELAY, KFACTOR)
    MITAnnotation.save_annotations(ANNOTS, fname)
    print('Record '+ str(rec) +' processed in '+ str(time.time() - T0) +'s')

print('The full database was sucessfully processed. Total branches: {0}'.format(
                                                       Interpretation.counter))
print('Reasoning statistics:')
pprint.pprint(reasoning.STATS.most_common())