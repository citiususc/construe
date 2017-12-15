# -*- coding: utf-8 -*-
# pylint: disable-msg=E1101, E0102, E0202
"""
Created on Wed Feb 18 12:37:17 2015

This module performs the full interpretation of every record in the training
set of the CinC challenge 2015.

@author: T. Teijeiro
"""
#Path setting
import sys
import os
sys.path.append(os.getcwd())
from construe.utils.units_helper import (set_ADCGain,
                                            set_sampling_freq,
                                            msec2samples as ms2sp)
#Amplitude values are always got in physical units
set_ADCGain(1.0)
set_sampling_freq(250.0)
import construe.utils.MIT.MITAnnotation as MITAnnotation
import construe.knowledge.base_evidence.energy as energy
from construe.tests.record_processing import process_record
import time
from construe.model.interpretation import Interpretation


#Searching settings
KFACTOR = 12
#Length of the fragments where we will perform the interpretation.
FR_LEN = 10240
#Storage management
ANNOTATOR = 'gqrs'      #Standard annotator used as base evidence.
DATABASE_DIR = ('/home/local/tomas.teijeiro/cinc_challenge15/training/'
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
    FR_OVERLAP = int(ms2sp(3000))
    MAX_DELAY = 20.0
    print 'Processing record {0} at 250.0 Hz '.format(rec)
    #Signal abduction in fragments of FR_LEN length. We start at one fragment
    #before the time of the alarms, that are in minute 5:00
    ANNOTS = process_record(DATABASE_DIR + str(rec), ANNOTATOR, 1000.0, FR_LEN,
                                 FR_OVERLAP, 0, MAX_DELAY, KFACTOR,
                                         int(5*60*250.0 + FR_OVERLAP - FR_LEN))
    MITAnnotation.save_annotations(ANNOTS, fname)
    print('Record '+ str(rec) +' processed in '+ str(time.time() - T0) +'s')

print('The full database was sucessfully processed. Total branches: {0}'.format(
                                                       Interpretation.counter))
