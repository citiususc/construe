# -*- coding: utf-8 -*-
# pylint: disable-msg=
"""
Created on Fri May  8 14:46:45 2015

This script generates a report with the interpretation results on a full
database. It includes the sequence of rhythms, the mean heart rate in every
rhythm, the number of extrasystoles, rhythm blocks and couplets.

@author: T. Teijeiro
"""

import construe.utils.MIT as MIT
import construe.utils.MIT.ECGCodes as ECGCodes
from construe.utils.units_helper import samples2msec as sp2ms
from collections import Counter

PATH = ('/home/local/tomas.teijeiro/Dropbox/Investigacion/tese/estadias/' +
        '2015_BMI/validation/')
RECORDS = [l.strip() for l in open(PATH + 'RECORDS')]
ANN = '.iqrs'

RHNAMES = {b'(N'   : 'Normal rhythm',
           b'(SVTA': 'Tachycardia',
           b'(SBR' : 'Bradycardia',
           b'(AFIB': 'Atrial Fibrillation',
           b'(T'   : 'Trigeminy',
           b'(B'   : 'Bigeminy',
           b'(VFL' : 'Ventricular Flutter',
           b'P'    : 'Absence of rhythm'}

for rec in RECORDS:
    rhctr = Counter()
    anns = MIT.read_annotations(PATH+rec+ANN)
    rhythms = (a for a in anns if a.code in (ECGCodes.RHYTHM, ECGCodes.VFON))
    try:
        start = next(rhythms)
    except StopIteration:
        continue
    print('Interpretation results for record {0}:'.format(rec))
    print('Rhythm analysis:')
    nect = len([a for a in anns if a.aux == b'(EXT'])
    nbk = len([a for a in anns if a.aux == b'(BK'])
    ncpt = len([a for a in anns if a.aux == b'(CPT'])
    while True:
        end = next(rhythms, anns[-1])
        if start.aux in RHNAMES:
            rhctr[start.aux] += end.time-start.time
        elif start.code == ECGCodes.VFON:
            rhctr['(VFL'] += end.time-start.time
        if end.code not in (ECGCodes.RHYTHM, ECGCodes.VFON):
            break
        start = end
    for rh, samples in rhctr.most_common():
        ms = int(sp2ms(samples))
        h = int(ms/3600000)
        ms -= h*3600000
        m = int(ms/60000)
        ms -= m*60000
        s = int(ms/1000)
        ms -= s*1000
        print('    {0:<20} - {1:02}:{2:02}:{3:02}.{4:03}'.format(
                                                     RHNAMES[rh], h, m, s, ms))
    print('Number of extrasystoles: {0}'.format(nect))
    print('Number of rhythm blocks: {0}'.format(nbk))
    print('Number of couplets: {0}'.format(ncpt))
    print('\n')

