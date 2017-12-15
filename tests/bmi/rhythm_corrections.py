# -*- coding: utf-8 -*-
# pylint: disable=
"""
Created on Thu May 14 14:54:36 2015

This script applies a series of heuristic rules to the annotations file
resulting from the interpretation process, in order to homogeinize and reduce
the number of detected episodes.

@author: T. Teijeiro
"""

import construe.utils.MIT as MIT
import construe.utils.MIT.interp2annots as i2a
import construe.utils.MIT.ECGCodes as COD
import construe.knowledge.observables as o
import construe.knowledge.constants as C
from construe.acquisition.signal_buffer import VALID_LEAD_NAMES
from construe.knowledge.abstraction_patterns.rhythm.afib import (
                                                           is_afib_rhythm_lian)
from construe.utils.clustering.xcorr_similarity import signal_match
from construe.utils.units_helper import msec2samples as ms2sp, SAMPLING_FREQ

import collections
import os
import numpy as np

DB = ('/home/local/tomas.teijeiro/Dropbox/Investigacion/tese/estadias/' +
      '2015_BMI/validation/training_dataset/')
#DB = '/home/tomas/Escritorio/afdb/'
#DB = '/tmp/mit/'
ANN = '.iqrs'
#ANN = '.ibatr'
OANN = '.rhy'

RECORDS = [l.strip() for l in open(DB + 'RECORDS')]

for rec in RECORDS:
    if os.path.isfile(DB+rec+OANN):
        print('Annotator "{0}" already exists. Skipping record {1}'.format(
                                                                   OANN, rec))
        continue
    print('Converting record {0}'.format(rec))
    anns = MIT.read_annotations(DB+rec+ANN)
    record = MIT.load_MIT_record(DB+rec)
    assert record.frequency == SAMPLING_FREQ
    record.leads = [VALID_LEAD_NAMES[l] for l in record.leads]
    interp = i2a.ann2interp(record, anns)
    afibs = list(interp.get_observations(o.Atrial_Fibrillation))
    i = 0
    while i < len(afibs):
        print('{0}/{1}'.format(i,len(afibs)))
        afib = afibs[i]
        beats = list(interp.get_observations(o.QRS, filt=lambda q, af=afib:
                                  af.earlystart <= q.time.start <= af.lateend))
        rpks = np.array([qrs.time.start for qrs in beats])
        #We obtain the shape representing the AFIB morphology as the qrs
        #matching the shape with more other qrss within the rhythm.
        ctr = collections.Counter()
        for qrs in beats:
            ctr[qrs] = len([q for q in beats
                            if signal_match(qrs.shape, q.shape)])
        nmatches = ctr.most_common(1)[0][1]
        refshape = max(q for q in ctr if ctr[q] == nmatches).shape
        #Now we try to include the following rhythms also as AFIB episodes.
        while True:
            nrhythm = next(interp.get_observations(o.Cardiac_Rhythm,
                                                   start=afib.lateend), None)
            if nrhythm is None:
                break
            elif isinstance(nrhythm, (o.Atrial_Fibrillation, o.RhythmBlock)):
                #Join consecutive AFIB episodes
                afib.end.value = nrhythm.end.value
                rpeaks = np.array([qrs.time.start for qrs in
                                       interp.get_observations(o.QRS,
                                          filt=lambda q, rh=nrhythm:
                                  rh.earlystart < q.time.start <= rh.lateend)])
                rpks = np.concatenate((rpks, rpeaks))
                if nrhythm in afibs:
                    afibs.remove(nrhythm)
                interp.observations.remove(nrhythm)
            elif not isinstance(nrhythm, o.Ventricular_Flutter):
                #Asystoles of more than 3 seconds always have to be reported.
                if (isinstance(nrhythm, o.Asystole) and
                        nrhythm.lateend-nrhythm.earlystart > ms2sp(3000)):
                    break
                #We check the shapeform and the RR of the new whole episode
                #candidate.
                rbeats = list(interp.get_observations(o.QRS, filt=
                                lambda q, rh=nrhythm:
                                   rh.earlystart < q.time.start <= rh.lateend))
                if all([signal_match(refshape, q.shape) for q in rbeats]):
                    rpeaks = np.array([q.time.start for q in rbeats])
                    tmprpks = np.concatenate((rpks, rpeaks))
                    rrs = (np.diff(rpeaks) if len(rpeaks) >= C.AFIB_MIN_NQRS
                                           else np.diff(tmprpks))
                    if is_afib_rhythm_lian(rrs):
                        #The rhythm is assumed to be part of the afib.
                        afib.end.value = nrhythm.end.value
                        interp.observations.remove(nrhythm)
                        rpks = tmprpks
                    else:
                        break
                else:
                    break
            else:
                break
        i += 1
    anns = i2a.interp2ann(interp)
    for b in interp.get_observations(o.BeatAnn):
        a = MIT.MITAnnotation.MITAnnotation()
        a.code = MIT.ECGCodes.ARFCT
        a.time = b.time.start
        anns.add(a)
    #Now we select only rhythm annotations and we remove repeated annotations.
    anns = [a for a in anns if a.code == COD.RHYTHM]
    i = 1
    while i < len(anns):
        if (anns[i].aux in ('(N', '(SVTA', '(SBR', '(AFIB', '(T', '(B', '(VFL')
                and anns[i].aux == anns[i-1].aux):
            anns.pop(i)
        else:
            i += 1
    MIT.save_annotations(anns, DB+rec+OANN)
    del interp
print('All records successfully processed')
