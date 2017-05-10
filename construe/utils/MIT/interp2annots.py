# -*- coding: utf-8 -*-
# pylint: disable=
"""
Created on Tue May 12 12:32:18 2015

Module to convert from interpretations to annotations lists, and back again.
Note that at this moment, some information is lost in every conversion.

@author: T. Teijeiro
"""

import construe.utils.MIT.ECGCodes as C
import construe.utils.MIT.MITAnnotation as MITAnnotation
import construe.utils.MIT as MIT
import construe.knowledge.observables as o
import construe.knowledge.constants as K
from construe.acquisition.signal_buffer import VALID_LEAD_NAMES
from construe.knowledge.abstraction_patterns.segmentation.QRS import _tag_qrs
from ..units_helper import msec2samples as ms2sp
from construe.model.interpretation import Interpretation
from construe.model.interval import Interval as Iv
from ..signal_processing.wave_extraction import Wave
import json
import sortedcontainers
import numpy as np
import warnings

def ann2interp(record, anns):
    """
    Returns an interpretation containing the observations represented in a list
    of annotations associated to a loaded MIT record. Note that only the
    *observations* field is properly set.
    """
    interp = Interpretation()
    observations = []
    for i in xrange(len(anns)):
        ann = anns[i]
        if ann.code in (C.PWAVE, C.TWAVE):
            obs = o.PWave() if ann.code == C.PWAVE else o.TWave()
            beg = next(a for a in reversed(anns[:i]) if a.time < ann.time
                                                     and a.code == C.WFON).time
            obs.start.value = Iv(beg, beg)
            end = beg+(ann.time-beg)*2
            obs.end.value = Iv(end, end)
            leads = (record.leads if ann.code is C.TWAVE
                     else set(K.PWAVE_LEADS).intersection(set(record.leads)))
            for lead in leads:
                sidx = record.leads.index(lead)
                s = record.signal[sidx][beg:end+1]
                mx, mn = np.amax(s), np.amin(s)
                pol = (1.0 if max(mx-s[0], mx-s[-1]) >= -min(mn-s[0],mn-s[1])
                       else -1.0)
                obs.amplitude[lead] = pol * np.ptp(s)
            observations.append(obs)
        elif MIT.is_qrs_annotation(ann):
            obs = o.QRS()
            obs.time.value = Iv(ann.time, ann.time)
            obs.tag = ann.code
            delin = json.loads(ann.aux)
            #QRS start and end is first tried to set according to delineation
            #info. If not present, it is done according to delineation
            #annotations.
            if delin:
                for l in delin.keys():
                    if l not in record.leads:
                        compatible = next((l2 for l2 in VALID_LEAD_NAMES
                                          if VALID_LEAD_NAMES[l2] == l), None)
                        if compatible is None:
                            raise ValueError('Unrecognized lead {0}'.format(l))
                        delin[compatible] = delin.pop(l)
                beg = ann.time + min(d[0] for d in delin.itervalues())
                end = ann.time + max(d[-1] for d in delin.itervalues())
            else:
                beg = next(a for a in reversed(anns[:i]) if a.code==C.WFON).time
                end = next(a for a in anns[i:] if a.code == C.WFOFF).time
            #Endpoints set
            obs.start.value = Iv(beg, beg)
            obs.end.value = Iv(end, end)
            for lead in delin:
                assert len(delin[lead]) % 3 == 0, 'Unrecognized delineation'
                sidx = record.leads.index(lead)
                beg = ann.time + delin[lead][0]
                end = ann.time + delin[lead][-1]
                obs.shape[lead] = o.QRSShape()
                sig = record.signal[sidx][beg:end+1]
                obs.shape[lead].sig = sig-sig[0]
                obs.shape[lead].amplitude = np.ptp(sig)
                obs.shape[lead].energy = np.sum(np.diff(sig)**2)
                obs.shape[lead].maxslope = np.max(np.abs(np.diff(sig)))
                waves = []
                for i in xrange(0, len(delin[lead]), 3):
                    wav = Wave()
                    wav.pts = tuple(delin[lead][i:i+3])
                    wav.move(-delin[lead][0])
                    if wav.r >= len(sig):
                        warnings.warn('Found delineation information after '
                         'the end of the signal in annotation {0}'.format(ann))
                        break
                    wav.amp = (np.sign(sig[wav.m]-sig[wav.l]) *
                               np.ptp(sig[wav.l:wav.r+1]))
                    wav.e = np.sum(np.diff(sig[wav.l:wav.r+1])**2)
                    wav.move(delin[lead][0])
                    wav.move(ann.time-obs.earlystart)
                    waves.append(wav)
                if not waves:
                    obs.shape.pop(lead)
                else:
                    obs.shape[lead].waves = tuple(waves)
                    obs.shape[lead].tag = _tag_qrs(waves)
            observations.append(obs)
        elif ann.code is C.RHYTHM and ann.aux in C.RHYTHM_AUX.values():
            rhclazz = next(rh for rh in C.RHYTHM_AUX
                           if C.RHYTHM_AUX[rh] == ann.aux)
            obs = rhclazz()
            obs.start.value = Iv(ann.time, ann.time)
            end = next((a.time for a in anns[i+1:] if a.code is C.RHYTHM),
                       anns[-1].time)
            obs.end.value = Iv(end, end)
            observations.append(obs)
        elif ann.code is C.ARFCT:
            obs = o.RDeflection()
            obs.time.value = Iv(ann.time, ann.time)
            observations.append(obs)
    interp.observations = sortedcontainers.SortedList(observations)
    return interp

def interp2ann(interp, btime=0, offset=0):
    """
    Generates a list of annotations representing the observations from
    an interpretation. The *btime* optional parameter allows to include only
    the observations after a specific time point, and *offset* allows to define
    a constant time to be added to the time point of each annotation.
    """
    annots = sortedcontainers.SortedList()
    beats = list(interp.get_observations(o.QRS,
                                         filt=lambda q: q.time.start >= btime))
    #We get the beat observations in the best explanation branch.
    for beat in beats:
        #We tag all beats as normal, and we include the delineation. The
        #delineation on each lead is included as a json string in the peak
        #annotation.
        beg = MITAnnotation.MITAnnotation()
        beg.code = C.WFON
        beg.time = int(offset + beat.earlystart)
        peak = MITAnnotation.MITAnnotation()
        peak.code = beat.tag
        peak.time = int(offset + beat.time.start)
        delin = {}
        for lead in beat.shape:
            shape = beat.shape[lead]
            displ = beg.time-peak.time
            shape.move(displ)
            waveseq = sum((w.pts for w in shape.waves), tuple())
            delin[lead] = tuple(int(w) for w in waveseq)
            shape.move(-displ)
        peak.aux = json.dumps(delin)
        end = MITAnnotation.MITAnnotation()
        end.code = C.WFOFF
        end.time = int(offset + beat.lateend)
        annots.add(beg)
        annots.add(peak)
        annots.add(end)
    #P and T wave annotations
    pstart = beats[0].earlystart - ms2sp(400) if beats else 0
    tend = beats[-1].lateend + ms2sp(400) if beats else 0
    for wtype in (o.PWave, o.TWave):
        for wave in interp.get_observations(wtype, pstart, tend):
            beg = MITAnnotation.MITAnnotation()
            beg.code = C.WFON
            beg.time = int(offset + wave.earlystart)
            end = MITAnnotation.MITAnnotation()
            end.code = C.WFOFF
            end.time = int(offset + wave.lateend)
            peak = MITAnnotation.MITAnnotation()
            peak.code = C.PWAVE if wtype is o.PWave else C.TWAVE
            peak.time = int((end.time+beg.time)/2.)
            annots.add(beg)
            annots.add(peak)
            annots.add(end)
    #Flutter annotations
    for flut in interp.get_observations(o.Ventricular_Flutter, btime):
        vfon = MITAnnotation.MITAnnotation()
        vfon.code = C.VFON
        vfon.time = int(offset + flut.earlystart)
        annots.add(vfon)
        for vfw in interp.get_observations(o.Deflection, flut.earlystart,
                                           flut.lateend):
            wav = MITAnnotation.MITAnnotation()
            wav.code = C.FLWAV
            wav.time = int(offset + vfw.time.start)
            annots.add(wav)
        vfoff = MITAnnotation.MITAnnotation()
        vfoff.code = C.VFOFF
        vfoff.time = int(offset + flut.lateend)
        annots.add(vfoff)
    #All rhythm annotations
    for rhythm in interp.get_observations(o.Cardiac_Rhythm, btime):
        if not isinstance(rhythm, o.RhythmStart):
            rhyon = MITAnnotation.MITAnnotation()
            rhyon.code = C.RHYTHM
            rhyon.aux = C.RHYTHM_AUX[type(rhythm)]
            rhyon.time = int(offset + rhythm.earlystart)
            annots.add(rhyon)
    #The end of the last rhythm is also added as an annotation
    try:
        rhyoff = MITAnnotation.MITAnnotation()
        rhyoff.code = C.RHYTHM
        rhyoff.aux = ')'
        rhyoff.time = int(offset + rhythm.earlyend)
        annots.add(rhyoff)
    except NameError:
        #If there are no rhythms ('rhythm' variable is undefined), we go on
        pass
    #Unintelligible R-Deflections
    for rdef in interp.get_observations(o.RDeflection, btime,
                                        filt=lambda a:
                                        a in interp.unintelligible or
                                        a in interp.focus):
        unint = MITAnnotation.MITAnnotation()
        #We store unintelligible annotations as artifacts
        unint.code = C.ARFCT
        unint.time = int(offset + rdef.earlystart)
        annots.add(unint)
    return annots
