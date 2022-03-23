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
from ..signal_processing.wave_extraction import Wave
import json
import sortedcontainers
import numpy as np
import warnings

#Specific string used to set the format of the annotations file
FMT_STRING = b'Construe_format_17.01'

def ann2interp(record, anns, fmt=False):
    """
    Returns an interpretation containing the observations represented in a list
    of annotations associated to a loaded MIT record. Note that only the
    *observations* field is properly set. The optional parameter *fmt* allows
    to specify if the specific Construe format for annotation files can be
    assumed. This parameter is also inferred from the first annotation in the
    list.
    """
    fmt = (fmt or len(anns) > 0 and anns[0].code is C.NOTE
                                                 and anns[0].aux == FMT_STRING)
    interp = Interpretation()
    observations = []
    RH_VALS = set(C.RHYTHM_AUX.values())
    for i in range(len(anns)):
        ann = anns[i]
        if ann.code in (C.PWAVE, C.TWAVE):
            obs = o.PWave() if ann.code == C.PWAVE else o.TWave()
            if fmt:
                beg = next(a for a in reversed(anns[:i]) if a.time < ann.time
                           and a.code == C.WFON and a.subtype == ann.code).time
                end = next(a for a in anns[i:] if a.time > ann.time
                          and a.code == C.WFOFF and a.subtype == ann.code).time
            else:
                beg = next(a for a in reversed(anns[:i]) if a.time < ann.time
                                                     and a.code == C.WFON).time
                end = next(a for a in anns[i:] if a.time > ann.time
                                                    and a.code == C.WFOFF).time
            obs.start.set(beg, beg)
            obs.end.set(end, end)
            if fmt:
                amp = json.loads(ann.aux)
                for l in amp.keys():
                    if l not in record.leads:
                        compatible = next((l2 for l2 in VALID_LEAD_NAMES
                                          if l2 in record.leads 
                                             and VALID_LEAD_NAMES[l2] == l), None)
                        if compatible is None:
                            raise ValueError('Unrecognized lead {0}'.format(l))
                    else:
                        compatible = l
                    obs.amplitude[compatible] = amp.pop(l)
            else:
                leads = (record.leads if ann.code is C.TWAVE
                         else set(K.PWAVE_LEADS) & set(record.leads))
                leads = record.leads
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
            obs.time.set(ann.time, ann.time)
            obs.tag = ann.code
            delin = json.loads(ann.aux)
            #QRS start and end is first tried to set according to delineation
            #info. If not present, it is done according to delineation
            #annotations.
            if delin:
                for l in delin.keys():
                    if l not in record.leads:
                        compatible = next((l2 for l2 in VALID_LEAD_NAMES
                                          if l2 in record.leads 
                                             and VALID_LEAD_NAMES[l2] == l), None)
                        if compatible is None:
                            raise ValueError('Unrecognized lead {0}'.format(l))
                        delin[compatible] = delin.pop(l)
                beg = ann.time + min(d[0] for d in delin.values())
                end = ann.time + max(d[-1] for d in delin.values())
            else:
                def extra_cond(a):
                    return a.subtype == C.SYSTOLE if fmt else True
                beg = next(a for a in reversed(anns[:i]) if a.code==C.WFON
                           and extra_cond(a)).time
                end = next(a for a in anns[i:] if a.code == C.WFOFF
                           and extra_cond(a)).time
            #Endpoints set
            obs.start.set(beg, beg)
            obs.end.set(end, end)
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
                for i in range(0, len(delin[lead]), 3):
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
        elif ann.code is C.RHYTHM and ann.aux in RH_VALS:
            rhclazz = next(rh for rh in C.RHYTHM_AUX
                           if C.RHYTHM_AUX[rh] == ann.aux)
            obs = rhclazz()
            obs.start.set(ann.time, ann.time)
            end = next((a.time for a in anns[i+1:] if a.code is C.RHYTHM),
                       anns[-1].time)
            obs.end.set(end, end)
            observations.append(obs)
        elif ann.code is C.ARFCT:
            obs = o.RDeflection()
            obs.time.set(ann.time, ann.time)
            observations.append(obs)
    interp.observations = sortedcontainers.SortedList(observations)
    return interp

def interp2ann(interp, btime=0, offset=0, include_format=True):
    """
    Generates a list of annotations representing the observations from
    an interpretation. The *btime* optional parameter allows to include only
    the observations after a specific time point, and *offset* allows to define
    a constant time to be added to the time point of each annotation. An
    optional format annotation of type NOTE can be included at the beginning.

    NOTE: A first annotation is included at the beginning of the list, with
    time=*offset*, to indicate that the annotations are created with the
    specific format for Construe interpretations. This format includes the
    following features (for version 17.01):
        - Beat annotations include the specific delineation information for
        each lead in a dictionary in JSON format. The keys in this dictionary
        are the lead names, and the values are a sequence of integer numbers.
        Each triple in this sequence determines a wave within the QRS complex.
        - WFON and WFOFF annotations include the type of wave they delimit in
        the *subtyp* field. QRS complexes are described by the SYSTOLE code,
        while P and T waves limits have the PWAVE or TWAVE code, respectively.
        - PWAVE and TWAVE annotations include the amplitude of each lead, in
        a dictionary in JSON format in the AUX field.
    """
    annots = sortedcontainers.SortedList()
    if include_format:
        fmtcode = MITAnnotation.MITAnnotation()
        fmtcode.code = C.NOTE
        fmtcode.time = int(offset)
        fmtcode.aux = FMT_STRING
        annots.add(fmtcode)
    beats = list(interp.get_observations(o.QRS,
                                         filt=lambda q: q.time.start >= btime))
    #We get the beat observations in the best explanation branch.
    for beat in beats:
        #We tag all beats as normal, and we include the delineation. The
        #delineation on each lead is included as a json string in the peak
        #annotation.
        beg = MITAnnotation.MITAnnotation()
        beg.code = C.WFON
        beg.subtype = C.SYSTOLE
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
        end.subtype = C.SYSTOLE
        end.time = int(offset + beat.lateend)
        annots.add(beg)
        annots.add(peak)
        annots.add(end)
    #P and T wave annotations
    pstart = beats[0].earlystart - ms2sp(400) if beats else 0
    tend = beats[-1].lateend + ms2sp(400) if beats else 0
    for wtype in (o.PWave, o.TWave):
        for wave in interp.get_observations(wtype, pstart, tend):
            if wave.earlystart >= btime:
                code = C.PWAVE if wtype is o.PWave else C.TWAVE
                beg = MITAnnotation.MITAnnotation()
                beg.code = C.WFON
                beg.subtype = code
                beg.time = int(offset + wave.earlystart)
                end = MITAnnotation.MITAnnotation()
                end.code = C.WFOFF
                end.subtype = code
                end.time = int(offset + wave.lateend)
                peak = MITAnnotation.MITAnnotation()
                peak.code = code
                peak.time = int((end.time+beg.time)/2.)
                peak.aux = json.dumps(wave.amplitude)
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
