# -*- coding: utf-8 -*-
# pylint: disable-msg=
"""
Created on Wed May  6 11:27:07 2015

This module contains a general function to obtain the interpretation of an ECG
record in a fragmented way. It can be used from any script that performs an
interpretation of a set of records or a full database.

@author: T. Teijeiro
"""

import construe.utils.MIT.MITAnnotation as MITAnnotation
import construe.utils.MIT.ECGCodes as ECGCodes
import construe.acquisition.record_acquisition as IN
import construe.inference.searching as searching
import construe.inference.reasoning as reasoning
import construe.knowledge.observables as o
from construe.model.interpretation import Interpretation
import time
import blist
import numpy as np
from construe.utils.MIT.interp2annots import interp2ann
from construe.utils.units_helper import (msec2samples as ms2sp,
                                            samples2msec as sp2ms)

def _merge_annots(annlst, interp, reftime):
    """
    Merges an annotations list and an interpretation by selecting on the
    overlap interval the sequence with highest coverage.
    """
    beg = next((ob.earlystart+reftime for ob in
                interp.get_observations(o.Cardiac_Rhythm)), np.inf) - ms2sp(150)
    #Ventricular flutter episodes change the reference point.
    vflut = next((a for a in reversed(annlst) if a.code is ECGCodes.VFOFF
                  and a.time >= beg), None)
    if vflut is not None:
        beg = vflut.time + 1
    bidx = next((i for i in xrange(len(annlst)) if annlst[i].time >= beg),
                len(annlst))
    end = next((a.time for a in reversed(annlst)
                if a.code is ECGCodes.RHYTHM and a.aux == ')'), annlst[-1].time)
    #First we calculate the possible 'join points' of the two sequences.
    jpts = (set(a.time for a in annlst[bidx:]
                if a.time <= end and a.code is ECGCodes.RHYTHM) &
            set(reftime+r.earlystart for r in interp.get_observations(
                                                        o.Cardiac_Rhythm,
                 filt=lambda rh: beg-reftime <= rh.earlystart <= end-reftime)))
    #If there are no join points, we give priority to the interpretation.
    if not jpts:
        jpt = beg
    else:
        #We select the join point with highest coverage.
        score = {}
        for jpt in jpts:
            score[jpt] = (len([a for a in annlst[bidx:] if a.time <= jpt and
                               (a.code in (ECGCodes.TWAVE, ECGCodes.PWAVE) or
                                MITAnnotation.is_qrs_annotation(a))]) +
                          len(list(interp.get_observations((o.QRS, o.PWave,
                                                            o.TWave),
                                                           jpt-reftime,
                                                           end-reftime))))
        jpt = max(jpts, key=lambda pt: score[pt])
    #We remove the discarded annotations (those after the selected join point),
    #ensuring the WFON/WFOFF pairs are consistent.
    offsets = 0
    while annlst and annlst[-1].time >= jpt:
        if annlst[-1].code is ECGCodes.WFOFF:
            offsets += 1
        elif annlst[-1].code is ECGCodes.WFON:
            offsets -= 1
        annlst.pop()
    while offsets > 0:
        annlst.pop(next(i for i in xrange(len(annlst)-1, -1, -1)
                        if annlst[i].code is ECGCodes.WFON))
        offsets -= 1
    return jpt-reftime


def process_record(path, ann='atr', tfactor=1.0, fr_len=23040, fr_overlap=1080,
                   min_delay=2560, max_delay=20.0, kfactor=12, initial_pos=0,
                   verbose=True):
    """
    This function performs a complete interpretation of a given MIT-BIH
    formatted record, using as initial evidence an external set of annotations.
    The interpretation is splitted in independent fragments of configurable
    length. The exploration factor is also configurable.

    Parameters
    ----------
    path:
        Complete name of the record to be processed (without any extension)
    ann:
        Annotator used to obtain the initial evidence (default: 'atr')
    tfactor:
        Time factor to control de duration of the interpretation. For example,
        if tfactor = 2.0 the interpretation can be working for two times the
        real duration of the interpreted record. **Note: This factor cannot
        be guaranteed**.
    fr_len:
        Length in samples of each independently interpreted fragment.
    fr_overlap:
        Lenght in samples of the overlapping between consecutive fragments, to
        prevent loss of information.
    min_delay:
        Minimum delay in samples between the acquisition time and the last
        interpretation time.
    max_delay:
        Maximum delay **in seconds**, that the interpretation can be without
        moving forward. If this threshold is exceeded, the searching process
        is pruned
    kfactor:
        Exploration factor. It is the number of interpretations expanded in
        each searching cycle.
    verbose:
        Boolean flag. If active, the algorithm will print to standard output
        the fragment being interpreted.

    Returns
    -------
    out:
        sortedlist of annotations resulting from the interpretation, including
        segmentation and rhythm annnotations.
    """
    #Input configuration
    IN.set_record(path, ann)
    IN.set_duration(fr_len)
    IN.set_tfactor(tfactor)
    #Annotations buffer
    annots = blist.sortedlist()
    pos = initial_pos
    while pos < IN.get_record_length():
        if verbose:
            print('Processing fragment {0}:{1}'.format(pos, pos+fr_len))
        #Input start
        IN.reset()
        IN.set_offset(pos)
        IN.start()
        time.sleep(sp2ms(min_delay)/(1000.0*tfactor))
        IN.get_more_evidence()

        #Reasoning and interpretation
        root = Interpretation()
        try:
            root.focus.append(next(IN.BUF.get_observations()))
            cntr = searching.Construe(root, kfactor)
        except (StopIteration, ValueError):
            pos += fr_len - fr_overlap
            continue
        ltime = (cntr.last_time, time.time())
        while cntr.best is None:
            IN.get_more_evidence()
            acq_time = IN.get_acquisition_point()
            filt = ((lambda n: acq_time + n[0][2] >= min_delay)
                    if IN.BUF.get_status() is IN.BUF.Status.ACQUIRING
                    else (lambda _: True))
            cntr.step(filt)
            if cntr.last_time > ltime[0]:
                ltime = (cntr.last_time, time.time())
            if time.time()-ltime[1] > max_delay:
                cntr.prune()
        best_explanation = cntr.best.node
        best_explanation.recover_old()
        #End of reasoning
        #We resolve possible conflicts on joining two fragments, selecting the
        #interpretation higher coverage.
        btime = _merge_annots(annots, best_explanation, pos) if annots else 0
        #We generate and add the annotations for the current fragment
        newanns = interp2ann(best_explanation, btime, pos)
        annots.update(newanns)
        #We go to the next fragment after deleting the current used branch and
        #clearing the reasoning cache.
        del cntr
        del root
        reasoning.reset()
        #We introduce an overlapping between consecutive fragments
        pos += fr_len - fr_overlap
    return annots
