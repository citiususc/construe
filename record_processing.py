#!/usr/bin/env python
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
import sortedcontainers
import warnings
import numpy as np
from construe.knowledge.abstraction_patterns.rhythm.afib import (
                                                           is_afib_rhythm_lian)
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

def _standardize_rhythm_annots(annots):
    """
    Standardizes a set of annotations obtained from the interpretation
    procedure to make them compatible with the criteria applied in the
    MIT-BIH Arrhythmia database in the labeling of rhythms.
    """
    dest = sortedcontainers.SortedList()
    for ann in annots:
        code = ann.code
        if code in (ECGCodes.RHYTHM, ECGCodes.VFON):
            #TODO remove this if not necessary
            if code is ECGCodes.VFON:
                newann = MITAnnotation.MITAnnotation()
                newann.code = ECGCodes.RHYTHM
                newann.aux = '(VFL'
                newann.time = ann.time
                dest.add(newann)
            ############################################################
            #For convention with original annotations, we only admit   #
            #bigeminies with more than two pairs, and trigeminies with #
            #more than two triplets,                                   #
            ############################################################
            if ann.aux == '(B':
                end = next((a for a in annots if a.time > ann.time
                           and a.code in (ECGCodes.RHYTHM, ECGCodes.VFON)),
                                                                annots[-1])
                nbeats = searching.ilen(a for a in annots if a.time >= ann.time
                                        and a.time<=end.time and
                                        MITAnnotation.is_qrs_annotation(a))
                if nbeats < 7:
                    continue
            if ann.aux == '(T':
                end = next((a for a in annots if a.time > ann.time
                           and a.code in (ECGCodes.RHYTHM, ECGCodes.VFON)),
                                                                annots[-1])
                nbeats = searching.ilen(a for a in annots if a.time >= ann.time
                                        and a.time<=end.time and
                                        MITAnnotation.is_qrs_annotation(a))
                if nbeats < 7:
                    continue
            #############################################################
            # Pauses and missed beats are replaced by bradycardias (for #
            # consistency with the reference annotations).              #
            #############################################################
            if ann.aux in ('(BK', 'P'):
                ann.aux = '(SBR'
            if ann.aux not in ('(EXT','(CPT'):
                prev = next((a for a in reversed(dest)
                                       if a.code is ECGCodes.RHYTHM), None)
                if prev is None or prev.aux != ann.aux:
                    dest.add(ann)
        else:
            dest.add(ann)
    #################################
    #Atrial fibrillation correction #
    #################################
    iterator = iter(dest)
    afibtime = 0
    while True:
        try:
            start = next(a.time for a in iterator
                         if a.code == ECGCodes.RHYTHM and a.aux == '(AFIB')
            end = next((a.time for a in iterator
                              if a.code == ECGCodes.RHYTHM), dest[-1].time)
            afibtime += end-start
        except StopIteration:
            break
    #If more than 1/20 of the time of atrial fibrillation...
    if annots and afibtime > (annots[-1].time-annots[0].time)/20.0:
        iterator = iter(dest)
        rhythms = ('(N', '(SVTA')
        start = next((a for a in iterator if a.code == ECGCodes.RHYTHM
                                               and a.aux in rhythms), None)
        while start is not None:
            end = next((a for a in iterator if a.code == ECGCodes.RHYTHM),
                                                                  dest[-1])
            #All normal rhythms that satisfy the Lian method to identify
            #afib by rhythm are now considered afib. We also check the
            #method considering alternate RRs to avoid false positives with
            #bigeminies.
            fragment = dest[dest.bisect_left(start):dest.bisect_right(end)]
            rrs = np.diff([a.time for a in fragment
                                        if MITAnnotation.is_qrs_annotation(a)])
            if (is_afib_rhythm_lian(rrs) and
                            is_afib_rhythm_lian(rrs[0::2]) and
                                           is_afib_rhythm_lian(rrs[1::2])):
                start.aux = '(AFIB'
            #Next rhythm
            start = (end if end.aux in rhythms else
                     next((a for a in iterator
                                        if a.code == ECGCodes.RHYTHM
                                              and a.aux in rhythms), None))
    ##############################
    #Paced rhythm identification #
    ##############################
    #To consider the presence of paced rhythms in a record, we require at
    #least a mean of one paced beat each 10 seconds.
    pacedrec = sum(1 for a in dest if a.code == ECGCodes.PACE) > 180
    if pacedrec:
        iterator = iter(dest)
        rhythms = ('(AFIB', '(N', '(SBR', '(SVTA')
        start = next((a for a in iterator if a.code == ECGCodes.RHYTHM
                                               and a.aux in rhythms), None)
        while start is not None:
            end = next((a for a in iterator if a.code == ECGCodes.RHYTHM),
                                                                  dest[-1])
            #If there are paced beats in a rhythm fragment, the full
            #rhythm is identified as paced.
            if any([start.time < a.time < end.time
                    and a.code == ECGCodes.PACE
                        for a in dest[dest.index(start):dest.index(end)]]):
                start.aux = '(P'
            #Next rhythm
            start = (end if end.aux in rhythms else
                     next((a for a in iterator
                                        if a.code == ECGCodes.RHYTHM
                                              and a.aux in rhythms), None))
    #########################################
    # Redundant rhythm description removing #
    #########################################
    i=1
    while i < len(dest):
        if dest[i].code is ECGCodes.RHYTHM:
            prev = next((a for a in reversed(dest[:i])
                                       if a.code is ECGCodes.RHYTHM), None)
            if prev is not None and prev.aux == dest[i].aux:
                dest.pop(i)
            else:
                i += 1
        else:
            i += 1
    return dest


def _clean_artifacts(annots):
    """Removes those artifact annotations that are close to a QRS annotation"""
    DISTANCE = ms2sp(150)
    banns = [a for a in annots if MITAnnotation.is_qrs_annotation(a) or
                                                      a.code == ECGCodes.ARFCT]
    i = 0
    while i < len(banns):
        if (banns[i].code == ECGCodes.ARFCT and
                ((i > 0 and
                    banns[i].time-banns[i-1].time < DISTANCE) or
                (i < len(banns)-1 and
                    banns[i+1].time-banns[i].time < DISTANCE))):
            #We cannot use 'remove' due to a bug in SortedList.
            j = annots.bisect_left(banns[i])
            while annots[j] is not banns[i]:
                j += 1
            annots.pop(j)
            banns.pop(i)
        else:
            i += 1
    return annots


def process_record(path, ann='atr', tfactor=1.0, fr_len=23040, fr_overlap=1080,
                   min_delay=2560, max_delay=20.0, kfactor=12, initial_pos=0,
                   final_pos=np.inf, verbose=True):
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
    initial_pos:
        Time position (in samples) where the interpretation should begin.
    final_pos:
        Time position (in samples) where the interpretation should finish.
    verbose:
        Boolean flag. If active, the algorithm will print to standard output
        the fragment being interpreted.

    Returns
    -------
    out:
        sortedlist of annotations resulting from the interpretation, including
        segmentation and rhythm annnotations.
    """
    if fr_len > final_pos-initial_pos:
        fr_len = int(final_pos-initial_pos)
        fr_overlap = 0
    if fr_len % IN._STEP != 0:
        fr_len += IN._STEP - (fr_len % IN._STEP)
        warnings.warn('Fragment length is not multiple of {0}. '
                      'Adjusted to {1}'.format(IN._STEP, fr_len))
    #Input configuration
    IN.set_record(path, ann)
    IN.set_duration(fr_len)
    IN.set_tfactor(tfactor)
    #Annotations buffer
    annots = sortedcontainers.SortedList()
    pos = initial_pos
    while pos < min(IN.get_record_length(), final_pos):
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
            root.focus.push(next(IN.BUF.get_observations()), None)
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
        best_explanation.recover_all()
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

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=
        'Interprets a MIT-BIH ECG record, generating as a result a set of '
        'annotations.')
    parser.add_argument('-r', metavar='record', required=True,
                        help='Name of the record to be processed')
    parser.add_argument('-a', metavar='ann', default='qrs',
                        help= ('Annotator used to set the initial evidence '
                              '(default: qrs)'))
    parser.add_argument('-o', metavar='oann', default='iqrs',
                        help= ('Save annotations as annotator oann '
                               '(default: iqrs)'))
    parser.add_argument('-f', metavar='init', default=0, type=int,
                        help= ('Begin the interpretation at the "init" time, '
                               'in samples'))
    parser.add_argument('-t', metavar='stop', default=np.inf, type=float,
                        help= ('Stop the interpretation at the "stop" time, '
                               'in samples'))
    parser.add_argument('-l', metavar='length', default=23040, type=int,
                        help= ('Length in samples of each independently '
                               'interpreted fragment. It has to be multiple '
                               'of 256. Default:23040.'))
    parser.add_argument('--overl', default=1080, type=int,
                        help= ('Length in samples of the overlapping between '
                               'consecutive fragments, to prevent loss of '
                               'information. Default: 1080'))
    parser.add_argument('--tfactor', default=1.0, type=float,
                        help= ('Time factor to control de duration of the '
                               'interpretation. For example, if --tfactor = '
                               '2.0 the interpretation can be working for two '
                               'times the real duration of the interpreted '
                               'record. Note: This factor cannot be '
                               'guaranteed. Default: 1.0'))
    parser.add_argument('-d', metavar='min_delay', default=2560, type=int,
                        help= ('Minimum delay in samples between the '
                               'acquisition time and the last interpretation '
                               'time. Default: 1080'))
    parser.add_argument('-D', metavar='max_delay', default=20.0, type=float,
                        help= ('Maximum delay in seconds that the '
                               'interpretation can be without moving forward. '
                               'If this threshold is exceeded, the searching '
                               'process is pruned. Default: 20.0'))
    parser.add_argument('-k', default=12, type=int,
                        help= ('Exploration factor. It is the number of '
                               'interpretations expanded in each searching '
                               'cycle. Default: 12'))
    parser.add_argument('-v', action = 'store_true',
                        help= ('Verbose mode. The algorithm will print to '
                               'standard output the fragment being '
                               'interpreted.'))
    parser.add_argument('--no-merge', action = 'store_true',
                        help= ('Avoids the use of a branch-merging strategy for'
                               ' interpretation exploration.'))
    args = parser.parse_args()
    reasoning.MERGE_STRATEGY = not args.no_merge
    result = _clean_artifacts(process_record(args.r, args.a, args.tfactor,
                                             args.l, args.overl, args.d,
                                             args.D, args.k, args.f, args.t,
                                             args.v))
    MITAnnotation.save_annotations(result, args.r + '.' + args.o)
    print('Record ' + args.r + ' succesfully processed')
