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
import construe.knowledge.abstraction_patterns as ap
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
                                        and a.time <= end.time and
                                        MITAnnotation.is_qrs_annotation(a))
                if nbeats < 7:
                    continue
            if ann.aux == '(T':
                end = next((a for a in annots if a.time > ann.time
                           and a.code in (ECGCodes.RHYTHM, ECGCodes.VFON)),
                                                                annots[-1])
                nbeats = searching.ilen(a for a in annots if a.time >= ann.time
                                        and a.time <= end.time and
                                        MITAnnotation.is_qrs_annotation(a))
                if nbeats < 7:
                    continue
            #############################################################
            # Pauses and missed beats are replaced by bradycardias (for #
            # consistency with the reference annotations).              #
            #############################################################
            if ann.aux in ('(BK', 'P'):
                ann.aux = '(SBR'
            if ann.aux not in ('(EXT', '(CPT'):
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
    i = 1
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


def _clean_artifacts_redundancy(annots):
    """
    Removes those artifact annotations that are close to a QRS annotation,  as
    well as redundant rhythm annotations.
    """
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
    #Redundant rhythms
    i = 1
    while i < len(annots):
        if annots[i].code is ECGCodes.RHYTHM:
            prev = next((a for a in reversed(annots[:i])
                                           if a.code is ECGCodes.RHYTHM), None)
            if prev is not None and prev.aux == annots[i].aux:
                annots.pop(i)
            else:
                i += 1
        else:
            i += 1
    return annots

def process_record_conduction(path, ann='atr', fr_len=512000, initial_pos=0,
                              final_pos=np.inf, exclude_pwaves=False,
                              exclude_twaves=False, verbose=True):
    """
    This function performs an interpretation in the conduction abstraction
    level of a given MIT-BIH formatted record, using as initial evidence an
    external set of annotations. The result is a delineation of the P waves,
    QRS complex, and T waves of each heartbeat in the initial evidence
    annotator. The interpretation is splitted in independent fragments of
    configurable length.

    Parameters
    ----------
    path:
        Complete name of the record to be processed (without any extension)
    ann:
        Annotator used to obtain the initial evidence (default: 'atr')
    fr_len:
        Length in samples of each independently interpreted fragment.
    initial_pos:
        Time position (in samples) where the interpretation should begin.
    final_pos:
        Time position (in samples) where the interpretation should finish.
    exclude_pwaves:
        Flag to avoid P-wave searching.
    exclude_twaves:
        Flag to avoid T-wave searching.
    verbose:
        Boolean flag. If active, the algorithm will print to standard output
        the fragment being interpreted.

    Returns
    -------
    out:
        sortedlist of annotations resulting from the interpretation, including
        only segmentation annnotations.
    """
    if fr_len > final_pos-initial_pos:
        fr_len = int(final_pos-initial_pos)
    if fr_len % IN._STEP != 0:
        fr_len += IN._STEP - (fr_len % IN._STEP)
        warnings.warn('Fragment length is not multiple of {0}. '
                      'Adjusted to {1}'.format(IN._STEP, fr_len))
    #Knowledge base configuration
    prev_knowledge = ap.KNOWLEDGE
    curr_knowledge = ap.SEGMENTATION_KNOWLEDGE[:]
    if exclude_twaves:
        curr_knowledge.remove(ap.TWAVE_PATTERN)
        curr_knowledge.remove(ap.PWAVE_PATTERN)
    elif exclude_pwaves:
        curr_knowledge.remove(ap.PWAVE_PATTERN)
    ap.set_knowledge_base(curr_knowledge)
    #Input configuration
    IN.set_record(path, ann)
    IN.set_duration(fr_len)
    IN.set_tfactor(1e20)
    #Annotations buffer
    annots = sortedcontainers.SortedList()
    pos = initial_pos
    ictr = Interpretation.counter
    while pos < min(IN.get_record_length(), final_pos):
        if verbose:
            print('Processing fragment {0}:{1}'.format(pos, pos+fr_len))
        #Input start
        IN.reset()
        IN.set_offset(pos)
        IN.start()
        while IN.BUF.get_status() == IN.BUF.Status.ACQUIRING:
            IN.get_more_evidence()

        #Reasoning and interpretation
        root = node = Interpretation()
        try:
            root.focus.push(next(IN.BUF.get_observations()), None)
        except (StopIteration, ValueError):
            pos += fr_len
            continue
        successors = {node:reasoning.firm_succ(node)}
        t0 = time.time()
        ########################
        ### Greedy searching ###
        ########################
        while True:
            try:
                node = next(successors[node])
                if node not in successors:
                    successors[node] = reasoning.firm_succ(node)
            except StopIteration:
                #If the focus contains a top-level hypothesis, then there is
                #no more evidence to explain.
                if isinstance(node.focus.top[0], o.CardiacCycle):
                    break
                else:
                    #In other case, we perform a backtracking operation
                    node = node.parent
            except KeyError:
                node = root
                break
        best_explanation = node
        best_explanation.recover_all()
        #End of reasoning
        #We generate and add the annotations for the current fragment
        newanns = interp2ann(best_explanation, 0, pos, pos == initial_pos)
        annots.update(newanns)
        #We go to the next fragment after deleting the current used branch and
        #clearing the reasoning cache.
        del root
        reasoning.reset()
        if verbose:
            idur = time.time() - t0
            print('Fragment finished in {0:.03f} seconds. Real-time factor: '
                  '{1:.03f}. Created {2} interpretations.'.format(idur,
                      sp2ms(IN.get_acquisition_point())/(idur*1000.),
                      Interpretation.counter-ictr))
        ictr = Interpretation.counter
        #We introduce an overlapping between consecutive fragments
        pos += fr_len
    #Restore the previous knowledge base
    ap.set_knowledge_base(prev_knowledge)
    return _clean_artifacts_redundancy(annots)


def process_record_rhythm(path, ann='atr', tfactor=1.0, fr_len=23040,
                          fr_overlap=1080, fr_tlimit=np.inf, min_delay=2560,
                          max_delay=20.0, kfactor=12, initial_pos=0,
                          final_pos=np.inf, exclude_pwaves=False,
                          exclude_twaves=False, verbose=True):
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
        Time factor to control the speed of the input signal. For example,
        if tfactor = 2.0 two seconds of new signal are added to the signal
        buffer each real second. Of course this can only be greater than 1 in
        offline interpretations.
    fr_len:
        Length in samples of each independently interpreted fragment.
    fr_overlap:
        Lenght in samples of the overlapping between consecutive fragments, to
        prevent loss of information.
    fr_tlimit:
        Time limit **in seconds** for the interpretation of each fragment.
    min_delay:
        Minimum delay **in samples** between the acquisition time and the last
        interpretation time.
    max_delay:
        Maximum delay **in seconds**, that the interpretation can be without
        moving forward. If this threshold is exceeded, the searching process
        is pruned.
    kfactor:
        Exploration factor. It is the number of interpretations expanded in
        each searching cycle.
    initial_pos:
        Time position (in samples) where the interpretation should begin.
    final_pos:
        Time position (in samples) where the interpretation should finish.
    exclude_pwaves:
        Flag to avoid P-wave searching.
    exclude_twaves:
        Flag to avoid T-wave searching.
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
    #Knowledge base configuration
    prev_knowledge = ap.KNOWLEDGE
    curr_knowledge = ap.RHYTHM_KNOWLEDGE[:]
    if exclude_twaves:
        curr_knowledge.remove(ap.TWAVE_PATTERN)
        curr_knowledge.remove(ap.PWAVE_PATTERN)
    elif exclude_pwaves:
        curr_knowledge.remove(ap.PWAVE_PATTERN)
    ap.set_knowledge_base(curr_knowledge)
    #Input configuration
    IN.set_record(path, ann)
    IN.set_duration(fr_len)
    IN.set_tfactor(tfactor)
    #Annotations buffer
    annots = sortedcontainers.SortedList()
    pos = initial_pos
    ictr = Interpretation.counter
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
        t0 = time.time()
        ltime = (cntr.last_time, t0)
        while cntr.best is None:
            IN.get_more_evidence()
            acq_time = IN.get_acquisition_point()
            def filt(node):
                """Filter function to enforce *min_delay*"""
                if IN.BUF.get_status() is IN.BUF.Status.ACQUIRING:
                    return acq_time + node[0][2] >= min_delay
                else:
                    return True
            cntr.step(filt)
            t = time.time()
            if cntr.last_time > ltime[0]:
                ltime = (cntr.last_time, t)
            if t-ltime[1] > max_delay:
                cntr.prune()
            if t-t0 > fr_tlimit:
                cntr.best = (min(cntr.open) if len(cntr.open) > 0
                                            else min(cntr.closed))
        best_explanation = cntr.best.node
        best_explanation.recover_all()
        #End of reasoning
        #We resolve possible conflicts on joining two fragments, selecting the
        #interpretation higher coverage.
        btime = _merge_annots(annots, best_explanation, pos) if annots else 0
        #We generate and add the annotations for the current fragment
        newanns = interp2ann(best_explanation, btime, pos, pos == initial_pos)
        annots.update(newanns)
        #We go to the next fragment after deleting the current used branch and
        #clearing the reasoning cache.
        del cntr
        del root
        reasoning.reset()
        if verbose:
            idur = time.time() - t0
            print('Fragment finished in {0:.03f} seconds. Real-time factor: '
                  '{1:.03f}. Created {2} interpretations.'.format(idur,
                      sp2ms(acq_time)/(idur*1000.),
                      Interpretation.counter-ictr))
        ictr = Interpretation.counter
        #We introduce an overlapping between consecutive fragments
        pos += fr_len - fr_overlap
    #Restore the previous knowledge base
    ap.set_knowledge_base(prev_knowledge)
    return _clean_artifacts_redundancy(annots)

if __name__ == '__main__':
    pass
