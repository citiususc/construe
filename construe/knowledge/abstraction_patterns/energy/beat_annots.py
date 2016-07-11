# -*- coding: utf-8 -*-
# pylint: disable-msg=
"""
Created on Mon Jun 16 12:02:52 2014

This module defines a purely deductive abstraction pattern to discover new
beat annotations using the *gqrs* application.

@author: T. Teijeiro
"""

from construe.model.automata import PatternAutomata
from construe.model import Interval as Iv
from construe.utils.MIT.MITAnnotation import read_annotations, MITAnnotation
from construe.utils.MIT import get_leads
from construe.model.constraint_network import verify
import construe.acquisition.record_acquisition as IN
import subprocess
import blist
import os
import numpy as np
import operator
import construe.knowledge.observables as o
import construe.knowledge.constants as C

ANNOTS = None

def reset():
    """
    Resets the static set of annotations, allowing to re-run the external QRS
    detection procedure.
    """
    global ANNOTS
    ANNOTS = None

def _merge_annots(annotators):
    """
    Merges several sets of beat annotators, by taking the union of all of them,
    and in the case that more than one annotator produce a result in the same
    area (within a predefined time window), then the annotation with highest
    amplitude is taken.
    """
    result = []
    dummy = MITAnnotation()
    dummy.time = np.inf
    iterators = [iter(annotator) for annotator in annotators]
    head = [next(iterator, dummy) for iterator in iterators]
    while True:
        nxt = min(head)
        if nxt is dummy:
            return result
        idx = head.index(nxt)
        head[idx] = next(iterators[idx], dummy)
        if result and nxt.time - result[-1].time <= C.BEATANN_MIN_DIST:
            result[-1] = max(nxt, result[-1],
                                         key = lambda ann:(ann.num, -ann.time))
        else:
            result.append(nxt)

def _load_annots():
    """
    Obtains the beat annotations of the current record using the gqrs
    application with a lower threshold, using some properties of the input
    system module.
    """
    global ANNOTS
    annotator = 'gqrs01'
    refann = blist.sortedlist(IN._ANNOTS)
    ANNOTS = blist.sortedlist()
    rec = IN.get_record_name()
    leads = get_leads(rec)
    annotators = []
    for lead in leads:
        command = ['gqrs', '-r', rec, '-outputName', annotator,
                                                       '-m', '0.1', '-s', lead]
        subprocess.check_call(command)
        annpath = rec + '.' + annotator
        annotators.append(read_annotations(annpath))
    for ann in _merge_annots(annotators):
        idx = refann.bisect_left(ann)
        #First we check that the annotation is not in the base evidence.
        if (ann.time-refann[idx-1].time > C.BEATANN_MIN_DIST and
                             (idx >= len(refann) or
                              refann[idx].time-ann.time > C.BEATANN_MIN_DIST)):
            #And now we select the most promising one between all leads.
            ANNOTS.add(ann)
    os.remove(annpath)

def _beatann_gconst(pattern, _):
    """
    General constraints of the beat annotation pattern, that simply looks in
    the global list for an appropriate annotation.
    """
    if ANNOTS is None:
        _load_annots()
    leads = IN.SIG.get_available_leads()
    #We find all the annotations in the given interval.
    beatann = pattern.hypothesis
    beg = min(int(beatann.earlystart), IN.get_acquisition_point()) + IN._OFFSET
    end = min(int(beatann.lateend), IN.get_acquisition_point()) + IN._OFFSET
    dummy = MITAnnotation()
    dummy.time = beg
    bidx = ANNOTS.bisect_left(dummy)
    dummy.time = end
    eidx = ANNOTS.bisect_right(dummy)
    verify(eidx > bidx)
    selected = max(ANNOTS[bidx:eidx], key= operator.attrgetter('num'))
    time = selected.time - IN._OFFSET
    beatann.time.value = Iv(time, time)
    beatann.start.value = Iv(time, time)
    beatann.end.value = Iv(time, time)
    beatann.level = {lead : 127 for lead in leads}
    beatann.level[leads[selected.chan]] = 127 - selected.num


BEATANN_PATTERN = PatternAutomata()
BEATANN_PATTERN.name = 'Beat Annotation'
BEATANN_PATTERN.Hypothesis = o.BeatAnn
BEATANN_PATTERN.add_transition(0, 1, gconst=_beatann_gconst)
BEATANN_PATTERN.final_states.add(1)
BEATANN_PATTERN.freeze()