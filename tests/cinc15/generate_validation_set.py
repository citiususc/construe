# -*- coding: utf-8 -*-
# pylint: disable-msg=
"""
Created on Mon Feb 23 09:07:59 2015

This script generates the classification answers for each record in the
training set, based on the abductive interpretation results.

@author: T. Teijeiro
"""

import csv
import numpy as np
import collections
import json
import bisect
import construe.utils.MIT as MIT
from construe.acquisition.signal_buffer import VALID_LEAD_NAMES
from construe.model import Interval as Iv
from construe.utils.clustering.xcorr_similarity import xcorr_full
from construe.utils.MIT import ECGCodes
from construe.utils.units_helper import (msec2samples as ms2sp,
                                            samples2msec as sp2ms,
                                            msec2bpm as ms2bpm,
                                            set_sampling_freq)
set_sampling_freq(250.0)

#Directories where the original records and interpretation results are stored
DB_DIR = '/home/local/tomas.teijeiro/cinc_challenge15/training/'
INTERP_DIR = '/home/local/tomas.teijeiro/cinc_challenge15/interp_results/'
#Alarm types
ASYST = 'Asystole'
BRAD = 'Bradycardia'
TACH = 'Tachycardia'
VFLUT = 'Ventricular_Flutter_Fib'
VTACH = 'Ventricular_Tachycardia'

#Structure for clustering
SIG = collections.namedtuple('Signal', ['sig'])

def _same_cluster(sig1, sig2):
    """
    Checks if two QRS complexes can be clustered together. Used to
    distinguish ventricular beats in the ventricular tachycardia alarms
    """
    cleads = set(sig1.keys()).intersection(set(sig2.keys()))
    corrs = []
    if not cleads:
        return False
    for lead in cleads:
        arr1, arr2 = sig1[lead].sig, sig2[lead].sig
        if len(arr2) > len(arr1):
            arr1, arr2 = arr2, arr1
        if len(arr1)/float(len(arr2)) > 1.5:
            return False
        corr, _ = xcorr_full(arr1, arr2)
        corrs.append(corr)
    corrs = np.array(corrs)
    return np.all(corrs > 0.8)


def eval_asyst(annotations, _):
    """Evaluates the asystole presence"""
    def check_vf(start, end):
        """Obtains the flutter waves present in a given interval"""
        return [a for a in annotations if start < a.time < end
                                                  and a.code is ECGCodes.FLWAV]

    lth, uth, dth = ms2sp((4*60+45)*1000), ms2sp(5*60*1000), ms2sp(3500)
    beats = np.array([b.time for b in annotations if MIT.is_qrs_annotation(b)
                                                     and lth <= b.time <= uth])
    if len(beats) < 2:
        return not check_vf(lth, uth)
    if uth-beats[-1] > dth:
        return not check_vf(beats[-1], uth)
    rrs = np.diff(beats)
    for i in range(len(rrs)):
        if rrs[i] > dth:
            if not check_vf(beats[i], beats[i+1]):
                return True
    return False


def eval_brad(annotations, _):
    """Evaluates the bradycardia presence"""
    lth, uth = ms2sp((4*60+45)*1000), ms2sp(5*60*1000)
    beats = np.array([b.time for b in annotations if MIT.is_qrs_annotation(b)])
    variability = np.std(np.diff(beats))
    #The default threshold is 40 bpm, but if the rhythm shows high variability,
    #we relax such threshold to 45 bpm.
    thres = 45 if variability > ms2sp(200) else 40
    lidx = bisect.bisect_left(beats, lth)
    uidx = bisect.bisect_right(beats, uth)
    for i in range(lidx, uidx-4):
        bpm = int(ms2bpm(sp2ms(beats[i+4]-beats[i])/4.0))
        if bpm <= thres:
            return True
    return False

def eval_tach(annotations, _):
    """Evaluates the tachycardia presence"""
    lth, uth = ms2sp((4*60+30)*1000), ms2sp(5*60*1000)
    beats = np.array([b.time for b in annotations if MIT.is_qrs_annotation(b)
                                                     and lth <= b.time <= uth])
    for i in range(len(beats)-16):
        if ms2bpm(sp2ms(beats[i+16]-beats[i])/16.0) > 120:
            return True
    return False

def eval_vflut(anns, _):
    """Evaluates the ventricular flutter presence"""
    lth, uth, dth = ms2sp((4*60+45)*1000), ms2sp(5*60*1000), ms2sp(3500)
    #We remove separations between consecutive flutter fragments
    i = 0
    while i < len(anns):
        if anns[i].code is ECGCodes.VFOFF:
            onset = next((j for j in range(i, len(anns))
                                       if anns[j].code is ECGCodes.VFON), None)
            if onset is not None and anns[i].time == anns[onset].time:
                anns.pop(onset)
                anns.pop(i)
                i -= 1
        i += 1
    vflim = (a for a in anns if a.code in (ECGCodes.VFON, ECGCodes.VFOFF))
    vfluts = []
    while True:
        try:
            beg = next(vflim)
            end = next(vflim)
            vfluts.append(Iv(beg.time, end.time))
        except StopIteration:
            break
    #If the record shows many flutter fragments, we simply check some flutter
    #waves in the last 15 seconds.
    if sum(fl.length for fl in vfluts) > ms2sp(20000):
        vfw = [a.time for a in anns if a.code is ECGCodes.FLWAV
                                                      and lth <= a.time <= uth]
        return len(vfw) > 5
    interv = Iv(lth, uth)
    return any([interv.intersection(vflut).length > dth for vflut in vfluts])


def eval_vtach(anns, rec):
    """Evaluates the ventricular tachycardia presence"""
    lth, uth = ms2sp((4*60+45)*1000), ms2sp(5*60*1000)
    #First we perform clustering on all beats
    qrsdur = {}
    clusters = []
    for ann in anns:
        if MIT.is_qrs_annotation(ann):
            delin = json.loads(ann.aux)
            qrs = {}
            for lead in delin:
                sidx = rec.leads.index(lead)
                qon = ann.time + delin[lead][0]
                qoff = ann.time + delin[lead][-1]
                qrs[lead] = SIG(sig=rec.signal[sidx][qon:qoff+1])
            qrsdur[ann] = max(len(s.sig) for s in qrs.values())
            clustered = False
            for cluster in clusters:
                if _same_cluster(cluster[0], qrs):
                    cluster[1].add(ann)
                    clustered = True
                    break
            if not clustered:
                clusters.append((qrs, set([ann]), qrsdur[ann]))
    if not clusters:
        return False
    #We take as normal beats the cluster with highest number of annotations.
    nclust = max(clusters, key= lambda cl:len(cl[1]))
    beats = [ann for ann in anns if MIT.is_qrs_annotation(ann)
                                                    and lth <= ann.time <= uth]
    if len(beats) < 5:
        return False
    for i in range(len(beats)-4):
        tach = ms2bpm(sp2ms(beats[i+4].time-beats[i].time)/4.0) >= 100
        bset = set(beats[i:i+5])
        ventr = (np.min([qrsdur[b] for b in bset]) > ms2sp(110)
                             or any([bset.issubset(cl[1]) for cl in clusters]))
        if (tach and ventr and all([b not in nclust[1] for b in bset])):
            return True
    return False

EVAL_ALARM_F = {ASYST: eval_asyst, BRAD: eval_brad, TACH: eval_tach,
              VFLUT: eval_vflut, VTACH: eval_vtach}

def challenge(record, alarm_type):
    """Evaluates the presence of a given alarm in a given record"""
    assert alarm_type in (ASYST, BRAD, TACH, VFLUT, VTACH), (
                                   'Unknown alarm type {0}'.format(alarm_type))
    annots = MIT.read_annotations(INTERP_DIR + str(record) + '.igqrs')
    rec = None
    if alarm_type == VTACH:
        rec = MIT.load_MIT_record(DB_DIR + str(record))
        rec.leads = [VALID_LEAD_NAMES[l] for l in rec.leads]
    return EVAL_ALARM_F[alarm_type](annots, rec)

if __name__ == "__main__":
    OUTFILE = '/home/local/tomas.teijeiro/cinc_challenge15/results/answers.txt'
    RECLIST = list(csv.reader(open(DB_DIR + 'ALARMS')))
    RECORDS = [r[0] for r in RECLIST]
    ALARMS = [r[1] for r in RECLIST]
    N = len(RECORDS)
    RESULTS = np.zeros(N)
    #Validation
    print('Processing records')
    for idx in range(N):
        fname = RECORDS[idx]
        print(fname)
        RESULTS[idx] = int(challenge(fname, ALARMS[idx]))
    #Result writing
    print('Generating results file')
    with open(OUTFILE, 'wb') as output:
        WRITER = csv.writer(output)
        for idx in range(N):
            WRITER.writerow([RECORDS[idx], ALARMS[idx], int(RESULTS[idx])])
