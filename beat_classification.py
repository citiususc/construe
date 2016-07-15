# -*- coding: utf-8 -*-
# pylint: disable-msg=
"""
Created on Mon May 11 17:48:17 2015

This module performs beat classification from a complete interpretation of an
ECG fragment, that is, all QRS observations are assigned a tag representing
its estimated origin.

@author: T. Teijeiro
"""

import construe.knowledge.observables as o
import construe.utils.MIT as MIT
import construe.utils.MIT.ECGCodes as C
import construe.utils.MIT.interp2annots as interp2annots
from construe.inference.searching import ilen
from construe.model.interval import Interval as Iv
from construe.acquisition.signal_buffer import Leads
from construe.utils.units_helper import (msec2samples as ms2sp,
                                            samples2msec as sp2ms,
                                            phys2digital as ph2dg,
                                            digital2phys as dg2ph, msec2bpm,
                                            set_sampling_freq)
import numpy as np
import collections
import blist
import pprint

#Histogram bins for the P-wave characterization
PW_BINS = [0.0, ph2dg(0.05), ph2dg(0.1), ph2dg(0.5), ph2dg(1.0)]

#Features vector
Feat = collections.namedtuple('Feat', ["RR", "dRR", "Dur", "dDur", "Axis",
                                       "dAxis", "Pw", "Rh", "Sim", "dAmp"])
#Cluster structure
Cluster = collections.namedtuple('Cluster', ['beats', 'info'])

#Codes for the rhythm.
REGULAR, AFIB, ADVANCED, DELAYED = range(4)
#Atrial fibrillation beats are tagged as NORMAL in the MIT-BIH Arrhythmia
#database, but during the classification, we marked them with a different code
#not used for other purposes, although semantically related with it
AFTAG = C.SYSTOLE

#Typical values to distinguish supraventricular and ventricular nature of
#a cluster, using only morphology features.
ORIGIN = {
    'S' : lambda f:np.count_nonzero([f.Axis >= 0, f.Dur < 2, f.Pw >= 0,
                                     f.Sim > 2, f.dAmp <= 0, f.dAxis == 0,
                                     f.dDur < 2]),
    'V' : lambda f:np.count_nonzero([f.Axis <= 0, f.Dur > 0, f.Pw == 0,
                                     f.Sim < 3, f.dAmp != 0, f.dAxis > 0,
                                     f.dDur > 0])
}

#QRS origin tag for each possible QRS code.
OTAG = {
    C.NORMAL: 'S', C.LBBB: 'S', C.RBBB: 'S', C.ABERR : 'S', C.PVC: 'V',
    C.FUSION: 'F', C.NPC: 'S', C.APC: 'S', C.SVPB: 'S', C.VESC: 'V',
    C.NESC: 'S', C.AESC: 'S', C.PACE: 'F', AFTAG: 'S', C.PFUS: 'F'
}


class BeatInfo(object):
    """
    This class defines the information needed by the classification for
    each individual QRS complex.
    """
    def __init__(self, qrs):
        self.qrs = qrs
        self.rr = ms2sp(800)
        self.pwave = False
        self.pos = REGULAR
        self.axis = get_axis(qrs)
        self.rh = None

    def __str__(self):
        return pprint.pformat(self.__dict__)

    def __repr__(self):
        return str(self)

def _in_vflutter(cluster, interp):
    """
    Checks if most beats in a cluster are interpreted within a ventricular
    flutter/fibrillation episode.
    """
    vfluts = [Iv(vf.earlystart, vf.earlyend)
              for vf in interp.get_observations(o.Ventricular_Flutter)]
    invf = ilen(b for b in cluster if any(b.time.start in vf for vf in vfluts))
    return invf/float(len(cluster)) > 0.5

def xcorr_full(sig1, sig2):
    """
    Performs a full normalized cross-correlation between two signals, returning
    the maximum value and the delay with respect to the first signal that
    achieves such value.

    Returns
    -------
    out:
        (corr, delay): Tuple with the maximum correlation factor (between -1
        and 1) and the delay that has to be applied to the second signal to get
        that correlation.
    """
    tr1 = sig1 - sig1[0] if sig1[0] != 0 else sig1
    tr2 = sig2 - sig2[0] if sig2[0] != 0 else sig2
    corr = np.correlate(tr1, tr2, mode='full')
    if np.any(tr1) and np.any(tr2):
        corr /= np.sqrt(np.dot(tr1, tr1) * np.dot(tr2, tr2))
    idx = np.argmax(corr)
    return (corr[idx], idx-len(tr2)+1)

def get_similarity(sig1, sig2):
    """
    Obtains a measure of the similarity between two multi-lead signals, as the
    mean of the cross-correlation maximum value for each lead.
    """
    cleads = set(sig1.keys()).intersection(sig2.keys())
    corrs = []
    for lead in set(sig1.keys()).union(sig2.keys()):
        if lead not in cleads:
            corrs.append(0.0)
        else:
            arr1, arr2 = sig1[lead].sig, sig2[lead].sig
            if len(arr2) > len(arr1):
                arr1, arr2 = arr2, arr1
            corr, _ = xcorr_full(arr1, arr2)
            corrs.append(corr)
    return np.mean(corrs)

def get_axis(beat):
    """
    Obtains the heart axis of a QRS complex, using only the information in
    the  MLII lead. The axis is therefore calculated only wrt this lead, and
    the value will be in the range [-90º,90º], being 90º the value obtained
    when all the amplitude is positive in lead MLII. In common heart axis scale,
    the range is [-120º,60º]. If no shapeform is recognized in lead MLII, the
    axis is undetermined (None).
    """
    try:
        shape = beat.shape[Leads.MLII]
        poswav = [w.amp for w in shape.waves if w.sign > 0]
        negwav = [w.amp for w in shape.waves if w.sign < 0]
        pamp = max(poswav) if poswav else 0.0
        namp = -max(negwav) if negwav else 0.0
        return int(90.0*(pamp-namp)/float(shape.amplitude))
    except KeyError:
        return None

def get_features(interpretation):
    """
    Obtains the relevant classification features for every QRS in the
    interpretation.
    """
    result = collections.OrderedDict()
    rhythms = interpretation.get_observations(o.Cardiac_Rhythm)
    beats = blist.sortedlist(interpretation.get_observations(o.QRS))
    rrs = np.diff([b.time.start for b in beats])
    beatiter = iter(beats)
    obs = interpretation.observations
    qrs = None
    for rh in rhythms:
        qidx0 = bidx = 0
        if qrs is None:
            i = 0
            qrs = next(beatiter)
        else:
            i = 1
        while qrs.time.start <= rh.lateend:
            info = BeatInfo(qrs)
            info.rh = rh
            bidx = beats.index(qrs)
            qidx0 = qidx0 or bidx
            if bidx > 0:
                info.rr = rrs[bidx-1]
            idx = obs.index(qrs)
            pw = None
            if idx > 0 and isinstance(obs[idx-1], o.PWave):
                pw = obs[idx-1]
            elif idx > 1 and isinstance(obs[idx-2], o.PWave):
                pw = obs[idx-2]
            info.pwave = pw.amplitude if pw is not None else {}
            if isinstance(rh, (o.Sinus_Rhythm, o.Bradycardia, o.Tachycardia)):
                info.pos = REGULAR
            elif isinstance(rh, o.Extrasystole):
                info.pos = ADVANCED if i == 1 else REGULAR
            elif isinstance(rh, o.Couplet):
                info.pos = ADVANCED if i in (1, 2) else REGULAR
            elif isinstance(rh, (o.RhythmBlock, o.Asystole)):
                info.pos = DELAYED
            elif isinstance(rh, o.Atrial_Fibrillation):
                info.pos = AFIB
            elif isinstance(rh, o.Bigeminy):
                info.pos = ADVANCED if i % 2 == 1 else REGULAR
            elif isinstance(rh, o.Trigeminy):
                info.pos = ADVANCED if i % 3 == 1 else REGULAR
            elif isinstance(rh, o.Ventricular_Flutter):
                info.pos = REGULAR
            result[qrs] = info
            qrs = next(beatiter, None)
            if qrs is None:
                break
            i += 1
        meanrr = np.mean(rrs[qidx0:bidx]) if qidx0 < bidx else rrs[bidx-1]
        rh.meas = o.CycleMeasurements((meanrr, 0), (0, 0), (0, 0))
    return result

def get_tag(cluster):
    """
    Obtains the tag assigned to a cluster. **Note: This tag is assumed to be
    the same for all beats in the cluster.**
    """
    return cluster.info.qrs.tag

def identical(cl1, cl2, feat=None):
    """
    Checks if two clusters are morphologically identical, which implies
    the classification algorithm considers both clusters to have the same
    origin (supraventricular or ventricular).
    """
    features = feat or get_qualitative_features(cl1.info, cl2)
    return features.Sim == 4 and features.dAmp == 0

def load_clustering(record, cluster_ext, observations):
    """
    Loads the clustering information for a given record. It receives as
    parameter the record name, the extension of the file containing the cluster
    information, and a list of observations corresponding to the annotations
    used in the matching. Returns a dictionary in which each key is the id of
    a cluster and the value is a tuple with the set of observations belonging
    to the cluster, and the real annotation code that should be assigned to the
    cluster.
    """
    clusters = collections.defaultdict(set)
    #We get only QRS observations and uninterpreted R-Deflections.
    observations = [ob for ob in observations
                                     if isinstance(ob, (o.QRS, o.RDeflection))]
    cluster_results = np.genfromtxt('{0}.{1}'.format(record,
                                                      cluster_ext)).astype(int)
    for idx, cl in cluster_results:
        clusters[cl].add(observations[idx])
    return clusters

def get_cluster_features(cluster, features):
    """
    Obtains a BeatInfo object as a summary of the features of a complete
    cluster. It is created by the calculation of the mean value of all the
    relevant features. It also involves the selection of a representant
    from the cluster by the minimization of the distance to the mean.
    """
    cl = [b for b in features if b in cluster]
    if not cl:
        return BeatInfo(o.QRS())
    leads = set.union(*[set(b.shape) for b in cl])
    cl = [b for b in cl if all(l in b.shape for l in leads)]
    if not cl:
        return BeatInfo(o.QRS())
    pwamps = {}
    amplitudes = {}
    qdurs = {}
    for l in leads:
        arr = np.array([features[b].pwave.get(l, 0.0) for b in cl])
        hist = np.histogram(arr, PW_BINS)
        pwamps[l] = dg2ph(hist[1][hist[0].argmax()])
        amplitudes[l] = np.array([b.shape[l].amplitude for b in cl])
        amplitudes[l] = (amplitudes[l]-np.mean(amplitudes[l]))/ph2dg(5.0)
        qdurs[l] = np.array([len(b.shape[l].sig)-1 for b in cl])
        qdurs[l] = (qdurs[l]-np.mean(qdurs[l]))/ms2sp(120)
    axis = (np.array([features[b].axis for b in cl])
                                 if Leads.MLII in leads else np.zeros(len(cl)))
    axis = (axis-np.mean(axis))/180.0
    #We calculate the euclidean distance of every QRS to the central measures
    eucdist = np.linalg.norm(np.matrix((tuple(qdurs.values()) +
                                        tuple(amplitudes.values()) +
                                        (axis,))), axis=0)
    #We select as representative the beat with minimum distance.
    info = BeatInfo(cl[np.argmin(eucdist)])
    info.pwave = np.mean(pwamps.values()) > 0.05
    #For the rhythm features, we use all beats
    cl = {b for b in cluster if b in features}
    info.rr = np.mean([features[b].rr for b in cl])
    info.pos = collections.Counter([features[b].pos for b in cl])
    rhpos = max(info.pos, key=lambda v:info.pos[v])
    n = float(sum(info.pos.values()))
    #Factor correction for advanced beats
    if rhpos != ADVANCED and info.pos[ADVANCED]/n > 0.2:
        nadv = ilen(b for b in cl if features[b].pos is REGULAR and
                                features[b].rr < features[b].rh.meas.rr[0])
        nadv -= info.pos[REGULAR]/2
        if 0 < info.pos[ADVANCED]+nadv > info.pos[REGULAR]-nadv:
            rhpos = ADVANCED
    #Factor correction for delayed beats
    elif rhpos != DELAYED and info.pos[DELAYED]/n > 0.2:
        ndel = ilen(b for b in cl if features[b].pos is REGULAR and
                                features[b].rr < features[b].rh.meas.rr[0])
        ndel -= info.pos[REGULAR]/2
        if 0 < info.pos[DELAYED]+ndel > info.pos[REGULAR]-ndel:
            rhpos = DELAYED
    info.rh = rhpos
    return info

def get_qualitative_features(nclust, clust):
    """
    Obtains a *Feat* object with the computed values of the features used
    for the classification based on comparison between two clusters.

    Parameters
    ----------
    nclust:
        Cluster structure already identified as normal.
    clust:
        It can be another cluster, or a single *BeatInfo* object.
    """
    if isinstance(clust, BeatInfo):
        info = clust
        rhpos = info.pos
        pwave = 1 if sum(info.pwave.values()) > 0.1 else 0
    else:
        info = clust.info
        rhpos = info.rh
        pwave = int(info.pwave)
    cleads = set(info.qrs.shape).intersection(nclust.qrs.shape)
    if cleads:
        mxl = max(cleads, key=lambda l:info.qrs.shape[l].amplitude)
        ampdf = (float(info.qrs.shape[mxl].amplitude)/
                                               nclust.qrs.shape[mxl].amplitude)
    else:
        ampdf = 1.0
    similarity = get_similarity(nclust.qrs.shape, info.qrs.shape)
    ndur = nclust.qrs.lateend-nclust.qrs.earlystart
    dur = info.qrs.lateend-info.qrs.earlystart
    durdf = dur-ndur
    ax = 0.0 if info.axis is None else info.axis
    axdf = (abs(nclust.axis-info.axis)
                              if None not in (nclust.axis, info.axis) else 0.0)
    rr = msec2bpm(sp2ms(info.rr))
    rrdf = rr-msec2bpm(sp2ms(nclust.rr))
    #QRS width: -1=narrow, 0=normal, 1=abnormal, 2=wide
    if dur < ms2sp(80):
        dur = -1
    elif dur < ms2sp(100):
        dur = 0
    elif dur < ms2sp(120):
        dur = 1
    else:
        dur = 2
    #QRS width difference: -1=narrower, 0:equal, 1=wider, 2=much wider
    if durdf <= ms2sp(-20):
        durdf = -1
    elif durdf < ms2sp(20):
        durdf = 0
    elif durdf < ms2sp(40):
        durdf = 1
    else:
        durdf = 2
    #Axis: -1 = Negative, 0=Balanced, 1=Positive
    if ax < -45:
        ax = -1
    elif ax < 45:
        ax = 0
    else:
        ax = 1
    #Axis difference: 0=equal, 1=different, 2=very different, 3=opposite
    if axdf < 45:
        axdf = 0
    elif axdf < 90:
        axdf = 1
    elif axdf < 135:
        axdf = 2
    else:
        axdf = 3
    #Rhythm: -1=Bradycardia, 0=Normal, 1=Tachycardia, 2=Extreme tachycardia
    if rr < 60:
        rr = -1
    elif rr < 100:
        rr = 0
    elif rr < 150:
        rr = 1
    else:
        rr = 2
    #Rhythm difference: -1=slower, 0=equal, 1=faster
    if rrdf <= -20:
        rrdf = -1
    elif rrdf < 20:
        rrdf = 0
    else:
        rrdf = 1
    #Similarity: 0=very different, 1=different, 2=similar,
    #            3=very similar, 4=identical
    if similarity < 0.25:
        similarity = 0
    elif similarity < 0.5:
        similarity = 1
    elif similarity < 0.75:
        similarity = 2
    elif similarity < 0.9:
        similarity = 3
    else:
        similarity = 4
    #Amplitude difference: -1=lower, 0=equal, 1=higher
    if ampdf < 0.75:
        ampdf = -1
    elif ampdf <= 1.25:
        ampdf = 0
    else:
        ampdf = 1
    return Feat(rr, rrdf, dur, durdf, ax, axdf, pwave, rhpos, similarity, ampdf)

def normal_classification(qrs):
    """
    Distinguishes between a normal QRS and complete Left and Right Bundle
    Branch Blocks, according to the typical duration and morphology criteria.
    """
    qdur = qrs.lateend-qrs.earlystart
    if qdur > ms2sp(100) and 'V1' in qrs.shape:
        v1tag = qrs.shape['V1'].tag
        if v1tag[-1] == 'R':
            #Right bundle branch block (complete or incomplete)
            return C.RBBB
        elif qdur > ms2sp(120) and v1tag in ('QS', 'rS'):
            #Left bundle branch block (only complete)
            return C.LBBB
    return C.NORMAL

def single_classification(cluster, features, interp):
    """
    Applies a simple set of classification rules to a cluster. These rules do
    not depend on the classification of other clusters of the same record.

    Parameters
    ----------
    cluster:
        Cluster structure.
    features:
        Dictionary of individual QRS features.
    interp:
        Interpretation of the record.

    Returns
    -------
    out:
        Tag corresponding to the classification of the cluster, or C.UNKNOWN.
    """
    #If most of the "beats" are within a ventricular flutter, we classify the
    #cluster as ventricular.
    if _in_vflutter(cluster[0], interp):
        return C.PVC
    cl = {b for b in cluster[0] if b in features}
    #If we have no features to classify, we consider the cluster does not
    #represent a QRS family.
    if not cl:
        return C.ARFCT
    n = float(sum(cluster.info.pos.values()))
    #The classification of paced clusters is straightforward, due to the high
    #specificity of the pacemaker spikes detection in the interpretation.
    npac = ilen(b for b in cl if b.paced)
    if npac > 1 and npac/n > 0.2:
        return C.PACE
    #We get the features array, ignoring those which are compared.
    feat = get_qualitative_features(cluster.info, cluster)
    #We require a significant number of beats to avoid the influence of
    #spurious features in the classification.
    if n > 30:
        if feat.Rh == 0 and feat.Pw:
            return normal_classification(cluster.info.qrs)
        if feat.Rh == 1 and feat.RR >= 0:
            return AFTAG
        if feat.Rh == 2 and feat.Dur == -1:
            return C.APC
        if feat.Pw and feat.Dur == -1:
            return C.NORMAL
    return C.UNKNOWN

def _vclass(feat):
    """
    Classification rules for clusters identified to have ventricular origin.

    Parameters:
    -----------
    feat:
        Feat object with the features of the cluster to classify.

    Returns:
    --------
    out:
        Class of the cluster, that can be one of (PVC, VESC, FUSION).
    """
    #Advanced ventricular beats, based on rhythm information
    if feat.Rh == 2 or feat.dRR == 1 or feat.RR == 2:
        return C.PVC
    #Escape beats
    if feat.Rh == 3:
        return C.VESC
    #Classification of fusion beats apply the same rule on supraventricular
    #and ventricular origin.
    if feat.Dur == 1 and feat.RR == 1 and feat.Rh == 0:
        return C.FUSION
    #Last distinction between premature and escape beats is based on the RR
    return C.PVC if feat.RR > 0 else C.VESC


def comparative_classification(cluster, classified):
    """
    Classifies a cluster using the information of other already-classified
    clusters to make easier the decision process.
    """
    #"Normal" cluster, established according to the context
    nclust = None
    #QRS origin, distinguishing only supraventricular/ventricular nature.
    orig = None
    #If there is a cluster already classified and with identical shape, the
    #origin is assumed to be the same.
    iclust = next((c[1] for c in classified if identical(c[1], cluster)), None)
    if iclust:
        if get_tag(iclust) is C.ARFCT:
            return C.ARFCT
        orig = OTAG[get_tag(iclust)]
    #Wide-QRS contexts. First, we evaluate the presence of an artificial
    #pacemaker, and then a bundle-branch-block morphology. The classification
    #rules are the same in both cases, only changin the NORMAL and FUSION tags.
    #Pacemaker context
    pclust = next((c[1] for c in classified if get_tag(c[1]) == C.PACE), None)
    if pclust:
        feat = get_qualitative_features(pclust.info, cluster)
        if orig == 'S':
            if feat.RR == feat.Dur == 0 and feat.Sim == 4:
                return C.PACE
            if feat.Rh == 2 or feat.dRR == 1  or feat.RR == 2:
                return C.APC
            if feat.Rh == 3 and feat.dRR == -1:
                return C.AESC
            return C.NORMAL
        if orig == 'V':
            return _vclass(feat)
        if orig == 'F':
            return C.PFUS
        if feat.Rh == 2 and feat.Dur > 0 and (feat.Sim < 4 or feat.dDur > 0):
                return C.PVC
        if feat.Rh == feat.dDur == 0 and (feat.Sim > 2 or feat.dAxis < 2):
            return C.PACE
        #If there is a pacemaker, this cluster is the normal situation
        nclust = pclust
    #Bundle branch block context (left or right)
    bbclust = next((c[1] for c in classified
                                   if get_tag(c[1]) in (C.LBBB, C.RBBB)), None)
    if bbclust:
        feat = get_qualitative_features(bbclust.info, cluster)
        if orig == 'S':
            if feat.RR == feat.Dur == 0 and feat.Sim == 4:
                return get_tag(bbclust)
            if feat.Rh == 2 or feat.dRR == 1 or feat.RR == 2:
                return C.APC
            if feat.Rh == 3 and feat.dRR == -1:
                return C.AESC
            return get_tag(bbclust) if feat.Dur > 0 else C.NORMAL
        if orig == 'V':
            return _vclass(feat)
        if orig == 'F':
            return C.FUSION
        if feat.Rh == 2 and feat.Dur > 0 and (feat.Sim < 4 or feat.dDur > 0):
                return C.PVC
        if feat.Rh == feat.dDur == 0 and (feat.Sim > 2 or feat.dAxis < 2):
            return get_tag(bbclust)
        #Branch block is assumed as normal unless a pacemaker is present
        nclust = nclust or bbclust
    #Atrial fibrillation context
    afclust = next((c[1] for c in classified
                     if get_tag(c[1]) == AFTAG and len(c[1].beats) > 10), None)
    #Normal cluster reference
    nclust = nclust or next((c[1] for c in classified
                                           if get_tag(c[1]) == C.NORMAL), None)
    if afclust:
        feat = get_qualitative_features(afclust.info, cluster)
        if (feat.Rh == 1 or cluster.info.pos[AFIB] ==
                                               max(cluster.info.pos.values())):
            if feat.RR >= 0:
                return AFTAG
            return C.AESC if feat.Dur <= 0 or orig == 'S' else C.VESC
        if not orig or orig == 'F':
            if feat.Dur == -1:
                orig = 'S'
            elif feat.Dur == 2 and feat.dAxis > 1:
                orig = 'V'
            else:
                orig = max(('V', 'S'), key=lambda t:ORIGIN[t](feat))
        if orig == 'S':
            if feat.Rh == feat.Dur == 0 and feat.Sim == 4:
                return normal_classification(cluster.info.qrs)
            if feat.Rh == 2:
                return C.APC if feat.dDur < 1 else C.ABERR
            if feat.Rh == 3:
                return C.AESC
            return C.NORMAL
        if orig == 'V':
            if feat.Rh == 2:
                return C.PVC
            if feat.Rh == 3:
                return C.VESC
            return C.PVC if feat.RR > 0 else C.VESC
    #Up to this point, if no normal clusters are present, we take as normality
    #reference the most-populated non-ventricular cluster.
    if not nclust:
        nclust = next(c[1] for c in classified
                                       if get_tag(c[1]) not in (C.PVC, C.VESC))
    feat = get_qualitative_features(nclust.info, cluster)
    if not orig or orig == 'F':
        if feat.Dur == -1:
            orig = 'S'
        elif feat.Dur == 2 and feat.dAxis > 1:
            orig = 'V'
        elif feat.dDur > 0 and feat.Sim < 4 and not feat.Pw:
            orig = 'V'
        else:
            orig = max(('V', 'S'), key=lambda t:ORIGIN[t](feat))
    if orig == 'S':
        if feat.RR == feat.Dur == 0 and feat.Sim == 4:
            return normal_classification(cluster.info.qrs)
        if feat.Rh == 2 or feat.dRR == 1 or feat.RR == 2:
            return C.APC if feat.dDur < 1 else C.ABERR
        if feat.Rh == 3 and feat.dRR == -1:
            return C.AESC
        if feat.Dur == 1 and feat.RR == 1 and feat.Rh == 0:
            return C.FUSION
        return normal_classification(cluster.info.qrs)
    if orig == 'V':
        return _vclass(feat)
    #At this point, we are unable to classify the cluster
    return C.ARFCT

def find_normal_cluster(clusters):
    """
    This function tries to obtain the most probable **normal** cluster from a
    list of non-classified clusters.

    Arguments:
    ----------
    - clusters: List of 2-tuples of QRS clusters, with (id, ClusterInfo) for
                each cluster.

    Returns
    -------
    out: 2-tuple with ((id, ClusterInfo), tag) assigned to the selected
         cluster. The tag must be in (AFTAG, C.NORMAL)
    """
    pwl = sorted(clusters, key=lambda cl:(cl[1].info.pwave, len(cl[1].beats)),
                                                                  reverse=True)
    _, pwcl = pwl[0]
    if (len(pwcl.beats) > 30 and pwcl.info.pwave and
                (pwcl.info.qrs.lateend-pwcl.info.qrs.earlystart) < ms2sp(120)):
        return (pwl[0], normal_classification(pwcl.info.qrs))
    else:
        for cl in clusters:
            _, nxt = cl
            if max(nxt.info.pos,key=lambda v:nxt.info.pos[v]) in (REGULAR,AFIB):
                afrel = nxt.info.pos[AFIB]/float(nxt.info.pos[REGULAR])
                tag = (AFTAG if afrel > 0.5 else
                                           normal_classification(nxt.info.qrs))
                return (cl, tag)
    #At this point, we select as normal cluster the cluster with
    #highest number of REGULAR or AFIB beats.
    ncl = max(clusters, key=lambda cl:max(cl[1].info.pos[AFIB],
                                          cl[1].info.pos[REGULAR]))
    tag = (AFTAG if ncl[1].info.pos[AFIB] > ncl[1].info.pos[REGULAR]
                                   else normal_classification(ncl[1].info.qrs))
    return ncl, tag

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=
        'Classifies the beat annotations in a MIT-BIH ECG record.')
    parser.add_argument('-r', metavar='record', required=True,
                        help='Name of the record to be processed')
    parser.add_argument('-a', metavar='ann', required=True,
                        help= ('Annotations resulting from the abductive '
                               'interpretation of the ECG signal'))
    parser.add_argument('-c', metavar='cluster', required=True,
                        help= ('Extension of the file containing the clustering'
                               ' information.'))
    parser.add_argument('-o', metavar='oann', default='cls',
                        help= ('Save annotations with classified QRS complexes'
                               ' as annotator oann (default: cls)'))
    args = parser.parse_args()
    rec = MIT.load_MIT_record(args.r)
    set_sampling_freq(rec.frequency)
    print('Classifying record {0}'.format(args.r))
    #Reconstruction of the abductive interpretation
    annots = MIT.read_annotations('{0}.{1}'.format(args.r, args.a))
    interp = interp2annots.ann2interp(rec, annots)
    #Cluster information
    clusters = load_clustering(args.r, args.c, interp.observations)
    #QRS feature extraction
    features = get_features(interp)
    #Cluster feature extraction
    for c in clusters:
        clusters[c] = Cluster(clusters[c],
                                   get_cluster_features(clusters[c], features))
    #Key function to compare clusters: First, we check clusters with more than
    #30 beats; then, the clusters with more REGULAR or AFIB beats, and finally
    #we sort by the number of beats in the cluster
    keyf = lambda cl, ft = features: (int(len(cl[1].beats) < 30),
                                      -ilen(b for b in cl[1].beats
                                                if b in ft and ft[b].pos
                                                       in (REGULAR, AFIB)),
                                      -len(cl[1].beats))
    #Cluster classification
    classified = []
    clist = sorted(clusters.iteritems(), key=keyf)
    #Single cluster classification
    i = 0
    while i < len(clist):
        c, nxt = clist[i]
        tag = single_classification(nxt, features, interp)
        if tag != C.UNKNOWN:
            for b in nxt.beats:
                b.tag = tag
            nxt.info.qrs.tag = tag
            classified.append((c, nxt))
            clist.pop(i)
        else:
            i += 1
    #No "normality" reference was identified in the single classification step
    if clist and all(c[1].info.qrs.tag not in (C.NORMAL, C.LBBB,
                                   C.RBBB, AFTAG, C.PACE) for c in classified):
        ncl, tag = find_normal_cluster(clist)
        for b in ncl[1].beats:
            b.tag = tag
        ncl[1].info.qrs.tag = tag
        classified.append(ncl)
        clist.remove(ncl)
    #Comparative classification
    while clist:
        c, nxt = clist.pop(0)
        tag = comparative_classification(nxt, classified)
        for b in nxt.beats:
            b.tag = tag
        #The cluster representant also is assigned the classification.
        nxt.info.qrs.tag = tag
        classified.append((c, nxt))
    #Afib code is now changed to normality to fit the convention
    for _, (beats, _) in classified:
        for b in (b for b in beats if isinstance(b, o.QRS)):
            if b.tag == AFTAG:
                b.tag = C.NORMAL
    annots = interp2annots.interp2ann(interp)
    #We also include the clustered artifacts.
    for b in interp.get_observations(o.RDeflection, filt=lambda ba:
                    any([ba in cl.beats and any(isinstance(b, o.QRS)
                        for b in cl.beats) for cl in clusters.itervalues()])):
        a = MIT.MITAnnotation.MITAnnotation()
        a.code = b.tag
        a.time = b.time.start
        annots.add(a)
    MIT.save_annotations(annots, '{0}.{1}'.format(args.r, args.o))
