# -*- coding: utf-8 -*-
# pylint: disable-msg=E1101, E0102, E0202
"""
Created on Mon Feb 27 18:26:34 2012

This file defines the numeric constants of the codes used to set the annotation
type in the WFDB library. Extracted from:
    http://physionet.org/physiotools/wfdb/lib/ecgcodes.h

@author: T. Teijeiro
"""

import construe.knowledge.observables as o

"""
Beat annotation codes:
NORMAL   N   Normal beat
LBBB     L   Left bundle branch block beat
RBBB     R   Right bundle branch block beat
BBB      B   Bundle branch block beat (unspecified)
APC      A   Atrial premature beat
ABERR    a   Aberrated atrial premature beat
NPC      J   Nodal (junctional) premature beat
SVPB     S   Supraventricular premature or ectopic beat (atrial or nodal)
PVC      V   Premature ventricular contraction
RONT     r   R-on-T premature ventricular contraction
FUSION   F   Fusion of ventricular and normal beat
AESC     e   Atrial escape beat
NESC     j   Nodal (junctional) escape beat
SVESC    n   Supraventricular escape beat (atrial or nodal) [1]
VESC     E   Ventricular escape beat
PACE     /   Paced beat
PFUS     f   Fusion of paced and normal beat
UNKNOWN  Q   Unclassifiable beat
LEARN    ?   Beat not classified during learning

Non-beat annotation codes:
VFON     [   Start of ventricular flutter/fibrillation
FLWAV    !   Ventricular flutter wave
VFOFF    ]   End of ventricular flutter/fibrillation
NAPC     x   Non-conducted P-wave (blocked APC) [4]
WFON     (   Waveform onset [4]
WFOFF    )   Waveform end [4]
PWAVE    p   Peak of P-wave [4]
TWAVE    t   Peak of T-wave [4]
UWAVE    u   Peak of U-wave [4]
PQ       ‘   PQ junction
JPT      ’   J-point
PACESP   ^   (Non-captured) pacemaker artifact
ARFCT    |   Isolated QRS-like artifact [2]
NOISE    ~   Change in signal quality [2]
RHYTHM   +   Rhythm change [3]
STCH     s   ST segment change [1,3]
TCH      T   T-wave change [1,3,4]
SYSTOLE  *   Systole [1]
DIASTOLE D   Diastole [1]
MEASURE  =   Measurement annotation [1,3]
NOTE     "   Comment annotation [3]
LINK     @   Link to external data [5]
"""

NOTQRS = 0  # not-QRS (not a getann/putann code)
NORMAL = 1  # normal beat
LBBB = 2    # left bundle branch block beat
RBBB = 3    # right bundle branch block beat
ABERR = 4   # aberrated atrial premature beat
PVC = 5     # premature ventricular contraction
FUSION = 6  # fusion of ventricular and normal beat
NPC = 7     # nodal (junctional) premature beat
APC = 8     # atrial premature contraction
SVPB = 9    # premature or ectopic supraventricular beat
VESC = 10   # ventricular escape beat
NESC = 11   # nodal (junctional) escape beat
PACE = 12   # paced beat
UNKNOWN = 13    # unclassifiable beat
NOISE = 14  # signal quality change
ARFCT = 16  # isolated QRS-like artifact
STCH = 18   # ST change
TCH = 19    # T-wave change
SYSTOLE = 20    # systole
DIASTOLE= 21# diastole
NOTE = 22   # comment annotation
MEASURE = 23# measurement annotation
PWAVE = 24  # P-wave peak
BBB = 25    # left or right bundle branch block
PACESP = 26 # non-conducted pacer spike
TWAVE = 27  # T-wave peak
RHYTHM = 28 # rhythm change
UWAVE = 29  # U-wave peak
LEARN = 30  # learning
FLWAV = 31  # ventricular flutter wave
VFON = 32   # start of ventricular flutter/fibrillation
VFOFF = 33  # end of ventricular flutter/fibrillation
AESC = 34   # atrial escape beat
SVESC = 35  # supraventricular escape beat
LINK = 36   # link to external data (aux contains URL)
NAPC = 37   # non-conducted P-wave (blocked APB)
PFUS = 38   # fusion of paced and normal beat
WFON = 39   # waveform onset
PQ = WFON   # PQ junction (beginning of QRS)
WFOFF = 40  # waveform end
JPT = WFOFF # J point (end of QRS)
RONT = 41   # R-on-T premature ventricular contraction

CHARMAP = {
#Beat annotations
    'N': NORMAL,
    'L': LBBB,
    'R': RBBB,
    'B': BBB,
    'A': APC,
    'a': ABERR,
    'J': NPC,
    'S': SVPB,
    'V': PVC,
    'r': RONT,
    'F': FUSION,
    'e': AESC,
    'j': NESC,
    'n': SVESC,
    'E': VESC,
    '/': PACE,
    'f': PFUS,
    'Q': UNKNOWN,
    '?': LEARN,
#Non-beat annotations
    '[': VFON,
    '!': FLWAV,
    ']': VFOFF,
    'x': NAPC,
    '(': WFON,
    ')': WFOFF,
    'p': PWAVE,
    't': TWAVE,
    'u': UWAVE,
    '‘': PQ,
    '’': JPT,
    '^': PACESP,
    '|': ARFCT,
    '~': NOISE,
    '+': RHYTHM,
    's': STCH,
    'T': TCH,
    '*': SYSTOLE,
    'D': DIASTOLE,
    '=': MEASURE,
    '"': NOTE,
    '@': LINK
}

def mit_to_aami(code):
    """Converts a beat code from the mit-bih to the aami scale"""
    if code in (1, 2, 3, 25):
        return 1
    elif code in (4, 7, 8, 9, 11, 34, 35):
        return 2
    elif code in (5, 10, 41):
        return 3
    elif code in (6,):
        return 5
    elif code in (12, 13, 38):
        return 6
    else:
        return LEARN

def aami_to_mit(code):
    """Converts a beat code from the aami scale to a corresponding mit-bih"""
    if code is 1:
        return NORMAL
    elif code is 2:
        return APC
    elif code is 3:
        return PVC
    elif code is 5:
        return FUSION
    elif code is 6:
        return UNKNOWN
    else:
        return LEARN

def ICHARMAP(code):
    """
    Obtains the character corresponding to a specific code, by performing an
    inverse search on the CHARMAP dict.
    """
    return next(k for k, v in CHARMAP.items() if v == code)

RHYTHM_AUX = {
#Stores the AUX text for the different rhythms.
    o.Sinus_Rhythm:         b'(N',
    o.Tachycardia:          b'(SVTA',
    o.Bradycardia:          b'(SBR',
    o.Atrial_Fibrillation:  b'(AFIB',
    o.Extrasystole:         b'(EXT', #Non-standard
    o.Couplet:              b'(CPT', #Non-standard
    o.Trigeminy:            b'(T',
    o.Bigeminy:             b'(B',
    o.RhythmBlock:          b'(BK',
    o.Ventricular_Flutter:  b'(VFL',
    o.Asystole:             b'P'
}