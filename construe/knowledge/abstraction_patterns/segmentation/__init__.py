# -*- coding: utf-8 -*-
# pylint: disable-msg=

"""

This package contains the knowledge definition for those patterns related
with signal segmentation.

Constants of the segmentation package
-------------------------------------

QUAL_THRES:
    This is a heuristically defined threshold on the absolute value of the
    quality indicator of a lead to consider it as 'good'.

QUAL_DIFF_THRES:
    This is a heuristically defined threshold on the difference of the quality
    indicator between two leads that indicates the quality difference is so
    high that one of the leads can be safely ignored.

@author: T. Teijeiro
"""

import construe.utils.signal_processing.signal_measures as sig_meas
import construe.acquisition.signal_buffer as sig_buf
from construe.utils.units_helper import msec2samples as ms2sp


###########################
## Constants definition ###
###########################

QUAL_THRES = 15.0
QUAL_DIFF_THRES = 10.0

def characterize_baseline(lead, beg, end):
    """
    Obtains the baseline estimation for a fragment delimited by two time
    points in a specific lead. It also obtains a quality estimator for the
    fragment.

    Parameters
    ----------
    lead:
        Selected lead to obtain the baseline estimator.
    beg:
        Starting sample of the interval.
    end:
        Ending sample of the interval.

    Returns
    ------
    out: (baseline, quality)
        Tuple with (baseline, quality) estimators. At the moment, the quality
        estimator is not yet numerically characterized, but we have strong
        evidence that the higher this value is, the higher the signal quality
        of the fragment where the baseline has been estimated.
    """
    assert beg >= 0 and end >= beg
    #We need at least 1 second of signal to estimate the baseline and the
    #quality.
    MIN_LENGTH = ms2sp(1000)
    if end-beg < MIN_LENGTH:
        center = beg + (end - beg)/2.0
        beg = max(0, int(center - MIN_LENGTH/2))
        end = int(center + MIN_LENGTH/2)
    signal = sig_buf.get_signal_fragment(beg, end, lead=lead)[0]
    return (sig_meas.mode(signal), sig_meas.kurtosis(signal))
