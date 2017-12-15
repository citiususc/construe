# -*- coding: utf-8 -*-
# pylint: disable-msg=
"""
Created on Fri Jul  4 09:03:16 2014

This script performs some visual validation tests on the procedures used to
characterize a signal fragment as a ventricular flutter.

@author: T. Teijeiro
"""

from construe.utils.MIT import load_MIT_record, read_annotations, ECGCodes
import collections
from construe.knowledge.abstraction_patterns.rhythm.vflutter import _is_VF
import matplotlib.pyplot as plt

DB = '/tmp/mit/'
RECORDS = [100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 111, 112, 113,
           114, 115, 116, 117, 118, 119, 121, 122, 123, 124, 200, 201, 202,
           203, 205, 207, 208, 209, 210, 212, 213, 214, 215, 217, 219, 220,
           221, 222, 223, 228, 230, 231, 232, 233, 234]

flut = collections.defaultdict(list)

plt.ioff()
for record in RECORDS:
    rpath = DB + str(record)
    try:
        annots = iter(read_annotations(rpath + '.wbr'))
    except IOError:
        continue
    rec = load_MIT_record(rpath)
    while True:
        try:
            vfon = next(a for a in annots if a.code == ECGCodes.VFON)
            vfoff = next(a for a in annots if a.code == ECGCodes.VFOFF)
            flut[record].append(rec.signal[:, vfon.time:vfoff.time+1])
        except StopIteration:
            break
    for i in xrange(len(flut[record])):
        fl = flut[record][i]
        for l in (0,1):
            isvf = _is_VF(fl[l])
            #FIXME this command only works if the _is_VF function internally
            #generates a matplotlib figure.
            plt.savefig('/tmp/flutters/{0}.{1}.{2}.{3}.png'.format(record, i,
                                                           rec.leads[l], isvf))
    print('{0} flutter observations in record {1}'.format(len(flut[record]),
                                                                       record))
plt.ion()