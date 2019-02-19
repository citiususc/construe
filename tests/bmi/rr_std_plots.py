# -*- coding: utf-8 -*-
# pylint: disable-msg=E1101
"""
Created on Wed May  6 10:02:14 2015

This script generates a set of RR standard deviation plots, as are used in
the BMI lab to assess the possible presence of atrial fibrillation episodes.

@author: T. Teijeiro
"""

import construe.utils.MIT as MIT
import construe.utils.MIT.ECGCodes as ECGCodes
from construe.utils.units_helper import samples2msec as sp2ms
import numpy as np
import matplotlib.pyplot as plt

PATH = '/home/tomas/Dropbox/Investigacion/tese/estadias/2015_BMI/data/'
RECORDS = [l.strip() for l in open(PATH + 'RECORDS')]
ANN = '.iqrs'


plt.ioff()
for rec in RECORDS:
    try:
        annots = MIT.read_annotations(PATH+rec+ANN)
    except IOError:
        print('No results found for record '+rec)
        continue
    rpeaks = sp2ms(np.array(
                         [a.time for a in annots if MIT.is_qrs_annotation(a)]))
    if len(rpeaks) < 2:
        print('No hearbeats found for record '+rec)
        continue
    pwaves = [a for a in annots if a.code == ECGCodes.PWAVE]
    #Plot creation
    fig, host = plt.subplots()
    par1 = host.twinx()
    rrstd = []
    pwf = []
    #We create one point by minute.
    minutes = int(rpeaks[-1]/60000)
    for m in range(minutes):
        mpks = rpeaks[np.logical_and(rpeaks > m*60000, rpeaks < (m+1)*60000)]
        if len(mpks) < 2:
            rrstd.append(0.0)
        else:
            #The standard deviation is limited to 500ms
            rrstd.append(min(490, np.std(np.diff(mpks))))
        npw = len([p for p in pwaves if m*60000 < sp2ms(p.time) < (m+1)*60000])
        if len(mpks) > 0:
            pwf.append(min(100, 100.0*npw/len(mpks)))
        else:
            pwf.append(0.0)
    #P wave detection plot on the background
    p1 = par1.bar(np.arange(minutes)-0.5, pwf, 1.0, alpha=0.5)
    for m in range(minutes):
        color = ((200-pwf[m]/100*200)/256.0, pwf[m]/100*200/256.0, 0)
        p1.patches[m].set_facecolor(color)
    par1.set_ylim(0, 100)
    par1.set_ylabel('P wave detection (%)')
    p0, = host.plot(rrstd, '-o', color='b')
    host.set_ylim(0, 500)
    host.set_xlim(0, minutes-1)
    host.set_ylabel('RR std (ms)')
    host.yaxis.label.set_color(p0.get_color())
    host.tick_params(axis='y', colors=p0.get_color(), **dict(size=4, width=1.5))

    plt.savefig(PATH + rec + '_RRstd.png', dpi=300)
    plt.close()
    print('Processed record '+rec)
plt.ion()
