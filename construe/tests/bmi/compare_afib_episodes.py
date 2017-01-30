# -*- coding: utf-8 -*-
# pylint: disable=
"""
Created on Wed May 13 12:09:00 2015

This script obtains the atrial fibrillation episodes obtained by the algorithms
in the mobiguide project, and converts them to a format that can be used to
compare with interpretation results.

@author: T. Teijeiro
"""

import construe.utils.MIT as MIT
import construe.utils.MIT.ECGCodes as ECGCodes
from construe.model.interval import Interval as Iv
from construe.tests.mit_validation.rhythm_validation import (Rhythm, epicmp,
                                                          print_epicmp_results)
from construe.utils.units_helper import (samples2msec as s2m,
                                            msec2samples as m2s)

import dateutil
import datetime as dt
import xml.etree.ElementTree as ET
import sortedcontainers
import glob
import copy

DB_DIR = '/home/tomas/Dropbox/Investigacion/tese/estadias/2015_BMI/data/'

#Namespaces used in the MobiGuide database
NS = {'ns'   : "http://interfaces.dataintegrator.atos.mobiguide",
      'mg_di': "http://mobiguide.atosresearch.eu/dataIntegrator",
      'ns2'  : "org.opencds.vmr.v1_0.schema.vmr"}

ANN = '.rhy'
RHTAG = '(AFIB'

DEVIDS = ['MG007', 'MG008', 'MG030']

for devid in DEVIDS:
    #The comparison will be done during intervals with available signal.
    sig_episodes = sortedcontainers.SortedList()

    mbg = sortedcontainers.SortedList()
    tree = ET.parse(DB_DIR + devid + '_episodes.xml')
    ep_seq = tree.find('ns:return/mg_di:diResponse/mg_di:additionalInfo', NS)
    for episode in ep_seq.findall('ns2:observationResult', NS):
        rhythm = Rhythm()
        rhythm.code = RHTAG
        tp = episode.find('ns2:observationEventTime', NS)
        start = dateutil.parser.parse(tp.attrib['low'])
        end = dateutil.parser.parse(tp.attrib['high'])
        #FIXME we need to ignore timezone for the moment
        rhythm.start = start.replace(tzinfo=None)
        rhythm.end = end.replace(tzinfo=None)
        mbg.add(rhythm)

    abd = sortedcontainers.SortedList()
    for f in glob.glob(DB_DIR+devid + '*' + ANN):
        reftime = MIT.get_datetime(f[:-len(ANN)])
        annots = MIT.read_annotations(f)
        if not annots:
            continue
        sig_episodes.add(Iv(reftime, reftime + dt.timedelta(
            milliseconds=s2m(annots[-1].time))))
        for r in (a for a in annots
                  if a.code == ECGCodes.RHYTHM and a.aux == RHTAG):
            rhythm = Rhythm()
            rhythm.code = r.aux
            rhythm.start = reftime + dt.timedelta(milliseconds=s2m(r.time))
            end = next((a.time for a in annots
                        if a.time > r.time and a.code == ECGCodes.RHYTHM),
                       annots[-1].time)
            rhythm.end = reftime + dt.timedelta(milliseconds=s2m(end))
            abd.add(rhythm)

    #We get only the sessions in common with the two annotators.
    sig_episodes = sortedcontainers.SortedList([e for e in sig_episodes
                                     if any([e.overlap(rh.iv) for rh in mbg])])
    #And we filter both lists.
    mbg = sortedcontainers.SortedList([rh for rh in mbg
                            if any([e.overlap(rh.iv) for e in sig_episodes])])
    abd = sortedcontainers.SortedList([rh for rh in abd
                            if any([e.overlap(rh.iv) for e in sig_episodes])])

    #HINT interval join in mobiguide if they are consecutive.
    i = 1
    while i < len(mbg):
        if (mbg[i].start-mbg[i-1].end).total_seconds() == 0:
            mbg[i-1].end = mbg[i].end
            mbg.pop(i)
        else:
            i += 1

    assert mbg and abd, 'Some of the annotators have no identified rhythms'

    #Now we get a common reference for the episodes to be timed.
    reftime = min(mbg[0].start, abd[0].start, sig_episodes[0].start)

    #Now we need to transform all timestamps to absolute references in terms of
    #samples, to compare the sequences using our epicmp implementation.
    rmbg = sortedcontainers.SortedList()
    rabd = sortedcontainers.SortedList()
    for lst in (mbg, abd):
        dst = rmbg if lst is mbg else rabd
        for rh in lst:
            relrh = copy.deepcopy(rh)
            relrh.start = int(m2s((rh.start-reftime).total_seconds()*1000))
            relrh.end = int(m2s((rh.end-reftime).total_seconds()*1000))
            dst.add(relrh)
    res = epicmp(rabd, rmbg, RHTAG)
    print('==== Records from {0} ===='.format(devid))
    print(print_epicmp_results(RHTAG, devid, res[0]+res[1], False))

    print(' Episodes detected by mobiguide but not by abduction:')
    for ep in (m for m in mbg if all([not m.iv.overlap(a.iv) for a in abd])):
        print('    {0}'.format(ep))
    print(' Episodes detected by abduction but not by mobiguide:')
    for ep in (a for a in abd if all([not a.iv.overlap(m.iv) for m in mbg])):
        print('    {0}'.format(ep))
    print('\n')

