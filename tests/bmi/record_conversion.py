# -*- coding: utf-8 -*-
# pylint: disable-msg=
"""
Created on Tue May  5 10:34:15 2015

This scripts converts files in the proprietary format used by the Mobiguide
project to MIT-BIH records. It also creates an annotations file including the
detected Atrial Fibrillation episodes.

@author: T. Teijeiro
"""

import csv
import numpy as np
import datetime as dt
import dateutil.parser
import os
import glob
import struct
import xml.etree.ElementTree as ET
import collections
import construe.utils.MIT.MITAnnotation as MITAnnotation
import construe.utils.MIT.ECGCodes as ECGCodes

Episode = collections.namedtuple('Episode', ['start', 'end'])

#Parameters for the resulting records
FREQ = 250
GAIN = 200
#Path with the records to convert
PATH = '/home/tomas/Dropbox/Investigacion/tese/estadias/2015_BMI/data/'
os.chdir(PATH)
#Device IDs in which we also convert the Atrial Fibrillation episodes
DEVIDS = ('MG007', 'MG008', 'MG030')
#Namespaces used in the MobiGuide database
NS = {'ns'   : "http://interfaces.dataintegrator.atos.mobiguide",
      'mg_di': "http://mobiguide.atosresearch.eu/dataIntegrator",
      'ns2'  : "org.opencds.vmr.v1_0.schema.vmr"}

def load_afib_episodes(devid):
    """
    Obtains a sorted list of episodes of atrial fibrillation detected by the
    mobiguide algorithm. If two episodes are consecutive, then they are merged
    in a single one.
    """
    result = []
    tree = ET.parse(PATH + devid + '_episodes.xml')
    ep_seq = tree.find('ns:return/mg_di:diResponse/mg_di:additionalInfo', NS)
    for episode in ep_seq.findall('ns2:observationResult', NS):
        etime = episode.find('ns2:observationEventTime', NS)
        #FIXME we need to ignore timezone for the moment
        start = dateutil.parser.parse(etime.attrib['low']).replace(tzinfo=None)
        end = dateutil.parser.parse(etime.attrib['high']).replace(tzinfo=None)
        #Join of consecutive episodes.
        if result and result[-1].end == start:
            result[-1] = Episode(result[-1].start, end)
        else:
            result.append(Episode(start, end))
    return result

if __name__ == "__main__":
    #We load the episodes of all the target devices.
    AF_EPISODES = {devid : load_afib_episodes(devid)
                   for devid in DEVIDS}
    #Now we convert all the monitoring sessions
    for rec in glob.glob('MG*.txt'):
        print('Converting record {0}'.format(rec))
        NAME, _ = os.path.splitext(rec)
        reader = csv.reader(open(PATH+rec), delimiter=' ')
        #The Header is skipped
        next(reader)
        #The first row is used to get the reference time.
        head = next(reader)
        tp = dt.datetime.strptime(head[1] + head[2] + '000',
                                  '%Y.%m.%d%H:%M:%S.%f').replace(tzinfo=None)
        sig = np.genfromtxt(head[3:])
        #Full ECG signal.
        sig = np.concatenate((sig,
            np.concatenate([np.genfromtxt(row[3:]) for row in reader if row])))
        #Signal convert to digital units
        sig = sig*GAIN
        #Center the range.
        sig = sig-np.ptp(sig)/2.0
        #Conversion to integer
        sig = sig.astype(int)
        #Verify the range
        assert -32768 <= np.max(sig) <= 32767, 'Signal exceeds digital range'
        assert -32768 <= np.min(sig) <= 32767, 'Signal exceeds digital range'
        #Now we write the .hea and the .dat files.
        heafmt = ('{0} 1 {1} {2} {3}\n'
                  '{0}.dat 16 {4} 16 0 0 0 0 MLII\n')
        with open(PATH + NAME + '.hea', 'w') as hea:
            hea.write(heafmt.format(NAME, FREQ, len(sig),
                                    tp.strftime('%H:%M:%S %d/%m/%Y'), GAIN))
        with open(PATH + NAME + '.dat', 'w') as dat:
            fmt = '<'+'h'*len(sig)
            dat.write(struct.pack(fmt, *sig))
        #And we create the (AFIB annotations according to the loaded episodes.
        etp = tp + dt.timedelta(milliseconds=len(sig)*4)
        devid = next((d for d in AF_EPISODES if NAME.startswith(d)), None)
        annots = []
        if devid is not None:
            afibs = [ep for ep in AF_EPISODES[devid] if tp <= ep.start <= etp]
            for af in afibs:
                #Two annotations for each episode
                bann = MITAnnotation.MITAnnotation()
                bann.code = ECGCodes.RHYTHM
                bann.time = int((af.start-tp).total_seconds()*FREQ)
                bann.aux = b'(AFIB'
                eann = MITAnnotation.MITAnnotation()
                eann.code = ECGCodes.RHYTHM
                eann.time = int((min(etp, af.end)-tp).total_seconds()*FREQ)
                #The end of AF is encoded as 'back to normality'
                eann.aux = b'(N'
                annots.append(bann)
                annots.append(eann)
        #Annotations are stored in a file with the '.mbg' extension.
        MITAnnotation.save_annotations(annots, PATH+NAME+'.mbg')
