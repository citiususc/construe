# -*- coding: utf-8 -*-
# pylint: disable-msg=E1101, E0102, E0202
"""
Created on Tue May 05 10:37:17 2015

Small test to try the abductive approach with the dataset from the Mobiguide
project at the BMI lab.

@author: T. Teijeiro
"""

import construe.utils.plotting.plotter as plotter
import construe.acquisition.record_acquisition as IN
import construe.acquisition.obs_buffer as obs_buffer
import construe.acquisition.signal_buffer as sig_buf
import construe.knowledge.observables as o
import construe.knowledge.constants as C
import construe.knowledge.abstraction_patterns as ap
import construe.inference.searching as searching
import construe.inference.reasoning as reasoning
import time
import itertools
import numpy as np
from pprint import pprint as pp
from construe.model import Interval as Iv
from construe.model.interpretation import Interpretation
from construe.utils.units_helper import (msec2samples as ms2sp,
                                            samples2msec as sp2ms,
                                            msec2bpm, bpm2msec)

#Signal reading
TFACTOR = 5.0
LENGTH = 23040
#Searching settings
KFACTOR = 12
MIN_DELAY = 1750
MAX_DELAY = int(ms2sp(20000)*TFACTOR)
#Overlapping between consecutive fragments
FR_OVERLAP = int(ms2sp(3000))
FR_IDX = 0
INIT = int(FR_IDX * (LENGTH - FR_OVERLAP))
IN.reset()
#Standard annotator used
ANNOTATOR = 'atr'
#Record used
REC = ('/home/tomas/Dropbox/Investigacion/tese/estadias/2015_BMI'
       '/validation/training_dataset/MG008-2015_07_11-ECG-1')
REC = ('/home/local/tomas.teijeiro/Dropbox/Investigacion/tese/validacions/'
       'loose_records/monitoring_160404-1003_SIM')
REC = '/datos/tomas.teijeiro/Servando/MonitorizacionDomiciliaria/mit/250Hz/100'
IN.set_record(REC, ANNOTATOR)
IN.set_offset(INIT)
IN.set_duration(LENGTH)
IN.set_tfactor(TFACTOR)
IN.start()
print('Preloading buffer...')
time.sleep(sp2ms(MIN_DELAY)/(1000.0*TFACTOR))
IN.get_more_evidence()

#Trivial interpretation
interp = Interpretation()
#The focus is initially set in the first observation
interp.focus.append(next(obs_buffer.get_observations()))
##########################
### Construe searching ###
##########################
print('Starting interpretation')
t0 = time.time()
cntr = searching.Construe(interp, KFACTOR)
ltime = (cntr.last_time, t0)
#Main loop
while cntr.best is None:
    IN.get_more_evidence()
    acq_time = IN.get_acquisition_point()
    #HINT debug code
    fstr = 'Int: {0:05d} '
    for i in xrange(int(sp2ms(acq_time - cntr.last_time)/1000.0)):
        fstr += '-'
    fstr += ' Acq: {1}'
    print(fstr.format(int(cntr.last_time), acq_time))
    #End of debug code
    filt = ((lambda n : acq_time + n[0][2] >= MIN_DELAY)
                if obs_buffer.get_status() is obs_buffer.Status.ACQUIRING
                                                        else (lambda _ : True))
    cntr.step(filt)
    if cntr.last_time > ltime[0]:
        ltime = (cntr.last_time, time.time())
    #If the distance between acquisition time and interpretation time is
    #excessive, the search tree is pruned.
    if ms2sp((time.time()-ltime[1])*1000.0)*TFACTOR > MAX_DELAY:
        print('Pruning search')
        if cntr.open:
            prevopen = cntr.open
        cntr.prune()
print('Finished in {0:.3f} seconds'.format(time.time()-t0))
print('Created {0} interpretations'.format(interp.counter))

#Best explanation
#be = cntr.best.node
#be.recover_old()
#brview = plotter.plot_observations(sig_buf.get_signal(
#                                         sig_buf.get_available_leads()[0]), be)
##Drawing of the search tree
#label_fncs = {}
#label_fncs['n'] = lambda br: str(br)
#label_fncs['e'] = lambda br: ''
#brview = plotter.plot_branch(interp, label_funcs=label_fncs, target=be)

