# -*- coding: utf-8 -*-
# pylint: disable-msg=E1101, E0102, E0202
"""
Created on Wed Nov 21 10:37:17 2012

This module implements a test to perform the rhythm interpretation of all the
records in the MIT-BIH arrhythmia database, in order to prove the capability
of improve a basic beat detection through abduction.

@author: T. Teijeiro
"""
from construe.utils.units_helper import (msec2samples as ms2sp,
                                            samples2msec as sp2ms,
                                            set_ADCGain, set_sampling_freq)
set_ADCGain(1.0)
set_sampling_freq(250.0)
import construe.utils.plotting.plotter as plotter
import construe.acquisition.record_acquisition as IN
import construe.acquisition.obs_buffer as obs_buffer
import construe.acquisition.signal_buffer as sig_buf
import construe.knowledge.observables as o
import construe.knowledge.constants as C
import construe.inference.searching as searching
import construe.inference.reasoning as reasoning
import time
import numpy as np
from construe.model import Interval as Iv
from construe.model.interpretation import Interpretation

#Searching settings
KFACTOR = 12
MAX_DELAY = int(ms2sp(20000))
LENGTH = 10240
#Overlapping between consecutive fragments
FR_OVERLAP = int(ms2sp(3000))
FR_IDX = 6
INIT = 65510#int(FR_IDX * (LENGTH - FR_OVERLAP))
IN.reset()
#Standard annotator used
ANNOTATOR = 'gqrs'
#Record used
REC = '/home/local/tomas.teijeiro/cinc_challenge15/training/v111l'

IN.set_record(REC, ANNOTATOR, True)
IN.set_offset(INIT)
IN.set_duration(LENGTH)
IN.set_tfactor(1000.0)
IN.start()
time.sleep(1)
IN.get_more_evidence()

#Trivial interpretation
interp = Interpretation()
#The focus is initially set in the first observation
interp.focus.append(next(obs_buffer.get_observations()))
########################
### PEKBFS searching ###
########################
print('Starting interpretation')
t0 = time.time()
pekbfs = searching.PEKBFS(interp, KFACTOR)
ltime = (pekbfs.last_time, t0)
while pekbfs.best is None:
    IN.get_more_evidence()
    acq_time = IN.get_acquisition_point()
    #HINT debug code
    fstr = 'Int: {0:05d} '
    for i in range(int(sp2ms(acq_time - pekbfs.last_time)/1000.0)):
        fstr += '-'
    fstr += ' Acq: {1}'
    print(fstr.format(int(pekbfs.last_time), acq_time))
    #End of debug code
    pekbfs.step()
    if pekbfs.last_time > ltime[0]:
        ltime = (pekbfs.last_time, time.time())
    if ms2sp((time.time()-ltime[1])*1000.0) > MAX_DELAY:
        print('Pruning search')
        if pekbfs.open:
            prevopen = pekbfs.open
        pekbfs.prune()
print('Finished in {0:.3f} seconds'.format(time.time()-t0))
print('Created {0} interpretations'.format(interp.counter))
be = pekbfs.best
brview = plotter.plot_observations(sig_buf.get_signal(
                                sig_buf.get_available_leads()[0]), pekbfs.best)

#Branches draw
label_fncs = {}
label_fncs['n'] = lambda br: str(br)
label_fncs['e'] = lambda br: ''
#brview = plotter.plot_branch(interp, label_funcs=label_fncs, target=pekbfs.best)
