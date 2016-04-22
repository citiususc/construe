# -*- coding: utf-8 -*-
# pylint: disable-msg=E1101, E0102, E0202
"""
Created on Mon March 21 11:37:17 2016

Small test to perform the full interpretation of a fragment of a record.

@author: T. Teijeiro
"""

import construe.utils.plotting.plotter as plotter
import construe.acquisition.record_acquisition as IN
import construe.acquisition.obs_buffer as obs_buffer
import construe.acquisition.signal_buffer as sig_buf
import construe.inference.searching as searching
import time
from construe.model.interpretation import Interpretation
from construe.utils.units_helper import (msec2samples as ms2sp,
                                            samples2msec as sp2ms)

#Record to interpret
REC = 'examples/fig10'
#Annotator used for the initial evidence
ANNOTATOR = 'qrs'
#Initial position and length for the interpretation (in samples)
INIT = 0
#Length has to be multiple of IN._STEP
LENGTH = 3840
#Searching settings
TFACTOR = 5.0
KFACTOR = 12
MIN_DELAY = 1750
MAX_DELAY = int(ms2sp(20000)*TFACTOR)
#Input system configuration
IN.reset()
IN.set_record(REC, ANNOTATOR)
IN.set_offset(INIT)
IN.set_duration(LENGTH)
IN.set_tfactor(TFACTOR)
IN.start()
print('Preloading buffer...')
time.sleep(sp2ms(MIN_DELAY)/(1000.0*TFACTOR))
#Load the initial evidence
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
be = cntr.best.node
be.recover_old()
#Drawing of the best explanation
brview = plotter.plot_observations(sig_buf.get_signal(
                                         sig_buf.get_available_leads()[0]), be)
#Drawing of the search tree
label_fncs = {}
label_fncs['n'] = lambda br: str(br)
label_fncs['e'] = lambda br: ''
brview = plotter.plot_branch(interp, label_funcs=label_fncs, target=be,
                             full_tree=True)
