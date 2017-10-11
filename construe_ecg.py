#!/usr/bin/env python2
# -*- coding: utf-8 -*-
# pylint: disable-msg= C0103, C0325
"""
Created on Mon Oct  9 12:58:30 2017

Main script to provide an WFDB-like command line interface for ECG records
interpretation using the Construe algorithm.

@author: tomas.teijeiro
"""

if __name__ == '__main__':
    import argparse
    import numpy as np
    import os
    import os.path
    import subprocess
    import time
    from construe.utils.MIT import (get_gain, get_sampling_frequency,
                                    read_annotations, save_annotations)
    from construe.utils.units_helper import set_ADCGain, set_sampling_freq

    parser = argparse.ArgumentParser(description=
        'Interprets a MIT-BIH ECG record in multiple abstraction levels, '
        'generating as a result a set of annotations encoding the observation '
        'hypotheses.')
    parser.add_argument('-r', metavar='record', required=True,
                        help='Name of the record to be processed')
    parser.add_argument('-a', metavar='ann', default=None,
                        help=('Annotator containing the initial evidence. If '
                               'not provided, the gqrs application is used.'))
    parser.add_argument('-o', metavar='oann', default='iqrs',
                        help=('Save annotations as annotator oann '
                              '(default: iqrs)'))
    parser.add_argument('--level', choices=['conduction', 'rhythm'],
                        default='rhythm',
                        help=('Highest abstraction level used in the '
                              'interpretation. Using the "conduction" level '
                              'produces just a wave delineation for each '
                              'QRS annotation in the initial evidence, while '
                              'the "rhythm" level also includes a rhythm '
                              'interpretation of the full signal, but at the '
                              'expense of a higher computational cost in '
                              'several orders of magnitude.'))
    parser.add_argument('--exclude-pwaves', action='store_true',
                        help=('Avoids searching for P-waves. Default:False'))
    parser.add_argument('--exclude-twaves', action='store_true',
                        help=('Avoids searching for T-waves. It also implies '
                              '--exclude-pwaves Default:False'))
    parser.add_argument('-f', metavar='init', default=0, type=int,
                        help=('Begin the interpretation at the "init" time, '
                              'in samples'))
    parser.add_argument('-t', metavar='stop', default=np.inf, type=float,
                        help=('Stop the interpretation at the "stop" time, '
                              'in samples'))
    parser.add_argument('-l', metavar='length', default=0, type=int,
                        help=('Length in samples of each independently '
                              'interpreted fragment. It has to be multiple '
                              'of 256. Default:23040 if the abstraction level'
                              ' is "rhythm", and 640000 if the abstraction '
                              'level is "conduction".'))
    parser.add_argument('--overl', default=1080, type=int,
                        help=('Length in samples of the overlapping between '
                              'consecutive fragments, to prevent loss of '
                              'information. If the selected abstraction level '
                              'is "conduction", this parameter is ignored. '
                              'Default: 1080.'))
    parser.add_argument('--tfactor', default=1e20, type=float,
                        help=('Time factor to control the speed of the input '
                              'signal. For example, if tfactor = 2.0 two '
                              'seconds of new signal are added to the signal '
                              'buffer each real second. A value of 1.0 '
                              'simulates real-time online interpretation. If '
                              'the selected abstraction level is "conduction",'
                              ' this parameter is ignored. Default: 1e20'))
    parser.add_argument('-d', metavar='min_delay', default=2560, type=int,
                        help=('Minimum delay in samples between the '
                              'acquisition time and the last interpretation '
                              'time. If the selected abstraction level is '
                              '"conduction", this parameter is ignored. '
                              'Default: 2560'))
    parser.add_argument('-D', metavar='max_delay', default=20.0, type=float,
                        help=('Maximum delay in seconds that the '
                              'interpretation can be without moving forward. '
                              'If this threshold is exceeded, the searching '
                              'process is pruned. If the selected abstraction '
                              'level is "conduction", this parameter is '
                              'ignored. Default: 20.0'))
    parser.add_argument('--time-limit', default=np.inf, type=float,
                        help=('Interpretation time limit *for each fragment*.'
                              'If the interpretation time exceeds this number '
                              'of seconds, the interpretation finishes '
                              'immediately, moving to the next fragment. '
                              'If the selected abstraction level is '
                              '"conduction", this parameter is ignored. '
                              'Default: Infinity'))
    parser.add_argument('-k', default=12, type=int,
                        help=('Exploration factor. It is the number of '
                              'interpretations expanded in each searching '
                              'cycle. Default: 12. If the selected '
                              'abstraction level is "conduction", this '
                              'parameter is ignored.'))
    parser.add_argument('-v', action='store_true',
                        help=('Verbose mode. The algorithm will print to '
                              'standard output the fragment being '
                              'interpreted.'))
    parser.add_argument('--no-merge', action='store_true',
                        help=('Avoids the use of a branch-merging strategy for'
                              ' interpretation exploration. If the selected '
                              'abstraction level is "conduction", this '
                              'parameter is ignored.'))
    args = parser.parse_args()
    #The first step is to set the global frequency and ADC gain variables that
    #determine the constant values in the knowledge base, which are initialized
    #in the first import of a construe knowledge module.
    set_ADCGain(get_gain(args.r))
    set_sampling_freq(get_sampling_frequency(args.r))
    #The initial evidence is now obtained
    if args.a is None:
        #A temporary annotations file with the 'gqrs' application is created,
        #loaded and immediately removed. We ensure that no collisions occur
        #with other annotators.
        aname = 0
        gqname = 'gq{0:02d}'
        rname, ext = os.path.splitext(args.r)
        while os.path.exists(rname + '.' + gqname.format(aname)):
            aname += 1
        command = ['gqrs', '-r', rname, '-outputName', gqname.format(aname)]
        subprocess.check_call(command)
        annpath = rname + '.' + gqname.format(aname)
        annots = read_annotations(annpath)
        os.remove(annpath)
    else:
        rname, ext = os.path.splitext(args.r)
        annots = read_annotations(rname + '.' + args.a)
    t0 = time.time()
    #Conduction or rhythm interpretation
    if args.level == 'conduction':
        from record_processing import process_record_conduction
        length = 512000 if args.l == 0 else args.l
        result = process_record_conduction(rname, annots, length, args.f,
                                           args.t, args.exclude_pwaves,
                                           args.exclude_twaves, args.v)
    else:
        from record_processing import process_record_rhythm
        #Merge strategy
        import construe.inference.reasoning as reasoning
        reasoning.MERGE_STRATEGY = not args.no_merge
        length = 23040 if args.l == 0 else args.l
        overl = 1080 if args.overl == -1 else args.overl
        result = process_record_rhythm(rname, annots, args.tfactor,
                                       length, overl, args.time_limit,
                                       args.d, args.D, args.k, args.f,
                                       args.t, args.exclude_pwaves,
                                       args.exclude_twaves, args.v)
    save_annotations(result, args.r + '.' + args.o)
    print('Record ' + args.r + ' succesfully processed')
    if args.v:
        from construe.model.interpretation import Interpretation
        from construe.utils.units_helper import samples2msec as sp2ms
        import construe.acquisition.record_acquisition as IN
        idur = time.time() - t0
        print('Interpretation time: {0:.03f} seconds. Global Real-time factor: '
              '{1:.03f}. Created {2} interpretations.'.format(idur,
              sp2ms(min(args.t, IN.get_record_length())-args.f)/(idur*1000.),
              Interpretation.counter))
