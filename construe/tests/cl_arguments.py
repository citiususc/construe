#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 11 13:52:20 2017

Tests a number of combinations of command line arguments with several
representative ECG records.

@author: tomas.teijeiro
"""

if __name__ == "__main__":
    import itertools
    import subprocess

    #Records are from the BroadInstitute collection, the MIT-BIH Arrhythmia
    #database, the Normal Sinus Rhythm Database, and the CinC 2017 challenge.
    RECORDS = ['1009638_6025_0_0', '100', '16272', '203', 'A00005']
    LEVELS = ['rhythm', 'conduction']
    EXCLUSIONS = ['', '--exclude-pwaves', '--exclude-twaves']
    LENGTHS = ['256', '2560', '25600', '256000']
    OVERLAPS = ['0', '256', '1080', '2560']
    TFACTORS = ['2', '20', '1e20']
    MAX_DELAYS = ['5', '10', '20']
    TIME_LIMITS = ['30', '60', '1e80']
    KVALS = ['1', '4', '12']
    MERGE = ['', '--no-merge']
    for r,l,x,ln,o,tf,MD,tl,k,m in itertools.product(RECORDS, LEVELS,
                                                        EXCLUSIONS, LENGTHS,
                                                        OVERLAPS, TFACTORS,
                                                        MAX_DELAYS, TIME_LIMITS,
                                                        KVALS, MERGE):
        if r == '1009638_6025_0_0':
            if (int(ln) < 25600 or float(tf) < 1e20 or int(MD) != 10
                    or float(tl) != 60 or int(k) == 1):
                continue
        command = ['python', 'construe_ecg.py', '-r', '/tmp/'+r, '--level', l,
                   '-l', ln, '--overl', o, '--tfactor', tf, '-D', MD,
                   '--time-limit', tl, '-k', k, '-v']
        if x != '':
            command.append(x)
        if m != '':
            command.append(m)
        print(command)
        p = subprocess.Popen(command, bufsize=1)
#        for line in iter(p.stdout.readline, b''):
#            print(line)
#        p.stdout.close()
        p.wait()




