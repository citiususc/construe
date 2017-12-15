#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 11 13:52:20 2017

Tests different combinations of command line arguments with several
representative ECG records.

@author: tomas.teijeiro
"""

if __name__ == "__main__":
    import subprocess
    import random

    #Records are from the BroadInstitute collection, the MIT-BIH Arrhythmia
    #database, the Normal Sinus Rhythm Database, and the CinC 2017 challenge.
    RECORDS = ['1009638_6025_0_0', '100', '16272', '203', 'A00005']
    LEVELS = ['rhythm', 'conduction']
    EXCLUSIONS = ['', '--exclude-pwaves', '--exclude-twaves']
    LENGTHS = ['256', '2560', '25600', '256000']
    OVERLAPS = ['0', '256', '1080', '2560']
    TFACTORS = ['2', '20', '1e20']
    MIN_DELAYS = ['256', '2560', '5120']
    MAX_DELAYS = ['5', '10', '20']
    TIME_LIMITS = ['30', '60', '1e80']
    KVALS = ['1', '4', '12']
    MERGE = ['', '--no-merge']
    #Continuous test with different options
    while True:
        r = random.choice(RECORDS)
        #We get the record length
        desc = subprocess.check_output(['wfdbdesc', '/tmp/' + r]).splitlines()
        for l in desc:
            if l.startswith('Length:'):
                break
        nsamples = int(l.split()[-3][1:])
        #Half of the tests are performed with the full record
        if random.randint(0, 1) == 1:
            f = random.randint(0, nsamples-1)
            t = random.randint(f, nsamples)
        else:
            f, t = 0, nsamples
        f, t = str(f), str(t)
        l = random.choice(LEVELS)
        x = random.choice(EXCLUSIONS)
        ln = random.choice(LENGTHS)
        o = random.choice(OVERLAPS)
        tf = random.choice(TFACTORS)
        md = random.choice(MIN_DELAYS)
        MD = random.choice(MAX_DELAYS)
        tl = random.choice(TIME_LIMITS)
        k = random.choice(KVALS)
        m = random.choice(MERGE)
        if int(o) >= int(ln):
            continue
        command = ['python', 'construe_ecg.py', '-r', '/tmp/'+r, '--level', l,
                   '-f', f, '-t', t, '-l', ln, '--overl', o, '--tfactor', tf,
                   '-d', md, '-D', MD, '--time-limit', tl, '-k', k, '-v']
        if x != '':
            command.append(x)
        if m != '':
            command.append(m)
        print(' '.join(command))
        p = subprocess.Popen(command, bufsize=1)
        p.wait()
