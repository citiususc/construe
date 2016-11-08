# -*- coding: utf-8 -*-
# pylint: disable=
"""
Created on Thu May 21 16:51:51 2015

Small utility script to slightly change the format of the reports given to
the medical doctor for the validation.

@author: T. Teijeiro
"""

import datetime as dt
import re

date = dt.date(2015, 4, 21)
time = dt.time(21, 58, 53)
ref = dt.datetime.combine(date, time)


with open('/tmp/f.txt') as f:
    for line in f:
        if line[-1]=='\n':
            line = line[:-1]
        beg, end = re.findall('(\d\d):(\d\d):(\d\d).(\d\d\d)', line)
        (bh, bm, bs, bms), (eh, em, es, ems) = re.findall(
                                         '(\d\d):(\d\d):(\d\d).(\d\d\d)', line)
        btime = dt.datetime.combine(date, dt.time(int(bh), int(bm), int(bs),
                                                  int(bms)*1000))
        etime = dt.datetime.combine(date, dt.time(int(eh), int(em), int(es),
                                                  int(ems)*1000))
        print('{0}, {1:04d}s - {2}, {3:04d}s ({4})'.format(line[:14],
              int((btime-ref).total_seconds()), line[19:],
              int((etime-ref).total_seconds()), str(etime-btime)[2:]+'.000'))
#        h, m, s, ms = re.findall('(\d\d):(\d\d):(\d\d).(\d\d\d)', line)[0]
#        dtime = dt.datetime.combine(date, dt.time(int(h), int(m), int(s),
#                                                  int(ms)*1000))
#        print('{0} - {1:04d}s {2}'.format(line[:15],
#                                int((dtime-ref).total_seconds()), line[16:-1]))

#    lines = list(f)
#    startimes = []
#    for line in lines:
#        h, m, s, ms = re.findall('(\d\d):(\d\d):(\d\d).(\d\d\d)', line)[0]
#        h, m, s, ms = int(h), int(m), int(s), int(ms)
#        startimes.append(dt.datetime(1900, 1, 1, hour=h, minute=m, second=s,
#                                     microsecond=ms*1000))
#    for i in xrange(len(lines)-1):
#        line = lines[i]
#        duration = startimes[i+1]-startimes[i]
#        print '{0} ({1}):{2}'.format(line[:15], str(duration)[2:-3], line[16:-1])

