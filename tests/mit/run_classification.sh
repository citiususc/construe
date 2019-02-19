#!/bin/bash
cd /datos/tomas.teijeiro/Dropbox/Investigacion/tese/python/interpreter

python -m interpreter.knowledge.beat_classification 0:16 > /tmp/d1.csv &
python -m interpreter.knowledge.beat_classification 16:32 > /tmp/d2.csv &
python -m interpreter.knowledge.beat_classification 32:48 > /tmp/d3.csv &

wait

cat /tmp/d1.csv /tmp/d2.csv /tmp/d3.csv > /tmp/d.csv

