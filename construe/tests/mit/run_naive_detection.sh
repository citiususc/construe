#!/bin/bash
#PBS -l nodes=1:ppn=64,walltime=24:00:00
#PBS -N naive_validation
#PBS -m ae -M tomas.teijeiro@usc.es
module load wfdb
cd $HOME/interpreter

#We test all the database
for i in $(seq 0 47); do
  let end=$i+1
  python interpreter/tests/mit_validation/regularity_beat_detector.py $i:$end\
    /sfs/users/tomas.teijeiro/mit/ /home/local/tomas.teijeiro/validacion/naive/ &
done
wait
