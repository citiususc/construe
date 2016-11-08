#!/bin/bash
#PBS -l nodes=2:ppn=32:intel:xeonl,walltime=48:00:00
#PBS -N bmi_data
#PBS -m ae -M tomas.teijeiro@usc.es
getent passwd
module load wfdb
cd $HOME/interpreter
#We load the job pool functionality
source interpreter/tests/cinc15/job_pool.sh

# initialize the job pool to allow 48 parallel jobs and echo commands
job_pool_init 48 1

#We test all the database using the job pool
for i in $(seq 0 24); do
  let end=$i+1
  job_pool_run python interpreter/tests/bmi/data_interpretation.py $i:$end\
    /home/local/tomas.teijeiro/databases/bmi/validation/ /home/local/tomas.teijeiro/validacion/bmi/validation/
done

# wait until all jobs complete before continuing
job_pool_wait
# shut down the job pool
job_pool_shutdown
# check the $job_pool_nerrors for the number of jobs that exited non-zero
echo "job_pool_nerrors: ${job_pool_nerrors}"
