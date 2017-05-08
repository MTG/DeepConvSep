#!/bin/bash

#$ -N dsd100_bss
#$ -l h_vmem=15G
#$ -cwd
#$ -q default.q
#$ -t 1-100:1
#$ -e $HOME/logs/$JOB_NAME-$JOB_ID.err
#$ -o $HOME/logs/$JOB_NAME-$JOB_ID.out
# ------------------------------------------------

# Start script
# --------------------------------
#
#printf "Starting execution of job $JOB_ID from user $SGE_O_LOGNAME\n"
#printf "Starting at `date`\n"
#printf "Calling Matlab now\n"
#printf "---------------------------\n"
echo ${SGE_TASK_ID}
/soft/MATLAB/R2013b/bin/matlab -nodisplay -nosplash -nodesktop  -singleCompThread -r "DSD100_eval_only('/homedtic/gerruz/data/DSD100/','/homedtic/gerruz/data/output100/',$SGE_TASK_ID);"
# Copy data back, if any
#printf "---------------------------\n"
#printf "Matlab processing done.\n"
#printf "Job done. Ending at `date`\n"
