#!/bin/bash

#$ -N bach10_bss
#$ -cwd
#$ -q short.q
#$ -t 1-60:1
# ------------------------------------------------

# Start script
# --------------------------------
#
#printf "Starting execution of job $JOB_ID from user $SGE_O_LOGNAME\n"
#printf "Starting at `date`\n"
#printf "Calling Matlab now\n"
#printf "---------------------------\n"
echo ${SGE_TASK_ID}
/soft/MATLAB/R2013b/bin/matlab -nodisplay -nosplash -nodesktop  -singleCompThread -r "cd ..;Bach10_eval_only('/homedtic/mmiron/data/Bach10/','/homedtic/mmiron/data/Bach10/output/','/homedtic/mmiron/data/Bach10/results/',$SGE_TASK_ID);"
# Copy data back, if any
#printf "---------------------------\n"
#printf "Matlab processing done.\n"
#printf "Job done. Ending at `date`\n"
