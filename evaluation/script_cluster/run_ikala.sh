#!/bin/bash

#$ -N ikala
#$ -cwd
#$ -q short.q
#$ -t 1-252:1
# ------------------------------------------------

# Start script
# --------------------------------
#
#printf "Starting execution of job $JOB_ID from user $SGE_O_LOGNAME\n"
#printf "Starting at `date`\n"
#printf "Calling Matlab now\n"
#printf "---------------------------\n"
echo ${SGE_TASK_ID}
/soft/MATLAB/R2013b/bin/matlab -nodisplay -nosplash -nodesktop  -singleCompThread -r "cd ..;evaluate_SS_iKala($SGE_TASK_ID,'/homedtic/mmiron/data/iKala/','fft_2048');"
# Copy data back, if any
#printf "---------------------------\n"
#printf "Matlab processing done.\n"
#printf "Job done. Ending at `date`\n"
