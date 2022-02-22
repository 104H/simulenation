#!/usr/bin/env bash

###############################################################################################
# Additional command line arguments are:
# $1 - offset   # increase for each submission
# $2 - ntasks    # should match --ntasks
#
# Script to run multiple simulations within a single Slurm submission. Each simulation will run
# as a separate task, with its own dedicated CPUs. 
#
# Usage: 
#   --ntasks=X # should correspond to the number of simulations to be ran as part of the job
#   --cpus-per-task=X # number of CPUs dedicated to each task (simulation)
#   Call this script multiple times (with increasing offsets) to cover all jobs defined in 
#   job_list.txt
#
# Example:
#   job_list.txt contains 100 jobs/simulations, and we want to run 20 simulations 
#   simultaneously within a single job. Then we need 5 calls (or create a Python wrapper):
#   > submit_multiple.sh 0 20
#   > submit_multiple.sh 1 20
#   > submit_multiple.sh 2 20
#   > submit_multiple.sh 3 20
#   > submit_multiple.sh 4 20
###############################################################################################

# Slurm job configuration
#SBATCH -A jinm60
#SBATCH --nodes=1
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=64
#SBATCH --time=00-02:00:00
#SBATCH --partition=dc-cpu-devel


# Load the required modules
module load GCC/9.3.0
module load ParaStationMPI/5.4.7-1
module load mpi4py/3.0.3-Python-3.8.5

# Activate python virtual env
export PATH="/p/project/jinm60/software/conda/anaconda3/bin:$PATH"
source activate modseq
export PYTHONPATH=$PYTHONPATH:/p/project/jinm60/software/conda/anaconda3/envs/modseq/lib/python3.8/site-packages

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export NUMEXPR_MAX_THREADS=${SLURM_CPUS_PER_TASK}
export NUMEXPR_NUM_THREADS=${SLURM_CPUS_PER_TASK}

echo "Submitting jobs with config \#CPUS: ${SLURM_CPUS_PER_TASK} and \#nTasks: ${SLURM_NTASKS}!";

cd $SLURM_SUBMIT_DIR

###############################################################################################
############################   Interesting stuff down here    #################################
###############################################################################################
cnt=0;      # job counter
offset=$1;  # offset to select corresponding subset of jobs/simulations in job_list.txt
ntasks=$2;   # should match --ntasks

# iterate through the corresponding subset of simulations (tasks) in job_list.txt, 
# and submit each one in the background.

while read p; do
    tmpOffsetStart=$(( offset * ntasks ))
    tmpOffsetEnd=$(( offset * ntasks + ntasks ))
        if [[ "$cnt" -ge "$tmpOffsetStart" && "$cnt" -lt "$tmpOffsetEnd" ]]; then
          echo "${p/\.sh/\.py}";
          srun --exclusive --ntasks=1 python3 "${p/\.sh/\.py}" &
        fi
    cnt=$(( cnt + 1 ))
done < job_list.txt  

wait  # this ensures that the Slurm job only exits when all simulations have ended and returned



