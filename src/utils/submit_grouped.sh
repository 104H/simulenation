#!/usr/bin/env bash

# Example script to generate scripts to be run on a cluster. This script should not called directly as
# as a template; some values will be replaced by NMSAT during export. For more details see the NMSAT d

# NOTE: this example is specific for a SLURM queueing system.

#SBATCH -A jinm60
#SBATCH -N 1
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=64
#SBATCH --time 02:00:00
#SBATCH --mem=500000
#SBATCH --mail-type=ALL
#SBATCH --mail-user=b.zajzon@fz-juelich.de
#SBATCH --partition=dc-cpu-devel
##SBATCH --output=mtest_rec-%j.out

x-modseq

#env | grep NEST
#env | grep PYTHONPATH

cd $SLURM_SUBMIT_DIR
# do not modify the following line!

while read p; do
    tmpOffsetStart=$(( offset * ntasks ))
    tmpOffsetEnd=$(( offset * ntasks + ntasks ))
        if [[ "$cnt" -ge "$tmpOffsetStart" && "$cnt" -lt "$tmpOffsetEnd" ]]; then
          echo "${p/\.sh/\.py}";
          srun --exclusive --ntasks=1 python3 "${p/\.sh/\.py}" &
        fi
    cnt=$(( cnt + 1 ))
done < job_list.txt

wait
