#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --time=00-24:00:00
#SBATCH --partition=batch
#SBATCH --mem=64000

## activate conda and set NEST variables
source /users/duarte/miniconda3/bin/activate
conda activate fna

module load mpi/openmpi/4.0.3rc4

source /users/duarte/nest-engine/nest-simulator-2.20.0-install/bin/nest_vars.sh

export PYTHONPATH=/users/duarte/fna-projects/:${PYTHONPATH}
export PYTHONPATH=/users/duarte/func-neurarch/:${PYTHONPATH}

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}

cd $SLURM_SUBMIT_DIR

orterun python /home/pbr-student/simulenation/src/demyelination/data/export/test_run/test_run_NE=100_T=1.py
