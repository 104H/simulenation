#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --time=00-24:00:00
##SBATCH --partition=batch
#SBATCH --partition=hambach,blaustein,hamstein
##SBATCH --mem={{ mem }}

## activate conda and set NEST variables
module load mpi/mpich/3.2
conda activate nest

#module load mpi/openmpi/4.0.3rc4

#export PYTHONPATH=/users/duarte/fna-projects/:${PYTHONPATH}
#export PYTHONPATH=/users/duarte/func-neurarch/:${PYTHONPATH}

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}

cd $SLURM_SUBMIT_DIR

srun python /users/hameed/simulenation/src/demyelination/data/export/test_run/test_run_nuX=9_gamma=4_T=1.py
