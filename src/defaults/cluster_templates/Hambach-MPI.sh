#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks={{ ppn }}
#SBATCH --cpus-per-task=1
#SBATCH --ntasks-per-node={{ ppn }}
#SBATCH --time={{ walltime }}
#SBATCH --partition={{ queue }}
#SBATCH --mem={{ mem }}

## activate conda and set NEST variables
module load mpi/mpich/3.2
#module load mpi/openmpi/4.0.3rc4

#export PYTHONPATH=/users/duarte/fna-projects/:${PYTHONPATH}
#export PYTHONPATH=/users/duarte/func-neurarch/:${PYTHONPATH}

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}

cd $SLURM_SUBMIT_DIR

mpirun -np {{ ppn }} python {{ computation_script }}
