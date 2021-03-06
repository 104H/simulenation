#!/bin/bash

#SBATCH --nodes={{ nodes }}
#SBATCH --ntasks=1
##SBATCH --cpus-per-task={{ ppn }}
#SBATCH --cpus-per-task=1
#SBATCH --time={{ walltime }}
##SBATCH --partition={{ queue }}
#SBATCH --partition=hambach,blaustein,hamstein
##SBATCH --mem={{ mem }}
#SBATCH --mem=6000

## activate conda and set NEST variables
module load mpi/mpich/3.2
conda activate nest

#module load mpi/openmpi/4.0.3rc4

#export PYTHONPATH=/users/duarte/fna-projects/:${PYTHONPATH}
#export PYTHONPATH=/users/duarte/func-neurarch/:${PYTHONPATH}

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}

cd $SLURM_SUBMIT_DIR

srun python {{ computation_script }}
