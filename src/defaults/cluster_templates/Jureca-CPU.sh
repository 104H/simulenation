#!/usr/bin/env bash

# Slurm job configuration
#SBATCH --nodes={{ nodes }}
#SBATCH --ntasks-per-node=24
#SBATCH --cpus-per-task={{ ppn }}
#SBATCH --time={{ walltime }}
#SBATCH --partition={{ queue }}

export PYTHONPATH=/p/project/jinm60/users/duarte1/fna-projects/:${PYTHONPATH}
export PYTHONPATH=/p/project/jinm60/users/duarte1/func-neurarch/:${PYTHONPATH}

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}

cd $SLURM_SUBMIT_DIR
srun python {{ computation_script }}


