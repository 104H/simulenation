#!/usr/bin/env bash

# Slurm job configuration
#SBATCH -A jinm60
#SBATCH --nodes={{ nodes }}
#SBATCH --ntasks=1
#SBATCH --cpus-per-task={{ ppn }}
#SBATCH --time={{ walltime }}
#SBATCH --partition={{ queue }}

# Load the required modules
module load GCC/9.3.0
module load ParaStationMPI/5.4.7-1
module load mpi4py/3.0.3-Python-3.8.5

# Activate python virtual env
export PATH="/p/project/jinm60/software/conda/anaconda3/bin:$PATH"
source activate modseq
export PYTHONPATH=$PYTHONPATH:/p/project/jinm60/software/conda/anaconda3/envs/modseq/lib/python3.8/site-packages

export OMP_NUM_THREADS={{ ppn }}
export MKL_NUM_THREADS={{ ppn }}
export NUMEXPR_MAX_THREADS={{ ppn }}
export NUMEXPR_NUM_THREADS={{ ppn }}

cd $SLURM_SUBMIT_DIR
srun python {{ computation_script }}


