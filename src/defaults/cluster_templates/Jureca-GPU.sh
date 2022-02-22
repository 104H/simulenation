#!/usr/bin/env bash

# Slurm job configuration
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --output=output_%j.out
#SBATCH --error=error_%j.er
#SBATCH --time={{ walltime }}
#SBATCH --gres=gpu:3 --partition={{ queue }}

module load TensorFlow/1.13.1-GPU-Python-3.6.8
module load Keras/2.2.4-GPU-Python-3.6.8

source /p/project/jinm60/software/nest-2.18.0-2019a-ParaStationMPI/bin/nest_vars.sh
source /p/project/jinm60/software/venv/bin/activate
export PYTHONPATH=/p/project/jinm60/software/venv/lib/python3.6/site-packages:${PYTHONPATH}

export PYTHONPATH=/p/project/jinm60/users/duarte1/fna-projects/:${PYTHONPATH}

cd $SLURM_SUBMIT_DIR
python {{ computation_script }}


