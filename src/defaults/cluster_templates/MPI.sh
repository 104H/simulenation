#!/bin/bash
#$ -cwd
#$ -q {{ queue }}.q
#$ -S /bin/bash

cd $SGE_O_WORKDIR
python {{ computation_script }}