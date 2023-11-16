#!/bin/bash

#SBATCH --qos=blanca-csdms
#SBATCH --job-name=igm-eqip-sermia
#SBATCH --output=igm-eqip-sermia.out
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1

module purge
module load anaconda
conda activate igm

export TF_CPP_MIN_LOG_LEVEL=2

igm_run
