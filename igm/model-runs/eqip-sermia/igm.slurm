#!/bin/bash

#SBATCH --qos=blanca-csdms
#SBATCH --job-name=igm-fast
#SBATCH --output=igm-fast.%j.out
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1

module purge
module load anaconda
conda activate igm

igm_run