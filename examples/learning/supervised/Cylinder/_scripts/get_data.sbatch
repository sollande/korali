#!/bin/bash -l
#
#SBATCH --time=10
#SBATCH --ntasks=1
#SBATCH --partition=xfer
SCRIPT_DIR=$(dirname "$0")
# TODO: script_dir is unfortunately not working with slurm permission somehow
srun rsync -avz /project/s929/pollakg/cyl/_data ./
