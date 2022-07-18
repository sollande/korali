#! /usr/bin/env bash

#setup run directory and copy necessary files
#SBATCH --ntasks-per-node=1
#SBATCH --partition=debug

#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12

RUNPATH="${SCRATCH}/korali/Test"
mkdir -p ${RUNPATH}
cp -R /users/dsolland/korali/examples/optimization/Test/* ${RUNPATH}
cd ${RUNPATH}

module load daint-gpu GSL cray-hdf5-parallel cray-python


cat <<EOF >daint_sbatch
#!/bin/bash -l
#SBATCH --job-name="opti 3x3 fans Motus data"
#SBATCH --account="s929"
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --constraint=gpu


srun ./run-cmaes.py
EOF

sbatch daint_sbatch
