#! /usr/bin/env bash

#setup run directory and copy necessary files
#SBATCH --ntasks-per-node=1
#SBATCH --partition=debug

#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12

RUNPATH="${SCRATCH}/korali/Opti_NP_fft"
APHROSPATH="${SCRATCH}/korali/Opti_NP_fft/Results"
mkdir -p ${APHROSPATH}
mkdir -p ${RUNPATH}
cp -R /users/dsolland/korali/examples/optimization/Opti_NP_fft/* ${RUNPATH}
cp -R /users/dsolland/aphros-dev/sim/sim45_fans/* ${APHROSPATH}
cd ${RUNPATH}

module load daint-gpu GSL cray-hdf5-parallel cray-python


cat <<EOF >daint_sbatch
#!/bin/bash -l
#SBATCH --job-name="3x3 FFT"
#SBATCH --account="s929"
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --constraint=gpu


python ./run-cmaes.py
EOF

sbatch daint_sbatch
