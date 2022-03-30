#! /usr/bin/env bash

if [ $# -lt 2 ] ; then
	echo "Usage: ./sbatch-compare-vracer-swimmer.sh RUNNAME TASK"
	exit 1
fi

RUNNAME=$1
TASK=$2

# path to previous run
RESULTSPATH=/scratch/snx3000/pweber/korali/2D/_trainingResults

# number of agents
NAGENTS=100

# number of nodes per worker
NRANKS=16

# number of cores per worker
NUMCORES=12

# setup run directory and copy necessary files
RUNPATH="${SCRATCH}/korali/${RUNNAME}"
mkdir -p ${RUNPATH}
cp compare-vracer-swimmer ${RUNPATH}
cd ${RUNPATH}

cat <<EOF >daint_sbatch
#!/bin/bash -l
#SBATCH --account=s929
#SBATCH --constraint=gpu
#SBATCH --job-name="${RUNNAME}"
#SBATCH --output=${RUNNAME}_out_%j.txt
#SBATCH --error=${RUNNAME}_err_%j.txt
#SBATCH --time=24:00:00
#SBATCH --partition=normal
# #SBATCH --time=00:30:00
# #SBATCH --partition=debug
#SBATCH --nodes=$((NRANKS+1))

srun --nodes=$NRANKS --ntasks-per-node=$NUMCORES --cpus-per-task=1 --threads-per-core=1 ./compare-vracer-swimmer -resultsPath $RESULTSPATH -task $TASK -nAgents $NAGENTS -nRanks $(( $NRANKS * $NUMCORES )) : --nodes=1 --ntasks-per-node=1 --cpus-per-task=$NUMCORES --threads-per-core=1 ./compare-vracer-swimmer -resultsPath $RESULTSPATH -task $TASK -nAgents $NAGENTS -nRanks $(( $NRANKS * $NUMCORES ))
EOF

echo "Starting task ${TASK} with ${NWORKER} simulations each using ${NRANKS} ranks with ${NUMCORES} cores"
echo "----------------------------"

chmod 755 daint_sbatch
sbatch daint_sbatch
