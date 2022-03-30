#! /usr/bin/env bash

if [ $# -lt 1 ] ; then
	echo "Usage: ./sbatch-run-vracer-swimmer.sh RUNNAME"
	exit 1
fi
if [ $# -gt 0 ] ; then
	RUNNAME=$1
fi

# number of agents in the environment
NAGENTS=1

# number of workers
NWORKER=32

# number of nodes per worker
NRANKS=4

# number of workers * number of nodes per worker
NNODES=$(( $NWORKER * $NRANKS ))

# setup run directory and copy necessary files
RUNPATH="${SCRATCH}/korali/${RUNNAME}"
mkdir -p ${RUNPATH}
cp run-vracer-swimmer ${RUNPATH}
cp settings.sh ${RUNPATH}
cp run-vracer-swimmer.cpp ${RUNPATH}
cp -r _model ${RUNPATH}
cd ${RUNPATH}

source settings.sh

cat <<EOF >daint_sbatch_training
#!/bin/bash -l
#SBATCH --account=s929
#SBATCH --constraint=gpu
#SBATCH --job-name="${RUNNAME}"
#SBATCH --time=24:00:00
#SBATCH --partition=normal
#SBATCH --nodes=$((NNODES+1))

# Korali engine gets 12 threads and 1 rank, CUP gets 1 threads and 12 ranks
srun --nodes=$NNODES --ntasks-per-node=12 --cpus-per-task=1 --threads-per-core=1 ./run-vracer-swimmer ${OPTIONS} -factory-content $(printf "%q" "${FACTORY}") -nAgents $NAGENTS -nRanks $(( $NRANKS * 12 )) : --nodes=1 --ntasks-per-node=1 --cpus-per-task=12 --threads-per-core=1 ./run-vracer-swimmer ${OPTIONS} -factory-content $(printf "%q" "${FACTORY}") -nAgents $NAGENTS -nRanks $(( $NRANKS * 12 ))
EOF

chmod 755 daint_sbatch_training
sbatch daint_sbatch_training
