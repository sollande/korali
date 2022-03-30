#! /usr/bin/env bash

if [ $# -lt 1 ] ; then
	echo "Usage: ./sbatch-eval-vracer-swimmer.sh RUNNAME "
	exit 1
fi

RUNNAME=$1

# number of workers
NWORKER=1

# number of nodes per worker
NRANKS=3

# number of cores per worker
NUMCORES=12

# number of workers * number of nodes per worker
NNODES=$(( $NWORKER * $NRANKS ))

# setup run directory and copy necessary files
RUNPATH="${SCRATCH}/korali/${RUNNAME}"
cp eval-vracer-swimmer ${RUNPATH}
cd ${RUNPATH}

source settings.sh

cat <<EOF >daint_sbatch_testing
#!/bin/bash -l
#SBATCH --account=s929
#SBATCH --constraint=gpu
#SBATCH --job-name="${RUNNAME}"
#SBATCH --partition=debug
#SBATCH --nodes=$((NNODES+1))


srun --nodes=$NNODES --ntasks-per-node=12 --cpus-per-task=1 --threads-per-core=1 ./eval-vracer-swimmer ${OPTIONS} -factory-content $(printf "%q" "${FACTORY}") -nAgents $NAGENTS -nRanks $(( $NRANKS * 12 )) : --nodes=1 --ntasks-per-node=1 --cpus-per-task=12 --threads-per-core=1 ./eval-vracer-swimmer ${OPTIONS} -factory-content $(printf "%q" "${FACTORY}") -nAgents $NAGENTS -nRanks $(( $NRANKS * 12 ))

EOF

echo "Starting testing with ${NWORKER} simulations each using ${NRANKS} ranks with ${NUMCORES} cores"
echo "----------------------------"

chmod 755 daint_sbatch_testing
sbatch daint_sbatch_testing
