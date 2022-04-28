#!/bin/bash

RUN=4
NGRID=1024

RUNNAME="kolmogorov_flow_$RUN_$NGRID"

mkdir -p ./runs/
launchname="${0##*/}"
cp $launchname "./runs/${RUNNAME}.sh"
git diff > "./runs/gitdiff_${RUNNAME}.txt"

module load daint-gpu gcc GSL/2.7-CrayGNU-21.09 cray-hdf5-parallel cray-python cdt-cuda craype-accel-nvidia60

./sbatch-run-kolmogorov-flow.sh $RUNNAME $NGRID
