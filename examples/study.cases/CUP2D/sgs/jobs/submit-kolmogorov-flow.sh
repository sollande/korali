#!/bin/bash

RUN=4
NGRID=1024

RUNNAME="kolmogorov_flow_$RUN_$NGRID"

mkdir -p ./runs/
launchname="${0##*/}"
cp $launchname "./runs/${RUNNAME}.sh"
git diff > "./runs/gitdiff_${RUNNAME}.txt"

./sbatch-run-kolmogorov-flow.sh $RUNNAME $NGRID
