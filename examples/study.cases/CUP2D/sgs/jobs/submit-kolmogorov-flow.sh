#!/bin/bash

RUN=1
NGRID=256

RUNNAME="kolmogorov_flow_$RUN"

mkdir -p ./runs/
launchname="${0##*/}"
cp $launchname "./runs/${RUNNAME}.sh"
git diff > "./runs/gitdiff_${RUNNAME}.txt"

./sbatch-run-kolmogorov-flow.sh $RUNNAME $NGRID
