#!/bin/bash

mlgnu

export RUN=2
export NGRID=256



mkdir -p ./runs/
launchname="${0##*/}"
cp $launchname "./runs/flow_launcher_${RUN}.sh"
git diff > "./runs/gitdiff_flow_${RUN}.txt"

./sbatch-run-kolmogorov-flow.sh $RUN
