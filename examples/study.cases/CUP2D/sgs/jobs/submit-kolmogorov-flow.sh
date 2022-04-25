#!/bin/bash

export RUN=1
export NGRID=2000000

launchname="${0##*/}"
cp $launchname "./flow_launcher_${RUN}.sh"


git diff > "./gitdiff_flow_${RUN}.txt"

./sbatch-run-kolmogorov-flow.sh $RUN
