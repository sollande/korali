#!/bin/bash

mlgnu

RUN=2

RUNNAME="vracer_$RUN"

mkdir -p ./runs/
launchname="${0##*/}"
cp $launchname "./runs/${RUNNAME}.sh"
git diff > "./runs/gitdiff_${RUNNAME}.txt"

./sbatch-run-vracer.sh $RUNNAME
