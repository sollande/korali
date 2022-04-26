#!/bin/bash

mlgnu

export RUN=2

mkdir -p ./runs/
launchname="${0##*/}"
cp $launchname "./runs/vracer_launcher_${RUN}.sh"
git diff > "./runs/gitdiff_vracer_${RUN}.txt"

./sbatch-run-vracer.sh $RUN
