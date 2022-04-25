#!/bin/bash

export RUN=1

launchname="${0##*/}"
cp $launchname "./vracer_launcher_${RUN}.sh"


git diff > "./gitdiff_vracer_${RUN}.txt"

./sbatch-run-vracer.sh $RUN
