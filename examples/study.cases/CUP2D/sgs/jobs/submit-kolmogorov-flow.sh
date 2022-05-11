#!/bin/bash

RUN=0
#GRIDS=( 128 )
GRIDS=( 256 512 1024 2048 )

for NGRID in "${GRIDS[@]}"
do
    RUNNAME="kolmogorov_flow_${RUN}_${NGRID}"
    echo Launching ${RUNNAME}

    mkdir -p ./runs/
    launchname="${0##*/}"
    cp $launchname "./runs/${RUNNAME}.sh"
    git diff > "./runs/gitdiff_${RUNNAME}.txt"

    module purge
    module load daint-gpu gcc GSL/2.7-CrayGNU-21.09 cray-hdf5-parallel cray-python cdt-cuda craype-accel-nvidia60

    ./sbatch-run-kolmogorov-flow.sh $RUNNAME $NGRID

done
