if [ $# -lt 1 ] ; then
	echo "Usage: ./sbatch-run-vracer.sh RUNNAME\n Exit.."
	exit 1
fi
RUNNAME="vracer_$1"

BASEPATH="${SCRATCH}/CUP2D/"
export OMP_PLACES=cores
export OMP_PROC_BIND=close
export OMP_NUM_THREADS=12

FOLDERNAME=${BASEPATH}/${RUNNAME}
mkdir -p ${FOLDERNAME}

cd ..

cp ./run-vracer.py ${FOLDERNAME}
cp -r ./_model ${FOLDERNAME}
cd ${FOLDERNAME}

slurmfile=daint_sbatch_${RUNNAME}_slurm

cat <<EOF >$slurmfile
#!/bin/bash -l

#SBATCH --account=s929
#SBATCH --job-name="${RUNNAME}"
#SBATCH --time=24:00:00
# #SBATCH --time=00:30:00
# #SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --constraint=gpu

srun python run-vracer.py
EOF

chmod 755 $slurmfile
sbatch $slurmfile
