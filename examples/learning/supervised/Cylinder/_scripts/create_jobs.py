#!/usr/bin/env python
import os
import sys
from datetime import datetime
import shutil
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from time import strftime
from utilities import initialize_constants
from utilities import make_parser
from utilities import exp_dir_str
from utilities import mkdir_p
from utilities import print_args
from utilities import bcolors
from utilities import copy_dir
import utilities as constants

if __name__ == "__main__":
    initialize_constants()
    parser = make_parser()
    parser.add_argument(
        "-N",
        "--nodes",
        help="[SLURM] Nodes to use",
        type=int,
        default=1,
        required=False
    )
    parser.add_argument(
        "-n",
        "--ntasks",
        help="[SLURM] Number of total tasks to use",
        type=int,
        default=1,
        required=False
    )
    parser.add_argument(
        "--ntasks-per-node",
        help="[SLURM] Number of total tasks to use",
        type=int,
        default=None,
        required=False
    )
    parser.add_argument(
        "-c",
        "--cpus-per-task",
        help="[SLURM] Number of cpus to use per task",
        type=int,
        default=12,
        required=False
    )
    parser.add_argument(
        "-t",
        "--time",
        help="[SLURM] time",
        default="0-1",
        required=False
    )
    parser.add_argument(
        "-p",
        "--partition",
        help="[SLURM] partition to use.",
        choices=["debug", "large", "long", "low", "normal", "prepost", "xfer"],
        default="normal",
        required=False
    )
    parser.add_argument(
        "--continute",
        help="TODO: Run latest model",
        required=False,
        action="store_true"
    )
    args = parser.parse_args()
    latent_dims = []
    # Check what kind of latent dimension: can pass range and others
    if hasattr(args.latent_dim, '__iter__'):
        if ":" in args.latent_dim:
            latent_dims = args.latent_dim.split(",")
            if len(latent_dims) == 2:
                latent_dims = range(latent_dims[0], latent_dims[1])
            if len(latent_dims) == 3:
                latent_dims = range(latent_dims[0], latent_dims[1], latent_dims[2])
        elif " " in args.latent_dim:
            latent_dims = [int(e) for e in args.latent_dim.split(" ")]
        else:
            latent_dims = [int(args.latent_dim)]
    else:
        latent_dims = [int(args.latent_dim)]

    if args.conduit == constants.DISTRIBUTED:
        assert(2*args.nodes == args.batch_concurrency)
    EXECUTABLE = "run-reconstruction.py"
    UTILITIES = "utilities.py"
    SCRIPT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(SCRIPT_DIR)
    CWD = os.getcwd()
    SCRATCH = os.getenv('SCRATCH')
    HOME = os.getenv('HOME')
    SCRIPT_DIR_WITHOUT_HOME = os.path.relpath(SCRIPT_DIR, constants.HOME)
    SCRIPT_DIR_ON_SCRATCH = os.path.join(SCRATCH, SCRIPT_DIR_WITHOUT_HOME)
    now = datetime.now().strftime(constants.DATE_FORMAT)
    args.output_dir_append = now
    # Copy executable to scratch for performance reasons.
    mkdir_p(SCRIPT_DIR_ON_SCRATCH)
    shutil.copy(os.path.join(SCRIPT_DIR, EXECUTABLE), os.path.join(SCRIPT_DIR_ON_SCRATCH, EXECUTABLE))
    shutil.copy(os.path.join(SCRIPT_DIR, UTILITIES), os.path.join(SCRIPT_DIR_ON_SCRATCH, UTILITIES))
    copy_dir(os.path.join(SCRIPT_DIR, "_models"), os.path.join(SCRIPT_DIR_ON_SCRATCH, "_models"))
    # CREATE RESULT_DIR
    for latent_dim in latent_dims:
        args.latent_dim = latent_dim
        args.latent_dim = latent_dim
        # pattern: _korali_result/model/lat10/timepoint
        EXPERIMENT_DIR = exp_dir_str(args)
        RESULT_DIR = os.path.join(CWD, EXPERIMENT_DIR)
        JOB_DIRECTORY = RESULT_DIR
        mkdir_p(JOB_DIRECTORY)
        lat_dim_str = str(latent_dim).zfill(2)
        jname = f"{args.model}_lat{lat_dim_str}_test" if args.test else f"{args.model}_lat{lat_dim_str}"
        jfile = os.path.join(JOB_DIRECTORY, "%s.job" % jname)
        RUNPATH = os.path.join(SCRATCH, )
        # Create batch file
        with open(jfile, "w+") as fh:
            fh.writelines("#!/bin/bash\n")
            fh.writelines(f"#SBATCH --chdir={SCRIPT_DIR_ON_SCRATCH}\n")
            fh.writelines(f"#SBATCH --job-name={jname}.job\n")
            fh.writelines(f"#SBATCH --output={os.path.join(RESULT_DIR, jname)}.out\n")
            fh.writelines(f"#SBATCH --time={args.time}\n")
            fh.writelines(f"#SBATCH --nodes={args.nodes}\n")
            if args.ntasks_per_node:
                fh.writelines(f"#SBATCH --ntasks-per-node={args.ntasks_per_node}\n")
                del args.ntasks
            else:
                del args.ntasks_per_node
                fh.writelines(f"#SBATCH --ntasks={args.ntasks}\n")
            fh.writelines(f"#SBATCH --cpus-per-task={args.cpus_per_task}\n")
            fh.writelines(f"#SBATCH --partition={args.partition}\n")
            fh.writelines("#SBATCH --account s929\n")
            fh.writelines("#SBATCH --constraint gpu\n")
            # fh.writelines("#SBATCH --mem=12000\n")
            fh.writelines("#SBATCH --mail-type=ALL\n")
            fh.writelines("#SBATCH --mail-user=$USER@student.ethz.ch\n")
            fh.writelines("#export OMP_NUM_THREADS=12\n")
            command = (
                f"srun python {EXECUTABLE}"
                f" --engine {args.engine}"
                f" --max-generations {args.max_generations}"
                f" --optimizer {args.optimizer}"
                f" --train-split {args.train_split}"
                f" --learning-rate {args.learning_rate}"
                f" --decay {args.decay}"
                f" --result-file {args.result_file}"
                f" --training-batch-size {args.training_batch_size}"
                f" --batch-concurrency {args.batch_concurrency}"
                f" --output-dir-append {args.output_dir_append}"
                f" --epochs {args.epochs}"
                f" --latent-dim {latent_dim}"
                f" --conduit {args.conduit}"
                f" --frequency {args.frequency}"
                f" --model {args.model}"
                f" --verbosity {args.verbosity}"
            )
            if args.data_path:
                command += f" --data-path {args.data_path}"
            else:
                command += f" --data-type {args.data_type}"
            if args.overwrite:
                command += " --overwrite"
            if args.file_output:
                command += " --file-output"
            if args.plot:
                command += " --plot"
            if args.overwrite:
                command += " --file-overwrite"
            fh.writelines(command)

        print(f"Submitting job {jname}")
        if args.verbosity != constants.SILENT and len(latent_dims) == 1:
            print_args(vars(args), color=bcolors.HEADER)
        os.system(f"sbatch {jfile}")
