#!/usr/bin/env python
import os
import sys
from datetime import datetime
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utilities import initialize_constants
from utilities import make_parser
from utilities import exp_dir_str
from utilities import mkdir_p
import utilities as constants

if __name__ == "__main__":
    SCRIPT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    iPython = False
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
        "-l"
        "--latest",
        help="Run latest model",
        required=False,
        action="store_true"
    )

    iPython = False
    if len(sys.argv) != 0:
        if sys.argv[0] == "/usr/bin/ipython":
            sys.argv = ['']
            ipython = True
    if iPython:
        args.test = False
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

    os.chdir(SCRIPT_DIR)
    CWD = os.getcwd()

    now = datetime.now().strftime("%d-%m-%y-%H:%M:%S")
    args.output_dir_append = now
    # CREATE RESULT_DIR
    for latent_dim in latent_dims:
        args.latent_dim = latent_dim
        args.latent_dim = latent_dim
        EXPERIMENT_DIR = exp_dir_str(args)
        RESULT_DIR = os.path.join(CWD, EXPERIMENT_DIR)
        JOB_DIRECTORY = RESULT_DIR
        mkdir_p(JOB_DIRECTORY)
        lat_dim_str = str(latent_dim).zfill(2)
        jname = f"{args.model}_lat{lat_dim_str}_test" if args.test else f"{args.model}_lat{lat_dim_str}"
        jfile = os.path.join(JOB_DIRECTORY, "%s.job" % jname)
        with open(jfile, "w+") as fh:
            fh.writelines("#!/bin/bash\n")
            fh.writelines(f"#SBATCH --chdir={SCRIPT_DIR}\n")
            fh.writelines("#SBATCH --job-name=%s.job\n" % jname)
            fh.writelines("#SBATCH --output=%s.out\n" % os.path.join(RESULT_DIR, jname))
            fh.writelines(f"#SBATCH --time={args.time}\n")
            fh.writelines(f"#SBATCH --nodes={args.nodes}\n")
            fh.writelines(f"#SBATCH --ntasks={args.ntasks}\n")
            fh.writelines(f"#SBATCH --cpus-per-task={args.cpus_per_task}\n")
            # fh.writelines("#SBATCH --mem=12000\n")
            # fh.writelines("#SBATCH --qos=normal\n")
            # fh.writelines("#SBATCH --chdir\n")
            fh.writelines("#SBATCH --mail-type=ALL\n")
            fh.writelines("#SBATCH --mail-user=$USER@student.ethz.ch\n")
            command = (
                f"srun python run-reconstruction.py"
                f" --engine {args.engine}"
                f" --max-generations {args.max_generations}"
                f" --optimizer {args.optimizer}"
                f" --train-split {args.train_split}"
                f" --learning-rate {args.learning_rate}"
                f" --decay {args.decay}"
                f" --test-file {args.test_file}"
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
            if args.test:
                command += " --test"
                command += f" --test-path {args.data_path}"
            else:
                command += f" --data-path {args.data_path}"
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
        os.system(f"sbatch -C gpu -A s929 {jfile}")
