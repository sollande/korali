#!/usr/bin/env python
import os
from datetime import datetime
from time import strftime
from utilities import initialize_constants
from utilities import make_parser
from utilities import exp_dir_str
from utilities import mkdir_p
import utilities as constants
import sys

if __name__ == "__main__":
    iPython = False
    parser = make_parser()
    parser.add_argument(
        "-N",
        "--nodes",
        help="Nodes to use",
        type=int,
        default=1,
        required=False
    )
    parser.add_argument(
        "-n",
        "--ntasks",
        help="Number of total tasks to use",
        type=int,
        default=1,
        required=False
    )
    parser.add_argument(
        "-c",
        "--cpus-per-task",
        help="Number of cpus to use per task",
        type=int,
        default=12,
        required=False
    )
    if len(sys.argv) != 0:
        if sys.argv[0] == "/usr/bin/ipython":
            sys.argv = ['']
            ipython = True

    args = parser.parse_args()
    if iPython:
        args.test = False

    latent_dims = []
    if len(args.latent_dim) > 2:
        if ":" in args.latent_dim:
            latent_dims = args.latent_dim.split(",")
            if len(latent_dims) == 2:
                latent_dims = range(latent_dims[0], latent_dims[1])
            if len(latent_dims) == 3:
                latent_dims = range(latent_dims[0], latent_dims[1], latent_dims[2])
        elif " " in args.latent_dim:
            latent_dims = [int(e) for e in args.latent_dim.split(" ")]
    else:
        latent_dims = args.latent_dim

    SCRIPT_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../Cylinder')
    os.chdir(SCRIPT_DIR)
    CWD = os.getcwd()

    now = datetime.now().strftime("%d/%m/%y-%H:%M:%S")
    # CREATE RESULT_DIR
    EXPERIMENT_DIR = exp_dir_str(args)
    RESULT_DIR = os.path.join(CWD, EXPERIMENT_DIR)
    JOB_DIRECTORY = os.path.join(RESULT_DIR, ".job")
    mkdir_p(RESULT_DIR)
    for latent_dim in latent_dims:
        jname = f"{args.model}_lat{args.latent_dim}_{args.output_dir_append}"
        jfile = os.path.join(JOB_DIRECTORY, "%s.job" % jname)
        with open(jfile) as fh:
            fh.writelines("#!/bin/bash\n")
            fh.writelines("#SBATCH --constraint=gpu\n" % jname)
            fh.writelines("#SBATCH -A=s929\n" % jname)
            fh.writelines("#SBATCH --job-name=%s.job\n" % jname)
            fh.writelines("#SBATCH --output=%s.out\n" % jname)
            fh.writelines("#SBATCH --time=0-1\n")
            fh.writelines(f"#SBATCH --nodes={args.nodes}\n")
            fh.writelines(f"#SBATCH --ntasks={args.ntasks}\n")
            fh.writelines(f"#SBATCH --cpus-per-task={args.cpus_per_task}\n")
            # fh.writelines("#SBATCH --mem=12000\n")
            # fh.writelines("#SBATCH --qos=normal\n")
            # fh.writelines("#SBATCH --chdir\n")
            # fh.writelines("#SBATCH --mail-type=ALL\n")
            # fh.writelines("#SBATCH --mail-user=$USER@student.ethz.ch\n")
            fh.writelines(f"python run-reconstruction.py\
            --overwrite {args.overwrite}\
            --engine {args.engine}\
            --max-generations {args.max_generations}\
            --optimizer {args.optimizer}\
            --data-path {args.data_path}\
            --test-path {args.test_path}\
            --test-file {args.test_file}\
            --train-split {args.train_split}\
            --learning-rate {args.learning_rate}\
            --decay {args.decay}\
            --training-batch-size {args.training_batch_size}\
            --batch-concurrency {args.batch_concurrency}\
            --output-dir-append {now}\
            --epochs {args.epochs}\
            --latent-dim {latent_dim}\
            --conduit {args.conduit}\
            --test {args.test}\
            --overwrite {args.overwrite}\
            --file-output {args.file_output}\
            --frequency {args.frequency}\
            --model {args.model}\
            --verbosity {args.verbosity}\
            --plot {args.plot}\
            ")

        print(f"Submitting job {jname}")
        os.system("sbatch %s" %jfile)
