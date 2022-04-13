#!/usr/bin/env python
import os
import sys
import pickle
import numpy as np
import argparse
import korali
import shutil
import time
from mpi4py import MPI
sys.path.append('./_models')
from cnn_autoencoder import configure_cnn_autencoder
from autoencoder import configure_autencoder
from utilities import min_max_scalar
from utilities import print_args
from utilities import print_header
from utilities import bcolors

SCRATCH = os.environ['SCRATCH']
HOME = os.environ['HOME']
CWD = os.getcwd()
TIMESTEPS = 0

parser = argparse.ArgumentParser()
parser.add_argument(
    "--engine", help="NN backend to use", default="OneDNN", required=False
)
parser.add_argument(
    "--max-generations",
    help="Maximum Number of generations to run",
    default=1,
    required=False,
)
parser.add_argument(
    "--optimizer",
    help="Optimizer to use for NN parameter updates",
    default="Adam",
    required=False,
)
parser.add_argument(
    "--data-path",
    help="Path to the training data",
    default="./_data/data.pickle",
    required=False,
)
parser.add_argument(
    "--test-path",
    help="Path to the training data",
    default="./_data/test.pickle",
    required=False,
)
parser.add_argument(
    "--test-file",
    help="Filename for testing error",
    default="testing_error.txt",
    required=False,
)
parser.add_argument(
    "--train-split", help="If 0<--train-split<=1 fraction of training samples; \
    else number of training samples", default=6*128, required=False
)
parser.add_argument(
    "--learningRate",
    help="Learning rate for the selected optimizer",
    default=0.0001,
    required=False,
)
parser.add_argument(
    "--decay",
    help="Decay of the learning rate.",
    default=0.0001,
    required=False,
)
parser.add_argument(
    "--trainingBatchSize",
    help="Batch size to use for training data; must divide the --train-split",
    type=int,
    default=128,
    required=False,
)
parser.add_argument(
    "--batch-concurrency",
    help="Batch Concurrency for the minibatches",
    type=int,
    default=1,
    required=False,
)

parser.add_argument("--epochs", help="Number of epochs", default=100, type=int, required=False)
parser.add_argument(
    "--latent-dim",
    help="Latent dimension of the encoder",
    default=10,
    required=False
    # [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 16, 20, 24, 28, 32, 36, 40, 64]
)
SEQUENTIAL = "Sequential"
DISTRIBUTED = "Distributed"
CONCURRENT = "Concurrent"
isMaster = lambda : args.conduit != DISTRIBUTED or (args.conduit == DISTRIBUTED and MPIrank == MPIroot)
parser.add_argument(
    "--conduit",
    help="Conduit to use [Sequential, Distributed, Concurrent]",
    choices=[SEQUENTIAL, CONCURRENT, DISTRIBUTED],
    default="Sequential",
    required=False,
)
parser.add_argument("--test", action="store_true")
parser.add_argument("--overwrite", action="store_true")
parser.add_argument("--file-output", action="store_true")
CNN_AUTOENCODER = 'cnn-autoencoder'
AUTOENCODER = 'autoencoder'
parser.add_argument('--model',
                    choices=[AUTOENCODER, CNN_AUTOENCODER],
                    help='Model to use.', default=AUTOENCODER)
SILENT = "Silent"
NORMAL = "Normal"
DETAILED = "Detailed"
parser.add_argument('--verbosity',
                    choices=[SILENT, NORMAL, DETAILED],
                    help='Verbosity Level', default="Detailed")
parser.add_argument(
    "--plot",
    help="Indicates whether to plot results after testing",
    default=False,
    required=False,
)

iPython = False
if len(sys.argv) != 0:
    if sys.argv[0] == "/usr/bin/ipython":
        sys.argv=['']
        iPython = True

args = parser.parse_args()
if iPython:
    args.test = False


k = korali.Engine()
### Initalize Korali Engine
k["Conduit"]["Type"] = args.conduit
####################### Model Selection ## #################################

if args.conduit == DISTRIBUTED:
    MPIcomm = MPI.COMM_WORLD
    MPIrank = MPIcomm.Get_rank()
    MPIsize = MPIcomm.Get_size()
    MPIroot = MPIsize - 1
    k.setMPIComm(MPI.COMM_WORLD)
if isMaster():
    if args.verbosity != SILENT:
        print_args(vars(args), color=bcolors.HEADER)
############################################################################

### Loading the data
if args.test:
    # Load 128 test sample file
    with open(args.test_path, "rb") as file:
        trajectories = pickle.load(file)
else:
    # Load 1000 sample file
    with open(args.data_path, "rb") as file:
        data = pickle.load(file)
        trajectories = data["trajectories"]
        del data

### flatten images 32x64 => 204
samples, img_height, img_width = np.shape(trajectories)
trajectories = np.reshape(trajectories, (samples, -1))
trajectories = min_max_scalar(trajectories)
### Permute
idx = np.random.permutation(samples)
if args.train_split >= 1:
    nb_train_samples = args.train_split
else:
    nb_train_samples = int(samples * args.train_split)

train_idx = idx[: nb_train_samples]

assert nb_train_samples % args.trainingBatchSize == 0, \
    "Batch Size {} must divide the number of training samples {}"\
    .format(args.trainingBatchSize,nb_train_samples)

trainingImages = trajectories[train_idx]
testingImages = trajectories[~train_idx]

### Converting images to Korali form (requires a time dimension)
trainingImageVector = [[x] for x in trainingImages.tolist()]
testingImageVector = [[x] for x in testingImages.tolist()]
testingGroundTruth = [ y[TIMESTEPS] for y in testingImageVector ]

### Calculate Epochs and iterations
stepsPerEpoch = int(len(trainingImageVector) / args.trainingBatchSize)
testingBatchSize = len(testingImageVector)
### If this is test mode, run only one epoch
if args.test:
    args.epochs = 1
    stepsPerEpoch = 1

e = korali.Experiment()
# Scatch enviroment variable
if args.file_output:
    CWD_WITHOUT_HOME = os.path.relpath(CWD, HOME)
    EXPERIMENT_DIR = os.path.join("_korali_result", args.model, f"lat{args.latent_dim}")
    RESULT_DIR = os.path.join(CWD, EXPERIMENT_DIR)
    RESULT_DIR_ON_SCRATCH = os.path.join(SCRATCH, CWD_WITHOUT_HOME, EXPERIMENT_DIR)
    e["File Output"]["Enabled"] = True
    e["File Output"]["Path"] = RESULT_DIR_ON_SCRATCH
    e["File Output"]["Frequency"] = 1

    if isMaster():
        os.makedirs(RESULT_DIR, exist_ok=True)
        os.makedirs(RESULT_DIR_ON_SCRATCH, exist_ok=True)
        if args.overwrite:
            shutil.rmtree(RESULT_DIR, ignore_errors=True)
    found = e.loadState(os.join.path(RESULT_DIR, "/latest"))
    if isMaster() and found == True and args.verbosity != SILENT:
        print("[Script] Evaluating previous run...\n")

e["Problem"]["Type"] = "Supervised Learning"
e["Random Seed"] = 0xC0FFEE
e["Console Output"]["Verbosity"] = args.verbosity
e["Problem"]["Max Timesteps"] = TIMESTEPS+1
e["Solver"]["Batch Concurrency"] = args.batch_concurrency
e["Problem"]["Training Batch Size"] = args.trainingBatchSize
e["Problem"]["Testing Batch Size"] = testingBatchSize
e["Problem"]["Input"]["Size"] = len(trainingImages[0])
e["Problem"]["Solution"]["Size"] = len(trainingImages[0])
# ====================================================================
e["Solver"]["Type"] = "Learner/DeepSupervisor"
e["Solver"]["Loss Function"] = "Mean Squared Error"
e["Solver"]["Termination Criteria"]["Max Generations"] = args.max_generations
e["Solver"]["Neural Network"]["Engine"] = args.engine
e["Solver"]["Neural Network"]["Optimizer"] = args.optimizer

## Set the autencoder layers
####################### Model Selection ####################################
if args.model == AUTOENCODER:
    configure_autencoder(e, args.latent_dim, img_width, img_height)
else:
    configure_cnn_autencoder(e, args.latent_dim, img_width, img_height)
############################################################################
if isMaster() and args.verbosity != SILENT:
    print("[Script] Running MNIST solver.")
    print("[Script] Nb. Training Images: %s" % len(trainingImages[0]))
    print("[Script] Nb. Testing Images: %s" % len(testingImages[0]))
    print("[Script] Algorithm: " + str(e["Solver"]["Neural Network"]["Optimizer"]))
    print("[Script] Database Size: " + str(len(trainingImageVector)))
    print("[Script] Batch Size: " + str(args.trainingBatchSize))
    print("[Script] Epochs: " + str(args.epochs))
    print("[Script] Initial Learning Rate: " + str(args.learningRate))
    print("[Script] Decay: " + str(args.decay))
    # ### Running SGD loop
times = []
testingErrors = []
# for e in experiments:
for epoch in range(args.epochs):
    if isMaster():
        time_start = time.time_ns()
    e["Solver"]["Mode"] = "Training"
    for step in range(stepsPerEpoch):
        # Creating minibatch
        miniBatchInput = trainingImageVector[
            step * args.trainingBatchSize : (step + 1) * args.trainingBatchSize
        ]  # N x T x C
        miniBatchSolution = [x[0] for x in miniBatchInput]  # N x C
        # Passing minibatch to Korali
        e["Problem"]["Input"]["Data"] = miniBatchInput
        e["Problem"]["Solution"]["Data"] = miniBatchSolution
        # Reconfiguring solver
        e["Solver"]["Learning Rate"] = args.learningRate
        e["Solver"]["Termination Criteria"]["Max Generations"] = (e["Solver"]["Termination Criteria"]["Max Generations"] + 1)
        # Running step
        if args.conduit == DISTRIBUTED:
            k.setMPIComm(MPI.COMM_WORLD)
        k.run(e)
    # Printing Information
    if isMaster() and args.verbosity != SILENT:
        print("[Script] --------------------------------------------------")
        print("[Script] Epoch: " + str(epoch) + "/" + str(args.epochs))
        print("[Script] Learning Rate: " + str(args.learningRate))
        print("[Script] Current Training Loss: " + str(e["Solver"]["Current Loss"]))
    args.learningRate = args.learningRate * (1.0 / (1.0 + args.decay * (epoch + 1)))
    # Evaluating testing set
    e["Solver"]["Mode"] = "Testing"
    e["Problem"]["Input"]["Data"] = testingImageVector
    e["Problem"]["Solution"]["Data"] = testingGroundTruth
    if args.conduit == DISTRIBUTED:
        k.setMPIComm(MPI.COMM_WORLD)
    k.run(e)
    # Getting MSE loss for testing set (only the korali master has the evaluated results)
    if isMaster():
        testingInferred = e["Solver"]["Evaluation"]
        assert len(testingInferred) == len(testingGroundTruth),\
            "Inferred vector does not have the same sample size as the ground truth"
        MSE = 0.0
        for yhat, y in zip(testingInferred, testingGroundTruth):
            diff = yhat - y
            MSE += diff * diff
        MSE /= (float(testingBatchSize) * 2.0)
        testingErrors.append(MSE)
        # Runtime of epochs
        times.append(time.time_ns()-time_start)
        if args.verbosity != SILENT:
            print("[Script] Current Testing Loss:  " + str(MSE))

if isMaster():
    if args.file_output:
        # Writing testing error to output
        with open(os.paht.join(RESULT_DIR_ON_SCRATCH, args.test_file), 'w') as f:
        f.write("MeanSqaured Testing Error\n")
        for e in testingErrors:
            f.write("{}\n".format(e))
        # move_dir(RESULT_DIR_ON_SCRATCH, RESULT_DIR)
        copy_dir(RESULT_DIR_ON_SCRATCH, RESULT_DIR)
    times = [e/(10**9) for e in times]
    print("[Script] Total Time {}s for {} Epochs".format(sum(times), args.epochs))
    print("[Script] Per Epoch Time: {}s ".format(sum(times)/len(times)))
