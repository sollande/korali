#!/usr/bin/env python
import os
import sys
import pickle
import numpy as np
import korali
import shutil
import time
from mpi4py import MPI
sys.path.append('./_models')
sys.path.append('./_scripts')
from cnn_autoencoder import configure_cnn_autencoder
from autoencoder import configure_autencoder
from utilities import min_max_scalar
from utilities import print_args
from utilities import print_header
from utilities import bcolors
from utilities import move_dir
from utilities import copy_dir
from utilities import make_parser
from utilities import initialize_constants
from utilities import exp_dir_str
import utilities as constants

initialize_constants()
CWD = os.getcwd()
REL_ROOT = os.path.relpath("/")
TIMESTEPS = 0

isMaster = lambda: args.conduit != constants.DISTRIBUTED or (args.conduit == constants.DISTRIBUTED and MPIrank == MPIroot)
parser = make_parser()

iPython = False
if len(sys.argv) != 0:
    if sys.argv[0] == "/usr/bin/ipython":
        sys.argv = ['']
        ipython = True

args = parser.parse_args()
# TODO: move this into argparser
args.latent_dim = int(args.latent_dim)
if iPython:
    args.test = False


k = korali.Engine()
### Initalize Korali Engine
k["Conduit"]["Type"] = args.conduit
####################### Model Selection ## #################################

if args.conduit == constants.DISTRIBUTED:
    MPIcomm = MPI.COMM_WORLD
    MPIrank = MPIcomm.Get_rank()
    MPIsize = MPIcomm.Get_size()
    MPIroot = MPIsize - 1
    k.setMPIComm(MPI.COMM_WORLD)
if isMaster():
    if args.verbosity != constants.SILENT:
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
    nb_train_samples = int(args.train_split)
else:
    nb_train_samples = int(samples * args.train_split)

train_idx = idx[: nb_train_samples]

assert nb_train_samples % args.training_batch_size == 0, \
    "Batch Size {} must divide the number of training samples {}"\
    .format(args.training_batch_size,nb_train_samples)

trainingImages = trajectories[train_idx]
testingImages = trajectories[~train_idx]

### Converting images to Korali form (requires a time dimension)
trainingImageVector = [[x] for x in trainingImages.tolist()]
testingImageVector = [[x] for x in testingImages.tolist()]
testingGroundTruth = [ y[TIMESTEPS] for y in testingImageVector ]

### Calculate Epochs and iterations
stepsPerEpoch = int(len(trainingImageVector) / args.training_batch_size)
testingBatchSize = len(testingImageVector)
### If this is test mode, run only one epoch
if args.test:
    args.epochs = 1
    stepsPerEpoch = 1

e = korali.Experiment()
if args.file_output:
    CWD_WITHOUT_HOME = os.path.relpath(CWD, constants.HOME)
    EXPERIMENT_DIR = exp_dir_str(args)
    RESULT_DIR = os.path.join(CWD, EXPERIMENT_DIR)
    if constants.SCRATCH:
        RESULT_DIR_ON_SCRATCH = os.path.join(constants.SCRATCH, CWD_WITHOUT_HOME, EXPERIMENT_DIR)
        # Note: korali appends ./ => requires relative path i.e. ../../../..
        # RESULT_DIR_ON_SCRATCH_REL = os.path.join(REL_ROOT, RESULT_DIR_ON_SCRATCH)
    e["File Output"]["Path"] = RESULT_DIR_ON_SCRATCH if constants.SCRATCH else RESULT_DIR

    if isMaster():
        if constants.SCRATCH:
            os.makedirs(RESULT_DIR_ON_SCRATCH, exist_ok=True)
        if args.overwrite:
            shutil.rmtree(RESULT_DIR, ignore_errors=True)
        os.makedirs(RESULT_DIR, exist_ok=True)
    isStateFound = e.loadState(os.path.join(RESULT_DIR, "/latest"))
    if isMaster() and isStateFound and args.verbosity != constants.SILENT:
        print("[Script] Evaluating previous run...\n")

e["File Output"]["Frequency"] = 0
e["Problem"]["Type"] = "Supervised Learning"
e["Random Seed"] = 0xC0FFEE
e["Console Output"]["Verbosity"] = args.verbosity
e["Problem"]["Max Timesteps"] = TIMESTEPS+1
e["Solver"]["Batch Concurrency"] = args.batch_concurrency
e["Problem"]["Training Batch Size"] = args.training_batch_size
e["Problem"]["Testing Batch Size"] = testingBatchSize
e["Problem"]["Input"]["Size"] = len(trainingImages[0])
e["Problem"]["Solution"]["Size"] = len(trainingImages[0])
# ====================================================================
e["Solver"]["Type"] = "Learner/DeepSupervisor"
e["Solver"]["Loss Function"] = "Mean Squared Error"
e["Solver"]["Termination Criteria"]["Max Generations"] = args.max_generations-1
e["Solver"]["Neural Network"]["Engine"] = args.engine
e["Solver"]["Neural Network"]["Optimizer"] = args.optimizer

## Set the autencoder layers
####################### Model Selection ####################################
if args.model == constants.AUTOENCODER:
    configure_autencoder(e, args.latent_dim, img_width, img_height)
else:
    configure_cnn_autencoder(e, args.latent_dim, img_width, img_height)
############################################################################
if isMaster() and args.verbosity != constants.SILENT:
    print("[Script] Running MNIST solver.")
    print("[Script] Nb. Training Images: %s" % len(trainingImages[0]))
    print("[Script] Nb. Testing Images: %s" % len(testingImages[0]))
    print("[Script] Algorithm: " + str(e["Solver"]["Neural Network"]["Optimizer"]))
    print("[Script] Database Size: " + str(len(trainingImageVector)))
    print("[Script] Batch Size: " + str(args.training_batch_size))
    print("[Script] Epochs: " + str(args.epochs))
    print("[Script] Initial Learning Rate: " + str(args.learning_rate))
    print("[Script] Decay: " + str(args.decay))
    # ### Running SGD loop
times = []
if isMaster() and args.file_output:
    ERROR_FILE = os.path.join(RESULT_DIR_ON_SCRATCH if constants.SCRATCH else RESULT_DIR, args.test_file)
    with open(ERROR_FILE, 'w') as f:
        f.write("Epoch\tMeanSquaredError\tTime\n")
for epoch in range(args.epochs):
    if isMaster():
        time_start = time.time_ns()
    e["Solver"]["Mode"] = "Training"
    for step in range(stepsPerEpoch):
        e["File Output"]["Enabled"] = True if (epoch % args.frequency == 0 and step+1 == stepsPerEpoch) else False
        # Creating minibatch
        miniBatchInput = trainingImageVector[
            step * args.training_batch_size: (step + 1) * args.training_batch_size
        ]  # N x T x C
        miniBatchSolution = [x[0] for x in miniBatchInput]  # N x C
        # Passing minibatch to Korali
        e["Problem"]["Input"]["Data"] = miniBatchInput
        e["Problem"]["Solution"]["Data"] = miniBatchSolution
        # Reconfiguring solver
        e["Solver"]["Learning Rate"] = args.learning_rate
        e["Solver"]["Termination Criteria"]["Max Generations"] = (e["Solver"]["Termination Criteria"]["Max Generations"] + 1)
        # Running step
        if args.conduit == constants.DISTRIBUTED:
            k.setMPIComm(MPI.COMM_WORLD)
        k.run(e)
    # Printing Information
    if isMaster() and args.verbosity != constants.SILENT:
        print("[Script] --------------------------------------------------")
        print("[Script] Epoch: " + str(epoch+1) + "/" + str(args.epochs))
        print("[Script] Learning Rate: " + str(args.learning_rate))
        print("[Script] Current Training Loss: " + str(e["Solver"]["Current Loss"]))
    args.learning_rate = args.learning_rate * (1.0 / (1.0 + args.decay * (epoch + 1)))
    # Evaluating testing set
    e["Solver"]["Mode"] = "Testing"
    e["Problem"]["Input"]["Data"] = testingImageVector
    e["Problem"]["Solution"]["Data"] = testingGroundTruth
    if args.conduit == constants.DISTRIBUTED:
        k.setMPIComm(MPI.COMM_WORLD)
    k.run(e)
    # Getting MSE loss for testing set (only the korali master has the evaluated results)
    if isMaster():
        testingInferred = e["Solver"]["Evaluation"]
        assert len(testingInferred) == len(testingGroundTruth),\
            "Inferred vector does not have the same sample size as the ground truth"
        MSE = 0.0
        for yhat, y in zip(testingInferred, testingGroundTruth):
            for yhat_i, y_i in zip(yhat, y):
                diff = yhat_i - y_i
                MSE += diff * diff
        MSE /= (float(testingBatchSize) * 2.0)
        # if epoch % args.frequency == 0:
        #     # Writing testing error to output
        # Runtime of epochs
        times.append((time.time_ns()-time_start)/(10**9) )
        if args.file_output:
            with open(ERROR_FILE, 'a') as f:
                f.write("{}\t{}\t{}\n".format(epoch+1, MSE, times[-1]))
        if args.verbosity != constants.SILENT:
            print("[Script] Current Testing Loss:  " + str(MSE))

if isMaster():
    if args.file_output:
        # Writing testing error to output
        if constants.SCRATCH:
            # move_dir(RESULT_DIR_ON_SCRATCH, RESULT_DIR)
            # copy_dir(RESULT_DIR_ON_SCRATCH, RESULT_DIR)
            pass
    print("[Script] Total Time {}s for {} Epochs".format(sum(times), args.epochs))
    print("[Script] Per Epoch Time: {}s ".format(sum(times)/len(times)))
    if args.plot:
        pass
