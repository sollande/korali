#!/usr/bin/env python
import os
import sys
import pickle
import numpy as np
import argparse
import korali
import shutil
sys.path.append('./_models')
from cnn_autoencoder import configure_cnn_autencoder
from autoencoder import configure_autencoder
from utilities import min_max_scalar
from utilities import print_args
from utilities import print_header
from utilities import bcolors


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
    "--trainSplit", help="Fraction of training samples", default=0.8, required=False
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
    help="Batch size to use for training data",
    type=int,
    default=34,
    required=False,
)
parser.add_argument("--epochs", help="Number of epochs", default=100, required=False)
parser.add_argument(
    "--latent-dim",
    help="Latent dimension of the encoder",
    default=10,
    required=False
    # [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 16, 20, 24, 28, 32, 36, 40, 64]
)
parser.add_argument(
    "--conduit",
    help="Conduit to use [Sequential, MPI, Concurrent]",
    default="Sequential",
    required=False,
)
parser.add_argument("--test", action="store_true")
parser.add_argument("--overwrite", action="store_false")
parser.add_argument("--file-output", action="store_false")
CNN_AUTOENCODER = 'cnn-autoencoder'
AUTOENCODER = 'autoencoder'
parser.add_argument('--model',
                    choices=[AUTOENCODER, CNN_AUTOENCODER],
                    help='Model to use.', default=AUTOENCODER)
parser.add_argument('--verbosity',
                    choices=["Silent", "Normal", "Detailed"],
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
    args.test = True


k = korali.Engine()
### Initalize Korali Engine
k["Conduit"]["Type"] = args.conduit
####################### Model Selection ## #################################

if args.conduit == "MPI":
    from mpi4py import MPI
    MPIcomm = MPI.COMM_WORLD
    MPIrank = MPIcomm.Get_rank()
    MPIsize = MPIcomm.Get_size()
    MPIroot = MPIsize - 1
    k.setMPIComm(MPI.COMM_WORLD)
    if MPIrank == MPIroot:
        if args.verbosity != "Silent":
            print_args(vars(args), color=bcolors.HEADER)
else:
        if args.verbosity != "Silent":
            print_args(vars(args), color=bcolors.HEADER)
############################################################################

### Loading the data
if args.test:
    with open(args.test_path, "rb") as file:
        trajectories = pickle.load(file)
else:
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
nb_train_samples = int(samples * args.trainSplit)
train_idx = idx[: nb_train_samples]

assert nb_train_samples % args.trainingBatchSize == 0, "Batch Size must divide the number of training samples"

trainingImages = trajectories[train_idx]
testingImages = trajectories[~train_idx]

### Converting images to Korali form (requires a time dimension)
trainingImageVector = [[x] for x in trainingImages.tolist()]
testingImageVector = [[x] for x in testingImages.tolist()]

### Calculate Epochs and iterations
stepsPerEpoch = int(len(trainingImageVector) / args.trainingBatchSize)
testingBatchSize = len(testingImageVector)
### If this is test mode, run only one epoch
if args.test:
    args.epochs = 1
    stepsPerEpoch = 1

e = korali.Experiment()
if args.file_output:
    e["File Output"]["Enabled"] = True
    e["File Output"]["Frequency"] = 1
    if args.overwrite:
        shutil.rmtree("./_korali_result", ignore_errors=True)
    found = e.loadState("./_korali_result" + "/latest")
    if found == True:
        print("[Korali] Evaluating previous run...\n")

e["Problem"]["Type"] = "Supervised Learning"
e["Random Seed"] = 0xC0FFEE
e["Console Output"]["Verbosity"] = args.verbosity
e["Problem"]["Max Timesteps"] = 1
e["Problem"]["Training Batch Size"] = args.trainingBatchSize
e["Problem"]["Testing Batch Size"] = testingBatchSize
e["Problem"]["Input"]["Size"] = len(trainingImages[0])
e["Problem"]["Solution"]["Size"] = len(trainingImages[0])
e["Solver"]["Mode"] = "Training"
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

print("[Korali] Running MNIST solver.")
print("[Korali] Nb. Training Images: %s" % len(trainingImages[0]))
print("[Korali] Nb. Testing Images: %s" % len(testingImages[0]))
print("[Korali] Algorithm: " + str(e["Solver"]["Neural Network"]["Optimizer"]))
print("[Korali] Database Size: " + str(len(trainingImageVector)))
print("[Korali] Batch Size: " + str(args.trainingBatchSize))
print("[Korali] Epochs: " + str(args.epochs))
print("[Korali] Initial Learning Rate: " + str(args.learningRate))
print("[Korali] Decay: " + str(args.decay))
# ### Running SGD loop
# for e in experiments:
for epoch in range(args.epochs):
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
        k.run(e)
    # Printing Information
    print("[Korali] --------------------------------------------------")
    print("[Korali] Epoch: " + str(epoch) + "/" + str(args.epochs))
    print("[Korali] Learning Rate: " + str(args.learningRate))
    print("[Korali] Current Training Loss: " + str(e["Solver"]["Current Loss"]))
    args.learningRate = args.learningRate * (1.0 / (1.0 + args.decay * (epoch + 1)))
    # Evaluating testing set
    e["Solver"]["Mode"] = "Testing"
    e["Problem"]["Input"]["Data"] = testingImageVector
    e["Problem"]["Solution"]["Data"] = [img[0] for img in testingImageVector]
    k.run(e)
    testingInferredVector = testInferredSet = e["Solver"]["Evaluation"]
    # Getting MSE loss for testing set
    squaredMeanError = 0.0
    for i, res in enumerate(testingInferredVector):
        sol = testingImageVector[i][0]
        for j, s in enumerate(sol):
            diff = res[j] - s
            squaredMeanError += diff * diff
    squaredMeanError = squaredMeanError / (float(testingBatchSize) * 2.0)
    print("[Korali] Current Testing Loss:  " + str(squaredMeanError))
