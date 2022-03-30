#!/usr/bin/env python3
import os
import sys
import pickle
from matplotlib.cbook import _premultiplied_argb32_to_unmultiplied_rgba8888
import numpy as np
import matplotlib.pyplot as plt
import argparse
import time
import korali
import random
from models import make_autencoder_experiment

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
### Load data
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
    default=32,
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
# parser.add_argument(
#     "--testMSEThreshold",
#     help="Threshold for the testing MSE, under which the run will report an error",
#     default=0.05,
#     required=False,
# )
parser.add_argument(
    "--plot",
    help="Indicates whether to plot results after testing",
    default=False,
    required=False,
)
args = parser.parse_args()
k = korali.Engine()
### Initalize Korali Engine
k["Conduit"]["Type"] = args.conduit
if args.conduit == "MPI":
    from mpi4py import MPI

    MPIcomm = MPI.COMM_WORLD
    MPIrank = MPIcomm.Get_rank()
    MPIsize = MPIcomm.Get_size()
    MPIroot = MPIsize - 1
    k.setMPIComm(MPI.COMM_WORLD)
    if MPIrank == MPIroot:
        print("Running FNN solver with arguments:")
        print(args)
else:
    print("Running FNN solver with arguments:")
    print(args)

min_max_scalar = lambda arr: (arr - arr.min()) / (arr.max() - arr.min())
### Loading the data
with open(args.data_path, "rb") as file:
    data = pickle.load(file)
    trajectories = data["trajectories"]
    del data

samples, img_height, img_width = np.shape(trajectories)
### flatten images 32x64 => 2048
trajectories = np.reshape(trajectories, (samples, -1))
trajectories = min_max_scalar(trajectories)
### Permute
idx = np.random.permutation(samples)
train_idx = idx[: int(samples * args.trainSplit)]

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
    latentDim = [10]
    args.epochs = 1
    stepsPerEpoch = 1

e = korali.Experiment()
found = e.loadState("./_korali_result" + "/latest")
if found == True:
    print("[Korali] Evaluating previous run...\n")

e["Problem"]["Type"] = "Supervised Learning"
e["Problem"]["Max Timesteps"] = 1
e["Problem"]["Training Batch Size"] = args.trainingBatchSize
e["Problem"]["Testing Batch Size"] = testingBatchSize
e["Problem"]["Input"]["Size"] = len(trainingImages[0])
e["Problem"]["Solution"]["Size"] = len(trainingImages[0])
e["Solver"]["Mode"] = "Training"
### Using a neural network solver (deep learning) for inference
### Configuring output
e["Console Output"]["Verbosity"] = "Normal"
# e["Console Output"]["Verbosity"] = "Silent"
e["File Output"]["Enabled"] = True
e["File Output"]["Frequency"] = 1
e["Random Seed"] = 0xC0FFEE
# ====================================================================
e["Solver"]["Type"] = "Learner/DeepSupervisor"
e["Solver"]["Loss Function"] = "Mean Squared Error"
e["Solver"]["Termination Criteria"]["Max Generations"] = args.max_generations
e["Solver"]["Neural Network"]["Engine"] = args.engine
e["Solver"]["Neural Network"]["Optimizer"] = args.optimizer
## Set the autencoder layers
make_autencoder_experiment(e, args.latent_dim, img_height, img_width)
### Printing Configuration
print("[Korali] Running MNIST solver.")
print("[Korali] Algorithm: " + str(e["Solver"]["Neural Network"]["Optimizer"]))
print("[Korali] Database Size: " + str(len(trainingImageVector)))
print("[Korali] Batch Size: " + str(args.trainingBatchSize))
print("[Korali] Epochs: " + str(args.epochs))
print("[Korali] Initial Learning Rate: " + str(args.learningRate))
print("[Korali] Decay: " + str(args.decay))
print("[Korali] Steps per epoch: " + str(stepsPerEpoch))
# ### Running SGD loop
# for e in experiments:
for epoch in range(args.epochs):
    print("[Georg] epoch: " + str(epoch))
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
        # Running step
        k.run(e)
        e["Solver"]["Termination Criteria"]["Max Generations"] = (
            e["Solver"]["Termination Criteria"]["Max Generations"] + 1
        )
    # Printing Information
    # print("[Korali] --------------------------------------------------")
    # print("[Korali] Epoch: " + str(epoch) + "/" + str(epochs))
    # print("[Korali] Learning Rate: " + str(args.learningRate))
    # print("[Korali] Current Training Loss: " + str(e["Solver"]["Current Loss"]))
    args.learningRate = args.learningRate * (1.0 / (1.0 + args.decay * (epoch + 1)))
    # Evaluating testing set
    print("[Georg] Mode Testing")
    e["Solver"]["Mode"] = "Testing"
    e["Problem"]["Input"]["Data"] = testingImageVector
    e["Problem"]["Solution"]["Data"] = [img[0] for img in testingImageVector]
    k.run(e)
    testingInferredVector = testInferredSet = e["Solver"]["Evaluation"]
    # Getting MSE loss for testing set
    squaredMeanError = 0.0
    for i, yhat in enumerate(testingInferredVector):
        y = testingImageVector[i][0]
        for j, pixel in enumerate(y):
            diff = yhat[j] - pixel
            squaredMeanError += diff * diff
    squaredMeanError = squaredMeanError / (float(testingBatchSize) * 2.0)
    print("[Korali] Current Testing Loss:  " + str(squaredMeanError))
    # e["Result"]["Testing Loss"] = squaredMeanError

if args.conduit == "MPI":
    pass
    # if MPIrank is MPIroot:
    #     ### Calc MSE on test set
    #     mse = np.mean((np.array(testInferredSet) - np.array(testOutputSet)) ** 2)
    #     print("MSE on test set: {}".format(mse))

    #     if mse > args.testMSEThreshold:
    #         print("Fail: MSE does not satisfy threshold: " + str(args.testMSEThreshold))
    #         exit(-1)
