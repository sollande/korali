#!/usr/bin/env python3
import os
import sys
import pickle
from matplotlib.cbook import _premultiplied_argb32_to_unmultiplied_rgba8888
import numpy as np
import matplotlib.pyplot as plt
import time
import korali
import random

# from utilities import configure_experiment

k = korali.Engine()
e = korali.Experiment()

# Hyperparameters

layers = 5
learningRate = 0.0001
trainingSize = 0.5
decay = 0.0001
trainingBatchSize = 32
epochs = 90
# width=height scaling factor of kernels
kernelSizeFactor = 9
autoencoderFactor = 4

### Loading the data
data_path = "./_data/data.pickle"
with open(data_path, "rb") as file:
    data = pickle.load(file)
    trajectories = data["trajectories"]
    del data

samples, img_height, img_width = np.shape(trajectories)
### flatten images 32x64 => 2048
trajectories = np.reshape(trajectories, (samples, -1))
### Permute
idx = np.random.permutation(samples)
train_idx = idx[: int(samples * trainingSize)]

trainingImages = trajectories[train_idx]
testingImages = trajectories[~train_idx]


### Converting images to Korali form (requires a time dimension)

trainingImageVector = [[x] for x in trainingImages.tolist()]
testingImageVector = [[x] for x in testingImages.tolist()]

### Shuffling training data set for stochastic gradient descent training

random.shuffle(trainingImageVector)

### LAYER DEFINITIONS ==================================================================
### ====================================================================================

### Calculating Derived Values

stepsPerEpoch = int(len(trainingImageVector) / trainingBatchSize)
testingBatchSize = len(testingImageVector)
latentDim = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 16, 20, 24, 28, 32, 36, 40, 64]
### If this is test mode, run only one epoch
if len(sys.argv) == 2:
    if sys.argv[1] == "--test":
        latentDim = [10]
        epochs = 1
        stepsPerEpoch = 1

# epochs=1
# stepsPerEpoch=1
### Configuring general problem settings
experiments = []
# latentDim = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 16, 20, 24, 28, 32, 36, 40, 64]
kernelSizeConv = 13
paddingConv = 6
strideConv = 1
kernelSizeActivat = 2
paddingActivat = 0
strideActivat = 2
# for latDim in latentDim:
latDim = 10
e["Problem"]["Type"] = "Supervised Learning"
e["Problem"]["Max Timesteps"] = 1
e["Problem"]["Training Batch Size"] = trainingBatchSize
e["Problem"]["Testing Batch Size"] = testingBatchSize
e["Problem"]["Input"]["Size"] = len(trainingImages[0])
e["Problem"]["Solution"]["Size"] = len(trainingImages[0])
e["Solver"]["Mode"] = "Training"
### Using a neural network solver (deep learning) for inference
### Configuring output
e["Console Output"]["Verbosity"] = "Detailed"
e["File Output"]["Enabled"] = True
e["Random Seed"] = 0xC0FFEE
# e_conf = configure_experiment(e, latDim, img_width, img_height, 3)
# ====================================================================
e["Solver"]["Termination Criteria"]["Max Generations"] = 1
e["Solver"]["Type"] = "Learner/DeepSupervisor"
e["Solver"]["Loss Function"] = "Mean Squared Error"
e["Solver"]["Neural Network"]["Engine"] = "OneDNN"
e["Solver"]["Neural Network"]["Optimizer"] = "Adam"
# CNN CONVOLUTION ====================================================
e["Solver"]["Neural Network"]["Hidden Layers"][0]["Type"] = "Layer/Convolution"
e["Solver"]["Neural Network"]["Hidden Layers"][0]["Image Height"]      = img_height
e["Solver"]["Neural Network"]["Hidden Layers"][0]["Image Width"]       = img_width
e["Solver"]["Neural Network"]["Hidden Layers"][0]["Padding Left"]      = paddingConv
e["Solver"]["Neural Network"]["Hidden Layers"][0]["Padding Right"]     = paddingConv
e["Solver"]["Neural Network"]["Hidden Layers"][0]["Padding Top"]       = paddingConv
e["Solver"]["Neural Network"]["Hidden Layers"][0]["Padding Bottom"]    = paddingConv
e["Solver"]["Neural Network"]["Hidden Layers"][0]["Kernel Height"]     = kernelSizeConv
e["Solver"]["Neural Network"]["Hidden Layers"][0]["Kernel Width"]      = kernelSizeConv
e["Solver"]["Neural Network"]["Hidden Layers"][0]["Vertical Stride"]   = strideConv
e["Solver"]["Neural Network"]["Hidden Layers"][0]["Horizontal Stride"] = strideConv
e["Solver"]["Neural Network"]["Hidden Layers"][0]["Output Channels"]   = 20*img_width*img_height
## Batch Normalization ===============
## TODO
## Pooling ===========================
e["Solver"]["Neural Network"]["Hidden Layers"][0+1]["Type"] = "Layer/Pooling"
e["Solver"]["Neural Network"]["Hidden Layers"][0+1]["Function"]          = "Exclusive Average"
e["Solver"]["Neural Network"]["Hidden Layers"][0+1]["Image Height"]      = img_height
e["Solver"]["Neural Network"]["Hidden Layers"][0+1]["Image Width"]       = img_width
e["Solver"]["Neural Network"]["Hidden Layers"][0+1]["Kernel Height"]     = kernelSizeActivat
e["Solver"]["Neural Network"]["Hidden Layers"][0+1]["Kernel Width"]      = kernelSizeActivat
e["Solver"]["Neural Network"]["Hidden Layers"][0+1]["Vertical Stride"]   = strideActivat
e["Solver"]["Neural Network"]["Hidden Layers"][0+1]["Horizontal Stride"] = strideActivat
e["Solver"]["Neural Network"]["Hidden Layers"][0+1]["Padding Left"]      = paddingActivat
e["Solver"]["Neural Network"]["Hidden Layers"][0+1]["Padding Right"]     = paddingActivat
e["Solver"]["Neural Network"]["Hidden Layers"][0+1]["Padding Top"]       = paddingActivat
e["Solver"]["Neural Network"]["Hidden Layers"][0+1]["Padding Bottom"]    = paddingActivat
img_height=img_height/2
img_width=img_width/2
e["Solver"]["Neural Network"]["Hidden Layers"][0+1]["Output Channels"]   = 20*img_width*img_height
## Activation ========================
e["Solver"]["Neural Network"]["Hidden Layers"][0+2]["Type"] = "Layer/Activation"
e["Solver"]["Neural Network"]["Hidden Layers"][0+2]["Function"] = "Elementwise/ReLU"
# Layer 2         ====================================================
e["Solver"]["Neural Network"]["Hidden Layers"][3]["Type"] = "Layer/Convolution"
e["Solver"]["Neural Network"]["Hidden Layers"][3]["Image Height"]      = img_height
e["Solver"]["Neural Network"]["Hidden Layers"][3]["Image Width"]       = img_width
e["Solver"]["Neural Network"]["Hidden Layers"][3]["Padding Left"]      = paddingConv
e["Solver"]["Neural Network"]["Hidden Layers"][3]["Padding Right"]     = paddingConv
e["Solver"]["Neural Network"]["Hidden Layers"][3]["Padding Top"]       = paddingConv
e["Solver"]["Neural Network"]["Hidden Layers"][3]["Padding Bottom"]    = paddingConv
e["Solver"]["Neural Network"]["Hidden Layers"][3]["Kernel Height"]     = kernelSizeConv
e["Solver"]["Neural Network"]["Hidden Layers"][3]["Kernel Width"]      = kernelSizeConv
e["Solver"]["Neural Network"]["Hidden Layers"][3]["Vertical Stride"]   = strideConv
e["Solver"]["Neural Network"]["Hidden Layers"][3]["Horizontal Stride"] = strideConv
e["Solver"]["Neural Network"]["Hidden Layers"][3]["Output Channels"]   = 20*img_width*img_height
## Batch Normalization ===============
## TODO
## Pooling ===========================
e["Solver"]["Neural Network"]["Hidden Layers"][3+1]["Type"] = "Layer/Pooling"
e["Solver"]["Neural Network"]["Hidden Layers"][3+1]["Function"]          = "Exclusive Average"
e["Solver"]["Neural Network"]["Hidden Layers"][3+1]["Image Height"]      = img_height
e["Solver"]["Neural Network"]["Hidden Layers"][3+1]["Image Width"]       = img_width
e["Solver"]["Neural Network"]["Hidden Layers"][3+1]["Kernel Height"]     = kernelSizeActivat
e["Solver"]["Neural Network"]["Hidden Layers"][3+1]["Kernel Width"]      = kernelSizeActivat
e["Solver"]["Neural Network"]["Hidden Layers"][3+1]["Vertical Stride"]   = strideActivat
e["Solver"]["Neural Network"]["Hidden Layers"][3+1]["Horizontal Stride"] = strideActivat
e["Solver"]["Neural Network"]["Hidden Layers"][3+1]["Padding Left"]      = paddingActivat
e["Solver"]["Neural Network"]["Hidden Layers"][3+1]["Padding Right"]     = paddingActivat
e["Solver"]["Neural Network"]["Hidden Layers"][3+1]["Padding Top"]       = paddingActivat
e["Solver"]["Neural Network"]["Hidden Layers"][3+1]["Padding Bottom"]    = paddingActivat
img_height=img_height/2
img_width=img_width/2
e["Solver"]["Neural Network"]["Hidden Layers"][3+1]["Output Channels"]   = 20*img_width*img_height
## Activation ========================
e["Solver"]["Neural Network"]["Hidden Layers"][3+2]["Type"] = "Layer/Activation"
e["Solver"]["Neural Network"]["Hidden Layers"][3+2]["Function"] = "Elementwise/ReLU"
# Layer 3         ====================================================
e["Solver"]["Neural Network"]["Hidden Layers"][6]["Type"] = "Layer/Convolution"
e["Solver"]["Neural Network"]["Hidden Layers"][6]["Image Height"]      = img_height
e["Solver"]["Neural Network"]["Hidden Layers"][6]["Image Width"]       = img_width
e["Solver"]["Neural Network"]["Hidden Layers"][6]["Padding Left"]      = paddingConv
e["Solver"]["Neural Network"]["Hidden Layers"][6]["Padding Right"]     = paddingConv
e["Solver"]["Neural Network"]["Hidden Layers"][6]["Padding Top"]       = paddingConv
e["Solver"]["Neural Network"]["Hidden Layers"][6]["Padding Bottom"]    = paddingConv
e["Solver"]["Neural Network"]["Hidden Layers"][6]["Kernel Height"]     = kernelSizeConv
e["Solver"]["Neural Network"]["Hidden Layers"][6]["Kernel Width"]      = kernelSizeConv
e["Solver"]["Neural Network"]["Hidden Layers"][6]["Vertical Stride"]   = strideConv
e["Solver"]["Neural Network"]["Hidden Layers"][6]["Horizontal Stride"] = strideConv
e["Solver"]["Neural Network"]["Hidden Layers"][6]["Output Channels"]   = 20*img_width*img_height
## Batch Normalization ===============
## TODO
## Pooling ===========================
print(img_height)
print(img_width)
print(paddingConv)
print(kernelSizeConv)
print(strideConv)
e["Solver"]["Neural Network"]["Hidden Layers"][6+1]["Type"] = "Layer/Pooling"
e["Solver"]["Neural Network"]["Hidden Layers"][6+1]["Function"]          = "Exclusive Average"
e["Solver"]["Neural Network"]["Hidden Layers"][6+1]["Image Height"]      = img_height
e["Solver"]["Neural Network"]["Hidden Layers"][6+1]["Image Width"]       = img_width
e["Solver"]["Neural Network"]["Hidden Layers"][6+1]["Kernel Height"]     = kernelSizeActivat
e["Solver"]["Neural Network"]["Hidden Layers"][6+1]["Kernel Width"]      = kernelSizeActivat
e["Solver"]["Neural Network"]["Hidden Layers"][6+1]["Vertical Stride"]   = strideActivat
e["Solver"]["Neural Network"]["Hidden Layers"][6+1]["Horizontal Stride"] = strideActivat
e["Solver"]["Neural Network"]["Hidden Layers"][6+1]["Padding Left"]      = paddingActivat
e["Solver"]["Neural Network"]["Hidden Layers"][6+1]["Padding Right"]     = paddingActivat
e["Solver"]["Neural Network"]["Hidden Layers"][6+1]["Padding Top"]       = paddingActivat
e["Solver"]["Neural Network"]["Hidden Layers"][6+1]["Padding Bottom"]    = paddingActivat
img_height=img_height/2
img_width=img_width/2
e["Solver"]["Neural Network"]["Hidden Layers"][6+1]["Output Channels"]   = 2*img_width*img_height
## Activation ========================
e["Solver"]["Neural Network"]["Hidden Layers"][6+2]["Type"] = "Layer/Activation"
e["Solver"]["Neural Network"]["Hidden Layers"][6+2]["Function"] = "Elementwise/ReLU"

# FFNN CONVOLUTION ===================================================
e["Solver"]["Neural Network"]["Hidden Layers"][9]["Type"] = "Layer/Linear"
e["Solver"]["Neural Network"]["Hidden Layers"][9]["Output Channels"] = img_height*img_width*latDim
## Activation ========================
e["Solver"]["Neural Network"]["Hidden Layers"][10]["Type"] = "Layer/Activation"
e["Solver"]["Neural Network"]["Hidden Layers"][10]["Function"] = "Elementwise/ReLU"

e["Solver"]["Neural Network"]["Hidden Layers"][11]["Type"] = "Layer/Linear"
e["Solver"]["Neural Network"]["Hidden Layers"][11]["Output Channels"] = img_height*img_width*2
## Activation ========================
e["Solver"]["Neural Network"]["Hidden Layers"][12]["Type"] = "Layer/Activation"
e["Solver"]["Neural Network"]["Hidden Layers"][12]["Function"] = "Elementwise/ReLU"

# CNN DECONVOLUTION ====================================================
# Layer 1 (Invert Pooling)   ===========================================
img_height=img_height*2
img_width=img_width*2
e["Solver"]["Neural Network"]["Hidden Layers"][13]["Type"] = "Layer/Deconvolution"
e["Solver"]["Neural Network"]["Hidden Layers"][13]["Image Height"]      = img_height
e["Solver"]["Neural Network"]["Hidden Layers"][13]["Image Width"]       = img_width
e["Solver"]["Neural Network"]["Hidden Layers"][13]["Kernel Height"]     = kernelSizeActivat
e["Solver"]["Neural Network"]["Hidden Layers"][13]["Kernel Width"]      = kernelSizeActivat
e["Solver"]["Neural Network"]["Hidden Layers"][13]["Vertical Stride"]   = strideActivat
e["Solver"]["Neural Network"]["Hidden Layers"][13]["Horizontal Stride"] = strideActivat
e["Solver"]["Neural Network"]["Hidden Layers"][13]["Padding Left"]      = paddingActivat
e["Solver"]["Neural Network"]["Hidden Layers"][13]["Padding Right"]     = paddingActivat
e["Solver"]["Neural Network"]["Hidden Layers"][13]["Padding Top"]       = paddingActivat
e["Solver"]["Neural Network"]["Hidden Layers"][13]["Padding Bottom"]    = paddingActivat
e["Solver"]["Neural Network"]["Hidden Layers"][13]["Output Channels"]   = 20*img_width*img_height
# Layer 2 (Invert Convolution) =====================================
e["Solver"]["Neural Network"]["Hidden Layers"][14]["Type"] = "Layer/Deconvolution"
e["Solver"]["Neural Network"]["Hidden Layers"][14]["Image Height"]      = img_height
e["Solver"]["Neural Network"]["Hidden Layers"][14]["Image Width"]       = img_width
e["Solver"]["Neural Network"]["Hidden Layers"][14]["Padding Left"]      = paddingConv
e["Solver"]["Neural Network"]["Hidden Layers"][14]["Padding Right"]     = paddingConv
e["Solver"]["Neural Network"]["Hidden Layers"][14]["Padding Top"]       = paddingConv
e["Solver"]["Neural Network"]["Hidden Layers"][14]["Padding Bottom"]    = paddingConv
e["Solver"]["Neural Network"]["Hidden Layers"][14]["Kernel Height"]     = kernelSizeConv
e["Solver"]["Neural Network"]["Hidden Layers"][14]["Kernel Width"]      = kernelSizeConv
e["Solver"]["Neural Network"]["Hidden Layers"][14]["Vertical Stride"]   = strideConv
e["Solver"]["Neural Network"]["Hidden Layers"][14]["Horizontal Stride"] = strideConv
e["Solver"]["Neural Network"]["Hidden Layers"][14]["Output Channels"]   = 20*img_width*img_height
## Batch Normalization ===============
## TODO
## Activation ========================
e["Solver"]["Neural Network"]["Hidden Layers"][15]["Type"] = "Layer/Activation"
e["Solver"]["Neural Network"]["Hidden Layers"][15]["Function"] = "Elementwise/ReLU"
# Layer 3 (Invert Pooling)    ==============================================
img_height=img_height*2
img_width=img_width*2
e["Solver"]["Neural Network"]["Hidden Layers"][16]["Type"] = "Layer/Deconvolution"
e["Solver"]["Neural Network"]["Hidden Layers"][16]["Image Height"]      = img_height
e["Solver"]["Neural Network"]["Hidden Layers"][16]["Image Width"]       = img_width
e["Solver"]["Neural Network"]["Hidden Layers"][16]["Kernel Height"]     = kernelSizeActivat
e["Solver"]["Neural Network"]["Hidden Layers"][16]["Kernel Width"]      = kernelSizeActivat
e["Solver"]["Neural Network"]["Hidden Layers"][16]["Vertical Stride"]   = strideActivat
e["Solver"]["Neural Network"]["Hidden Layers"][16]["Horizontal Stride"] = strideActivat
e["Solver"]["Neural Network"]["Hidden Layers"][16]["Padding Left"]      = paddingActivat
e["Solver"]["Neural Network"]["Hidden Layers"][16]["Padding Right"]     = paddingActivat
e["Solver"]["Neural Network"]["Hidden Layers"][16]["Padding Top"]       = paddingActivat
e["Solver"]["Neural Network"]["Hidden Layers"][16]["Padding Bottom"]    = paddingActivat
e["Solver"]["Neural Network"]["Hidden Layers"][16]["Output Channels"]   = 20*img_width*img_height
# Layer 4 (Invert Convolution) =====================================
e["Solver"]["Neural Network"]["Hidden Layers"][17]["Type"] = "Layer/Deconvolution"
e["Solver"]["Neural Network"]["Hidden Layers"][17]["Image Height"]      = img_height
e["Solver"]["Neural Network"]["Hidden Layers"][17]["Image Width"]       = img_width
e["Solver"]["Neural Network"]["Hidden Layers"][17]["Padding Left"]      = paddingConv
e["Solver"]["Neural Network"]["Hidden Layers"][17]["Padding Right"]     = paddingConv
e["Solver"]["Neural Network"]["Hidden Layers"][17]["Padding Top"]       = paddingConv
e["Solver"]["Neural Network"]["Hidden Layers"][17]["Padding Bottom"]    = paddingConv
e["Solver"]["Neural Network"]["Hidden Layers"][17]["Kernel Height"]     = kernelSizeConv
e["Solver"]["Neural Network"]["Hidden Layers"][17]["Kernel Width"]      = kernelSizeConv
e["Solver"]["Neural Network"]["Hidden Layers"][17]["Vertical Stride"]   = strideConv
e["Solver"]["Neural Network"]["Hidden Layers"][17]["Horizontal Stride"] = strideConv
e["Solver"]["Neural Network"]["Hidden Layers"][17]["Output Channels"]   = 20*img_width*img_height
## Batch Normalization ===============
## TODO
## Activation ========================
e["Solver"]["Neural Network"]["Hidden Layers"][18]["Type"] = "Layer/Activation"
e["Solver"]["Neural Network"]["Hidden Layers"][18]["Function"] = "Elementwise/ReLU"
# Layer 5         ====================================================
img_height=img_height*2
img_width=img_width*2
e["Solver"]["Neural Network"]["Hidden Layers"][19]["Type"] = "Layer/Deconvolution"
e["Solver"]["Neural Network"]["Hidden Layers"][19]["Image Height"]      = img_height
e["Solver"]["Neural Network"]["Hidden Layers"][19]["Image Width"]       = img_width
e["Solver"]["Neural Network"]["Hidden Layers"][19]["Kernel Height"]     = kernelSizeActivat
e["Solver"]["Neural Network"]["Hidden Layers"][19]["Kernel Width"]      = kernelSizeActivat
e["Solver"]["Neural Network"]["Hidden Layers"][19]["Vertical Stride"]   = strideActivat
e["Solver"]["Neural Network"]["Hidden Layers"][19]["Horizontal Stride"] = strideActivat
e["Solver"]["Neural Network"]["Hidden Layers"][19]["Padding Left"]      = paddingActivat
e["Solver"]["Neural Network"]["Hidden Layers"][19]["Padding Right"]     = paddingActivat
e["Solver"]["Neural Network"]["Hidden Layers"][19]["Padding Top"]       = paddingActivat
e["Solver"]["Neural Network"]["Hidden Layers"][19]["Padding Bottom"]    = paddingActivat
e["Solver"]["Neural Network"]["Hidden Layers"][19]["Output Channels"]   = 1*img_width*img_height
# Layer 6 (Invert Convolution) =====================================
e["Solver"]["Neural Network"]["Hidden Layers"][20]["Type"] = "Layer/Deconvolution"
e["Solver"]["Neural Network"]["Hidden Layers"][20]["Image Height"]      = img_height
e["Solver"]["Neural Network"]["Hidden Layers"][20]["Image Width"]       = img_width
e["Solver"]["Neural Network"]["Hidden Layers"][20]["Padding Left"]      = paddingConv
e["Solver"]["Neural Network"]["Hidden Layers"][20]["Padding Right"]     = paddingConv
e["Solver"]["Neural Network"]["Hidden Layers"][20]["Padding Top"]       = paddingConv
e["Solver"]["Neural Network"]["Hidden Layers"][20]["Padding Bottom"]    = paddingConv
e["Solver"]["Neural Network"]["Hidden Layers"][20]["Kernel Height"]     = kernelSizeConv
e["Solver"]["Neural Network"]["Hidden Layers"][20]["Kernel Width"]      = kernelSizeConv
e["Solver"]["Neural Network"]["Hidden Layers"][20]["Vertical Stride"]   = strideConv
e["Solver"]["Neural Network"]["Hidden Layers"][20]["Horizontal Stride"] = strideConv
e["Solver"]["Neural Network"]["Hidden Layers"][20]["Output Channels"]   = 1*img_width*img_height
## Batch Normalization ===============
## TODO
## Activation ========================
e["Solver"]["Neural Network"]["Hidden Layers"][21]["Type"] = "Layer/Activation"
e["Solver"]["Neural Network"]["Hidden Layers"][21]["Function"] = "Elementwise/Tanh"

    # experiments.append(e)

k["Conduit"]["Type"] = "Sequential"
### Printing Configuration

print("[Korali] Running MNIST solver.")
print("[Korali] Algorithm: " + str(e["Solver"]["Neural Network"]["Optimizer"]))
print("[Korali] Database Size: " + str(len(trainingImageVector)))
print("[Korali] Batch Size: " + str(trainingBatchSize))
print("[Korali] Epochs: " + str(epochs))
print("[Korali] Initial Learning Rate: " + str(learningRate))
print("[Korali] Decay: " + str(decay))
# ### Running SGD loop
# for e in experiments:
for epoch in range(epochs):
    for step in range(stepsPerEpoch):

        # Creating minibatch
        miniBatchInput = trainingImageVector[
            step * trainingBatchSize : (step + 1) * trainingBatchSize
        ]  # N x T x C
        miniBatchSolution = [x[0] for x in miniBatchInput]  # N x C
        # Passing minibatch to Korali
        e["Problem"]["Input"]["Data"] = miniBatchInput
        e["Problem"]["Solution"]["Data"] = miniBatchSolution
        # Reconfiguring solver
        e["Solver"]["Learning Rate"] = learningRate
        e["Solver"]["Termination Criteria"]["Max Generations"] = (e["Solver"]["Termination Criteria"]["Max Generations"] + 1)
        # Running step
        k.run(e)
    # Printing Information
    print("[Korali] --------------------------------------------------")
    print("[Korali] Epoch: " + str(epoch) + "/" + str(epochs))
    print("[Korali] Learning Rate: " + str(learningRate))
    print("[Korali] Current Training Loss: " + str(e["Solver"]["Current Loss"]))
    learningRate = learningRate * (1.0 / (1.0 + decay * (epoch + 1)))
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
