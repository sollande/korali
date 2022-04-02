#!/usr/bin/env python3
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import time
import korali
import random
from mnist import MNIST

k = korali.Engine()
e = korali.Experiment()

### Hyperparameters
 
learningRate = 0.0001
decay = 0.0001
trainingBatchSize = 12
epochs = 90

### Loading MNIST data [28x28 images with {0,..,9} as label - http://yann.lecun.com/exdb/mnist/]

mndata = MNIST('./_data')
mndata.gz = True
trainingImages, _ = mndata.load_training()
testingImages, _ = mndata.load_testing()

### Converting images to Korali form (requires a time dimension)

trainingImageVector = [ [ x ] for x in trainingImages ]
testingImageVector = [ [ x ] for x in testingImages ]

### Shuffling training data set for stochastic gradient descent training

random.shuffle(trainingImageVector)

### Calculating Derived Values

stepsPerEpoch = int(len(trainingImageVector) / trainingBatchSize)
testingBatchSize = len(testingImageVector)

### If this is test mode, run only one epoch
if len(sys.argv) == 2:
 if sys.argv[1] == '--test':
  epochs=1
  stepsPerEpoch=1

### Configuring general problem settings
k["Conduit"]["Type"] = "Sequential"
e["Problem"]["Type"] = "Supervised Learning"
e["Problem"]["Max Timesteps"] = 1
e["Problem"]["Training Batch Size"] = trainingBatchSize
e["Problem"]["Testing Batch Size"] = testingBatchSize
e["Problem"]["Input"]["Size"] = len(trainingImages[0])
e["Problem"]["Solution"]["Size"] = len(trainingImages[0])
e["Solver"]["Mode"] = "Training"

### Using a neural network solver (deep learning) for inference
e["Solver"]["Termination Criteria"]["Max Generations"] = 1
e["Solver"]["Type"] = "Learner/DeepSupervisor"
e["Solver"]["Loss Function"] = "Mean Squared Error"
e["Solver"]["Neural Network"]["Engine"] = "OneDNN"
e["Solver"]["Neural Network"]["Optimizer"] = "Adam"


### Defining the shape of the neural network [autoencoder version of LeNet-1 - http://yann.lecun.com/exdb/publis/pdf/lecun-90c.pdf (fig. 2)]
## Convolutional Layer with tanh activation function [1x28x28] -> [6x24x24]
e["Solver"]["Neural Network"]["Hidden Layers"][0]["Type"] = "Layer/Convolution"
e["Solver"]["Neural Network"]["Hidden Layers"][0]["Image Height"]      = 28
e["Solver"]["Neural Network"]["Hidden Layers"][0]["Image Width"]       = 28
e["Solver"]["Neural Network"]["Hidden Layers"][0]["Kernel Height"]     = 5
e["Solver"]["Neural Network"]["Hidden Layers"][0]["Kernel Width"]      = 5
e["Solver"]["Neural Network"]["Hidden Layers"][0]["Vertical Stride"]   = 1
e["Solver"]["Neural Network"]["Hidden Layers"][0]["Horizontal Stride"] = 1
e["Solver"]["Neural Network"]["Hidden Layers"][0]["Padding Left"]      = 0
e["Solver"]["Neural Network"]["Hidden Layers"][0]["Padding Right"]     = 0
e["Solver"]["Neural Network"]["Hidden Layers"][0]["Padding Top"]       = 0
e["Solver"]["Neural Network"]["Hidden Layers"][0]["Padding Bottom"]    = 0
e["Solver"]["Neural Network"]["Hidden Layers"][0]["Output Channels"]   = 4*24*24

e["Solver"]["Neural Network"]["Hidden Layers"][1]["Type"] = "Layer/Activation"
e["Solver"]["Neural Network"]["Hidden Layers"][1]["Function"] = "Elementwise/Tanh"

## Pooling Layer [4x24x24] -> [4x12x12]
e["Solver"]["Neural Network"]["Hidden Layers"][2]["Type"] = "Layer/Pooling"
e["Solver"]["Neural Network"]["Hidden Layers"][2]["Function"]          = "Exclusive Average"
e["Solver"]["Neural Network"]["Hidden Layers"][2]["Image Height"]      = 24
e["Solver"]["Neural Network"]["Hidden Layers"][2]["Image Width"]       = 24
e["Solver"]["Neural Network"]["Hidden Layers"][2]["Kernel Height"]     = 2
e["Solver"]["Neural Network"]["Hidden Layers"][2]["Kernel Width"]      = 2
e["Solver"]["Neural Network"]["Hidden Layers"][2]["Vertical Stride"]   = 2
e["Solver"]["Neural Network"]["Hidden Layers"][2]["Horizontal Stride"] = 2
e["Solver"]["Neural Network"]["Hidden Layers"][2]["Padding Left"]      = 0
e["Solver"]["Neural Network"]["Hidden Layers"][2]["Padding Right"]     = 0
e["Solver"]["Neural Network"]["Hidden Layers"][2]["Padding Top"]       = 0
e["Solver"]["Neural Network"]["Hidden Layers"][2]["Padding Bottom"]    = 0
e["Solver"]["Neural Network"]["Hidden Layers"][2]["Output Channels"]   = 4*12*12

## Convolutional Layer with tanh activation function [4x12x12] -> [12x8x8]
e["Solver"]["Neural Network"]["Hidden Layers"][3]["Type"] = "Layer/Convolution"
e["Solver"]["Neural Network"]["Hidden Layers"][3]["Image Height"]      = 12
e["Solver"]["Neural Network"]["Hidden Layers"][3]["Image Width"]       = 12
e["Solver"]["Neural Network"]["Hidden Layers"][3]["Kernel Height"]     = 5
e["Solver"]["Neural Network"]["Hidden Layers"][3]["Kernel Width"]      = 5
e["Solver"]["Neural Network"]["Hidden Layers"][3]["Vertical Stride"]   = 1
e["Solver"]["Neural Network"]["Hidden Layers"][3]["Horizontal Stride"] = 1
e["Solver"]["Neural Network"]["Hidden Layers"][3]["Padding Left"]      = 0
e["Solver"]["Neural Network"]["Hidden Layers"][3]["Padding Right"]     = 0
e["Solver"]["Neural Network"]["Hidden Layers"][3]["Padding Top"]       = 0
e["Solver"]["Neural Network"]["Hidden Layers"][3]["Padding Bottom"]    = 0
e["Solver"]["Neural Network"]["Hidden Layers"][3]["Output Channels"]   = 12*8*8

e["Solver"]["Neural Network"]["Hidden Layers"][4]["Type"] = "Layer/Activation"
e["Solver"]["Neural Network"]["Hidden Layers"][4]["Function"] = "Elementwise/Tanh"

## Pooling Layer [12x8x8] -> [12x4x4]
e["Solver"]["Neural Network"]["Hidden Layers"][5]["Type"] = "Layer/Pooling"
e["Solver"]["Neural Network"]["Hidden Layers"][5]["Function"]          = "Exclusive Average"
e["Solver"]["Neural Network"]["Hidden Layers"][5]["Image Height"]      = 8
e["Solver"]["Neural Network"]["Hidden Layers"][5]["Image Width"]       = 8
e["Solver"]["Neural Network"]["Hidden Layers"][5]["Kernel Height"]     = 2
e["Solver"]["Neural Network"]["Hidden Layers"][5]["Kernel Width"]      = 2
e["Solver"]["Neural Network"]["Hidden Layers"][5]["Vertical Stride"]   = 2
e["Solver"]["Neural Network"]["Hidden Layers"][5]["Horizontal Stride"] = 2
e["Solver"]["Neural Network"]["Hidden Layers"][5]["Padding Left"]      = 0
e["Solver"]["Neural Network"]["Hidden Layers"][5]["Padding Right"]     = 0
e["Solver"]["Neural Network"]["Hidden Layers"][5]["Padding Top"]       = 0
e["Solver"]["Neural Network"]["Hidden Layers"][5]["Padding Bottom"]    = 0
e["Solver"]["Neural Network"]["Hidden Layers"][5]["Output Channels"]   = 12*4*4

## Convolutional Fully Connected Latent Representation Layer [12x4x4] -> [10x1x1]
e["Solver"]["Neural Network"]["Hidden Layers"][6]["Type"] = "Layer/Convolution"
e["Solver"]["Neural Network"]["Hidden Layers"][6]["Image Height"]      = 4
e["Solver"]["Neural Network"]["Hidden Layers"][6]["Image Width"]       = 4
e["Solver"]["Neural Network"]["Hidden Layers"][6]["Kernel Height"]     = 4
e["Solver"]["Neural Network"]["Hidden Layers"][6]["Kernel Width"]      = 4
e["Solver"]["Neural Network"]["Hidden Layers"][6]["Vertical Stride"]   = 1
e["Solver"]["Neural Network"]["Hidden Layers"][6]["Horizontal Stride"] = 1
e["Solver"]["Neural Network"]["Hidden Layers"][6]["Padding Left"]      = 0
e["Solver"]["Neural Network"]["Hidden Layers"][6]["Padding Right"]     = 0
e["Solver"]["Neural Network"]["Hidden Layers"][6]["Padding Top"]       = 0
e["Solver"]["Neural Network"]["Hidden Layers"][6]["Padding Bottom"]    = 0
e["Solver"]["Neural Network"]["Hidden Layers"][6]["Output Channels"]   = 10*1*1

## Deconvolutional of Fully Connected Latent Representation Layer [10x1x1] -> [12x4x4] 
e["Solver"]["Neural Network"]["Hidden Layers"][7]["Type"] = "Layer/Deconvolution"
e["Solver"]["Neural Network"]["Hidden Layers"][7]["Image Height"]      = 4
e["Solver"]["Neural Network"]["Hidden Layers"][7]["Image Width"]       = 4
e["Solver"]["Neural Network"]["Hidden Layers"][7]["Kernel Height"]     = 4
e["Solver"]["Neural Network"]["Hidden Layers"][7]["Kernel Width"]      = 4
e["Solver"]["Neural Network"]["Hidden Layers"][7]["Vertical Stride"]   = 1
e["Solver"]["Neural Network"]["Hidden Layers"][7]["Horizontal Stride"] = 1
e["Solver"]["Neural Network"]["Hidden Layers"][7]["Padding Left"]      = 0
e["Solver"]["Neural Network"]["Hidden Layers"][7]["Padding Right"]     = 0
e["Solver"]["Neural Network"]["Hidden Layers"][7]["Padding Top"]       = 0
e["Solver"]["Neural Network"]["Hidden Layers"][7]["Padding Bottom"]    = 0
e["Solver"]["Neural Network"]["Hidden Layers"][7]["Output Channels"]   = 12*4*4

## Deonvolutional of Pooling Layer [12x4x4] -> [12x8x8]
e["Solver"]["Neural Network"]["Hidden Layers"][8]["Type"] = "Layer/Deconvolution"
e["Solver"]["Neural Network"]["Hidden Layers"][8]["Image Height"]      = 8
e["Solver"]["Neural Network"]["Hidden Layers"][8]["Image Width"]       = 8
e["Solver"]["Neural Network"]["Hidden Layers"][8]["Kernel Height"]     = 2
e["Solver"]["Neural Network"]["Hidden Layers"][8]["Kernel Width"]      = 2
e["Solver"]["Neural Network"]["Hidden Layers"][8]["Vertical Stride"]   = 2
e["Solver"]["Neural Network"]["Hidden Layers"][8]["Horizontal Stride"] = 2
e["Solver"]["Neural Network"]["Hidden Layers"][8]["Padding Left"]      = 0
e["Solver"]["Neural Network"]["Hidden Layers"][8]["Padding Right"]     = 0
e["Solver"]["Neural Network"]["Hidden Layers"][8]["Padding Top"]       = 0
e["Solver"]["Neural Network"]["Hidden Layers"][8]["Padding Bottom"]    = 0
e["Solver"]["Neural Network"]["Hidden Layers"][8]["Output Channels"]   = 12*8*8

## Deconvolutional of Convolutional Layer [12x8x8] -> [4x12x12]
e["Solver"]["Neural Network"]["Hidden Layers"][9]["Type"] = "Layer/Deconvolution"
e["Solver"]["Neural Network"]["Hidden Layers"][9]["Image Height"]      = 12
e["Solver"]["Neural Network"]["Hidden Layers"][9]["Image Width"]       = 12
e["Solver"]["Neural Network"]["Hidden Layers"][9]["Kernel Height"]     = 5
e["Solver"]["Neural Network"]["Hidden Layers"][9]["Kernel Width"]      = 5
e["Solver"]["Neural Network"]["Hidden Layers"][9]["Vertical Stride"]   = 1
e["Solver"]["Neural Network"]["Hidden Layers"][9]["Horizontal Stride"] = 1
e["Solver"]["Neural Network"]["Hidden Layers"][9]["Padding Left"]      = 0
e["Solver"]["Neural Network"]["Hidden Layers"][9]["Padding Right"]     = 0
e["Solver"]["Neural Network"]["Hidden Layers"][9]["Padding Top"]       = 0
e["Solver"]["Neural Network"]["Hidden Layers"][9]["Padding Bottom"]    = 0
e["Solver"]["Neural Network"]["Hidden Layers"][9]["Output Channels"]   = 4*12*12

## Deconvolutional of Pooling Layer [4x12x12] -> [4x24x24] 
e["Solver"]["Neural Network"]["Hidden Layers"][10]["Type"] = "Layer/Deconvolution"
e["Solver"]["Neural Network"]["Hidden Layers"][10]["Image Height"]      = 24
e["Solver"]["Neural Network"]["Hidden Layers"][10]["Image Width"]       = 24
e["Solver"]["Neural Network"]["Hidden Layers"][10]["Kernel Height"]     = 2
e["Solver"]["Neural Network"]["Hidden Layers"][10]["Kernel Width"]      = 2
e["Solver"]["Neural Network"]["Hidden Layers"][10]["Vertical Stride"]   = 2
e["Solver"]["Neural Network"]["Hidden Layers"][10]["Horizontal Stride"] = 2
e["Solver"]["Neural Network"]["Hidden Layers"][10]["Padding Left"]      = 0
e["Solver"]["Neural Network"]["Hidden Layers"][10]["Padding Right"]     = 0
e["Solver"]["Neural Network"]["Hidden Layers"][10]["Padding Top"]       = 0
e["Solver"]["Neural Network"]["Hidden Layers"][10]["Padding Bottom"]    = 0
e["Solver"]["Neural Network"]["Hidden Layers"][10]["Output Channels"]   = 4*24*24

## Deconvolutional of Convolutional Layer [6x28x28] -> [1x28x28]
e["Solver"]["Neural Network"]["Hidden Layers"][11]["Type"] = "Layer/Deconvolution"
e["Solver"]["Neural Network"]["Hidden Layers"][11]["Image Height"]      = 28
e["Solver"]["Neural Network"]["Hidden Layers"][11]["Image Width"]       = 28
e["Solver"]["Neural Network"]["Hidden Layers"][11]["Kernel Height"]     = 5
e["Solver"]["Neural Network"]["Hidden Layers"][11]["Kernel Width"]      = 5
e["Solver"]["Neural Network"]["Hidden Layers"][11]["Vertical Stride"]   = 1
e["Solver"]["Neural Network"]["Hidden Layers"][11]["Horizontal Stride"] = 1
e["Solver"]["Neural Network"]["Hidden Layers"][11]["Padding Left"]      = 0
e["Solver"]["Neural Network"]["Hidden Layers"][11]["Padding Right"]     = 0
e["Solver"]["Neural Network"]["Hidden Layers"][11]["Padding Top"]       = 0
e["Solver"]["Neural Network"]["Hidden Layers"][11]["Padding Bottom"]    = 0
e["Solver"]["Neural Network"]["Hidden Layers"][11]["Output Channels"]   = 1*28*28

### Configuring output

e["Console Output"]["Verbosity"] = "Silent"
e["File Output"]["Enabled"] = False
e["Random Seed"] = 0xC0FFEE

### Printing Configuration

print("[Korali] Running MNIST solver.")
print("[Korali] Algorithm: " + str(e["Solver"]["Neural Network"]["Optimizer"]))
print("[Korali] Database Size: " + str(len(trainingImageVector)))
print("[Korali] Batch Size: " + str(trainingBatchSize))
print("[Korali] Epochs: " + str(epochs))
print("[Korali] Initial Learning Rate: " + str(learningRate))
print("[Korali] Decay: " + str(decay))

### Running SGD loop

for epoch in range(epochs):
    e["Solver"]["Mode"] = "Training"
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
        e["Solver"]["Termination Criteria"]["Max Generations"] = (
            e["Solver"]["Termination Criteria"]["Max Generations"] + 1
        )

        # Running step
        k.run(e)

    # Printing Information
    print("[Korali] --------------------------------------------------")
    print("[Korali] Epoch: " + str(epoch) + "/" + str(epochs))
    print("[Korali] Learning Rate: " + str(learningRate))
    print("[Korali] Current Training Loss: " + str(e["Solver"]["Current Loss"]))
    learningRate = learningRate * (1.0 / (1.0 + decay * (epoch + 1)))
    # Evaluating testing set
    e["Problem"]["Input"]["Data"] = testingImageVector
    e["Problem"]["Solution"]["Data"] = [img[0] for img in testingImageVector]
    e["Solver"]["Mode"] = "Testing"
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

    # Adjusting learning rate via decay
