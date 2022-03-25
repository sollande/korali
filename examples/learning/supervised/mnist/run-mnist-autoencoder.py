#!/usr/bin/env python3
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import time
import korali
import random
from mnist import MNIST
from mpi4py import MPI
from statistics import mean
import argparse

sys.argv=['--test']

k = korali.Engine()
e = korali.Experiment()

# MPIcomm = MPI.COMM_WORLD
# MPIrank = MPIcomm.Get_rank()
# MPIsize = MPIcomm.Get_size()
# MPIroot = MPIsize - 1
# k.setMPIComm(MPI.COMM_WORLD)
k["Conduit"]["Type"] = "Sequential"
# e["Solver"]["Batch Concurrency"] = 1
### Hyperparameters

learningRate = 0.0001
decay = 0.0001
trainingBatchSize = 80
epochs = 3
validation_split = 0.1
MAX_RGB = 255.0

### Loading MNIST data [28x28 images with {0,..,9} as label - http://yann.lecun.com/exdb/mnist/]

mndata = MNIST("./_data")
mndata.gz = True
trainingImages, _ = mndata.load_training()
testingImages, _ = mndata.load_testing()
trainingImages = [[p/MAX_RGB for p in img] for img in trainingImages]
testingImages = [[p/MAX_RGB for p in img] for img in testingImages]

nb_validation_samples = int(validation_split*len(trainingImages))
nb_training_samples = len(trainingImages)-nb_validation_samples
nb_testing_samples = 10
# TODO: shuffle
### If this is test mode, run only one epoch
# if len(sys.argv) == 2:
#     if sys.argv[1] == "--test":
if False:
    epochs = 3
    nb_validation_samples = int(validation_split*len(trainingImages[:6000]))
    nb_training_samples = len(trainingImages[:6000])-nb_validation_samples
    nb_testing_samples = 10

# e["File Output"]["Path"] = "resultsPath"
# Loading previous results, if they exist.
# found = e.loadState()
random.shuffle(trainingImages)

trainingInput = trainingImages[:nb_training_samples]
validationImages = trainingImages[nb_training_samples+1:nb_training_samples+1+nb_validation_samples]
testingImages = testingImages[:nb_testing_samples]
trainingBatchSize = 100
validationBatchSize = nb_validation_samples

img_size = len(trainingInput[0])

### Converting images to Korali form (requires a time dimension)

flattendTrainingSamples = [[img] for img in trainingInput]
flattendTrainingTargets = trainingInput
flattendValidationSamples = [[img] for img in validationImages]
flattendValidationTargets = validationImages
flattendTestingSamples = [[img] for img in testingImages]

### Shuffling training data set for stochastic gradient descent training
# miniBatchIndices = list(range(nb_training_samples))
# random.shuffle(miniBatchIndices)

### Calculating Derived Values
stepsPerEpoch = int(nb_training_samples / trainingBatchSize)
testingBatchSize = nb_testing_samples

### Configuring general problem settings

e["Problem"]["Type"] = "Supervised Learning"
e["Problem"]["Input"]["Data"] = flattendTrainingSamples
e["Problem"]["Solution"]["Data"] = flattendTrainingTargets
# e["Problem"]["Validation Set"]["Data"] = flattendValidationSamples[:2]
# e["Problem"]["Validation Set"]["Solution"] = flattendValidationTargets[:2]
# e["Problem"]["Validation Batch Size"] = 1

e["Problem"]["Max Timesteps"] = 1

e["Problem"]["Training Batch Size"] = trainingBatchSize
e["Problem"]["Testing Batch Size"] = testingBatchSize
# e["Problem"]["Testing Batch Size"] = testingBatchSize

e["Problem"]["Input"]["Size"] = img_size
e["Problem"]["Solution"]["Size"] = img_size

### Using a neural network solver (deep learning) for inference
e["Solver"]["Termination Criteria"]["Max Generations"] = epochs
e["Solver"]["Learning Rate"] = learningRate

e["Solver"]["Type"] = "Learner/DeepSupervisor"
e["Solver"]["Mode"] = "Training"
e["Solver"]["Loss Function"] = "Mean Squared Error"
e["Solver"]["Neural Network"]["Engine"] = "OneDNN"
e["Solver"]["Neural Network"]["Optimizer"] = "Adam"


### Defining the shape of the neural network [autoencoder version of LeNet-1 - http://yann.lecun.com/exdb/publis/pdf/lecun-90c.pdf (fig. 2)]
## Convolutional Layer with tanh activation function [1x28x28] -> [6x24x24]
e["Solver"]["Neural Network"]["Hidden Layers"][0]["Type"] = "Layer/Convolution"
e["Solver"]["Neural Network"]["Hidden Layers"][0]["Image Height"] = 28
e["Solver"]["Neural Network"]["Hidden Layers"][0]["Image Width"] = 28
e["Solver"]["Neural Network"]["Hidden Layers"][0]["Kernel Height"] = 5
e["Solver"]["Neural Network"]["Hidden Layers"][0]["Kernel Width"] = 5
e["Solver"]["Neural Network"]["Hidden Layers"][0]["Vertical Stride"] = 1
e["Solver"]["Neural Network"]["Hidden Layers"][0]["Horizontal Stride"] = 1
e["Solver"]["Neural Network"]["Hidden Layers"][0]["Padding Left"] = 0
e["Solver"]["Neural Network"]["Hidden Layers"][0]["Padding Right"] = 0
e["Solver"]["Neural Network"]["Hidden Layers"][0]["Padding Top"] = 0
e["Solver"]["Neural Network"]["Hidden Layers"][0]["Padding Bottom"] = 0
e["Solver"]["Neural Network"]["Hidden Layers"][0]["Output Channels"] = 4 * 24 * 24

e["Solver"]["Neural Network"]["Hidden Layers"][1]["Type"] = "Layer/Activation"
e["Solver"]["Neural Network"]["Hidden Layers"][1]["Function"] = "Elementwise/Tanh"

## Pooling Layer [4x24x24] -> [4x12x12]
e["Solver"]["Neural Network"]["Hidden Layers"][2]["Type"] = "Layer/Pooling"
e["Solver"]["Neural Network"]["Hidden Layers"][2]["Function"] = "Exclusive Average"
e["Solver"]["Neural Network"]["Hidden Layers"][2]["Image Height"] = 24
e["Solver"]["Neural Network"]["Hidden Layers"][2]["Image Width"] = 24
e["Solver"]["Neural Network"]["Hidden Layers"][2]["Kernel Height"] = 2
e["Solver"]["Neural Network"]["Hidden Layers"][2]["Kernel Width"] = 2
e["Solver"]["Neural Network"]["Hidden Layers"][2]["Vertical Stride"] = 2
e["Solver"]["Neural Network"]["Hidden Layers"][2]["Horizontal Stride"] = 2
e["Solver"]["Neural Network"]["Hidden Layers"][2]["Padding Left"] = 0
e["Solver"]["Neural Network"]["Hidden Layers"][2]["Padding Right"] = 0
e["Solver"]["Neural Network"]["Hidden Layers"][2]["Padding Top"] = 0
e["Solver"]["Neural Network"]["Hidden Layers"][2]["Padding Bottom"] = 0
e["Solver"]["Neural Network"]["Hidden Layers"][2]["Output Channels"] = 4 * 12 * 12

## Convolutional Layer with tanh activation function [4x12x12] -> [12x8x8]
e["Solver"]["Neural Network"]["Hidden Layers"][3]["Type"] = "Layer/Convolution"
e["Solver"]["Neural Network"]["Hidden Layers"][3]["Image Height"] = 12
e["Solver"]["Neural Network"]["Hidden Layers"][3]["Image Width"] = 12
e["Solver"]["Neural Network"]["Hidden Layers"][3]["Kernel Height"] = 5
e["Solver"]["Neural Network"]["Hidden Layers"][3]["Kernel Width"] = 5
e["Solver"]["Neural Network"]["Hidden Layers"][3]["Vertical Stride"] = 1
e["Solver"]["Neural Network"]["Hidden Layers"][3]["Horizontal Stride"] = 1
e["Solver"]["Neural Network"]["Hidden Layers"][3]["Padding Left"] = 0
e["Solver"]["Neural Network"]["Hidden Layers"][3]["Padding Right"] = 0
e["Solver"]["Neural Network"]["Hidden Layers"][3]["Padding Top"] = 0
e["Solver"]["Neural Network"]["Hidden Layers"][3]["Padding Bottom"] = 0
e["Solver"]["Neural Network"]["Hidden Layers"][3]["Output Channels"] = 12 * 8 * 8

e["Solver"]["Neural Network"]["Hidden Layers"][4]["Type"] = "Layer/Activation"
e["Solver"]["Neural Network"]["Hidden Layers"][4]["Function"] = "Elementwise/Tanh"

## Pooling Layer [12x8x8] -> [12x4x4]
e["Solver"]["Neural Network"]["Hidden Layers"][5]["Type"] = "Layer/Pooling"
e["Solver"]["Neural Network"]["Hidden Layers"][5]["Function"] = "Exclusive Average"
e["Solver"]["Neural Network"]["Hidden Layers"][5]["Image Height"] = 8
e["Solver"]["Neural Network"]["Hidden Layers"][5]["Image Width"] = 8
e["Solver"]["Neural Network"]["Hidden Layers"][5]["Kernel Height"] = 2
e["Solver"]["Neural Network"]["Hidden Layers"][5]["Kernel Width"] = 2
e["Solver"]["Neural Network"]["Hidden Layers"][5]["Vertical Stride"] = 2
e["Solver"]["Neural Network"]["Hidden Layers"][5]["Horizontal Stride"] = 2
e["Solver"]["Neural Network"]["Hidden Layers"][5]["Padding Left"] = 0
e["Solver"]["Neural Network"]["Hidden Layers"][5]["Padding Right"] = 0
e["Solver"]["Neural Network"]["Hidden Layers"][5]["Padding Top"] = 0
e["Solver"]["Neural Network"]["Hidden Layers"][5]["Padding Bottom"] = 0
e["Solver"]["Neural Network"]["Hidden Layers"][5]["Output Channels"] = 12 * 4 * 4

## Convolutional Fully Connected Latent Representation Layer [12x4x4] -> [10x1x1]
e["Solver"]["Neural Network"]["Hidden Layers"][6]["Type"] = "Layer/Convolution"
e["Solver"]["Neural Network"]["Hidden Layers"][6]["Image Height"] = 4
e["Solver"]["Neural Network"]["Hidden Layers"][6]["Image Width"] = 4
e["Solver"]["Neural Network"]["Hidden Layers"][6]["Kernel Height"] = 4
e["Solver"]["Neural Network"]["Hidden Layers"][6]["Kernel Width"] = 4
e["Solver"]["Neural Network"]["Hidden Layers"][6]["Vertical Stride"] = 1
e["Solver"]["Neural Network"]["Hidden Layers"][6]["Horizontal Stride"] = 1
e["Solver"]["Neural Network"]["Hidden Layers"][6]["Padding Left"] = 0
e["Solver"]["Neural Network"]["Hidden Layers"][6]["Padding Right"] = 0
e["Solver"]["Neural Network"]["Hidden Layers"][6]["Padding Top"] = 0
e["Solver"]["Neural Network"]["Hidden Layers"][6]["Padding Bottom"] = 0
e["Solver"]["Neural Network"]["Hidden Layers"][6]["Output Channels"] = 10 * 1 * 1

## Deconvolutional of Fully Connected Latent Representation Layer [10x1x1] -> [12x4x4]
e["Solver"]["Neural Network"]["Hidden Layers"][7]["Type"] = "Layer/Deconvolution"
e["Solver"]["Neural Network"]["Hidden Layers"][7]["Image Height"] = 4
e["Solver"]["Neural Network"]["Hidden Layers"][7]["Image Width"] = 4
e["Solver"]["Neural Network"]["Hidden Layers"][7]["Kernel Height"] = 4
e["Solver"]["Neural Network"]["Hidden Layers"][7]["Kernel Width"] = 4
e["Solver"]["Neural Network"]["Hidden Layers"][7]["Vertical Stride"] = 1
e["Solver"]["Neural Network"]["Hidden Layers"][7]["Horizontal Stride"] = 1
e["Solver"]["Neural Network"]["Hidden Layers"][7]["Padding Left"] = 0
e["Solver"]["Neural Network"]["Hidden Layers"][7]["Padding Right"] = 0
e["Solver"]["Neural Network"]["Hidden Layers"][7]["Padding Top"] = 0
e["Solver"]["Neural Network"]["Hidden Layers"][7]["Padding Bottom"] = 0
e["Solver"]["Neural Network"]["Hidden Layers"][7]["Output Channels"] = 12 * 4 * 4

## Deonvolutional of Pooling Layer [12x4x4] -> [12x8x8]
e["Solver"]["Neural Network"]["Hidden Layers"][8]["Type"] = "Layer/Deconvolution"
e["Solver"]["Neural Network"]["Hidden Layers"][8]["Image Height"] = 8
e["Solver"]["Neural Network"]["Hidden Layers"][8]["Image Width"] = 8
e["Solver"]["Neural Network"]["Hidden Layers"][8]["Kernel Height"] = 2
e["Solver"]["Neural Network"]["Hidden Layers"][8]["Kernel Width"] = 2
e["Solver"]["Neural Network"]["Hidden Layers"][8]["Vertical Stride"] = 2
e["Solver"]["Neural Network"]["Hidden Layers"][8]["Horizontal Stride"] = 2
e["Solver"]["Neural Network"]["Hidden Layers"][8]["Padding Left"] = 0
e["Solver"]["Neural Network"]["Hidden Layers"][8]["Padding Right"] = 0
e["Solver"]["Neural Network"]["Hidden Layers"][8]["Padding Top"] = 0
e["Solver"]["Neural Network"]["Hidden Layers"][8]["Padding Bottom"] = 0
e["Solver"]["Neural Network"]["Hidden Layers"][8]["Output Channels"] = 12 * 8 * 8

## Deconvolutional of Convolutional Layer [12x8x8] -> [4x12x12]
e["Solver"]["Neural Network"]["Hidden Layers"][9]["Type"] = "Layer/Deconvolution"
e["Solver"]["Neural Network"]["Hidden Layers"][9]["Image Height"] = 12
e["Solver"]["Neural Network"]["Hidden Layers"][9]["Image Width"] = 12
e["Solver"]["Neural Network"]["Hidden Layers"][9]["Kernel Height"] = 5
e["Solver"]["Neural Network"]["Hidden Layers"][9]["Kernel Width"] = 5
e["Solver"]["Neural Network"]["Hidden Layers"][9]["Vertical Stride"] = 1
e["Solver"]["Neural Network"]["Hidden Layers"][9]["Horizontal Stride"] = 1
e["Solver"]["Neural Network"]["Hidden Layers"][9]["Padding Left"] = 0
e["Solver"]["Neural Network"]["Hidden Layers"][9]["Padding Right"] = 0
e["Solver"]["Neural Network"]["Hidden Layers"][9]["Padding Top"] = 0
e["Solver"]["Neural Network"]["Hidden Layers"][9]["Padding Bottom"] = 0
e["Solver"]["Neural Network"]["Hidden Layers"][9]["Output Channels"] = 4 * 12 * 12

## Deconvolutional of Pooling Layer [4x12x12] -> [4x24x24]
e["Solver"]["Neural Network"]["Hidden Layers"][10]["Type"] = "Layer/Deconvolution"
e["Solver"]["Neural Network"]["Hidden Layers"][10]["Image Height"] = 24
e["Solver"]["Neural Network"]["Hidden Layers"][10]["Image Width"] = 24
e["Solver"]["Neural Network"]["Hidden Layers"][10]["Kernel Height"] = 2
e["Solver"]["Neural Network"]["Hidden Layers"][10]["Kernel Width"] = 2
e["Solver"]["Neural Network"]["Hidden Layers"][10]["Vertical Stride"] = 2
e["Solver"]["Neural Network"]["Hidden Layers"][10]["Horizontal Stride"] = 2
e["Solver"]["Neural Network"]["Hidden Layers"][10]["Padding Left"] = 0
e["Solver"]["Neural Network"]["Hidden Layers"][10]["Padding Right"] = 0
e["Solver"]["Neural Network"]["Hidden Layers"][10]["Padding Top"] = 0
e["Solver"]["Neural Network"]["Hidden Layers"][10]["Padding Bottom"] = 0
e["Solver"]["Neural Network"]["Hidden Layers"][10]["Output Channels"] = 4 * 24 * 24

## Deconvolutional of Convolutional Layer [6x28x28] -> [1x28x28]
e["Solver"]["Neural Network"]["Hidden Layers"][11]["Type"] = "Layer/Deconvolution"
e["Solver"]["Neural Network"]["Hidden Layers"][11]["Image Height"] = 28
e["Solver"]["Neural Network"]["Hidden Layers"][11]["Image Width"] = 28
e["Solver"]["Neural Network"]["Hidden Layers"][11]["Kernel Height"] = 5
e["Solver"]["Neural Network"]["Hidden Layers"][11]["Kernel Width"] = 5
e["Solver"]["Neural Network"]["Hidden Layers"][11]["Vertical Stride"] = 1
e["Solver"]["Neural Network"]["Hidden Layers"][11]["Horizontal Stride"] = 1
e["Solver"]["Neural Network"]["Hidden Layers"][11]["Padding Left"] = 0
e["Solver"]["Neural Network"]["Hidden Layers"][11]["Padding Right"] = 0
e["Solver"]["Neural Network"]["Hidden Layers"][11]["Padding Top"] = 0
e["Solver"]["Neural Network"]["Hidden Layers"][11]["Padding Bottom"] = 0
e["Solver"]["Neural Network"]["Hidden Layers"][11]["Output Channels"] = 1 * 28 * 28

### Configuring output

e["Console Output"]["Verbosity"] = "Detailed"
e["File Output"]["Enabled"] = False
e["Random Seed"] = 0xC0FFEE
### Printing Configuration

# print("[Korali] Running MNIST solver.")
# print("[Korali] Algorithm: " + str(e["Solver"]["Neural Network"]["Optimizer"]))
# print("[Korali] Database Size: " + str(len(flattendTrainingSamples)))
# print("[Korali] Batch Size: " + str(trainingBatchSize))
# print("[Korali] Epochs: " + str(epochs))
# print("[Korali] Initial Learning Rate: " + str(learningRate))
# print("[Korali] Decay: " + str(decay))

### Training the neural network
# k["Conduit"]["Concurrent Jobs"] = 1
### Running SGD loop
# for epoch in range(epochs):
#     for step in range(stepsPerEpoch):

#         # Creating minibatch
#         # miniBatchIdx = miniBatchIndices[step * trainingBatchSize : (step+1) * trainingBatchSize]
#         miniBatchInput = flattendTrainingSamples[
#             step * trainingBatchSize : (step + 1) * trainingBatchSize
#         ]  # N x T x C
#         # Solution is the same as the input without the time index
#         miniBatchSolution = [x[0] for x in miniBatchInput]  # N x C
#         # Passing minibatch to Korali
#         e["Problem"]["Input"]["Data"] = miniBatchInput
#         e["Problem"]["Solution"]["Data"] = miniBatchSolution
#         # Reconfiguring solver
#         learningRate = learningRate * (1.0 / (1.0 + decay * (epoch + 1)))
#         e["Solver"]["Learning Rate"] = learningRate
#         # e["Solver"]["Termination Criteria"]["Max Generations"] = int(e["Solver"]["Termination Criteria"]["Max Generations"]) + 1
#         # Running step
#         k.run(e)

#     # Printing Information
#     # print("[Korali] --------------------------------------------------")
#     # print("[Korali] Epoch: " + str(epoch) + "/" + str(epochs))
#     # print("[Korali] Learning Rate: " + str(learningRate))
#     # print('[Korali] Current Training Loss: ' + str(e["Solver"]["Current Loss"]))

print("Start run")
k.run(e)
print("Finished run")
#Evaluating testing set
e["Problem"]["Input"]["Data"] = flattendTestingSamples
e["Solver"]["Mode"] = "Testing"
k.run(e)

# img = e["Solver"]["Evaluation"][0]
# pixels = first_image.reshape((28, 28))
images_real = [np.array(img[0]).reshape((28,28)) for img in flattendTestingSamples]  # N x C
images = [np.array(img).reshape((28, 28))  for img in e["Solver"]["Evaluation"]]

f, axarr = plt.subplots(nb_testing_samples, 2)
for i, (img, img_real) in enumerate(zip(images, images_real)):
    plt.imshow(img, cmap='gray')
    axarr[i,0].imshow(img_real, cmap='gray')
    axarr[i,1].imshow(img, cmap='gray')

plt.show()

# k.run(e)
# Getting MSE loss for testing set
# mse = 0.0
# for i, yhat in enumerate(e["Solver"]["Evaluation"]):
#   y = flattendTestingSamples[i][0]
#   mse.append(mean([(img-imghat)*(img-imghat) for (img, imghat) in zip(y, yhat)]))
# for i, yhat in enumerate(e["Solver"]["Evaluation"]):
#     y = flattendTestingSamples[i][0]
#     mse += sum([(px - pxhat) * (px - pxhat) for (px, pxhat) in zip(y, yhat)])

# mse = mse / (nb_testing_samples * 2)
# print("[Korali] Current Testing Loss:  " + str(mse))

# Adjusting learning rate via decay

# for i, yhat in enumerate(e["Solver"]["Evaluation"]):
