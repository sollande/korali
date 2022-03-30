#!/usr/bin/env python3
import os
import sys
import pickle
import numpy as np
import matplotlib.pyplot as plt
import time
import korali
import random

from utilities import getSamePadding

k = korali.Engine()

# Hyperparameters
 
layers = 5
learningRate = 0.0001
trainingSize = 0.5
decay = 0.0001
trainingBatchSize = 12
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
train_idx = idx[:int(samples*trainingSize)]

trainingImages = trajectories[train_idx]
testingImages = trajectories[~train_idx]


### Converting images to Korali form (requires a time dimension)

trainingImageVector = [ [ x ] for x in trainingImages.tolist()]
testingImageVector = [ [ x ] for x in testingImages.tolist()]

### Shuffling training data set for stochastic gradient descent training

random.shuffle(trainingImageVector)

### LAYER DEFINITIONS ==================================================================
### CNNS =====================
inputDim_CNN = [{"height": img_height, "width": img_width},
                {"height": img_height/2, "width": img_width/2},
                {"height": img_height/4, "width": img_width/4},
                {"height": img_height/8, "width": img_width/8},
                {"height": img_height/16, "width": img_width/16},
                {"height": 1, "width": 1}
                ]
outputDim_CNN = inputDim_CNN[0:5]

kernelDim_CNN = [{"width": kernelSizeFactor, "height": kernelSizeFactor},
                 {"width": kernelSizeFactor-2, "height": kernelSizeFactor-2},
                 {"width": kernelSizeFactor-4, "height":  kernelSizeFactor-4},
                 {"width": 3, "height": 3},
                 {"width": 1, "height": 1}]
kernelStrides_CNN = [{"width": 1, "height": 1},
                    {"width": 1, "height": 1},
                    {"width": 1, "height": 1},
                    {"width": 1, "height": 1},
                    {"width": 1, "height": 1},
                    ]
# padding left, right, top, bottom
# autoencoderFactor, 4*autoencoderFactor, 16*autoencoderFactor, 64*autoencoderFactor, 1
outputChannels_CNN = [autoencoderFactor,
                      4**1*autoencoderFactor,
                      4**2*autoencoderFactor,
                      4**3*autoencoderFactor,
                      1,
                      ]
# infered from the output channels
output_dim_pooling = {}
paddings_CNN = []
for l in range(layers):
    padding_horizontal = getSamePadding(kernelStrides_CNN[l]["width"], inputDim_CNN[l]["width"], kernelDim_CNN[l]["width"])
    padding_vertical = getSamePadding(kernelStrides_CNN[l]["height"], inputDim_CNN[l]["height"], kernelDim_CNN[l]["height"])
    paddings_CNN.append({"left": padding_horizontal, "right": padding_horizontal, "top": padding_vertical, "bottom": padding_vertical})

# Innput dimension+Padding-Kernel_dim/Kernel_Stride

# kernels*OD_per_kernel["width"]*OD_per_kernel["height"]
### POOLING ================================
poolingFunction = layers*["Exclusive Average"]
outputChannels_pooling = outputChannels_CNN
inputDim_pooling = outputDim_CNN
kernelDim_pooling = [{"width": 2, "height": 2},
                 {"width": 2, "height": 2},
                 {"width": 2, "height": 2},
                 {"width": 2, "height": 2},
                 {"width": 2, "height": 2}]

kernelStrides_pooling = kernelDim_pooling
# padding left, right, top, bottom
paddings_pooling = [{"left": 0, "right": 0, "top": 0, "bottom": 0},
                     {"left": 0, "right": 0, "top": 0, "bottom": 0},
                     {"left": 0, "right": 0, "top": 0, "bottom": 0},
                     {"left": 0, "right": 0, "top": 0, "bottom": 0},
                     {"left": 0, "right": 0, "top": 0, "bottom": 0},
                     ]

outputDim_pooling = []
for I, P, K, S in zip(inputDim_pooling, paddings_pooling, kernelDim_pooling, kernelStrides_pooling):
    output_dim = lambda I, P1, P2, K, S: (I+P1+P2-K)/S+1
    output_height = output_dim(I["height"], P["top"], P["bottom"], K["height"], S["height"])
    output_width = output_dim(I["width"], P["left"], P["right"], K["width"], S["width"])
    outputDim_pooling.append({"height": output_height, "width": output_width})

### Activations
activations = ["Elementwise/ReLU", "Elementwise/ReLU", "Elementwise/ReLU", "Elementwise/ReLU", "Elementwise/Tanh"]
### ====================================================================================

### Calculating Derived Values

stepsPerEpoch = int(len(trainingImageVector) / trainingBatchSize)
testingBatchSize = len(testingImageVector)
 
### If this is test mode, run only one epoch
if len(sys.argv) == 2:
 if sys.argv[1] == '--test':
  epochs=1
  stepsPerEpoch=1

# epochs=1
# stepsPerEpoch=1
### Configuring general problem settings
e = korali.Experiment()

e["Problem"]["Type"] = "Supervised Learning"
e["Problem"]["Max Timesteps"] = 1
e["Problem"]["Training Batch Size"] = trainingBatchSize
e["Problem"]["Testing Batch Size"] = testingBatchSize
e["Problem"]["Input"]["Size"] = len(trainingImages[0])
e["Problem"]["Solution"]["Size"] = len(trainingImages[0])

e["Problem"]["Input"]["Data"] = trainingImageVector
e["Problem"]["Input"]["Size"] = 1
e["Problem"]["Solution"]["Data"] = trainingImageVector
e["Problem"]["Solution"]["Size"] = 1

### Using a neural network solver (deep learning) for inference

e["Solver"]["Type"] = "Learner/DeepSupervisor"
e["Solver"]["Loss Function"] = "Mean Squared Error"
e["Solver"]["Learning Rate"] = learningRate
e["Solver"]["Batch Concurrency"] = 1
# This is only for evolutionary algorithms
# e["Solver"]["Steps Per Generation"] = 1
e["Solver"]["Neural Network"]["Engine"] = "OneDNN"
e["Solver"]["Neural Network"]["Optimizer"] = "Adam"

### ENCODER ==========================================================================
# layers*3 = CNN, Pooling, Activation
### Defining the shape of the neural network [autoencoder version of LeNet-1 - http://yann.lecun.com/exdb/publis/pdf/lecun-90c.pdf (fig. 2)]
## CONVOLUTIONAL LAYER
## with tanh activation function [1x32x64] -> [6x24x24]
e["Solver"]["Neural Network"]["Hidden Layers"][0]["Type"] = "Layer/Convolution"
e["Solver"]["Neural Network"]["Hidden Layers"][0]["Image Height"]      = 32 # img_width
e["Solver"]["Neural Network"]["Hidden Layers"][0]["Image Width"]       = 64 # img_height
e["Solver"]["Neural Network"]["Hidden Layers"][0]["Kernel Height"]     = 9  # kernel_size_factor
e["Solver"]["Neural Network"]["Hidden Layers"][0]["Kernel Width"]      = 9  # kernel_size_factor
e["Solver"]["Neural Network"]["Hidden Layers"][0]["Vertical Stride"]   = 1
e["Solver"]["Neural Network"]["Hidden Layers"][0]["Horizontal Stride"] = 1
e["Solver"]["Neural Network"]["Hidden Layers"][0]["Padding Left"]      = 4
e["Solver"]["Neural Network"]["Hidden Layers"][0]["Padding Right"]     = 4
e["Solver"]["Neural Network"]["Hidden Layers"][0]["Padding Top"]       = 4
e["Solver"]["Neural Network"]["Hidden Layers"][0]["Padding Bottom"]    = 4
e["Solver"]["Neural Network"]["Hidden Layers"][0]["Output Channels"]   = 4*32*64 # autoencoder_factor*input_height*img_width
## POOLING LAYER
e["Solver"]["Neural Network"]["Hidden Layers"][1]["Type"] = "Layer/Pooling"
e["Solver"]["Neural Network"]["Hidden Layers"][1]["Function"]          = "Exclusive Average"
e["Solver"]["Neural Network"]["Hidden Layers"][1]["Image Height"]      = 32
e["Solver"]["Neural Network"]["Hidden Layers"][1]["Image Width"]       = 64
e["Solver"]["Neural Network"]["Hidden Layers"][1]["Kernel Height"]     = 2
e["Solver"]["Neural Network"]["Hidden Layers"][1]["Kernel Width"]      = 2
e["Solver"]["Neural Network"]["Hidden Layers"][1]["Vertical Stride"]   = 2
e["Solver"]["Neural Network"]["Hidden Layers"][1]["Horizontal Stride"] = 2
e["Solver"]["Neural Network"]["Hidden Layers"][1]["Padding Left"]      = 0
e["Solver"]["Neural Network"]["Hidden Layers"][1]["Padding Right"]     = 0
e["Solver"]["Neural Network"]["Hidden Layers"][1]["Padding Top"]       = 0
e["Solver"]["Neural Network"]["Hidden Layers"][1]["Padding Bottom"]    = 0
e["Solver"]["Neural Network"]["Hidden Layers"][1]["Output Channels"]   = 4*16*32 # output channels CNN

e["Solver"]["Neural Network"]["Hidden Layers"][2]["Type"] = "Layer/Activation"
e["Solver"]["Neural Network"]["Hidden Layers"][2]["Function"] = "Elementwise/ReLU"
# ==============================================================================================================================
e["Solver"]["Neural Network"]["Hidden Layers"][3]["Type"] = "Layer/Convolution"
e["Solver"]["Neural Network"]["Hidden Layers"][3]["Image Height"]      = 16 # img_width
e["Solver"]["Neural Network"]["Hidden Layers"][3]["Image Width"]       = 32 # img_height
e["Solver"]["Neural Network"]["Hidden Layers"][3]["Kernel Height"]     = 9-2  # kernel_size_factor
e["Solver"]["Neural Network"]["Hidden Layers"][3]["Kernel Width"]      = 9-2  # kernel_size_factor
e["Solver"]["Neural Network"]["Hidden Layers"][3]["Vertical Stride"]   = 1
e["Solver"]["Neural Network"]["Hidden Layers"][3]["Horizontal Stride"] = 1
e["Solver"]["Neural Network"]["Hidden Layers"][3]["Padding Left"]      = 3
e["Solver"]["Neural Network"]["Hidden Layers"][3]["Padding Right"]     = 3
e["Solver"]["Neural Network"]["Hidden Layers"][3]["Padding Top"]       = 3
e["Solver"]["Neural Network"]["Hidden Layers"][3]["Padding Bottom"]    = 3
e["Solver"]["Neural Network"]["Hidden Layers"][3]["Output Channels"]   = 16*16*32 # autoencoder_factor*input_height*img_width
## POOLING LAYER
e["Solver"]["Neural Network"]["Hidden Layers"][4]["Type"] = "Layer/Pooling"
e["Solver"]["Neural Network"]["Hidden Layers"][4]["Function"]          = "Exclusive Average"
e["Solver"]["Neural Network"]["Hidden Layers"][4]["Image Height"]      = 16
e["Solver"]["Neural Network"]["Hidden Layers"][4]["Image Width"]       = 32
e["Solver"]["Neural Network"]["Hidden Layers"][4]["Kernel Height"]     = 2
e["Solver"]["Neural Network"]["Hidden Layers"][4]["Kernel Width"]      = 2
e["Solver"]["Neural Network"]["Hidden Layers"][4]["Vertical Stride"]   = 2
e["Solver"]["Neural Network"]["Hidden Layers"][4]["Horizontal Stride"] = 2
e["Solver"]["Neural Network"]["Hidden Layers"][4]["Padding Left"]      = 0
e["Solver"]["Neural Network"]["Hidden Layers"][4]["Padding Right"]     = 0
e["Solver"]["Neural Network"]["Hidden Layers"][4]["Padding Top"]       = 0
e["Solver"]["Neural Network"]["Hidden Layers"][4]["Padding Bottom"]    = 0
e["Solver"]["Neural Network"]["Hidden Layers"][4]["Output Channels"]   = 16*8*16 # output channels CNN

e["Solver"]["Neural Network"]["Hidden Layers"][5]["Type"] = "Layer/Activation"
e["Solver"]["Neural Network"]["Hidden Layers"][5]["Function"] = "Elementwise/ReLU"
# ==============================================================================================================================
e["Solver"]["Neural Network"]["Hidden Layers"][6]["Type"] = "Layer/Convolution"
e["Solver"]["Neural Network"]["Hidden Layers"][6]["Image Height"]      = 8 # img_width
e["Solver"]["Neural Network"]["Hidden Layers"][6]["Image Width"]       = 16 # img_height
e["Solver"]["Neural Network"]["Hidden Layers"][6]["Kernel Height"]     = 9-4  # kernel_size_factor
e["Solver"]["Neural Network"]["Hidden Layers"][6]["Kernel Width"]      = 9-4  # kernel_size_factor
e["Solver"]["Neural Network"]["Hidden Layers"][6]["Vertical Stride"]   = 1
e["Solver"]["Neural Network"]["Hidden Layers"][6]["Horizontal Stride"] = 1
e["Solver"]["Neural Network"]["Hidden Layers"][6]["Padding Left"]      = 2
e["Solver"]["Neural Network"]["Hidden Layers"][6]["Padding Right"]     = 2
e["Solver"]["Neural Network"]["Hidden Layers"][6]["Padding Top"]       = 2
e["Solver"]["Neural Network"]["Hidden Layers"][6]["Padding Bottom"]    = 2
e["Solver"]["Neural Network"]["Hidden Layers"][6]["Output Channels"]   = 64*8*16 # autoencoder_factor*input_height*img_width
## POOLING LAYER
e["Solver"]["Neural Network"]["Hidden Layers"][7]["Type"] = "Layer/Pooling"
e["Solver"]["Neural Network"]["Hidden Layers"][7]["Function"]          = "Exclusive Average"
e["Solver"]["Neural Network"]["Hidden Layers"][7]["Image Height"]      = 8
e["Solver"]["Neural Network"]["Hidden Layers"][7]["Image Width"]       = 16
e["Solver"]["Neural Network"]["Hidden Layers"][7]["Kernel Height"]     = 2
e["Solver"]["Neural Network"]["Hidden Layers"][7]["Kernel Width"]      = 2
e["Solver"]["Neural Network"]["Hidden Layers"][7]["Vertical Stride"]   = 2
e["Solver"]["Neural Network"]["Hidden Layers"][7]["Horizontal Stride"] = 2
e["Solver"]["Neural Network"]["Hidden Layers"][7]["Padding Left"]      = 0
e["Solver"]["Neural Network"]["Hidden Layers"][7]["Padding Right"]     = 0
e["Solver"]["Neural Network"]["Hidden Layers"][7]["Padding Top"]       = 0
e["Solver"]["Neural Network"]["Hidden Layers"][7]["Padding Bottom"]    = 0
e["Solver"]["Neural Network"]["Hidden Layers"][7]["Output Channels"]   = 64*4*8 # output channels CNN

e["Solver"]["Neural Network"]["Hidden Layers"][8]["Type"] = "Layer/Activation"
e["Solver"]["Neural Network"]["Hidden Layers"][8]["Function"] = "Elementwise/ReLU"
# ==============================================================================================================================
e["Solver"]["Neural Network"]["Hidden Layers"][9]["Type"] = "Layer/Convolution"
e["Solver"]["Neural Network"]["Hidden Layers"][9]["Image Height"]      = 4 # img_width
e["Solver"]["Neural Network"]["Hidden Layers"][9]["Image Width"]       = 8 # img_height
e["Solver"]["Neural Network"]["Hidden Layers"][9]["Kernel Height"]     = 3  # kernel_size_factor
e["Solver"]["Neural Network"]["Hidden Layers"][9]["Kernel Width"]      = 3  # kernel_size_factor
e["Solver"]["Neural Network"]["Hidden Layers"][9]["Vertical Stride"]   = 1
e["Solver"]["Neural Network"]["Hidden Layers"][9]["Horizontal Stride"] = 1
e["Solver"]["Neural Network"]["Hidden Layers"][9]["Padding Left"]      = 1
e["Solver"]["Neural Network"]["Hidden Layers"][9]["Padding Right"]     = 1
e["Solver"]["Neural Network"]["Hidden Layers"][9]["Padding Top"]       = 1
e["Solver"]["Neural Network"]["Hidden Layers"][9]["Padding Bottom"]    = 1
e["Solver"]["Neural Network"]["Hidden Layers"][9]["Output Channels"]   = 256*4*8 # autoencoder_factor*input_height*img_width
## POOLING LAYER
e["Solver"]["Neural Network"]["Hidden Layers"][10]["Type"] = "Layer/Pooling"
e["Solver"]["Neural Network"]["Hidden Layers"][10]["Function"]          = "Exclusive Average"
e["Solver"]["Neural Network"]["Hidden Layers"][10]["Image Height"]      = 4
e["Solver"]["Neural Network"]["Hidden Layers"][10]["Image Width"]       = 8
e["Solver"]["Neural Network"]["Hidden Layers"][10]["Kernel Height"]     = 2
e["Solver"]["Neural Network"]["Hidden Layers"][10]["Kernel Width"]      = 2
e["Solver"]["Neural Network"]["Hidden Layers"][10]["Vertical Stride"]   = 2
e["Solver"]["Neural Network"]["Hidden Layers"][10]["Horizontal Stride"] = 2
e["Solver"]["Neural Network"]["Hidden Layers"][10]["Padding Left"]      = 0
e["Solver"]["Neural Network"]["Hidden Layers"][10]["Padding Right"]     = 0
e["Solver"]["Neural Network"]["Hidden Layers"][10]["Padding Top"]       = 0
e["Solver"]["Neural Network"]["Hidden Layers"][10]["Padding Bottom"]    = 0
e["Solver"]["Neural Network"]["Hidden Layers"][10]["Output Channels"]   = 256*2*4 # output channels CNN

e["Solver"]["Neural Network"]["Hidden Layers"][11]["Type"] = "Layer/Activation"
e["Solver"]["Neural Network"]["Hidden Layers"][11]["Function"] = "Elementwise/ReLU"
#
# ==============================================================================================================================
e["Solver"]["Neural Network"]["Hidden Layers"][12]["Type"] = "Layer/Convolution"
e["Solver"]["Neural Network"]["Hidden Layers"][12]["Image Height"]      = 2 # img_height
e["Solver"]["Neural Network"]["Hidden Layers"][12]["Image Width"]       = 4 # img_width
e["Solver"]["Neural Network"]["Hidden Layers"][12]["Kernel Height"]     = 1  # kernel_size_factor
e["Solver"]["Neural Network"]["Hidden Layers"][12]["Kernel Width"]      = 1  # kernel_size_factor
e["Solver"]["Neural Network"]["Hidden Layers"][12]["Vertical Stride"]   = 1
e["Solver"]["Neural Network"]["Hidden Layers"][12]["Horizontal Stride"] = 1
e["Solver"]["Neural Network"]["Hidden Layers"][12]["Padding Left"]      = 0
e["Solver"]["Neural Network"]["Hidden Layers"][12]["Padding Right"]     = 0
e["Solver"]["Neural Network"]["Hidden Layers"][12]["Padding Top"]       = 0
e["Solver"]["Neural Network"]["Hidden Layers"][12]["Padding Bottom"]    = 0
e["Solver"]["Neural Network"]["Hidden Layers"][12]["Output Channels"]   = 1*2*4 # autoencoder_factor*input_height*img_width
## POOLING LAYER
e["Solver"]["Neural Network"]["Hidden Layers"][13]["Type"] = "Layer/Pooling"
e["Solver"]["Neural Network"]["Hidden Layers"][13]["Function"]          = "Exclusive Average"
e["Solver"]["Neural Network"]["Hidden Layers"][13]["Image Height"]      = 2
e["Solver"]["Neural Network"]["Hidden Layers"][13]["Image Width"]       = 4
e["Solver"]["Neural Network"]["Hidden Layers"][13]["Kernel Height"]     = 2
e["Solver"]["Neural Network"]["Hidden Layers"][13]["Kernel Width"]      = 2
e["Solver"]["Neural Network"]["Hidden Layers"][13]["Vertical Stride"]   = 2
e["Solver"]["Neural Network"]["Hidden Layers"][13]["Horizontal Stride"] = 2
e["Solver"]["Neural Network"]["Hidden Layers"][13]["Padding Left"]      = 0
e["Solver"]["Neural Network"]["Hidden Layers"][13]["Padding Right"]     = 0
e["Solver"]["Neural Network"]["Hidden Layers"][13]["Padding Top"]       = 0
e["Solver"]["Neural Network"]["Hidden Layers"][13]["Padding Bottom"]    = 0
e["Solver"]["Neural Network"]["Hidden Layers"][13]["Output Channels"]   = 1*1*2 # output channels CNN

e["Solver"]["Neural Network"]["Hidden Layers"][14]["Type"] = "Layer/Activation"
e["Solver"]["Neural Network"]["Hidden Layers"][14]["Function"] = "Elementwise/Tanh"
###
### LATENT SPACE ==========================================================================
###

# ###
# ### DECODER ==========================================================================
# ###
e["Solver"]["Neural Network"]["Hidden Layers"][15]["Type"] = "Layer/Deconvolution"
e["Solver"]["Neural Network"]["Hidden Layers"][15]["Image Height"]      = 2
e["Solver"]["Neural Network"]["Hidden Layers"][15]["Image Width"]       = 4
e["Solver"]["Neural Network"]["Hidden Layers"][15]["Kernel Height"]     = 2
e["Solver"]["Neural Network"]["Hidden Layers"][15]["Kernel Width"]      = 2
e["Solver"]["Neural Network"]["Hidden Layers"][15]["Vertical Stride"]   = 2
e["Solver"]["Neural Network"]["Hidden Layers"][15]["Horizontal Stride"] = 2
e["Solver"]["Neural Network"]["Hidden Layers"][15]["Padding Left"]      = 0
e["Solver"]["Neural Network"]["Hidden Layers"][15]["Padding Right"]     = 0
e["Solver"]["Neural Network"]["Hidden Layers"][15]["Padding Top"]       = 0
e["Solver"]["Neural Network"]["Hidden Layers"][15]["Padding Bottom"]    = 0
e["Solver"]["Neural Network"]["Hidden Layers"][15]["Output Channels"]   = 1*2*4
# ### ====================================================================================
e["Solver"]["Neural Network"]["Hidden Layers"][16]["Type"] = "Layer/Deconvolution"
e["Solver"]["Neural Network"]["Hidden Layers"][16]["Image Height"]      = 2
e["Solver"]["Neural Network"]["Hidden Layers"][16]["Image Width"]       = 4
e["Solver"]["Neural Network"]["Hidden Layers"][16]["Kernel Height"]     = 1
e["Solver"]["Neural Network"]["Hidden Layers"][16]["Kernel Width"]      = 1
e["Solver"]["Neural Network"]["Hidden Layers"][16]["Vertical Stride"]   = 1
e["Solver"]["Neural Network"]["Hidden Layers"][16]["Horizontal Stride"] = 1
e["Solver"]["Neural Network"]["Hidden Layers"][16]["Padding Left"]      = 0
e["Solver"]["Neural Network"]["Hidden Layers"][16]["Padding Right"]     = 0
e["Solver"]["Neural Network"]["Hidden Layers"][16]["Padding Top"]       = 0
e["Solver"]["Neural Network"]["Hidden Layers"][16]["Padding Bottom"]    = 0
e["Solver"]["Neural Network"]["Hidden Layers"][16]["Output Channels"]   = 256*2*4

e["Solver"]["Neural Network"]["Hidden Layers"][17]["Type"] = "Layer/Activation"
e["Solver"]["Neural Network"]["Hidden Layers"][17]["Function"] = "Elementwise/Tanh"
# ### ====================================================================================
e["Solver"]["Neural Network"]["Hidden Layers"][18]["Type"] = "Layer/Deconvolution"
e["Solver"]["Neural Network"]["Hidden Layers"][18]["Image Height"]      = 4
e["Solver"]["Neural Network"]["Hidden Layers"][18]["Image Width"]       = 8
e["Solver"]["Neural Network"]["Hidden Layers"][18]["Kernel Height"]     = 2
e["Solver"]["Neural Network"]["Hidden Layers"][18]["Kernel Width"]      = 2
e["Solver"]["Neural Network"]["Hidden Layers"][18]["Vertical Stride"]   = 2
e["Solver"]["Neural Network"]["Hidden Layers"][18]["Horizontal Stride"] = 2
e["Solver"]["Neural Network"]["Hidden Layers"][18]["Padding Left"]      = 0
e["Solver"]["Neural Network"]["Hidden Layers"][18]["Padding Right"]     = 0
e["Solver"]["Neural Network"]["Hidden Layers"][18]["Padding Top"]       = 0
e["Solver"]["Neural Network"]["Hidden Layers"][18]["Padding Bottom"]    = 0
e["Solver"]["Neural Network"]["Hidden Layers"][18]["Output Channels"]   = 256*4*8
# ### ====================================================================================
e["Solver"]["Neural Network"]["Hidden Layers"][19]["Type"] = "Layer/Deconvolution"
e["Solver"]["Neural Network"]["Hidden Layers"][19]["Image Height"]      = 4
e["Solver"]["Neural Network"]["Hidden Layers"][19]["Image Width"]       = 8
e["Solver"]["Neural Network"]["Hidden Layers"][19]["Kernel Height"]     = 3
e["Solver"]["Neural Network"]["Hidden Layers"][19]["Kernel Width"]      = 3
e["Solver"]["Neural Network"]["Hidden Layers"][19]["Vertical Stride"]   = 1
e["Solver"]["Neural Network"]["Hidden Layers"][19]["Horizontal Stride"] = 1
e["Solver"]["Neural Network"]["Hidden Layers"][19]["Padding Left"]      = 1
e["Solver"]["Neural Network"]["Hidden Layers"][19]["Padding Right"]     = 1
e["Solver"]["Neural Network"]["Hidden Layers"][19]["Padding Top"]       = 1
e["Solver"]["Neural Network"]["Hidden Layers"][19]["Padding Bottom"]    = 1
e["Solver"]["Neural Network"]["Hidden Layers"][19]["Output Channels"]   = 64*4*8

e["Solver"]["Neural Network"]["Hidden Layers"][20]["Type"] = "Layer/Activation"
e["Solver"]["Neural Network"]["Hidden Layers"][20]["Function"] = "Elementwise/Tanh"

# # ### ====================================================================================
e["Solver"]["Neural Network"]["Hidden Layers"][21]["Type"] = "Layer/Deconvolution"
e["Solver"]["Neural Network"]["Hidden Layers"][21]["Image Height"]      = 8
e["Solver"]["Neural Network"]["Hidden Layers"][21]["Image Width"]       = 16
e["Solver"]["Neural Network"]["Hidden Layers"][21]["Kernel Height"]     = 2
e["Solver"]["Neural Network"]["Hidden Layers"][21]["Kernel Width"]      = 2
e["Solver"]["Neural Network"]["Hidden Layers"][21]["Vertical Stride"]   = 2
e["Solver"]["Neural Network"]["Hidden Layers"][21]["Horizontal Stride"] = 2
e["Solver"]["Neural Network"]["Hidden Layers"][21]["Padding Left"]      = 0
e["Solver"]["Neural Network"]["Hidden Layers"][21]["Padding Right"]     = 0
e["Solver"]["Neural Network"]["Hidden Layers"][21]["Padding Top"]       = 0
e["Solver"]["Neural Network"]["Hidden Layers"][21]["Padding Bottom"]    = 0
e["Solver"]["Neural Network"]["Hidden Layers"][21]["Output Channels"]   = 64*8*16
# ### ====================================================================================
e["Solver"]["Neural Network"]["Hidden Layers"][22]["Type"] = "Layer/Deconvolution"
e["Solver"]["Neural Network"]["Hidden Layers"][22]["Image Height"]      = 8
e["Solver"]["Neural Network"]["Hidden Layers"][22]["Image Width"]       = 16
e["Solver"]["Neural Network"]["Hidden Layers"][22]["Kernel Height"]     = 9-4
e["Solver"]["Neural Network"]["Hidden Layers"][22]["Kernel Width"]      = 9-4
e["Solver"]["Neural Network"]["Hidden Layers"][22]["Vertical Stride"]   = 1
e["Solver"]["Neural Network"]["Hidden Layers"][22]["Horizontal Stride"] = 1
e["Solver"]["Neural Network"]["Hidden Layers"][22]["Padding Left"]      = 2
e["Solver"]["Neural Network"]["Hidden Layers"][22]["Padding Right"]     = 2
e["Solver"]["Neural Network"]["Hidden Layers"][22]["Padding Top"]       = 2
e["Solver"]["Neural Network"]["Hidden Layers"][22]["Padding Bottom"]    = 2
e["Solver"]["Neural Network"]["Hidden Layers"][22]["Output Channels"]   = 16*8*16

e["Solver"]["Neural Network"]["Hidden Layers"][23]["Type"] = "Layer/Activation"
e["Solver"]["Neural Network"]["Hidden Layers"][23]["Function"] = "Elementwise/Tanh"
# ### ====================================================================================
e["Solver"]["Neural Network"]["Hidden Layers"][24]["Type"] = "Layer/Deconvolution"
e["Solver"]["Neural Network"]["Hidden Layers"][24]["Image Height"]      = 16
e["Solver"]["Neural Network"]["Hidden Layers"][24]["Image Width"]       = 32
e["Solver"]["Neural Network"]["Hidden Layers"][24]["Kernel Height"]     = 2
e["Solver"]["Neural Network"]["Hidden Layers"][24]["Kernel Width"]      = 2
e["Solver"]["Neural Network"]["Hidden Layers"][24]["Vertical Stride"]   = 2
e["Solver"]["Neural Network"]["Hidden Layers"][24]["Horizontal Stride"] = 2
e["Solver"]["Neural Network"]["Hidden Layers"][24]["Padding Left"]      = 0
e["Solver"]["Neural Network"]["Hidden Layers"][24]["Padding Right"]     = 0
e["Solver"]["Neural Network"]["Hidden Layers"][24]["Padding Top"]       = 0
e["Solver"]["Neural Network"]["Hidden Layers"][24]["Padding Bottom"]    = 0
e["Solver"]["Neural Network"]["Hidden Layers"][24]["Output Channels"]   = 16*16*32
# ### ====================================================================================
e["Solver"]["Neural Network"]["Hidden Layers"][25]["Type"] = "Layer/Deconvolution"
e["Solver"]["Neural Network"]["Hidden Layers"][25]["Image Height"]      = 16
e["Solver"]["Neural Network"]["Hidden Layers"][25]["Image Width"]       = 32
e["Solver"]["Neural Network"]["Hidden Layers"][25]["Kernel Height"]     = 9-2
e["Solver"]["Neural Network"]["Hidden Layers"][25]["Kernel Width"]      = 9-2
e["Solver"]["Neural Network"]["Hidden Layers"][25]["Vertical Stride"]   = 1
e["Solver"]["Neural Network"]["Hidden Layers"][25]["Horizontal Stride"] = 1
e["Solver"]["Neural Network"]["Hidden Layers"][25]["Padding Left"]      = 3
e["Solver"]["Neural Network"]["Hidden Layers"][25]["Padding Right"]     = 3
e["Solver"]["Neural Network"]["Hidden Layers"][25]["Padding Top"]       = 3
e["Solver"]["Neural Network"]["Hidden Layers"][25]["Padding Bottom"]    = 3
e["Solver"]["Neural Network"]["Hidden Layers"][25]["Output Channels"]   = 4*16*32

e["Solver"]["Neural Network"]["Hidden Layers"][26]["Type"] = "Layer/Activation"
e["Solver"]["Neural Network"]["Hidden Layers"][26]["Function"] = "Elementwise/Tanh"
# ### ====================================================================================
e["Solver"]["Neural Network"]["Hidden Layers"][27]["Type"] = "Layer/Deconvolution"
e["Solver"]["Neural Network"]["Hidden Layers"][27]["Image Height"]      = 32
e["Solver"]["Neural Network"]["Hidden Layers"][27]["Image Width"]       = 64
e["Solver"]["Neural Network"]["Hidden Layers"][27]["Kernel Height"]     = 2
e["Solver"]["Neural Network"]["Hidden Layers"][27]["Kernel Width"]      = 2
e["Solver"]["Neural Network"]["Hidden Layers"][27]["Vertical Stride"]   = 2
e["Solver"]["Neural Network"]["Hidden Layers"][27]["Horizontal Stride"] = 2
e["Solver"]["Neural Network"]["Hidden Layers"][27]["Padding Left"]      = 0
e["Solver"]["Neural Network"]["Hidden Layers"][27]["Padding Right"]     = 0
e["Solver"]["Neural Network"]["Hidden Layers"][27]["Padding Top"]       = 0
e["Solver"]["Neural Network"]["Hidden Layers"][27]["Padding Bottom"]    = 0
e["Solver"]["Neural Network"]["Hidden Layers"][27]["Output Channels"]   = 4*32*64
# ### ====================================================================================
e["Solver"]["Neural Network"]["Hidden Layers"][28]["Type"] = "Layer/Deconvolution"
e["Solver"]["Neural Network"]["Hidden Layers"][28]["Image Height"]      = 32
e["Solver"]["Neural Network"]["Hidden Layers"][28]["Image Width"]       = 64
e["Solver"]["Neural Network"]["Hidden Layers"][28]["Kernel Height"]     = 9
e["Solver"]["Neural Network"]["Hidden Layers"][28]["Kernel Width"]      = 9
e["Solver"]["Neural Network"]["Hidden Layers"][28]["Vertical Stride"]   = 1
e["Solver"]["Neural Network"]["Hidden Layers"][28]["Horizontal Stride"] = 1
e["Solver"]["Neural Network"]["Hidden Layers"][28]["Padding Left"]      = 4
e["Solver"]["Neural Network"]["Hidden Layers"][28]["Padding Right"]     = 4
e["Solver"]["Neural Network"]["Hidden Layers"][28]["Padding Top"]       = 4
e["Solver"]["Neural Network"]["Hidden Layers"][28]["Padding Bottom"]    = 4
e["Solver"]["Neural Network"]["Hidden Layers"][28]["Output Channels"]   = 1*32*64



### Configuring output

e["Console Output"]["Verbosity"] = "Detailed"
e["File Output"]["Enabled"] = True
e["Random Seed"] = 0xC0FFEE
k["Conduit"]["Type"] = "Concurrent"
### Printing Configuration

print("[Korali] Running MNIST solver.")
print("[Korali] Algorithm: " + str(e["Solver"]["Neural Network"]["Optimizer"]))
print("[Korali] Database Size: " + str(len(trainingImageVector)))
print("[Korali] Batch Size: " + str(trainingBatchSize))
print("[Korali] Epochs: " + str(epochs))
print("[Korali] Initial Learning Rate: " + str(learningRate))
print("[Korali] Decay: " + str(decay))
### Running SGD loop
e["Solver"]["Mode"] = "Training"
e["Solver"]["Termination Criteria"]["Max Generations"] = 1
k.run(e)

e["Solver"]["Mode"] = "Testing"
e["Problem"]["Input"]["Data"] = testingImageVector
k.run(e)
testInferredSet = [ x[0] for x in e["Solver"]["Evaluation"] ]
mse = np.mean((np.array(testInferredSet) - np.array(testingImageVector))**2)
print("MSE on test set: {}".format(mse))
