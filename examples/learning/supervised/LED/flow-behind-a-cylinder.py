#!/usr/bin/env python3
import os
import sys
import pickle
import numpy as np
import matplotlib.pyplot as plt
import time
import korali
import random
os.chdir("/home/pollakg/polybox/CSE/master/6th_term/master_thesis/korali/examples/learning/supervised/LED/")

k = korali.Engine()
e = korali.Experiment()

### Hyperparameters
 
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

trainingImageVector = [ [ x ] for x in trainingImages ]
testingImageVector = [ [ x ] for x in testingImages ]

### Shuffling training data set for stochastic gradient descent training

random.shuffle(trainingImageVector)

### LAYER DEFINITIONS ==================================================================
### CNNS =====================

def getSamePadding(stride, image_size, filter_size):
    # Input image (W_i,W_i)
    # Output image (W_o,W_o) with W_o = (W_i - F + 2P)/S + 1
    # W_i == W_o -> P = ((S-1)W + F - S)/2
    S = stride
    W = image_size  # width or height
    F = filter_size
    half_pad = int((S - 1) * W - S + F)
    if half_pad % 2 == 1:
        raise ValueError(
            "(S-1) * W  - S + F has to be divisible by two ({:}-1)*{:} - {:} + {:} = {:}"
            .format(S, W, S, F, half_pad))
    else:
        pad = int(half_pad / 2)
    if (pad > image_size / 2):
        raise ValueError(
            "Very large padding P={:}, compared to input width {:}. Reduce the strides."
            .format(pad, image_size))
    return pad


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

### Configuring general problem settings

e["Problem"]["Type"] = "Supervised Learning"
e["Problem"]["Max Timesteps"] = 1
e["Problem"]["Training Batch Size"] = trainingBatchSize
e["Problem"]["Inference Batch Size"] = testingBatchSize
e["Problem"]["Input"]["Size"] = len(trainingImages[0])
e["Problem"]["Solution"]["Size"] = len(trainingImages[0])

### Using a neural network solver (deep learning) for inference

e["Solver"]["Termination Criteria"]["Max Generations"] = 1
e["Solver"]["Type"] = "Learner/DeepSupervisor"
e["Solver"]["Loss Function"] = "Mean Squared Error"
e["Solver"]["Steps Per Generation"] = 1
e["Solver"]["Neural Network"]["Engine"] = "OneDNN"
e["Solver"]["Neural Network"]["Optimizer"] = "Adam"

### ENCODER ==========================================================================
# layers*3 = CNN, Pooling, Activation
layer_idx = 0
for l in range(0, layers):
    ### Defining the shape of the neural network [autoencoder version of LeNet-1 - http://yann.lecun.com/exdb/publis/pdf/lecun-90c.pdf (fig. 2)]
    ## CONVOLUTIONAL LAYER
    ## with tanh activation function [1x32x64] -> [6x24x24]
    e["Solver"]["Neural Network"]["Hidden Layers"][layer_idx]["Type"] = "Layer/Convolution"
    e["Solver"]["Neural Network"]["Hidden Layers"][layer_idx]["Image Height"]      = inputDim_CNN[l]["height"]
    e["Solver"]["Neural Network"]["Hidden Layers"][layer_idx]["Image Width"]       = inputDim_CNN[l]["width"]
    e["Solver"]["Neural Network"]["Hidden Layers"][layer_idx]["Kernel Height"]     = kernelDim_CNN[l]["height"]
    e["Solver"]["Neural Network"]["Hidden Layers"][layer_idx]["Kernel Width"]      = kernelDim_CNN[l]["width"]
    e["Solver"]["Neural Network"]["Hidden Layers"][layer_idx]["Vertical Stride"]   = kernelStrides_CNN[l]["height"]
    e["Solver"]["Neural Network"]["Hidden Layers"][layer_idx]["Horizontal Stride"] = kernelStrides_CNN[l]["width"]
    e["Solver"]["Neural Network"]["Hidden Layers"][layer_idx]["Padding Left"]      = paddings_CNN[l]["left"]
    e["Solver"]["Neural Network"]["Hidden Layers"][layer_idx]["Padding Right"]     = paddings_CNN[l]["right"]
    e["Solver"]["Neural Network"]["Hidden Layers"][layer_idx]["Padding Top"]       = paddings_CNN[l]["top"]
    e["Solver"]["Neural Network"]["Hidden Layers"][layer_idx]["Padding Bottom"]    = paddings_CNN[l]["bottom"]
    e["Solver"]["Neural Network"]["Hidden Layers"][layer_idx]["Output Channels"]   = outputChannels_CNN[l]*outputDim_CNN[l]["width"]*outputDim_CNN[l]["height"]
    ## POOLING LAYER
    e["Solver"]["Neural Network"]["Hidden Layers"][layer_idx+1]["Type"] = "Layer/Pooling"
    e["Solver"]["Neural Network"]["Hidden Layers"][layer_idx+1]["Function"]          = "Exclusive Average"
    e["Solver"]["Neural Network"]["Hidden Layers"][layer_idx+1]["Image Height"]      = inputDim_pooling[l]["height"]
    e["Solver"]["Neural Network"]["Hidden Layers"][layer_idx+1]["Image Width"]       = inputDim_pooling[l]["width"]
    e["Solver"]["Neural Network"]["Hidden Layers"][layer_idx+1]["Kernel Height"]     = kernelDim_pooling[l]["height"]
    e["Solver"]["Neural Network"]["Hidden Layers"][layer_idx+1]["Kernel Width"]      = kernelDim_pooling[l]["width"]
    e["Solver"]["Neural Network"]["Hidden Layers"][layer_idx+1]["Vertical Stride"]   = kernelStrides_pooling[l]["height"]
    e["Solver"]["Neural Network"]["Hidden Layers"][layer_idx+1]["Horizontal Stride"] = kernelStrides_pooling[l]["width"]
    e["Solver"]["Neural Network"]["Hidden Layers"][layer_idx+1]["Padding Left"]      = paddings_pooling[l]["left"]
    e["Solver"]["Neural Network"]["Hidden Layers"][layer_idx+1]["Padding Right"]     = paddings_pooling[l]["right"]
    e["Solver"]["Neural Network"]["Hidden Layers"][layer_idx+1]["Padding Top"]       = paddings_pooling[l]["top"]
    e["Solver"]["Neural Network"]["Hidden Layers"][layer_idx+1]["Padding Bottom"]    = paddings_pooling[l]["bottom"]
    e["Solver"]["Neural Network"]["Hidden Layers"][layer_idx+1]["Output Channels"]   = outputChannels_pooling[l]*outputDim_pooling[l]["width"]*outputDim_pooling[l]["height"]
    # Activations                                  layer_idx
    e["Solver"]["Neural Network"]["Hidden Layers"][layer_idx+2]["Type"] = "Layer/Activation"
    e["Solver"]["Neural Network"]["Hidden Layers"][layer_idx+2]["Function"] = activations[l]
    layer_idx+=3
###
### LATENT SPACE ==========================================================================
###

###
### DECODER ==========================================================================
###
for l, l_reversed in enumerate(range(layers-1, -1, -1)):
    e["Solver"]["Neural Network"]["Hidden Layers"][layer_idx+l]["Type"] = "Layer/Convolution"
    e["Solver"]["Neural Network"]["Hidden Layers"][layer_idx+l]["Image Height"]      = inputDim_CNN[l_reversed]["height"]
    e["Solver"]["Neural Network"]["Hidden Layers"][layer_idx+l]["Image Width"]       = inputDim_CNN[l_reversed]["width"]
    e["Solver"]["Neural Network"]["Hidden Layers"][layer_idx+l]["Kernel Height"]     = kernelDim_CNN[l_reversed]["height"]
    e["Solver"]["Neural Network"]["Hidden Layers"][layer_idx+l]["Kernel Width"]      = kernelDim_CNN[l_reversed]["width"]
    e["Solver"]["Neural Network"]["Hidden Layers"][layer_idx+l]["Vertical Stride"]   = kernelStrides_CNN[l_reversed]["height"]
    e["Solver"]["Neural Network"]["Hidden Layers"][layer_idx+l]["Horizontal Stride"] = kernelStrides_CNN[l_reversed]["width"]
    e["Solver"]["Neural Network"]["Hidden Layers"][layer_idx+l]["Padding Left"]      = paddings_CNN[l_reversed]["left"]
    e["Solver"]["Neural Network"]["Hidden Layers"][layer_idx+l]["Padding Right"]     = paddings_CNN[l_reversed]["right"]
    e["Solver"]["Neural Network"]["Hidden Layers"][layer_idx+l]["Padding Top"]       = paddings_CNN[l_reversed]["top"]
    e["Solver"]["Neural Network"]["Hidden Layers"][layer_idx+l]["Padding Bottom"]    = paddings_CNN[l_reversed]["bottom"]
    e["Solver"]["Neural Network"]["Hidden Layers"][layer_idx+l]["Output Channels"]   = outputChannels_CNN[l_reversed-1]*outputDim_CNN[l_reversed-1]["width"]*outputDim_CNN[l_reversed-1]["height"]
    ## POOLING LAYER
    e["Solver"]["Neural Network"]["Hidden Layers"][layer_idx+1]["Type"] = "Layer/Pooling"
    e["Solver"]["Neural Network"]["Hidden Layers"][layer_idx+1]["Function"]          = "Exclusive Average"
    e["Solver"]["Neural Network"]["Hidden Layers"][layer_idx+1]["Image Height"]      = inputDim_pooling[l_reversed]["height"]
    e["Solver"]["Neural Network"]["Hidden Layers"][layer_idx+1]["Image Width"]       = inputDim_pooling[l_reversed]["width"]
    e["Solver"]["Neural Network"]["Hidden Layers"][layer_idx+1]["Kernel Height"]     = kernelDim_pooling[l_reversed]["height"]
    e["Solver"]["Neural Network"]["Hidden Layers"][layer_idx+1]["Kernel Width"]      = kernelDim_pooling[l_reversed]["width"]
    e["Solver"]["Neural Network"]["Hidden Layers"][layer_idx+1]["Vertical Stride"]   = kernelStrides_pooling[l_reversed]["height"]
    e["Solver"]["Neural Network"]["Hidden Layers"][layer_idx+1]["Horizontal Stride"] = kernelStrides_pooling[l_reversed]["width"]
    e["Solver"]["Neural Network"]["Hidden Layers"][layer_idx+1]["Padding Left"]      = paddings_pooling[l_reversed]["left"]
    e["Solver"]["Neural Network"]["Hidden Layers"][layer_idx+1]["Padding Right"]     = paddings_pooling[l_reversed]["right"]
    e["Solver"]["Neural Network"]["Hidden Layers"][layer_idx+1]["Padding Top"]       = paddings_pooling[l_reversed]["top"]
    e["Solver"]["Neural Network"]["Hidden Layers"][layer_idx+1]["Padding Bottom"]    = paddings_pooling[l_reversed]["bottom"]
    e["Solver"]["Neural Network"]["Hidden Layers"][layer_idx+1]["Output Channels"]   = outputChannels_pooling[l_reversed-1]*outputDim_CNN[l_reversed]["width"]*outputDim_CNN[l_reversed]["height"]
    # Activations
    if layers > 2:
        e["Solver"]["Neural Network"]["Hidden Layers"][layer_idx+l+2]["Type"] = "Layer/Activation"
        e["Solver"]["Neural Network"]["Hidden Layers"][layer_idx+l+2]["Function"] = activations[l_reversed-1]
    layer_idx+=3
### Configuring output

e["Console Output"]["Verbosity"] = "Detailed"
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
 for step in range(stepsPerEpoch):
 
  # Creating minibatch
  miniBatchInput = trainingImageVector[step * trainingBatchSize : (step+1) * trainingBatchSize] # N x T x C
  miniBatchSolution = [ x[0] for x in miniBatchInput ] # N x C
  
  # Passing minibatch to Korali
  e["Problem"]["Input"]["Data"] = miniBatchInput
  e["Problem"]["Solution"]["Data"] = miniBatchSolution
 
  # Reconfiguring solver
  e["Solver"]["Learning Rate"] = learningRate
  e["Solver"]["Termination Criteria"]["Max Generations"] = e["Solver"]["Termination Criteria"]["Max Generations"] + 1
  
  # Running step
  k.run(e)
  
 # Printing Information
 print("[Korali] --------------------------------------------------")
 print("[Korali] Epoch: " + str(epoch) + "/" + str(epochs))
 print("[Korali] Learning Rate: " + str(learningRate))
 print('[Korali] Current Training Loss: ' + str(e["Solver"]["Current Loss"])) 
    
 # Evaluating testing set
 testingInferredVector = testInferredSet = e.getEvaluation(testingImageVector)
 
 # Getting MSE loss for testing set
 squaredMeanError = 0.0
 for i, res in enumerate(testingInferredVector):
  sol = testingImageVector[i]
  for j, s in enumerate(sol):
   diff = res[j] - s
   squaredMeanError += diff * diff 
 squaredMeanError = squaredMeanError / (float(testingBatchSize) * 2.0)
 print('[Korali] Current Testing Loss:  ' + str(squaredMeanError))
 
 # Adjusting learning rate via decay
 learningRate = learningRate * (1.0 / (1.0 + decay * (epoch+1)));
 
