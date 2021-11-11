#!/usr/bin/env python3
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.patches as mpatch
from random import randrange
import time
import korali
import argparse
sys.argv=['']
k = korali.Engine()

parser = argparse.ArgumentParser()
parser.add_argument(
    '--optimizer',
    help='Optimizer to use for NN parameter updates',
    default='Adam',
    required=False)
parser.add_argument(
    '--maxGenerations',
     help='Maximum Number of generations to run',
     default=10,
     required=False)   
parser.add_argument(
    '--rnnType',
    help='Type of the RNN (GRU or LSTM)',
    default='GRU',
    required=False)
parser.add_argument(
    '--learningRate',
    help='Learning rate for the selected optimizer',
    default=0.001,
    required=False)
parser.add_argument(
    '--trainingBatchSize',
    help='Batch size to use for training data',
    default=500,
    required=False)
parser.add_argument(
    '--testBatchSize',
    help='Batch size to use for test data',
    default=100,
    required=False)
parser.add_argument(
    '--testMSEThreshold',
    help='Threshold for the testing MSE, under which the run will report an error',
    default=0.20,
    required=False)
parser.add_argument(
    '--plot',
    help='Indicates whether to plot results after testing',
    default=False,
    required=False)
args = parser.parse_args()

print("Running RNN solver with arguments:")
print(args)

# Setting random seed for reproducibility
np.random.seed(0xC0FFEE)

# Input parameters
tf = 2.0 # Total Time
dt = 0.4 # Time Differential
s = 1.0  # Parameter for peak separation
w = np.pi # Parameter for wave speed
a = 1.0 # Scaling

# Transformation Function
def y(x, t): return np.sin(x * s +  w * t)  

X = np.random.uniform(0, np.pi*2, args.trainingBatchSize)
T = np.arange(0, tf, dt)

# Providing inputs batches with varying timesequence lengths
trainingInputSetX = [ ]
trainingInputSetT = [ ]
for j, x in enumerate(X):
 trainingInputSetX.append([ ])
 trainingInputSetT.append([ ])
 for t in range(randrange(len(T)) + 1):
  trainingInputSetX[j].append([x])
  trainingInputSetT[j].append([T[t]])

# Giving the solution for the last time step of each batch sequence
trainingSolutionSet = [ ]
for j, x in enumerate(X):
 t = trainingInputSetT[j][-1][0]
 trainingSolutionSet.append([ y(x, t) ]) 

### Defining a learning problem to infer values of sin(x,t)

e = korali.Experiment()
e["Problem"]["Type"] = "Supervised Learning"
e["Problem"]["Max Timesteps"] = len(T)
e["Problem"]["Training Batch Size"] = args.trainingBatchSize
e["Problem"]["Inference Batch Size"] = args.testBatchSize
e["Problem"]["Input"]["Data"] = trainingInputSetX
e["Problem"]["Input"]["Size"] = 1
e["Problem"]["Solution"]["Data"] = trainingSolutionSet
e["Problem"]["Solution"]["Size"] = 1
 
### Using a neural network solver (deep learning) for inference

e["Solver"]["Type"] = "Learner/DeepSupervisor"
e["Solver"]["Loss Function"] = "Mean Squared Error"
e["Solver"]["Steps Per Generation"] = 20
e["Solver"]["Learning Rate"] = float(args.learningRate)

### Defining the shape of the neural network

e["Solver"]["Neural Network"]["Engine"] = "OneDNN"
e["Solver"]["Neural Network"]["Optimizer"] = args.optimizer

e["Solver"]["Neural Network"]["Hidden Layers"][0]["Type"] = "Layer/Recurrent/" + args.rnnType
e["Solver"]["Neural Network"]["Hidden Layers"][0]["Depth"] = 1
e["Solver"]["Neural Network"]["Hidden Layers"][0]["Output Channels"] = 32

### Configuring output 

e["Console Output"]["Frequency"] = 1
e["File Output"]["Enabled"] = False

### Training the neural network

e["Solver"]["Termination Criteria"]["Max Generations"] = int(args.maxGenerations)

### If this is test mode, run only a couple generations
if len(sys.argv) == 2:
 if sys.argv[1] == '--test':
  e["Solver"]["Termination Criteria"]["Max Generations"] = 2
  
e["Random Seed"] = 0xC0FFEE
k.run(e)

### Obtaining inferred results from the NN and comparing them to the actual solution

X = np.random.uniform(0, np.pi*2, args.testBatchSize)

# Providing inputs batches with varying timesequence lengths
testInputSetX = [ ]
testInputSetT = [ ]
for j, x in enumerate(X):
 testInputSetX.append([ ])
 testInputSetT.append([ ])
 for t in range(randrange(len(T)) + 1):
  testInputSetX[j].append([x])
  testInputSetT[j].append([T[t]])

# Giving the solution for the last time step of each batch sequence
testSolutionSet = [ ]
for j, x in enumerate(X):
 t = testInputSetT[j][-1][0]
 testSolutionSet.append([ y(x, t) ]) 
testInferredSet = e.getEvaluation(testInputSetX) 

### Calc MSE on test set

mse = np.mean((np.array(testInferredSet) - np.array(testSolutionSet))**2)
print("MSE on test set: {}".format(mse))

if (mse > args.testMSEThreshold):
 print("Fail: MSE does not satisfy threshold: " + str(args.testMSEThreshold))
 exit(-1)
 
### Plotting inferred result
if args.plot:
 cmap = cm.get_cmap(name='Set1')
 xAxis = [ x[-1][0] for x in testInputSetX ]

 for i, x in enumerate(testInputSetX):
  t = len(x)-1  
  plt.plot(xAxis[i], testSolutionSet[i], "o", color=cmap(t))
  plt.plot(xAxis[i], testInferredSet[i], "x", color=cmap(t))
 
 labelPatches = [ ] 
 for i, t in enumerate(T):
  labelPatches.append(mpatch.Patch(color=cmap(i), label='Seq Length: ' + str(i+1)))
 plt.legend(handles=labelPatches, loc='lower right')

 plt.show()
