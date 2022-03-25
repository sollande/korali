import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import time
import korali
import argparse
sys.argv=['']
k = korali.Engine()

parser = argparse.ArgumentParser()
parser.add_argument(
    '--engine',
    help='NN backend to use',
    default='OneDNN',
    required=False)
parser.add_argument(
    '--maxGenerations',
    help='Maximum Number of generations to run',
    default=5,
    required=False)    
parser.add_argument(
    '--optimizer',
    help='Optimizer to use for NN parameter updates',
    default='Adam',
    required=False)
parser.add_argument(
    '--learningRate',
    help='Learning rate for the selected optimizer',
    default=0.005,
    required=False)
parser.add_argument(
    '--trainingBatchSize',
    help='Batch size to use for training data',
    default=80,
    required=False)
parser.add_argument(
    '--testBatchSize',
    help='Batch size to use for test data',
    default=40,
    required=False)
parser.add_argument(
    '--testMSEThreshold',
    help='Threshold for the testing MSE, under which the run will report an error',
    default=0.05,
    required=False)
parser.add_argument(
    '--plot',
    help='Indicates whether to plot results after testing',
    default=True,
    required=False)
args = parser.parse_args()

print("Running FNN solver with arguments:")
print(args)

scaling = 5.0
np.random.seed(0xC0FFEE)

# The input set has scaling and a linear element to break symmetry
trainingInputSet = np.random.uniform(0, 2 * np.pi, args.trainingBatchSize)
trainingSolutionSet = np.tanh(np.exp(np.sin(trainingInputSet))) * scaling 

trainingInputSet = [ [ [ i ] ] for i in trainingInputSet.tolist() ]
trainingSolutionSet = [ [ i ] for i in trainingSolutionSet.tolist() ]

### Defining a learning problem to infer values of sin(x)

e = korali.Experiment()
e["Problem"]["Type"] = "Supervised Learning"
e["Problem"]["Max Timesteps"] = 1
e["Problem"]["Training Batch Size"] = args.trainingBatchSize
e["Problem"]["Testing Batch Size"] = args.testBatchSize

e["Problem"]["Input"]["Data"] = trainingInputSet
e["Problem"]["Input"]["Size"] = 1
e["Problem"]["Solution"]["Data"] = trainingSolutionSet
e["Problem"]["Solution"]["Size"] = 1

### Using a neural network solver (deep learning) for training

e["Solver"]["Type"] = "Learner/DeepSupervisor"
e["Solver"]["Mode"] = "Training"
e["Solver"]["Loss Function"] = "Mean Squared Error"
e["Solver"]["Learning Rate"] = float(args.learningRate)
e["Solver"]["Batch Concurrency"] = 1

### Defining the shape of the neural network

e["Solver"]["Neural Network"]["Engine"] = args.engine
e["Solver"]["Neural Network"]["Optimizer"] = args.optimizer

e["Solver"]["Neural Network"]["Hidden Layers"][0]["Type"] = "Layer/Linear"
e["Solver"]["Neural Network"]["Hidden Layers"][0]["Output Channels"] = 32

e["Solver"]["Neural Network"]["Hidden Layers"][1]["Type"] = "Layer/Activation"
e["Solver"]["Neural Network"]["Hidden Layers"][1]["Function"] = "Elementwise/Tanh"

e["Solver"]["Neural Network"]["Hidden Layers"][2]["Type"] = "Layer/Linear"
e["Solver"]["Neural Network"]["Hidden Layers"][2]["Output Channels"] = 32

e["Solver"]["Neural Network"]["Hidden Layers"][3]["Type"] = "Layer/Activation"
e["Solver"]["Neural Network"]["Hidden Layers"][3]["Function"] = "Elementwise/Tanh"

### Configuring output

e["Console Output"]["Frequency"] = 1
e["Console Output"]["Verbosity"] = "Detailed"
e["File Output"]["Enabled"] = False
e["Random Seed"] = 0xC0FFEE

### Training the neural network

e["Solver"]["Termination Criteria"]["Max Generations"] = int(args.maxGenerations)
k["Conduit"]["Type"] = "Sequential"
# k["Conduit"]["Concurrent Jobs"] = 2
#k["Conduit"]["Type"] = "Distributed"
#k.setMPIComm(MPI.COMM_WORLD)
k.run(e)

### Obtaining inferred results from the NN and comparing them to the actual solution

testInputSet = np.random.uniform(0, 2 * np.pi, args.testBatchSize)
testInputSet = [ [ [ i ] ] for i in testInputSet.tolist() ]
testOutputSet = [ x[0][0] for x in np.tanh(np.exp(np.sin(testInputSet))) * scaling ]

e["Solver"]["Mode"] = "Testing"
e["Problem"]["Input"]["Data"] = testInputSet
### Running Testing and getting results
k.run(e)
testInferredSet = [ x[0] for x in e["Solver"]["Evaluation"] ]
print("training finished")

# ### Calc MSE on test set
mse = np.mean((np.array(testInferredSet) - np.array(testOutputSet))**2)
print("MSE on test set: {}".format(mse))

if (mse > args.testMSEThreshold):
 print("Fail: MSE does not satisfy threshold: " + str(args.testMSEThreshold))
 exit(-1)

# ### Plotting Results

# if (args.plot):
#  plt.plot(testInputSet, testOutputSet, "o")
#  plt.plot(testInputSet, testInferredSet, "x")
#  plt.show()
