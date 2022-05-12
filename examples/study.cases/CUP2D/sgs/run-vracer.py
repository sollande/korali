#!/usr/bin/env python3
import os
import sys
import argparse
sys.path.append('_model')

import math
from environment import *

####### Parsing arguments
parser = argparse.ArgumentParser()
parser.add_argument('--environment', help='Specifies which environment to run.', required=False, type=str, default="kolmogorovFlow")
parser.add_argument('--pathToIC', help='Specifies the path to the initial condition.', required=False, type=str, default="../../_model/Field_N=128_Cs=0.0_T=10.0.npz")
parser.add_argument('--pathToGroundtruth', help='Specifies the path to the data.', required=False, type=str, default="../../_model/Energy_N=128_Cs=0.0.out")
parser.add_argument('--nEnvironments', help='Specifies the number of environments to run in parallel.', required=False, type=int, default=1)
parser.add_argument('--multiPolicy', help='Whether to use multiple policies.', action='store_true', required=False)
parser.add_argument('--numBlocks', help='(Number of blocks)^2 == agents to use', required=False, type=int, default=2)
parser.add_argument('--stepsPerAction', help='Number of simulation steps between actions', required=False, type=int, default=10)
parser.add_argument('--numExp', help='Total number of experiences', required=False, type=int, default=1000000)
args = parser.parse_args()

############ Setup Korali ############
######################################
import korali
k = korali.Engine()
e = korali.Experiment()

nAgents = args.numBlocks*args.numBlocks

if args.nEnvironments > 1:
	from mpi4py import MPI
	k["Conduit"]["Type"] = "Distributed"
	k.setMPIComm(MPI.COMM_WORLD)

####### Defining Korali Problem
e["Problem"]["Type"] = "Reinforcement Learning / Continuous"
e["Problem"]["Environment Function"] = lambda x : runEnvironment(x, args.environment, args.numBlocks, args.stepsPerAction, args.pathToIC, args.pathToGroundtruth)
e["Problem"]["Agents Per Environment"] = nAgents
if args.multiPolicy:
	e["Problem"]["Policies Per Environment"] = nAgents

### Defining Agent Configuration 
e["Solver"]["Type"] = "Agent / Continuous / VRACER"
e["Solver"]["Mode"] = "Training"
e["Solver"]["Episodes Per Generation"] = 1
e["Solver"]["Experiences Between Policy Updates"] = 1
e["Solver"]["Learning Rate"] = 0.0001
e["Solver"]["Discount Factor"] = 0.95
e["Solver"]["Mini Batch"]["Size"] = 128
# e["Solver"]["Multi Agent Relationship"] = 'Individual'
# e["Solver"]["Multi Agent Correlation"] = False
# e["Solver"]["Multi Agent Sampling"] = "Baseline"

## Defining State Variables
stateVariableCount = 2*64 # get velocity field at every gridpoint
for i in range(stateVariableCount):
	e["Variables"][i]["Name"] = "State Variable " + str(i)
	e["Variables"][i]["Type"] = "State"

## Defining Action Variables
actionVariableCount = 64 # learn Cs for every gridpoint
for i in range(actionVariableCount):
	e["Variables"][stateVariableCount + i]["Name"] = "Action Variable " + str(i) 
	e["Variables"][stateVariableCount + i]["Type"] = "Action"
	e["Variables"][stateVariableCount + i]["Lower Bound"] = -5
	e["Variables"][stateVariableCount + i]["Upper Bound"] = 5
	e["Variables"][stateVariableCount + i]["Initial Exploration Noise"] = 0.1

### Setting RL Algorithm settings
e["Solver"]["Experience Replay"]["Start Size"] = 1024 #65536
e["Solver"]["Experience Replay"]["Maximum Size"] = 262144
e["Solver"]["Experience Replay"]["Off Policy"]["Annealing Rate"] = 5.0e-8
e["Solver"]["Experience Replay"]["Off Policy"]["Cutoff Scale"] = 5.0
e["Solver"]["Experience Replay"]["Off Policy"]["REFER Beta"] = 0.3
e["Solver"]["Experience Replay"]["Off Policy"]["Target"] = 0.1

e["Solver"]["Policy"]["Distribution"] = "Clipped Normal"
e["Solver"]["State Rescaling"]["Enabled"] = True
e["Solver"]["Reward"]["Rescaling"]["Enabled"] = True
  
### Configuring the neural network and its hidden layers
e["Solver"]["Neural Network"]["Engine"] = "OneDNN"
e["Solver"]['Neural Network']['Optimizer'] = "Adam"
e["Solver"]["L2 Regularization"]["Enabled"] = False
e["Solver"]["L2 Regularization"]["Importance"] = 1.0

e["Solver"]["Neural Network"]["Hidden Layers"][0]["Type"] = "Layer/Linear"
e["Solver"]["Neural Network"]["Hidden Layers"][0]["Output Channels"] = 256

e["Solver"]["Neural Network"]["Hidden Layers"][1]["Type"] = "Layer/Activation"
e["Solver"]["Neural Network"]["Hidden Layers"][1]["Function"] = "Elementwise/Tanh"

e["Solver"]["Neural Network"]["Hidden Layers"][2]["Type"] = "Layer/Linear"
e["Solver"]["Neural Network"]["Hidden Layers"][2]["Output Channels"] = 256

e["Solver"]["Neural Network"]["Hidden Layers"][3]["Type"] = "Layer/Activation"
e["Solver"]["Neural Network"]["Hidden Layers"][3]["Function"] = "Elementwise/Tanh"

### Setting file output configuration
e["Solver"]["Termination Criteria"]["Max Experiences"] = args.numExp
e["Solver"]["Experience Replay"]["Serialize"] = True
e["Console Output"]["Verbosity"] = "Detailed"
e["File Output"]["Enabled"] = True
e["File Output"]["Frequency"] = 1
e["File Output"]["Use Multiple Files"] = False
e["File Output"]["Path"] = "_trainingResults/"

### Running Experiment
k.run(e)
