#!/usr/bin/env python3

## In this example, we demonstrate how Korali finds values for the
## variables that maximize the objective function, given by a
## user-provided computational model.

# Importing computational model
import sys
import math
sys.path.append('./_model')
from model import *
import numpy as np

# Starting Korali's Engine
import korali
k = korali.Engine()

# Creating new experiment
e = korali.Experiment()

# Configuring Problem
e["Random Seed"] = 0xC0FEE
e["Problem"]["Type"] = "Optimization"
e["Problem"]["Objective Function"] = model

dim = 3

# Defining the problem's variables.
for i in range(dim):
    e["Variables"][i]["Name"] = "XA" + str(i)
    e["Variables"][i]["Lower Bound"] = 0
    e["Variables"][i]["Upper Bound"] = 1

for i in range(dim,2*dim):
    e["Variables"][i]["Name"] = "XP" + str(i)
    e["Variables"][i]["Lower Bound"] = 0
    e["Variables"][i]["Upper Bound"] = 10

for i in range(2*dim,3*dim):
    e["Variables"][i]["Name"] = "XPH" + str(i)
    e["Variables"][i]["Lower Bound"] = -np.pi
    e["Variables"][i]["Upper Bound"] = np.pi


# Configuring CMA-ES parameters
e["Solver"]["Type"] = "Optimizer/CMAES"
e["Solver"]["Population Size"] = 8
e["Solver"]["Termination Criteria"]["Min Value Difference Threshold"] = 1e-3
e["Solver"]["Termination Criteria"]["Max Generations"] = 100

# Configuring results path
e["File Output"]["Enabled"] = True
e["File Output"]["Path"] = '_korali_result_cmaes'
e["File Output"]["Frequency"] = 1

# Running Korali
k.run(e)
