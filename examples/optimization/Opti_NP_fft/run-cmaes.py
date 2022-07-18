#!/usr/bin/env python3

## In this example, we demonstrate how Korali finds values for the
## variables that maximize the objective function, given by a
## user-provided computational model.

# Importing computational model
import sys
import math
import numpy as np
sys.path.append('./_model')
from model import *

pi = np.pi

# Starting Korali's Engine
import korali
k = korali.Engine()

# Creating new experiment
e = korali.Experiment()

# Configuring Problem
e["Problem"]["Type"] = "Optimization"
e["Problem"]["Objective Function"] = model

# Defining the problem's variables.
dim = 9

# Defining the problem's variables.
for i in range(0,dim):
    e["Variables"][i]["Name"] = "XA" + str(i)
    e["Variables"][i]["Lower Bound"] = 0
    e["Variables"][i]["Upper Bound"] = +1
    e["Variables"][i]["Initial Standard Deviation"] = 0.5
    e["Variables"][i]["Initial Mean"] = 0.5

for i in range(dim,2*dim):
    e["Variables"][i]["Name"] = "XP" + str(i)
    e["Variables"][i]["Lower Bound"] = 0.5
    e["Variables"][i]["Upper Bound"] = 5
    

for i in range(2*dim,3*dim):
    e["Variables"][i]["Name"] = "XPH" + str(i)
    e["Variables"][i]["Lower Bound"] = -pi
    e["Variables"][i]["Upper Bound"] = +pi
    

# Configuring CMA-ES parameters
e["Solver"]["Type"] = "Optimizer/CMAES"
e["Solver"]["Population Size"] = 10
e["Solver"]["Termination Criteria"]["Min Value Difference Threshold"] = 1e-2
e["Solver"]["Termination Criteria"]["Max Generations"] = 100

# Reproducibility Settings
e["Random Seed"] = 0xC0FFEE
e["Preserve Random Number Generator States"] = True

# Configuring results path
e["File Output"]["Enabled"] = True
e["File Output"]["Path"] = '_korali_result_cmaes'
e["File Output"]["Frequency"] = 1

#k["Conduit"]["Type"] = "Concurrent"
#k["Conduit"]["Concurrent Jobs"] = 10

# Running Korali
k.run(e)
