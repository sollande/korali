#!/usr/bin/env python3
import sys
import threading
sys.path.append('./model')
from ackley import ackley
import libkorali

korali = libkorali.Engine(ackley)

korali["Verbosity"] = "Normal"

korali["Problem"]["Type"] = "Direct"
for i in range(4):
 korali["Problem"]["Variables"][i]["Name"] = "X" + str(i)
 korali["Problem"]["Variables"][i]["Distribution"] = "Uniform"
 korali["Problem"]["Variables"][i]["Type"] = "Computational"
 korali["Problem"]["Variables"][i]["Minimum"] = -32.0
 korali["Problem"]["Variables"][i]["Maximum"] = +32.0


korali["Solver"]["Method"] = "CMA-ES"
korali["Solver"]["Termination Criteria"]["Min DeltaX"] = 1e-11
korali["Solver"]["Termination Criteria"]["Max Generations"] = 600
korali["Solver"]["Lambda"] = 128

korali.run()
