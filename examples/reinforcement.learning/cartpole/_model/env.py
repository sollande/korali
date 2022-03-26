#!/usr/bin/env python3
from cartpole import *

######## Defining Environment Storage

cart = CartPole()
maxSteps = 500

def env(s):

 # Initializing environment and random seed
 sampleId = s["Sample Id"]
 cart.reset(sampleId)
 s["State"] = cart.getState().tolist()
 step = 0
 done = False
  
 # Storing State Gradient
 #s["State Gradient"] = [[1.,1.],[1.,1.],[1.,1.],[1.,1.]]
 s["State Gradient"] = [[0.,0.],[0.,0.],[0.,0.],[0.,0.]]

 while not done and step < maxSteps:

  # Getting new action
  s.update()
 
  #print(s["Action"])
  # Performing the action
  done = cart.advance(s["Action"])
  
  # Getting Reward
  s["Reward"] = cart.getReward()
   
  # Storing New State
  s["State"] = cart.getState().tolist()
  
  # Storing State Gradient
  s["State Gradient"] = [[1.,1.],[1.,1.],[1.,1.],[1.,1.]]
  #s["State Gradient"] = [[0.,0.],[0.,0.],[0.,0.],[0.,0.]]

  # Advancing step counter
  step = step + 1

 # Setting finalization status
 if (cart.isOver()):
  s["Termination"] = "Terminal"
 else:
  s["Termination"] = "Truncated"

def multienv(s):

 # Initializing environment and random seed
 sampleId = s["Sample Id"]
 launchId = s["Launch Id"]
 if s["Mode"] == "Training":
    envId = sampleId % 3
 else:
    envId = 0
 cart.reset(sampleId * 1024 + launchId)
 s["Environment Id"] = envId
 s["State"] = cart.getState().tolist()
 step = 0
 done = False

 while not done and step < maxSteps:

  # Getting new action
  s.update()
  
  # Performing the action
  done = cart.advance(s["Action"])
  
  # Getting Reward
  reward = cart.getReward()
  if (envId == 0):
    s["Reward"] = cart.getReward()
  elif (envId == 1):
    s["Reward"] = cart.getReward() - 1
  else:
    s["Reward"] = cart.getReward() * 0.1
   
  # Storing New State
  s["State"] = cart.getState().tolist()
  
  # Advancing step counter
  step = step + 1

 # Setting finalization status
 if (cart.isOver()):
  s["Termination"] = "Terminal"
 else:
  s["Termination"] = "Truncated"
