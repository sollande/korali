import sys
sys.path.append('./_model/normal')
sys.path.append('./_model')
from model import *

import numpy as np
import korali


def main():
  # * Initialize the assumed data distribution
  distrib = NormalConditionalDistribution()

  k = korali.Engine()
  e = korali.Experiment()

  e["Problem"]["Type"] = "Bayesian/Latent/HierarchicalCustom"

  # * Next, define the computational model for the log-likelihood, log[ p(data point | latent) ]

  ## Warning: The i=i below is necessary to capture the current i.
  ## Just writing
  ##   lambda sample, i: logisticModelFunction(sample, x_vals[i])
  ## will capture i by reference and thus not do what is intended.

  e["Problem"]["Log Likelihood Functions"] = [
      lambda sample, i=i: distrib.conditional_logp(sample, distrib._p.data[i])
      for i in range(distrib._p.nIndividuals)
  ]
  e["Problem"]["Diagonal Covariance"] = True

  e["Solver"]["Type"] = "HSAEM"
  e["Solver"]["Number Samples Per Step"] = 5
  e["Solver"]["MCMC Outer Steps"] = 1
  e["Solver"]["MCMC Target Acceptance Rate"] = 0.4
  e["Solver"]["N1"] = 2
  e["Solver"]["N2"] = 2
  e["Solver"]["N3"] = 2
  e["Solver"]["K1"] = 200
  e["Solver"]["Ka"] = 200
  e["Solver"]["Alpha 1"] = 0.25
  e["Solver"]["Alpha 2"] = 0.5
  e["Solver"]["Use Simulated Annealing"] = True
  e["Solver"]["Simulated Annealing Decay Factor"] = 0.95
  e["Solver"]["Simulated Annealing Initial Variance"] = 1
  e["Solver"]["Termination Criteria"]["Max Generations"] = 250

  e["Distributions"][0]["Name"] = "Uniform 0"
  e["Distributions"][0]["Type"] = "Univariate/Uniform"
  e["Distributions"][0]["Minimum"] = -100
  e["Distributions"][0]["Maximum"] = 100

  e["Distributions"][1]["Name"] = "Uniform 1"
  e["Distributions"][1]["Type"] = "Univariate/Uniform"
  e["Distributions"][1]["Minimum"] = 0
  e["Distributions"][1]["Maximum"] = 100

  # * Define the variables:
  #   We only define one prototype latent variable vector for individual 0.
  #   The others will be automatically generated by Korali, as well as all hyperparameters.

  # We define one normal and one lognormal variable.

  e["Variables"][0]["Name"] = "Theta 1"
  e["Variables"][0]["Initial Value"] = 2
  e["Variables"][0]["Latent Variable Distribution Type"] = "Normal"
  e["Variables"][0][
      "Prior Distribution"] = "Uniform 0"  # not used, but required

  e["Variables"][1]["Name"] = "Theta 2"
  e["Variables"][1]["Initial Value"] = 2
  e["Variables"][1]["Latent Variable Distribution Type"] = "Log-Normal"
  e["Variables"][1][
      "Prior Distribution"] = "Uniform 1"  # not used, but required

  # Configure how results will be stored to a file:
  e["File Output"]["Frequency"] = 1
  e["File Output"]["Path"] = "_korali_result_normal_custom/"
  # We choose a non-default output directory -
  # for plotting results, we can later set the directory with:
  #   python3 -m korali.plotter --dir _korali_result_logistic/

  # Configure console output:
  e["Console Output"]["Frequency"] = 1
  e["Console Output"][
      "Verbosity"] = "Normal"  # "Detailed" results in all latent variable means being printed -
  # we have 200 of them here, so we suppress this by choosing a less
  # detailed output option.

  k.run(e)


if __name__ == '__main__':
  # # ** For debugging, try this: **
  # import sys, trace
  # sys.stdout = sys.stderr
  # tracer = trace.Trace(trace=1, count=0, ignoredirs=["/usr", sys.prefix])
  # tracer.runfunc(main)
  # # ** Else: **
  main()
