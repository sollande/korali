import sys
sys.path.append('./_model/simple_example_Lavielle')
from model import *

import korali


def main():
  # Initialize the distribution
  distrib = ConditionalDistribution4()

  k = korali.Engine()
  e = korali.Experiment()

  # We need to add one dimension to _p.data, because one individual in the general case could have
  # more than one data point assigned
  data_vector = [[] for _ in range(distrib._p.nIndividuals)]
  for i in range(distrib._p.nIndividuals):
    data_vector[i].append([distrib._p.data[i]])

  e["Problem"]["Type"] = "Bayesian/Latent/HierarchicalLatentCustom"
  # The computational model for the log-likelihood, log[ p(data point | latent) ]
  e["Problem"][
      "Log Likelihood Functions"] = [lambda sample: distrib.conditional_p(
          sample, data_vector[i])  for i in range(distrib._p.nIndividuals)]

  e["Problem"]["Latent Space Dimensions"] = 1

  e["Solver"]["Type"] = "HSAEM"
  e["Solver"]["Number Samples Per Step"] = 5
  e["Solver"]["mcmc Outer Steps"] = 1
  e["Solver"]["N1"] = 2
  e["Solver"]["N2"] = 2
  e["Solver"]["N3"] = 2
  e["Solver"]["Termination Criteria"]["Max Generations"] = 30

  e["Distributions"][0]["Name"] = "Uniform 0"
  e["Distributions"][0]["Type"] = "Univariate/Uniform"
  e["Distributions"][0]["Minimum"] = -100
  e["Distributions"][0]["Maximum"] = 100

  # * Define the variables:
  #   We only define one prototype latent variable vector (one-dimensional here) for individual 0.
  #   The others will be automatically generated by Korali, as well as all hyperparameters.
  e["Variables"][0]["Name"] = "latent mean " + str(0)
  e["Variables"][0]["Initial Value"] = -5
  # e["Variables"][0]["Bayesian Type"] = "Latent"
  e["Variables"][0]["Latent Variable Distribution Type"] = "Normal"
  e["Variables"][0]["Prior Distribution"] = "Uniform 0"

  e["File Output"]["Frequency"] = 50
  e["File Output"]["Path"] = "_korali_result_hierarchical/"
  e["Console Output"]["Frequency"] = 10
  e["Console Output"]["Verbosity"] = "Detailed"

  k.run(e)

  print("------------ Experiment finished ------------\n")
  print("   Compare results to true optimizer in ")
  print("     '_data/simple_example_Lavielle/data_925_info.txt' ")
  print("   Plot experiment stats with: ")
  print("     'python3 -m korali.plotter'")


if __name__ == '__main__':
  # # ** For debugging, try this: **
  # import sys, trace
  # sys.stdout = sys.stderr
  # tracer = trace.Trace(trace=1, count=0, ignoredirs=["/usr", sys.prefix])
  # tracer.runfunc(main)
  # # ** Else: **
  main()
