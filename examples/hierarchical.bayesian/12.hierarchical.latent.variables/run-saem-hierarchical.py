
import sys
sys.path.append('./_model')
from model import *
from tutorial_samplers import *

import numpy as np

import korali


def main():
    # Initialize the distribution
    distrib = ConditionalDistribution4()

    # # rng = np.random.default_rng()
    # # initial_hyperparams = rng.standard_normal(2) # 1d mean and cov
    # initial_hyperparams = np.random.standard_normal(2) # 1d mean and cov

    k = korali.Engine()
    e = korali.Experiment()

    e["Problem"]["Type"] = "Bayesian/Latent/HierarchicalLatent"
    # The computational model for the log-likelihood, log[ p(data point | latent) ]
    e["Problem"]["Conditional Log Likelihood Function"] = lambda sample : distrib.conditional_p(sample)

    # We need to add one dimension to _p.data, because one individual in the general case could have
    # more than one data point assigned
    data_vector = [[]] * distrib._p.nIndividuals
    for i in range(distrib._p.nIndividuals):
        data_vector[i].append([distrib._p.data[i]])
    # e["Problem"]["Data"] = data_vector
    e["Problem"]["Data"] = data_vector
    e["Problem"]["Data Dimensions"] = 1
    e["Problem"]["Number Individuals"] = distrib._p.nIndividuals
    e["Problem"]["Latent Space Dimensions"] = 1

    e["Solver"]["Type"] = "HSAEM"
    e["Solver"]["Number Samples Per Step"] = 5
    e["Solver"]["Termination Criteria"]["Max Generations"] = 30

    e["Distributions"][0]["Name"] = "Uniform 0"
    e["Distributions"][0]["Type"] = "Univariate/Uniform"
    e["Distributions"][0]["Minimum"] = -100
    e["Distributions"][0]["Maximum"] = 100

    # * Define the variables:
    #   We only define one prototype latent variable vector (one-dimensional here) for individual 0.
    #   The others will be automatically generated by Korali, as well as all hyperparameters.
    for i in range(1): # range(distrib._p.nIndividuals):<
        e["Variables"][i]["Name"] = "latent mean "+str(i)
        e["Variables"][i]["Initial Value"] = -5
        e["Variables"][i]["Bayesian Type"] = "Latent"
        e["Variables"][i]["Latent Variable Distribution Type"] = "Normal"
        e["Variables"][i]["Prior Distribution"] = "Uniform 0"  # not used (?) but required

    e["File Output"]["Frequency"] = 50
    e["Console Output"]["Frequency"] = 10
    e["Console Output"]["Verbosity"] = "Detailed"

    k.run(e)




if __name__ == '__main__':
    # # ** For debugging, try this: **
    # import sys, trace
    # sys.stdout = sys.stderr
    # tracer = trace.Trace(trace=1, count=0, ignoredirs=["/usr", sys.prefix])
    # tracer.runfunc(main)
    # # ** Else: **
    main()
