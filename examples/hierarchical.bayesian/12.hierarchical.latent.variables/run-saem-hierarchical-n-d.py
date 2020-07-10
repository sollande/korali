import numpy as np
import sys
sys.path.append('./_model/n-d/')
from model import *

import korali


def main():
    # Initialize the distribution
    distrib = ConditionalDistribution5()

    # # rng = np.random.default_rng()
    # # initial_hyperparams = rng.standard_normal(2) # 1d mean and cov
    # initial_hyperparams = np.random.standard_normal(2) # 1d mean and cov

    k = korali.Engine()
    e = korali.Experiment()

    e["Problem"]["Type"] = "Bayesian/Latent/HierarchicalLatent"
    # The computational model for the log-likelihood, log[ p(data point | latent) ]
    e["Problem"]["Conditional Log Likelihood Function"] = lambda sample : distrib.conditional_p(sample)

    data_vector = [[] for _ in range(distrib._p.nIndividuals)]
    for i in range(distrib._p.nIndividuals):
        data_vector[i] = distrib._p.data[i].tolist()
    e["Problem"]["Data"] = data_vector
    e["Problem"]["Data Dimensions"] = distrib._p.nDimensions
    e["Problem"]["Number Individuals"] = distrib._p.nIndividuals
    e["Problem"]["Latent Space Dimensions"] = distrib._p.nDimensions

    e["Solver"]["Type"] = "HSAEM"
    e["Solver"]["Number Samples Per Step"] = 10
    e["Solver"]["mcmc Outer Steps"] = 1
    e["Solver"]["N1"] = 2
    e["Solver"]["N2"] = 2
    e["Solver"]["N3"] = 2
    e["Solver"]["Termination Criteria"]["Max Generations"] = 100

    e["Distributions"][0]["Name"] = "Uniform 0"
    e["Distributions"][0]["Type"] = "Univariate/Uniform"
    e["Distributions"][0]["Minimum"] = -100
    e["Distributions"][0]["Maximum"] = 100

    e["Distributions"][1]["Name"] = "Uniform 1"
    e["Distributions"][1]["Type"] = "Univariate/Uniform"
    e["Distributions"][1]["Minimum"] = 0
    e["Distributions"][1]["Maximum"] = 100

    e["Distributions"][2]["Name"] = "Uniform 2"
    e["Distributions"][2]["Type"] = "Univariate/Uniform"
    e["Distributions"][2]["Minimum"] = 0.0
    e["Distributions"][2]["Maximum"] = 1.0

    # * Define the variables:
    #   We only define one prototype latent variable vector for individual 0.
    #   The others will be automatically generated by Korali, as well as all hyperparameters.
    dimCounter = 0
    for i in range(distrib._p.dNormal):
        e["Variables"][dimCounter]["Name"] = "(Normal) latent mean "+str(i)
        e["Variables"][dimCounter]["Initial Value"] = -5.
        e["Variables"][dimCounter]["Bayesian Type"] = "Latent"
        e["Variables"][dimCounter]["Latent Variable Distribution Type"] = "Normal"
        e["Variables"][dimCounter]["Prior Distribution"] = "Uniform 0"  # not used, but required
        dimCounter += 1
    for i in range(distrib._p.dLognormal):
        e["Variables"][dimCounter]["Name"] = "(Log-normal) latent mean "+str(i)
        e["Variables"][dimCounter]["Initial Value"] = 5. # Valid range: (0, infinity)
        e["Variables"][dimCounter]["Bayesian Type"] = "Latent"
        e["Variables"][dimCounter]["Latent Variable Distribution Type"] = "Log-Normal"
        e["Variables"][dimCounter]["Prior Distribution"] = "Uniform 1" # not used, but required
        dimCounter += 1
    for i in range(distrib._p.dLogitnormal):
        e["Variables"][dimCounter]["Name"] = "(Logit-normal) latent mean "+str(i)
        e["Variables"][dimCounter]["Initial Value"] = 0.5
        e["Variables"][dimCounter]["Bayesian Type"] = "Latent"
        e["Variables"][dimCounter]["Latent Variable Distribution Type"] = "Logit-Normal"
        e["Variables"][dimCounter]["Prior Distribution"] = "Uniform 2"  # not used, but required
        dimCounter += 1
    assert dimCounter == distrib._p.nDimensions

    e["File Output"]["Frequency"] = 1
    e["File Output"]["Path"] = "_korali_result_n-d/"
    e["Console Output"]["Frequency"] = 1
    e["Console Output"]["Verbosity"] = "Detailed"

    k.run(e)

    print("------------ Experiment finished ------------\n")
    print("   Plot experiment stats with:")
    print("       'python3 -m korali.plotter'")
    print("   Compare results to original hyperparameter")
    print("    values in '_data/n-d/data_advanced.txt' ")



if __name__ == '__main__':
    # # ** For debugging, try this: **
    # import sys, trace
    # sys.stdout = sys.stderr
    # tracer = trace.Trace(trace=1, count=0, ignoredirs=["/usr", sys.prefix])
    # tracer.runfunc(main)
    # # ** Else: **
    main()
