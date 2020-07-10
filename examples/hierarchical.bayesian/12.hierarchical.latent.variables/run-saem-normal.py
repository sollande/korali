
import sys
sys.path.append('./_model/normal')
sys.path.append('./_model')
from model import *
from utils import generate_variable

import numpy as np
import korali



def main():
    # Initialize the distribution
    distrib = NormalConditionalDistribution()

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
    # e["Problem"]["Data"] = data_vector
    e["Problem"]["Data"] = data_vector
    e["Problem"]["Data Dimensions"] = distrib._p.nDataDimensions
    e["Problem"]["Number Individuals"] = distrib._p.nIndividuals
    e["Problem"]["Latent Space Dimensions"] = distrib._p.nLatentSpaceDimensions

    e["Solver"]["Type"] = "HSAEM"
    e["Solver"]["Number Samples Per Step"] = 5
    e["Solver"]["mcmc Outer Steps"] = 1
    e["Solver"]["mcmc Target Acceptance Rate"] = 0.4
    e["Solver"]["N1"] = 2
    e["Solver"]["N2"] = 2
    e["Solver"]["N3"] = 2
    e["Solver"]["K1"] = 200
    e["Solver"]["Ka"] = 200
    e["Solver"]["Alpha 1"] = 0.9999
    e["Solver"]["Alpha 2"] = 0.9999
    e["Solver"]["Use Simulated Annealing"] = True
    e["Solver"]["Simulated Annealing Decay Factor"] = 0.95
    e["Solver"]["Simulated Annealing Initial Variance"] = 1
    e["Solver"]["Diagonal Covariance" ] = True
    e["Solver"]["Termination Criteria"]["Max Generations"] = 250

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
    if np.isscalar(distrib._p.transf):
        distrib._p.transf = [distrib._p.transf]
    if np.isscalar(distrib._p.err_transf):
        distrib._p.err_transf = [distrib._p.err_transf]
    dimCounter = 0
    distribs = {"Normal": "Uniform 0", "Log-Normal": "Uniform 1",
                "Logit-Normal": "Uniform 2", "Probit-Normal": "Uniform XX"}
    for transf in distrib._p.transf:
        generate_variable(transf, e, dimCounter, "latent parameter "+str(dimCounter), distribs, initial=distrib._p.beta[dimCounter])
        dimCounter += 1

    for i, err_transf in enumerate(distrib._p.err_transf):
        generate_variable(err_transf,  e, dimCounter, "standard deviation "+str(i), distribs, initial=distrib._p.beta[dimCounter])
        dimCounter += 1

    assert dimCounter == distrib._p.dNormal + distrib._p.dLognormal + distrib._p.dLogitnormal + distrib._p.dProbitnormal

    e["File Output"]["Frequency"] = 1
    e["File Output"]["Path"] = "_korali_result_normal/"
    e["Console Output"]["Frequency"] = 1
    e["Console Output"]["Verbosity"] = "Normal" # "Detailed" results in all latent variable means being printed - we have 200 of them here, so better suppress this.

    k.run(e)




if __name__ == '__main__':
    # # ** For debugging, try this: **
    # import sys, trace
    # sys.stdout = sys.stderr
    # tracer = trace.Trace(trace=1, count=0, ignoredirs=["/usr", sys.prefix])
    # tracer.runfunc(main)
    # # ** Else: **
    main()
