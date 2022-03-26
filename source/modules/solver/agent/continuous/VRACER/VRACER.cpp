#include "engine.hpp"
#include "modules/solver/agent/continuous/VRACER/VRACER.hpp"
#include "omp.h"
#include "sample/sample.hpp"

#include <gsl/gsl_sf_psi.h>

//#define DEBUG
namespace korali
{
namespace solver
{
namespace agent
{
namespace continuous
{
;

void VRACER::initializeAgent()
{
  // Initializing common discrete agent configuration
  Continuous::initializeAgent();

  // Init statistics
  _statisticsAverageActionSigmas.resize(_problem->_actionVectorSize);

  /*********************************************************************
   * Initializing Critic/Policy Neural Network Optimization Experiment
   *********************************************************************/

  _criticPolicyExperiment["Problem"]["Type"] = "Supervised Learning";
  _criticPolicyExperiment["Problem"]["Max Timesteps"] = _timeSequenceLength;
  _criticPolicyExperiment["Problem"]["Training Batch Size"] = _miniBatchSize;
  _criticPolicyExperiment["Problem"]["Inference Batch Size"] = 1;
  _criticPolicyExperiment["Problem"]["Input"]["Size"] = _problem->_stateVectorSize;
  _criticPolicyExperiment["Problem"]["Solution"]["Size"] = 1 + _policyParameterCount;

  _criticPolicyExperiment["Solver"]["Type"] = "DeepSupervisor";
  _criticPolicyExperiment["Solver"]["L2 Regularization"]["Enabled"] = _l2RegularizationEnabled;
  _criticPolicyExperiment["Solver"]["L2 Regularization"]["Importance"] = _l2RegularizationImportance;
  _criticPolicyExperiment["Solver"]["Learning Rate"] = _currentLearningRate;
  _criticPolicyExperiment["Solver"]["Loss Function"] = "Direct Gradient";
  _criticPolicyExperiment["Solver"]["Steps Per Generation"] = 1;
  _criticPolicyExperiment["Solver"]["Neural Network"]["Optimizer"] = _neuralNetworkOptimizer;
  _criticPolicyExperiment["Solver"]["Neural Network"]["Engine"] = _neuralNetworkEngine;
  _criticPolicyExperiment["Solver"]["Neural Network"]["Hidden Layers"] = _neuralNetworkHiddenLayers;
  _criticPolicyExperiment["Solver"]["Output Weights Scaling"] = 0.001;

  // No transformations for the state value output
  _criticPolicyExperiment["Solver"]["Neural Network"]["Output Layer"]["Scale"][0] = 1.0f;
  _criticPolicyExperiment["Solver"]["Neural Network"]["Output Layer"]["Shift"][0] = 0.0f;
  _criticPolicyExperiment["Solver"]["Neural Network"]["Output Layer"]["Transformation Mask"][0] = "Identity";

  // Setting transformations for the selected policy distribution output
  for (size_t i = 0; i < _policyParameterCount; i++)
  {
    _criticPolicyExperiment["Solver"]["Neural Network"]["Output Layer"]["Scale"][i + 1] = _policyParameterScaling[i];
    _criticPolicyExperiment["Solver"]["Neural Network"]["Output Layer"]["Shift"][i + 1] = _policyParameterShifting[i];
    _criticPolicyExperiment["Solver"]["Neural Network"]["Output Layer"]["Transformation Mask"][i + 1] = _policyParameterTransformationMasks[i];
  }

  // Running initialization to verify that the configuration is correct
  _criticPolicyExperiment.initialize();
  _criticPolicyProblem = dynamic_cast<problem::SupervisedLearning *>(_criticPolicyExperiment._problem);
  _criticPolicyLearner = dynamic_cast<solver::DeepSupervisor *>(_criticPolicyExperiment._solver);

  _maxMiniBatchPolicyMean.resize(_problem->_actionVectorSize);
  _maxMiniBatchPolicyStdDev.resize(_problem->_actionVectorSize);

  _minMiniBatchPolicyMean.resize(_problem->_actionVectorSize);
  _minMiniBatchPolicyStdDev.resize(_problem->_actionVectorSize);
}

void VRACER::trainPolicy()
{
  // Obtaining Minibatch experience ids
  const auto miniBatch = generateMiniBatch(_miniBatchSize);

  // Now calculating policy gradients
  calculatePolicyGradients(miniBatch);
}

void VRACER::calculatePolicyGradients(const std::vector<size_t> &miniBatch)
{
  // Resetting statistics
  std::fill(_statisticsAverageActionSigmas.begin(), _statisticsAverageActionSigmas.end(), 0.0);

  for (size_t i = 0; i < _problem->_actionVectorSize; i++)
  {
    _maxMiniBatchPolicyMean[i] = -Inf;
    _maxMiniBatchPolicyStdDev[i] = -Inf;
    _minMiniBatchPolicyMean[i] = +Inf;
    _minMiniBatchPolicyStdDev[i] = +Inf;
  }

  const size_t miniBatchSize = miniBatch.size();

  // Gathering state sequences for selected minibatch
  const auto stateSequence = getMiniBatchStateSequence(miniBatch);

  // Running policy NN on the Minibatch experiences
  auto policyInfo = runPolicy(stateSequence);

  // Using policy information to update experience's metadata
  updateExperienceMetadata(miniBatch, policyInfo);

  // Calculating gradient of action probability wrt state
#pragma omp parallel for
  for (size_t b = 0; b < miniBatchSize; b++)
  {
    // Getting index of experience
    const size_t expId = miniBatch[b];

    // Getting sampled action
    const auto &expAction = _actionBuffer[expId];

    // Getting current policy
    const auto &curPolicy = _curPolicyBuffer[expId];

    // Storage for the gradient to backpropagate
    std::vector<float> gradientInput(1 + _policyParameterCount, 0.f);

    // Gradient of policy wrt value is zero
    gradientInput[0] = 0.f;

    // Compute policy gradient only if inside trust region (or offPolicy disabled)
    if (_isOnPolicyBuffer[expId])
    {
      // Compute gradient of action probability wrt policy
      const auto actionProbGrad = calculateActionProbabilityGradient(expAction, curPolicy);

      // Set gradient for backpropagation
      for (size_t i = 0; i < _policyParameterCount; i++)
        gradientInput[1 + i] = actionProbGrad[i];
    }

    // Validate gradient
    for (size_t i = 0; i < gradientInput.size(); i++)
      if (std::isfinite(gradientInput[i]) == false)
        KORALI_LOG_ERROR("Action gradient returned an invalid value: %f\n", gradientInput[i]);

    // Set gradient of Loss as Solution
    _criticPolicyProblem->_solutionData[b] = gradientInput;
  }

  // Calculate gradient of action probability wrt state
  _criticPolicyLearner->_neuralNetwork->backward(_criticPolicyProblem->_solutionData);
  auto policyStateGradient = _criticPolicyLearner->_neuralNetwork->getInputGradients(miniBatchSize);

#ifdef DEBUG
  for (auto grad : policyStateGradient)
  {
    for (auto d : grad)
      printf("pig [%f]\t", d);
    printf("\n");
  }
  printf("\n\n");
#endif

  std::vector<std::vector<float>> policyStateActionGradient(miniBatchSize, std::vector<float>(_problem->_actionVectorSize, 0.f));

#pragma omp parallel for
  for (size_t b = 0; b < miniBatchSize; b++)
  {
    // Getting index of experience
    const size_t expId = miniBatch[b];

    const auto &stateActionGradient = _stateGradientBuffer[expId];

    for (size_t i = 0; i < _problem->_actionVectorSize; ++i)
    {
      // Calculating product of policy gradient wrt input and gradient of state wrt previous action
      for (size_t j = 0; j < _problem->_stateVectorSize; ++j)
        policyStateActionGradient[b][i] += policyStateGradient[b][j] * stateActionGradient[j][i];
    }
  }

#ifdef DEBUG
  for (auto grad : policyStateActionGradient)
  {
    for (auto d : grad)
      printf("psag [%f]\t", d);
    printf("\n");
  }
  printf("\n\n");
#endif

  // Gathering state sequences for preceeding experiences
  const auto previousStateSequence = getMiniBatchPreviousStateSequence(miniBatch);

  // Running policy NN on the Minibatch of preceeding experiences
  auto previousPolicyInfo = runPolicy(previousStateSequence);

  // Storage for the gradient of the action wrt policy parameter
  std::vector<std::vector<float>> stateActionGradients(miniBatchSize, std::vector<float>(_policyParameterCount));

  for (size_t b = 0; b < miniBatchSize; b++)
  {
    // Getting index of current and previous experience
    const size_t expId = miniBatch[b];
    const size_t prevExpId = (expId - 1) % _episodeIdBuffer.size();

    // Gathering metadata
    const float V = _stateValueBuffer[expId];
    const auto &curPolicy = _curPolicyBuffer[prevExpId];
    const auto &oldPolicy = _expPolicyBuffer[prevExpId];
    const auto &expAction = _actionBuffer[prevExpId];

    // Resetting gradient to backpropagate
    _criticPolicyProblem->_solutionData[b] = std::vector<float>(1 + _policyParameterCount, 0.f);

    // Compute policy gradient only if inside trust region and previous action exists in RM (=is part of the same episode)
    if (_isOnPolicyBuffer[expId] && _episodeIdBuffer[expId] == _episodeIdBuffer[prevExpId])
    {
      // Qret for terminal state is just reward
      float Qret = getScaledReward(_environmentIdBuffer[expId], _rewardBuffer[expId]);

      // If experience is non-terminal, add Vtbc
      if (_terminationBuffer[expId] == e_nonTerminal)
      {
        float nextExpVtbc = _retraceValueBuffer[expId + 1];
        Qret += _discountFactor * nextExpVtbc;
      }

      // If experience is truncated, add truncated state value
      if (_terminationBuffer[expId] == e_truncated)
      {
        float nextExpVtbc = _truncatedStateValueBuffer[expId];
        Qret += _discountFactor * nextExpVtbc;
      }

      // Compute Off-Policy Objective (eq. 5)
      float lossOffPolicy = Qret - V;

      // Calcuation gradient of action wrt policy parameter
      stateActionGradients[b] = calculateActionPolicyGradient(expAction, curPolicy, oldPolicy);
      for(auto g : stateActionGradients[b])
         if(std::isfinite(g) == false)
             KORALI_LOG_ERROR("State action gradient not finite\n");

      const float oldActionProbability = calculateActionProbability(expAction, oldPolicy);

      // Set Gradient of action wrt mean
      for (size_t i = 0; i < _policyParameterCount; i++)
      {
        _criticPolicyProblem->_solutionData[b][1 + i] = _experienceReplayOffPolicyREFERBeta * lossOffPolicy / oldActionProbability * policyStateActionGradient[b][i] * stateActionGradients[b][i];
        if(std::isfinite(_criticPolicyProblem->_solutionData[b][1 + i]) == false)
            KORALI_LOG_ERROR("Policy State Gradient wrt previous action not finite (old p %f)\n", oldActionProbability);
      }
    }
  }

  // Backpropagate gradients
  _criticPolicyLearner->_neuralNetwork->backward(_criticPolicyProblem->_solutionData);

  // Retrieve hyperparameter gradients
  auto nnPolicyGradientPreviousActionParams = _criticPolicyLearner->_neuralNetwork->getHyperparameterGradients(miniBatchSize);

#ifdef DEBUG
  for (auto grad : nnPolicyGradientPreviousActionParams)
  {
    printf("pg [%f]\t", grad);
  }
  printf("\n\n");
#endif

  // Running policy NN on the Minibatch experiences
  policyInfo = runPolicy(stateSequence);

#pragma omp parallel for
  for (size_t b = 0; b < miniBatchSize; b++)
  {
    // Getting index of current experiment
    size_t expId = miniBatch[b];

    // Get state, action and policy for this experience
    const auto &expPolicy = _expPolicyBuffer[expId];
    const auto &expAction = _actionBuffer[expId];

    // Gathering metadata
    const float V = _stateValueBuffer[expId];
    const auto &curPolicy = _curPolicyBuffer[expId];
    const float expVtbc = _retraceValueBuffer[expId];

    // Storage for the update gradient
    std::vector<float> gradientLoss(1 + _policyParameterCount, 0.f);

    // Gradient of Value Function V(s) (eq. (9); *-1 because the optimizer is maximizing)
    gradientLoss[0] = expVtbc - V;

    // Compute policy gradient only if inside trust region (or offPolicy disabled)
    if (_isOnPolicyBuffer[expId])
    {
      // Qret for terminal state is just reward
      float Qret = getScaledReward(_environmentIdBuffer[expId], _rewardBuffer[expId]);

      // If experience is non-terminal, add Vtbc
      if (_terminationBuffer[expId] == e_nonTerminal)
      {
        float nextExpVtbc = _retraceValueBuffer[expId + 1];
        Qret += _discountFactor * nextExpVtbc;
      }

      // If experience is truncated, add truncated state value
      if (_terminationBuffer[expId] == e_truncated)
      {
        float nextExpVtbc = _truncatedStateValueBuffer[expId];
        Qret += _discountFactor * nextExpVtbc;
      }

      // Compute Off-Policy Objective (eq. 5)
      float lossOffPolicy = Qret - V;

      // Compute IW Gradient wrt params
      auto iwGrad = calculateImportanceWeightGradient(expAction, curPolicy, expPolicy);

      // Set Gradient of Loss wrt Params
      for (size_t i = 0; i < _policyParameterCount; i++)
        gradientLoss[1 + i] = _experienceReplayOffPolicyREFERBeta * lossOffPolicy * iwGrad[i];
    }

    // Compute derivative of kullback-leibler divergence wrt current distribution params
    auto klGrad = calculateKLDivergenceGradient(expPolicy, curPolicy);

    // Step towards old policy (gradient pointing to larger difference between old and current policy)
    const float klGradMultiplier = -(1.0f - _experienceReplayOffPolicyREFERBeta);
    for (size_t i = 0; i < _policyParameterCount; i++)
      gradientLoss[1 + i] += klGradMultiplier * klGrad[i];

    for (size_t i = 0; i < _problem->_actionVectorSize; i++)
    {
      if (expPolicy.distributionParameters[i] > _maxMiniBatchPolicyMean[i]) _maxMiniBatchPolicyMean[i] = expPolicy.distributionParameters[i];
      if (expPolicy.distributionParameters[_problem->_actionVectorSize + i] > _maxMiniBatchPolicyStdDev[i]) _maxMiniBatchPolicyStdDev[i] = expPolicy.distributionParameters[_problem->_actionVectorSize + i];
      if (expPolicy.distributionParameters[i] < _minMiniBatchPolicyMean[i]) _minMiniBatchPolicyMean[i] = expPolicy.distributionParameters[i];
      if (expPolicy.distributionParameters[_problem->_actionVectorSize + i] < _minMiniBatchPolicyStdDev[i]) _minMiniBatchPolicyStdDev[i] = expPolicy.distributionParameters[_problem->_actionVectorSize + i];
    }

    // Set Gradient of Loss as Solution
    for (size_t i = 0; i < gradientLoss.size(); i++)
      if (std::isfinite(gradientLoss[i]) == false)
        KORALI_LOG_ERROR("Gradient loss returned an invalid value: %f\n", gradientLoss[i]);
    _criticPolicyProblem->_solutionData[b] = gradientLoss;
  }

  // Backpropagate gradients
  _criticPolicyLearner->_neuralNetwork->backward(_criticPolicyProblem->_solutionData);

  // Getting hyperparameter gradients
  auto nnHyperparameterPolicyGradients = _criticPolicyLearner->_neuralNetwork->getHyperparameterGradients(miniBatchSize);

  // Calculating full gradients
  std::vector<float> nnHyperparameterFullGradients(nnHyperparameterPolicyGradients.size(), 0.f);

#pragma omp parallel for simd
  for (size_t i = 0; i < nnHyperparameterFullGradients.size(); ++i)
  {
    nnHyperparameterFullGradients[i] = nnHyperparameterPolicyGradients[i] + nnPolicyGradientPreviousActionParams[i];
  }

  // Setting learning rate
  _criticPolicyLearner->_learningRate = _currentLearningRate;

  // Passing hyperparameter gradients through the optimzier
  _criticPolicyLearner->_optimizer->processResult(nnHyperparameterFullGradients);

  // Getting new set of hyperparameters from optimizer
  _criticPolicyLearner->_neuralNetwork->setHyperparameters(_criticPolicyLearner->_optimizer->_currentValue);

  // Compute average action stadard deviation
  for (size_t j = 0; j < _problem->_actionVectorSize; j++) _statisticsAverageActionSigmas[j] /= (float)miniBatchSize;
}

std::vector<policy_t> VRACER::runPolicy(const std::vector<std::vector<std::vector<float>>> &stateBatch)
{
  // Getting batch size
  size_t batchSize = stateBatch.size();

  // Storage for policy
  std::vector<policy_t> policyVector(batchSize);

  // Forward the neural network for this state
  const auto evaluation = _criticPolicyLearner->getEvaluation(stateBatch);

#pragma omp parallel for
  for (size_t b = 0; b < batchSize; b++)
  {
    // Getting state value
    policyVector[b].stateValue = evaluation[b][0];

    // Getting distribution parameters
    policyVector[b].distributionParameters.assign(evaluation[b].begin() + 1, evaluation[b].end());
  }

  return policyVector;
}

knlohmann::json VRACER::getPolicy()
{
  knlohmann::json hyperparameters;
  hyperparameters["Policy"] = _criticPolicyLearner->getHyperparameters();
  return hyperparameters;
}

void VRACER::setPolicy(const knlohmann::json &hyperparameters)
{
  _criticPolicyLearner->setHyperparameters(hyperparameters["Policy"].get<std::vector<float>>());
}

void VRACER::printInformation()
{
  _k->_logger->logInfo("Normal", " + [VRACER] Policy Learning Rate: %.3e\n", _currentLearningRate);
  _k->_logger->logInfo("Detailed", " + [VRACER] Max Policy Parameters (Mu & Sigma):\n");
  for (size_t i = 0; i < _problem->_actionVectorSize; i++)
    _k->_logger->logInfo("Detailed", " + [VRACER] Action %zu: (%.3e,%.3e)\n", i, _maxMiniBatchPolicyMean[i], _maxMiniBatchPolicyStdDev[i]);
  _k->_logger->logInfo("Detailed", " + [VRACER] Min Policy Parameters (Mu & Sigma):\n");
  for (size_t i = 0; i < _problem->_actionVectorSize; i++)
    _k->_logger->logInfo("Detailed", " + [VRACER] Action %zu: (%.3e,%.3e)\n", i, _minMiniBatchPolicyMean[i], _minMiniBatchPolicyStdDev[i]);
}

void VRACER::setConfiguration(knlohmann::json& js) 
{
 if (isDefined(js, "Results"))  eraseValue(js, "Results");

 if (isDefined(js, "Statistics", "Average Action Sigmas"))
 {
 try { _statisticsAverageActionSigmas = js["Statistics"]["Average Action Sigmas"].get<std::vector<float>>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ VRACER ] \n + Key:    ['Statistics']['Average Action Sigmas']\n%s", e.what()); } 
   eraseValue(js, "Statistics", "Average Action Sigmas");
 }

 if (isDefined(_k->_js.getJson(), "Variables"))
 for (size_t i = 0; i < _k->_js["Variables"].size(); i++) { 
 if (isDefined(_k->_js["Variables"][i], "Initial Exploration Noise"))
 {
 try { _k->_variables[i]->_initialExplorationNoise = _k->_js["Variables"][i]["Initial Exploration Noise"].get<float>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ VRACER ] \n + Key:    ['Initial Exploration Noise']\n%s", e.what()); } 
   eraseValue(_k->_js["Variables"][i], "Initial Exploration Noise");
 }
  else   KORALI_LOG_ERROR(" + No value provided for mandatory setting: ['Initial Exploration Noise'] required by VRACER.\n"); 

 } 
 Continuous::setConfiguration(js);
 _type = "agent/continuous/VRACER";
 if(isDefined(js, "Type")) eraseValue(js, "Type");
 if(isEmpty(js) == false) KORALI_LOG_ERROR(" + Unrecognized settings for Korali module: VRACER: \n%s\n", js.dump(2).c_str());
} 

void VRACER::getConfiguration(knlohmann::json& js) 
{

 js["Type"] = _type;
   js["Statistics"]["Average Action Sigmas"] = _statisticsAverageActionSigmas;
 for (size_t i = 0; i <  _k->_variables.size(); i++) { 
   _k->_js["Variables"][i]["Initial Exploration Noise"] = _k->_variables[i]->_initialExplorationNoise;
 } 
 Continuous::getConfiguration(js);
} 

void VRACER::applyModuleDefaults(knlohmann::json& js) 
{

 std::string defaultString = "{}";
 knlohmann::json defaultJs = knlohmann::json::parse(defaultString);
 mergeJson(js, defaultJs); 
 Continuous::applyModuleDefaults(js);
} 

void VRACER::applyVariableDefaults() 
{

 std::string defaultString = "{\"Initial Exploration Noise\": -1.0}";
 knlohmann::json defaultJs = knlohmann::json::parse(defaultString);
 if (isDefined(_k->_js.getJson(), "Variables"))
  for (size_t i = 0; i < _k->_js["Variables"].size(); i++) 
   mergeJson(_k->_js["Variables"][i], defaultJs); 
 Continuous::applyVariableDefaults();
} 

bool VRACER::checkTermination()
{
 bool hasFinished = false;

 hasFinished = hasFinished || Continuous::checkTermination();
 return hasFinished;
}

;

} //continuous
} //agent
} //solver
} //korali
;
