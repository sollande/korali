#include "engine.hpp"
#include "modules/experiment/experiment.hpp"
#include "modules/solver/deepSupervisor/deepSupervisor.hpp"
#include "sample/sample.hpp"
#include <omp.h>

namespace korali
{
namespace solver
{
;

void DeepSupervisor::initialize()
{
  // Getting problem pointer
  _problem = dynamic_cast<problem::SupervisedLearning *>(_k->_problem);

  // Don't reinitialize if experiment was already initialized
  if (_k->_isInitialized == true) return;

  /*****************************************************************
   * Setting up Neural Networks
   *****************************************************************/

  // Configuring neural network's inputs
  knlohmann::json neuralNetworkConfig;
  neuralNetworkConfig["Type"] = "Neural Network";
  neuralNetworkConfig["Engine"] = _neuralNetworkEngine;
  neuralNetworkConfig["Timestep Count"] = _problem->_maxTimesteps;

  // Iterator for the current layer id
  size_t curLayer = 0;

  // Setting the number of input layer nodes as number of input vector size
  neuralNetworkConfig["Layers"][curLayer]["Type"] = "Layer/Input";
  neuralNetworkConfig["Layers"][curLayer]["Output Channels"] = _problem->_inputSize;
  curLayer++;

  // Adding user-defined hidden layers
  for (size_t i = 0; i < _neuralNetworkHiddenLayers.size(); i++)
  {
    neuralNetworkConfig["Layers"][curLayer]["Weight Scaling"] = _outputWeightsScaling;
    neuralNetworkConfig["Layers"][curLayer] = _neuralNetworkHiddenLayers[i];
    curLayer++;
  }

  // Adding linear transformation layer to convert hidden state to match output channels
  neuralNetworkConfig["Layers"][curLayer]["Type"] = "Layer/Linear";
  neuralNetworkConfig["Layers"][curLayer]["Output Channels"] = _problem->_solutionSize;
  neuralNetworkConfig["Layers"][curLayer]["Weight Scaling"] = _outputWeightsScaling;
  curLayer++;

  // Applying a user-defined pre-activation function
  if (_neuralNetworkOutputActivation != "Identity")
  {
    neuralNetworkConfig["Layers"][curLayer]["Type"] = "Layer/Activation";
    neuralNetworkConfig["Layers"][curLayer]["Function"] = _neuralNetworkOutputActivation;
    curLayer++;
  }

  // Applying output layer configuration
  neuralNetworkConfig["Layers"][curLayer] = _neuralNetworkOutputLayer;
  neuralNetworkConfig["Layers"][curLayer]["Type"] = "Layer/Output";

  // Instancing training neural network
  auto trainingNeuralNetworkConfig = neuralNetworkConfig;
  trainingNeuralNetworkConfig["Batch Sizes"] = {_problem->_trainingBatchSize, _problem->_inferenceBatchSize};
  trainingNeuralNetworkConfig["Mode"] = "Training";
  _neuralNetwork = dynamic_cast<NeuralNetwork *>(getModule(trainingNeuralNetworkConfig, _k));
  _neuralNetwork->applyModuleDefaults(trainingNeuralNetworkConfig);
  _neuralNetwork->setConfiguration(trainingNeuralNetworkConfig);
  _neuralNetwork->initialize();

  /*****************************************************************
   * Initializing NN hyperparameters
   *****************************************************************/

  // If the hyperparameters have not been specified, produce new initial ones
  if (_hyperparameters.size() == 0) _hyperparameters = _neuralNetwork->generateInitialHyperparameters();

  /*****************************************************************
   * Setting up weight and bias optimization experiment
   *****************************************************************/

  if (_neuralNetworkOptimizer == "Adam") _optimizer = new korali::fAdam(_hyperparameters.size());
  if (_neuralNetworkOptimizer == "AdaBelief") _optimizer = new korali::fAdaBelief(_hyperparameters.size());
  if (_neuralNetworkOptimizer == "MADGRAD") _optimizer = new korali::fMadGrad(_hyperparameters.size());
  if (_neuralNetworkOptimizer == "RMSProp") _optimizer = new korali::fRMSProp(_hyperparameters.size());
  if (_neuralNetworkOptimizer == "Adagrad") _optimizer = new korali::fAdagrad(_hyperparameters.size());

  // Setting hyperparameter structures in the neural network and optmizer
  setHyperparameters(_hyperparameters);

  // Resetting Optimizer
  _optimizer->reset();

  // Setting current loss
  _currentLoss = 0.0f;
}

void DeepSupervisor::runGeneration()
{
  // Grabbing constants
  const size_t N = _problem->_trainingBatchSize;
  const size_t OC = _problem->_solutionSize;

  // Updating optimizer's learning rate, in case it changed
  _optimizer->_eta = _learningRate;

  for (size_t step = 0; step < _stepsPerGeneration; step++)
  {
    // If we use an MSE loss function, we need to update the gradient vector with its difference with each of batch's last timestep of the NN output
    if (_lossFunction == "Mean Squared Error")
    {
      // Checking that incoming data has a correct format
      _problem->verifyData();

      // Creating gradient vector
      auto gradientVector = _problem->_solutionData;

      // Forward propagating the input values through the training neural network
      _neuralNetwork->forward(_problem->_inputData);

      // Getting a reference to the neural network output
      const auto &results = _neuralNetwork->getOutputValues(N);

      // Calculating gradients via the loss function
      for (size_t b = 0; b < N; b++)
        for (size_t i = 0; i < OC; i++)
          gradientVector[b][i] = gradientVector[b][i] - results[b][i];

      // Backward propagating the gradients through the training neural network
      _neuralNetwork->backward(gradientVector);

      // Calculating loss across the batch size
      _currentLoss = 0.0;
      for (size_t b = 0; b < N; b++)
        for (size_t i = 0; i < OC; i++)
          _currentLoss += gradientVector[b][i] * gradientVector[b][i];
      _currentLoss = _currentLoss / ((float)N * 2.0f);
    }

    // If using direct gradient, backward propagating the gradients directly through the training neural network
    if (_lossFunction == "Direct Gradient")
      _neuralNetwork->backward(_problem->_solutionData);

    // Getting hyperparameter gradients
    auto nnHyperparameterGradients = _neuralNetwork->getHyperparameterGradients(N);

    // Apply gradient of L2 regularizer
    if (_l2RegularizationEnabled)
    {
      const auto nnHyperparameters = _neuralNetwork->getHyperparameters();
#pragma omp parallel for simd
      for (size_t i = 0; i < nnHyperparameterGradients.size(); ++i)
        nnHyperparameterGradients[i] -= _l2RegularizationImportance * nnHyperparameters[i];
    }

    // Passing hyperparameter gradients through an ADAM update
    _optimizer->processResult(nnHyperparameterGradients);

    // Getting new set of hyperparameters from Adam
    _neuralNetwork->setHyperparameters(_optimizer->_currentValue);
  }
}

std::vector<float> DeepSupervisor::getHyperparameters()
{
  return _neuralNetwork->getHyperparameters();
}

void DeepSupervisor::setHyperparameters(const std::vector<float> &hyperparameters)
{
  // Update evaluation network
  _neuralNetwork->setHyperparameters(hyperparameters);

  // Updating optimizer's current value
  _optimizer->_currentValue = hyperparameters;
}

std::vector<std::vector<float>> &DeepSupervisor::getEvaluation(const std::vector<std::vector<std::vector<float>>> &input)
{
  // Grabbing constants
  const size_t N = input.size();

  // Running the input values through the neural network
  _neuralNetwork->forward(input);

  // Returning the output values for the last given timestep
  return _neuralNetwork->getOutputValues(N);
}

void DeepSupervisor::printGenerationAfter()
{
  // Printing results so far
  if (_lossFunction == "Mean Squared Error") _k->_logger->logInfo("Normal", " + Training Loss: %.15f\n", _currentLoss);
  if (_lossFunction == "Direct Gradient") _k->_logger->logInfo("Normal", " + Gradient L2-Norm: %.15f\n", std::sqrt(_currentLoss));
}

void DeepSupervisor::setConfiguration(knlohmann::json& js) 
{
 if (isDefined(js, "Results"))  eraseValue(js, "Results");

 if (isDefined(js, "Current Loss"))
 {
 try { _currentLoss = js["Current Loss"].get<float>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ deepSupervisor ] \n + Key:    ['Current Loss']\n%s", e.what()); } 
   eraseValue(js, "Current Loss");
 }

 if (isDefined(js, "Normalization Means"))
 {
 try { _normalizationMeans = js["Normalization Means"].get<std::vector<float>>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ deepSupervisor ] \n + Key:    ['Normalization Means']\n%s", e.what()); } 
   eraseValue(js, "Normalization Means");
 }

 if (isDefined(js, "Normalization Variances"))
 {
 try { _normalizationVariances = js["Normalization Variances"].get<std::vector<float>>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ deepSupervisor ] \n + Key:    ['Normalization Variances']\n%s", e.what()); } 
   eraseValue(js, "Normalization Variances");
 }

 if (isDefined(js, "Neural Network", "Hidden Layers"))
 {
 _neuralNetworkHiddenLayers = js["Neural Network"]["Hidden Layers"].get<knlohmann::json>();

   eraseValue(js, "Neural Network", "Hidden Layers");
 }
  else   KORALI_LOG_ERROR(" + No value provided for mandatory setting: ['Neural Network']['Hidden Layers'] required by deepSupervisor.\n"); 

 if (isDefined(js, "Neural Network", "Output Activation"))
 {
 _neuralNetworkOutputActivation = js["Neural Network"]["Output Activation"].get<knlohmann::json>();

   eraseValue(js, "Neural Network", "Output Activation");
 }
  else   KORALI_LOG_ERROR(" + No value provided for mandatory setting: ['Neural Network']['Output Activation'] required by deepSupervisor.\n"); 

 if (isDefined(js, "Neural Network", "Output Layer"))
 {
 _neuralNetworkOutputLayer = js["Neural Network"]["Output Layer"].get<knlohmann::json>();

   eraseValue(js, "Neural Network", "Output Layer");
 }
  else   KORALI_LOG_ERROR(" + No value provided for mandatory setting: ['Neural Network']['Output Layer'] required by deepSupervisor.\n"); 

 if (isDefined(js, "Neural Network", "Engine"))
 {
 try { _neuralNetworkEngine = js["Neural Network"]["Engine"].get<std::string>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ deepSupervisor ] \n + Key:    ['Neural Network']['Engine']\n%s", e.what()); } 
   eraseValue(js, "Neural Network", "Engine");
 }
  else   KORALI_LOG_ERROR(" + No value provided for mandatory setting: ['Neural Network']['Engine'] required by deepSupervisor.\n"); 

 if (isDefined(js, "Neural Network", "Optimizer"))
 {
 try { _neuralNetworkOptimizer = js["Neural Network"]["Optimizer"].get<std::string>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ deepSupervisor ] \n + Key:    ['Neural Network']['Optimizer']\n%s", e.what()); } 
{
 bool validOption = false; 
 if (_neuralNetworkOptimizer == "Adam") validOption = true; 
 if (_neuralNetworkOptimizer == "AdaBelief") validOption = true; 
 if (_neuralNetworkOptimizer == "MADGRAD") validOption = true; 
 if (_neuralNetworkOptimizer == "RMSProp") validOption = true; 
 if (_neuralNetworkOptimizer == "Adagrad") validOption = true; 
 if (validOption == false) KORALI_LOG_ERROR(" + Unrecognized value (%s) provided for mandatory setting: ['Neural Network']['Optimizer'] required by deepSupervisor.\n", _neuralNetworkOptimizer.c_str()); 
}
   eraseValue(js, "Neural Network", "Optimizer");
 }
  else   KORALI_LOG_ERROR(" + No value provided for mandatory setting: ['Neural Network']['Optimizer'] required by deepSupervisor.\n"); 

 if (isDefined(js, "Hyperparameters"))
 {
 try { _hyperparameters = js["Hyperparameters"].get<std::vector<float>>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ deepSupervisor ] \n + Key:    ['Hyperparameters']\n%s", e.what()); } 
   eraseValue(js, "Hyperparameters");
 }
  else   KORALI_LOG_ERROR(" + No value provided for mandatory setting: ['Hyperparameters'] required by deepSupervisor.\n"); 

 if (isDefined(js, "Loss Function"))
 {
 try { _lossFunction = js["Loss Function"].get<std::string>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ deepSupervisor ] \n + Key:    ['Loss Function']\n%s", e.what()); } 
{
 bool validOption = false; 
 if (_lossFunction == "Direct Gradient") validOption = true; 
 if (_lossFunction == "Mean Squared Error") validOption = true; 
 if (validOption == false) KORALI_LOG_ERROR(" + Unrecognized value (%s) provided for mandatory setting: ['Loss Function'] required by deepSupervisor.\n", _lossFunction.c_str()); 
}
   eraseValue(js, "Loss Function");
 }
  else   KORALI_LOG_ERROR(" + No value provided for mandatory setting: ['Loss Function'] required by deepSupervisor.\n"); 

 if (isDefined(js, "Steps Per Generation"))
 {
 try { _stepsPerGeneration = js["Steps Per Generation"].get<size_t>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ deepSupervisor ] \n + Key:    ['Steps Per Generation']\n%s", e.what()); } 
   eraseValue(js, "Steps Per Generation");
 }
  else   KORALI_LOG_ERROR(" + No value provided for mandatory setting: ['Steps Per Generation'] required by deepSupervisor.\n"); 

 if (isDefined(js, "Learning Rate"))
 {
 try { _learningRate = js["Learning Rate"].get<float>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ deepSupervisor ] \n + Key:    ['Learning Rate']\n%s", e.what()); } 
   eraseValue(js, "Learning Rate");
 }
  else   KORALI_LOG_ERROR(" + No value provided for mandatory setting: ['Learning Rate'] required by deepSupervisor.\n"); 

 if (isDefined(js, "L2 Regularization", "Enabled"))
 {
 try { _l2RegularizationEnabled = js["L2 Regularization"]["Enabled"].get<int>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ deepSupervisor ] \n + Key:    ['L2 Regularization']['Enabled']\n%s", e.what()); } 
   eraseValue(js, "L2 Regularization", "Enabled");
 }
  else   KORALI_LOG_ERROR(" + No value provided for mandatory setting: ['L2 Regularization']['Enabled'] required by deepSupervisor.\n"); 

 if (isDefined(js, "L2 Regularization", "Importance"))
 {
 try { _l2RegularizationImportance = js["L2 Regularization"]["Importance"].get<int>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ deepSupervisor ] \n + Key:    ['L2 Regularization']['Importance']\n%s", e.what()); } 
   eraseValue(js, "L2 Regularization", "Importance");
 }
  else   KORALI_LOG_ERROR(" + No value provided for mandatory setting: ['L2 Regularization']['Importance'] required by deepSupervisor.\n"); 

 if (isDefined(js, "Output Weights Scaling"))
 {
 try { _outputWeightsScaling = js["Output Weights Scaling"].get<float>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ deepSupervisor ] \n + Key:    ['Output Weights Scaling']\n%s", e.what()); } 
   eraseValue(js, "Output Weights Scaling");
 }
  else   KORALI_LOG_ERROR(" + No value provided for mandatory setting: ['Output Weights Scaling'] required by deepSupervisor.\n"); 

 if (isDefined(js, "Termination Criteria", "Target Loss"))
 {
 try { _targetLoss = js["Termination Criteria"]["Target Loss"].get<float>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ deepSupervisor ] \n + Key:    ['Termination Criteria']['Target Loss']\n%s", e.what()); } 
   eraseValue(js, "Termination Criteria", "Target Loss");
 }
  else   KORALI_LOG_ERROR(" + No value provided for mandatory setting: ['Termination Criteria']['Target Loss'] required by deepSupervisor.\n"); 

 if (isDefined(_k->_js.getJson(), "Variables"))
 for (size_t i = 0; i < _k->_js["Variables"].size(); i++) { 
 } 
 Solver::setConfiguration(js);
 _type = "deepSupervisor";
 if(isDefined(js, "Type")) eraseValue(js, "Type");
 if(isEmpty(js) == false) KORALI_LOG_ERROR(" + Unrecognized settings for Korali module: deepSupervisor: \n%s\n", js.dump(2).c_str());
} 

void DeepSupervisor::getConfiguration(knlohmann::json& js) 
{

 js["Type"] = _type;
   js["Neural Network"]["Hidden Layers"] = _neuralNetworkHiddenLayers;
   js["Neural Network"]["Output Activation"] = _neuralNetworkOutputActivation;
   js["Neural Network"]["Output Layer"] = _neuralNetworkOutputLayer;
   js["Neural Network"]["Engine"] = _neuralNetworkEngine;
   js["Neural Network"]["Optimizer"] = _neuralNetworkOptimizer;
   js["Hyperparameters"] = _hyperparameters;
   js["Loss Function"] = _lossFunction;
   js["Steps Per Generation"] = _stepsPerGeneration;
   js["Learning Rate"] = _learningRate;
   js["L2 Regularization"]["Enabled"] = _l2RegularizationEnabled;
   js["L2 Regularization"]["Importance"] = _l2RegularizationImportance;
   js["Output Weights Scaling"] = _outputWeightsScaling;
   js["Termination Criteria"]["Target Loss"] = _targetLoss;
   js["Current Loss"] = _currentLoss;
   js["Normalization Means"] = _normalizationMeans;
   js["Normalization Variances"] = _normalizationVariances;
 for (size_t i = 0; i <  _k->_variables.size(); i++) { 
 } 
 Solver::getConfiguration(js);
} 

void DeepSupervisor::applyModuleDefaults(knlohmann::json& js) 
{

 std::string defaultString = "{\"Steps Per Generation\": 1, \"L2 Regularization\": {\"Enabled\": false, \"Importance\": 0.0001}, \"Neural Network\": {\"Output Activation\": \"Identity\", \"Output Layer\": {}}, \"Termination Criteria\": {\"Target Loss\": -1.0}, \"Hyperparameters\": [], \"Output Weights Scaling\": 1.0}";
 knlohmann::json defaultJs = knlohmann::json::parse(defaultString);
 mergeJson(js, defaultJs); 
 Solver::applyModuleDefaults(js);
} 

void DeepSupervisor::applyVariableDefaults() 
{

 std::string defaultString = "{}";
 knlohmann::json defaultJs = knlohmann::json::parse(defaultString);
 if (isDefined(_k->_js.getJson(), "Variables"))
  for (size_t i = 0; i < _k->_js["Variables"].size(); i++) 
   mergeJson(_k->_js["Variables"][i], defaultJs); 
 Solver::applyVariableDefaults();
} 

bool DeepSupervisor::checkTermination()
{
 bool hasFinished = false;

 if ((_k->_currentGeneration > 1) && (_targetLoss > 0.0) && (_currentLoss <= _targetLoss))
 {
  _terminationCriteria.push_back("deepSupervisor['Target Loss'] = " + std::to_string(_targetLoss) + ".");
  hasFinished = true;
 }

 hasFinished = hasFinished || Solver::checkTermination();
 return hasFinished;
}

;

} //solver
} //korali
;
