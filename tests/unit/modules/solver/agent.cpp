#include "gtest/gtest.h"
#include "korali.hpp"
#include "modules/problem/reinforcementLearning/reinforcementLearning.hpp"
#include "modules/problem/reinforcementLearning/continuous/continuous.hpp"
#include "modules/problem/reinforcementLearning/discrete/discrete.hpp"
#include "modules/solver/agent/continuous/VRACER/VRACER.hpp"
#include "modules/solver/agent/discrete/dVRACER/dVRACER.hpp"
#include "modules/conduit/sequential/sequential.hpp"

namespace korali
{
 namespace problem
 {
  extern Sample *__currentSample;
  extern size_t __envFunctionId;
  extern solver::Agent *_agent;
  extern Conduit* _conduit;
  extern cothread_t _envThread;
  extern size_t _launchId;
  extern void __environmentWrapper();
 }
}

namespace
{
 using namespace korali;
 using namespace korali::solver;
 using namespace korali::solver::agent;
 using namespace korali::solver::agent::continuous;
 using namespace korali::solver::agent::discrete;
 using namespace korali::problem;
 using namespace korali::problem::reinforcementLearning;
 using namespace korali::conduit;


 //////////////// Base Agent CLASS ////////////////////////

 TEST(a, baseAgent)
 {
  // Creating base experiment
  Experiment e;
  e._logger = new Logger("Detailed", stdout);
  auto& experimentJs = e._js.getJson();
  experimentJs["Variables"][0]["Name"] = "X";
  Variable v;
  e._variables.push_back(&v);

  // Creating optimizer configuration Json
  knlohmann::json agentJs;

  // Configuring Problem
  e["Problem"]["Type"] = "Reinforcement Learning / Continuous";
  reinforcementLearning::Continuous* pC;
  knlohmann::json problemRefJs;

  problemRefJs["Type"] = "Reinforcement Learning / Continuous";
  problemRefJs["Environment Function"] = [](Sample &s){};

  e["Variables"][0]["Name"] = "State0";
  e["Variables"][1]["Name"] = "Action0";
  e["Variables"][1]["Type"] = "Action";
  e["Variables"][1]["Initial Exploration Noise"] = 0.45;
  e["Variables"][1]["Lower Bound"] = 0.00;
  e["Variables"][1]["Upper Bound"] = 1.00;

  e["Solver"]["Type"] = "Agent / Continuous / VRACER";

  Variable vState;
  vState._name = "State0";
  vState._type = "State";

  Variable vAction;
  vAction._name = "Action0";
  vAction._type = "Action";
  vAction._initialExplorationNoise = 0.45;
  vAction._lowerBound = 0.00;
  vAction._upperBound = 1.00;

  e._variables.push_back(&vState);
  e._variables.push_back(&vAction);

  ASSERT_NO_THROW(pC = dynamic_cast<reinforcementLearning::Continuous *>(Module::getModule(problemRefJs, &e)));

  // Defaults should be applied without a problem
  ASSERT_NO_THROW(pC->applyModuleDefaults(problemRefJs));

  // Covering variable functions (no effect)
  ASSERT_NO_THROW(pC->applyVariableDefaults());

  // Setting up problem correctly
  ASSERT_NO_THROW(pC->setConfiguration(problemRefJs));

  // Intitialize problem
  e._problem = pC;
  ASSERT_NO_THROW(pC->initialize());

  // Using a agent solver

  agentJs["Type"] = "Agent / Continuous / VRACER";
  agentJs["Mode"] = "Training";
  agentJs["Episodes Per Generation"] = 10;
  agentJs["Experiences Between Policy Updates"] = 1;
  agentJs["Discount Factor"] = 0.99;
  agentJs["Learning Rate"] = 0.0001;
  agentJs["Mini Batch"]["Size"] = 32;
  agentJs["Experience Replay"]["Start Size"] = 1000;
  agentJs["Experience Replay"]["Maximum Size"] = 10000;

  /// Configuring the neural network and its hidden layers

  agentJs["Neural Network"]["Engine"] = "OneDNN";
  agentJs["Neural Network"]["Optimizer"] = "Adam";

  agentJs["Neural Network"]["Hidden Layers"][0]["Type"] = "Layer/Linear";
  agentJs["Neural Network"]["Hidden Layers"][0]["Output Channels"] = 16;

  // Creating module
  VRACER* a;
  ASSERT_NO_THROW(a = dynamic_cast<VRACER *>(Module::getModule(agentJs, &e)));

  // Defaults should be applied without a problem
  ASSERT_NO_THROW(a->applyModuleDefaults(agentJs));

  // Covering variable functions (no effect)
  ASSERT_NO_THROW(a->applyVariableDefaults());

  // Backup the correct base configuration
  auto baseOptJs = agentJs;
  auto baseExpJs = experimentJs;

  // Setting up optimizer correctly
  agentJs = baseOptJs;
  experimentJs = baseExpJs;
  ASSERT_NO_THROW(a->setConfiguration(agentJs));

  // Running initial configuration correctly
  ASSERT_NO_THROW(a->initialize());

  // Case with no ER size maximum
  a->_experienceReplayMaximumSize = 0;
  ASSERT_NO_THROW(a->initialize());
  ASSERT_NE(a->_experienceReplayMaximumSize, 0);

  // Case with no ER size maximum
  a->_experienceReplayStartSize = 0;
  ASSERT_NO_THROW(a->initialize());
  ASSERT_NE(a->_experienceReplayStartSize, 0);

  // Case Cutoff scale
  a->_experienceReplayOffPolicyCutoffScale = -1.0f;
  ASSERT_ANY_THROW(a->initialize());
  a->_experienceReplayOffPolicyCutoffScale = 1.0f;

  // Case testing with testing best curPolicy empty
  a->_mode = "Testing";
  a->_testingSampleIds = std::vector<size_t>();
  ASSERT_ANY_THROW(a->initialize()); // No sample ids defined

  a->_testingSampleIds = std::vector<size_t>({1});
  a->_trainingCurrentPolicy = std::vector<float>({1.0});
  ASSERT_NO_THROW(a->initialize());

  // Testing Process Episode corner cases
  knlohmann::json episode;
  episode["Environment Id"] = 0;
  episode["Experiences"][0]["State"] = std::vector<float>({0.0f});
  episode["Experiences"][0]["Action"] = std::vector<float>({0.0f});
  episode["Experiences"][0]["Reward"] = 1.0f;
  episode["Experiences"][0]["Termination"] = "Terminal";
  episode["Experiences"][0]["Policy"]["State Value"] = 1.0;
  a->processEpisode(episode);
  // ASSERT_NO_THROW(a->processEpisode(episode));

  // No state value provided error
  episode["Experiences"][0]["Policy"].erase("State Value");
  ASSERT_ANY_THROW(a->processEpisode(episode));
  episode["Experiences"][0]["Policy"]["State Value"] = 1.0;

  // Correct handling of truncated state
  episode["Experiences"][0]["Termination"] = "Truncated";
  episode["Experiences"][0]["Truncated State"] = std::vector<float>({0.0f});
  ASSERT_NO_THROW(a->processEpisode(episode));

  // Correct handling of truncated state
  episode["Experiences"][0]["Termination"] = "Truncated";
  episode["Experiences"][0]["Truncated State"] = std::vector<float>({std::numeric_limits<float>::infinity()});
  ASSERT_ANY_THROW(a->processEpisode(episode));
  episode["Experiences"][0]["Truncated State"] = std::vector<float>({0.0f});

  // Check truncated state sequence for sequences > 1
  episode["Experiences"][0]["Environment Id"] = 0;
  episode["Experiences"][0]["State"] = std::vector<float>({0.0f});
  episode["Experiences"][0]["Action"] = std::vector<float>({0.0f});
  episode["Experiences"][0]["Reward"] = 1.0f;
  episode["Experiences"][0]["Termination"] = "Non Terminal";
  episode["Experiences"][0]["Policy"]["State Value"] = 1.0;

  episode["Experiences"][1]["Environment Id"] = 0;
  episode["Experiences"][1]["State"] = std::vector<float>({0.0f});
  episode["Experiences"][1]["Action"] = std::vector<float>({0.0f});
  episode["Experiences"][1]["Reward"] = 1.0f;
  episode["Experiences"][1]["Termination"] = "Non Terminal";
  episode["Experiences"][1]["Policy"]["State Value"] = 1.0;

  episode["Experiences"][2]["Environment Id"] = 0;
  episode["Experiences"][2]["State"] = std::vector<float>({0.0f});
  episode["Experiences"][2]["Action"] = std::vector<float>({0.0f});
  episode["Experiences"][2]["Reward"] = 1.0f;
  episode["Experiences"][2]["Termination"] = "Truncated";
  episode["Experiences"][2]["Policy"]["State Value"] = 1.0;
  episode["Experiences"][2]["Truncated State"] = std::vector<float>({0.0f});

  ASSERT_NO_THROW(a->processEpisode(episode));
  a->_timeSequenceLength = 2;
  ASSERT_NO_THROW(a->getTruncatedStateSequence(a->_terminationBuffer.size()-1));

  // Triggering bad path in serialization routine
  e._fileOutputPath = "/dev/null/\%*Incorrect Path*";
  ASSERT_ANY_THROW(a->serializeExperienceReplay());
  ASSERT_ANY_THROW(a->deserializeExperienceReplay());

  // Some specific printing cases
  a->_mode = "Training";
  a->_maxEpisodes = 0;
  ASSERT_NO_THROW(a->printGenerationAfter());
  a->_maxEpisodes = 1;
  ASSERT_NO_THROW(a->printGenerationAfter());
  a->_maxExperiences = 0;
  ASSERT_NO_THROW(a->printGenerationAfter());
  a->_maxExperiences = 1;
  ASSERT_NO_THROW(a->printGenerationAfter());
  a->_maxPolicyUpdates = 0;
  ASSERT_NO_THROW(a->printGenerationAfter());
  a->_maxPolicyUpdates = 1;
  ASSERT_NO_THROW(a->printGenerationAfter());

  // Testing optional parameters
  agentJs = baseOptJs;
  experimentJs = baseExpJs;
  agentJs["Action Lower Bounds"] = "Not a Number";
  ASSERT_ANY_THROW(a->setConfiguration(agentJs));

  agentJs = baseOptJs;
  experimentJs = baseExpJs;
  agentJs["Action Lower Bounds"] = std::vector<float>({0.0});
  ASSERT_NO_THROW(a->setConfiguration(agentJs));

  agentJs = baseOptJs;
  experimentJs = baseExpJs;
  agentJs["Action Upper Bounds"] = "Not a Number";
  ASSERT_ANY_THROW(a->setConfiguration(agentJs));

  agentJs = baseOptJs;
  experimentJs = baseExpJs;
  agentJs["Action Upper Bounds"] = std::vector<float>({0.0});
  ASSERT_NO_THROW(a->setConfiguration(agentJs));

  agentJs = baseOptJs;
  experimentJs = baseExpJs;
  agentJs["Current Episode"] = "Not a Number";
  ASSERT_ANY_THROW(a->setConfiguration(agentJs));

  agentJs = baseOptJs;
  experimentJs = baseExpJs;
  agentJs["Current Episode"] = 1;
  ASSERT_NO_THROW(a->setConfiguration(agentJs));

  agentJs = baseOptJs;
  experimentJs = baseExpJs;
  agentJs["Training"]["Reward History"] = "Not a Number";
  ASSERT_ANY_THROW(a->setConfiguration(agentJs));

  agentJs = baseOptJs;
  experimentJs = baseExpJs;
  agentJs["Training"]["Reward History"] = std::vector<float>({1.0});
  ASSERT_NO_THROW(a->setConfiguration(agentJs));

  agentJs = baseOptJs;
  experimentJs = baseExpJs;
  agentJs["Training"]["Experience History"] = "Not a Number";
  ASSERT_ANY_THROW(a->setConfiguration(agentJs));

  agentJs = baseOptJs;
  experimentJs = baseExpJs;
  agentJs["Training"]["Experience History"] = std::vector<float>({1});
  ASSERT_NO_THROW(a->setConfiguration(agentJs));

  agentJs = baseOptJs;
  experimentJs = baseExpJs;
  agentJs["Training"]["Average Reward"] = "Not a Number";
  ASSERT_ANY_THROW(a->setConfiguration(agentJs));

  agentJs = baseOptJs;
  experimentJs = baseExpJs;
  agentJs["Training"]["Average Reward"] = 1.0;
  ASSERT_NO_THROW(a->setConfiguration(agentJs));

  agentJs = baseOptJs;
  experimentJs = baseExpJs;
  agentJs["Training"]["Last Reward"] = "Not a Number";
  ASSERT_ANY_THROW(a->setConfiguration(agentJs));

  agentJs = baseOptJs;
  experimentJs = baseExpJs;
  agentJs["Training"]["Last Reward"] = 1.0;
  ASSERT_NO_THROW(a->setConfiguration(agentJs));

  agentJs = baseOptJs;
  experimentJs = baseExpJs;
  agentJs["Training"]["Best Reward"] = "Not a Number";
  ASSERT_ANY_THROW(a->setConfiguration(agentJs));

  agentJs = baseOptJs;
  experimentJs = baseExpJs;
  agentJs["Training"]["Best Reward"] = 1.0;
  ASSERT_NO_THROW(a->setConfiguration(agentJs));

  agentJs = baseOptJs;
  experimentJs = baseExpJs;
  agentJs["Training"]["Best Episode Id"] = "Not a Number";
  ASSERT_ANY_THROW(a->setConfiguration(agentJs));

  agentJs = baseOptJs;
  experimentJs = baseExpJs;
  agentJs["Training"]["Best Episode Id"] = 1;
  ASSERT_NO_THROW(a->setConfiguration(agentJs));

  agentJs = baseOptJs;
  experimentJs = baseExpJs;
  agentJs["Testing"]["Reward"] = "Not a Number";
  ASSERT_ANY_THROW(a->setConfiguration(agentJs));

  agentJs = baseOptJs;
  experimentJs = baseExpJs;
  agentJs["Testing"]["Reward"] = std::vector<float>({1.0});
  ASSERT_NO_THROW(a->setConfiguration(agentJs));

  agentJs = baseOptJs;
  experimentJs = baseExpJs;
  agentJs["Testing"]["Best Episode Id"] = "Not a Number";
  ASSERT_ANY_THROW(a->setConfiguration(agentJs));
  
  agentJs = baseOptJs;
  experimentJs = baseExpJs;
  agentJs["Testing"]["Candidate Count"] = "Not a Number";
  ASSERT_ANY_THROW(a->setConfiguration(agentJs));
  
  agentJs = baseOptJs;
  experimentJs = baseExpJs;
  agentJs["Testing"]["Best Reward"] = "Not a Number";
  ASSERT_ANY_THROW(a->setConfiguration(agentJs));

  agentJs = baseOptJs;
  experimentJs = baseExpJs;
  agentJs["Testing"]["Worst Reward"] = "Not a Number";
  ASSERT_ANY_THROW(a->setConfiguration(agentJs));

  agentJs = baseOptJs;
  experimentJs = baseExpJs;
  agentJs["Testing"]["Average Reward"] = "Not a Number";
  ASSERT_ANY_THROW(a->setConfiguration(agentJs));
 
  agentJs = baseOptJs;
  experimentJs = baseExpJs;
  agentJs["Testing"]["Average Reward History"] = "Not a Vector";
  ASSERT_ANY_THROW(a->setConfiguration(agentJs));

  agentJs = baseOptJs;
  experimentJs = baseExpJs;
  agentJs["Testing"]["Best Average Reward"] = "Not a Number";
  ASSERT_ANY_THROW(a->setConfiguration(agentJs));
  
  agentJs = baseOptJs;
  experimentJs = baseExpJs;
  agentJs["Experience Replay"]["Off Policy"]["Count"] = "Not a Number";
  ASSERT_ANY_THROW(a->setConfiguration(agentJs));

  agentJs = baseOptJs;
  experimentJs = baseExpJs;
  agentJs["Experience Replay"]["Off Policy"]["Count"] = 1;
  ASSERT_NO_THROW(a->setConfiguration(agentJs));

  agentJs = baseOptJs;
  experimentJs = baseExpJs;
  agentJs["Experience Replay"]["Off Policy"]["Ratio"] = "Not a Number";
  ASSERT_ANY_THROW(a->setConfiguration(agentJs));

  agentJs = baseOptJs;
  experimentJs = baseExpJs;
  agentJs["Experience Replay"]["Off Policy"]["Ratio"] = 1.0;
  ASSERT_NO_THROW(a->setConfiguration(agentJs));

  agentJs = baseOptJs;
  experimentJs = baseExpJs;
  agentJs["Experience Replay"]["Off Policy"]["Current Cutoff"] = "Not a Number";
  ASSERT_ANY_THROW(a->setConfiguration(agentJs));

  agentJs = baseOptJs;
  experimentJs = baseExpJs;
  agentJs["Experience Replay"]["Off Policy"]["Current Cutoff"] = 1.0;
  ASSERT_NO_THROW(a->setConfiguration(agentJs));

  agentJs = baseOptJs;
  experimentJs = baseExpJs;
  agentJs["Current Learning Rate"] = "Not a Number";
  ASSERT_ANY_THROW(a->setConfiguration(agentJs));

  agentJs = baseOptJs;
  experimentJs = baseExpJs;
  agentJs["Current Learning Rate"] = 1.0;
  ASSERT_NO_THROW(a->setConfiguration(agentJs));

  agentJs = baseOptJs;
  experimentJs = baseExpJs;
  agentJs["Policy Update Count"] = "Not a Number";
  ASSERT_ANY_THROW(a->setConfiguration(agentJs));

  agentJs = baseOptJs;
  experimentJs = baseExpJs;
  agentJs["Policy Update Count"] = 1;
  ASSERT_NO_THROW(a->setConfiguration(agentJs));

  agentJs = baseOptJs;
  experimentJs = baseExpJs;
  agentJs["Current Sample ID"] = "Not a Number";
  ASSERT_ANY_THROW(a->setConfiguration(agentJs));

  agentJs = baseOptJs;
  experimentJs = baseExpJs;
  agentJs["Current Sample ID"] = 1;
  ASSERT_NO_THROW(a->setConfiguration(agentJs));

  agentJs = baseOptJs;
  experimentJs = baseExpJs;
  agentJs["Experience Count Per Environment"] = "Not a Number";
  ASSERT_ANY_THROW(a->setConfiguration(agentJs));

  agentJs = baseOptJs;
  experimentJs = baseExpJs;
  agentJs["Experience Count Per Environment"] = std::vector<size_t>({1});
  ASSERT_NO_THROW(a->setConfiguration(agentJs));

  agentJs = baseOptJs;
  experimentJs = baseExpJs;
  agentJs["Experience Count"] = "Not a Number";
  ASSERT_ANY_THROW(a->setConfiguration(agentJs));

  agentJs = baseOptJs;
  experimentJs = baseExpJs;
  agentJs["Experience Count"] = 1;
  ASSERT_NO_THROW(a->setConfiguration(agentJs));

  agentJs = baseOptJs;
  experimentJs = baseExpJs;
  agentJs["Reward"]["Rescaling"]["Sum Squared Rewards"] = "Not a Number";
  ASSERT_ANY_THROW(a->setConfiguration(agentJs));

  agentJs = baseOptJs;
  experimentJs = baseExpJs;
  agentJs["Reward"]["Rescaling"]["Sum Squared Rewards"] = std::vector<float>({1.0});
  ASSERT_NO_THROW(a->setConfiguration(agentJs));

  agentJs = baseOptJs;
  experimentJs = baseExpJs;
  agentJs["Reward"]["Rescaling"]["Sigma"] = "Not a Number";
  ASSERT_ANY_THROW(a->setConfiguration(agentJs));

  agentJs = baseOptJs;
  experimentJs = baseExpJs;
  agentJs["Reward"]["Rescaling"]["Sigma"] = std::vector<float>({1.0});
  ASSERT_NO_THROW(a->setConfiguration(agentJs));

  agentJs = baseOptJs;
  experimentJs = baseExpJs;
  agentJs["State Rescaling"]["Means"] = "Not a Number";
  ASSERT_ANY_THROW(a->setConfiguration(agentJs));

  agentJs = baseOptJs;
  experimentJs = baseExpJs;
  agentJs["State Rescaling"]["Means"] = std::vector<float>({1.0});
  ASSERT_NO_THROW(a->setConfiguration(agentJs));

  agentJs = baseOptJs;
  experimentJs = baseExpJs;
  agentJs["State Rescaling"]["Sigmas"] = "Not a Number";
  ASSERT_ANY_THROW(a->setConfiguration(agentJs));

  agentJs = baseOptJs;
  experimentJs = baseExpJs;
  agentJs["State Rescaling"]["Sigmas"] = std::vector<float>({1.0});
  ASSERT_NO_THROW(a->setConfiguration(agentJs));

  // Testing mandatory parameters

  agentJs = baseOptJs;
  experimentJs = baseExpJs;
  agentJs.erase("Mode");
  ASSERT_ANY_THROW(a->setConfiguration(agentJs));

  agentJs = baseOptJs;
  experimentJs = baseExpJs;
  agentJs["Mode"] = 1.0;
  ASSERT_ANY_THROW(a->setConfiguration(agentJs));

  agentJs = baseOptJs;
  experimentJs = baseExpJs;
  agentJs["Mode"] = "Training";
  ASSERT_NO_THROW(a->setConfiguration(agentJs));

  agentJs = baseOptJs;
  experimentJs = baseExpJs;
  agentJs.erase("Concurrent Workers");
  ASSERT_ANY_THROW(a->setConfiguration(agentJs));

  agentJs = baseOptJs;
  experimentJs = baseExpJs;
  agentJs["Concurrent Workers"] = "Not a Number";
  ASSERT_ANY_THROW(a->setConfiguration(agentJs));

  agentJs = baseOptJs;
  experimentJs = baseExpJs;
  agentJs["Concurrent Workers"] = 1;
  ASSERT_NO_THROW(a->setConfiguration(agentJs));

  agentJs = baseOptJs;
  experimentJs = baseExpJs;
  agentJs.erase("Episodes Per Generation");
  ASSERT_ANY_THROW(a->setConfiguration(agentJs));

  agentJs = baseOptJs;
  experimentJs = baseExpJs;
  agentJs["Episodes Per Generation"] = "Not a Number";
  ASSERT_ANY_THROW(a->setConfiguration(agentJs));

  agentJs = baseOptJs;
  experimentJs = baseExpJs;
  agentJs["Episodes Per Generation"] = 1;
  ASSERT_NO_THROW(a->setConfiguration(agentJs));

  agentJs = baseOptJs;
  experimentJs = baseExpJs;
  agentJs["Mini Batch"].erase("Size");
  ASSERT_ANY_THROW(a->setConfiguration(agentJs));

  agentJs = baseOptJs;
  experimentJs = baseExpJs;
  agentJs["Mini Batch"]["Size"] = "Not a Number";
  ASSERT_ANY_THROW(a->setConfiguration(agentJs));

  agentJs = baseOptJs;
  experimentJs = baseExpJs;
  agentJs["Mini Batch"]["Size"] = 1;
  ASSERT_NO_THROW(a->setConfiguration(agentJs));

  agentJs = baseOptJs;
  experimentJs = baseExpJs;
  agentJs.erase("Time Sequence Length");
  ASSERT_ANY_THROW(a->setConfiguration(agentJs));

  agentJs = baseOptJs;
  experimentJs = baseExpJs;
  agentJs["Time Sequence Length"] = "Not a Number";
  ASSERT_ANY_THROW(a->setConfiguration(agentJs));

  agentJs = baseOptJs;
  experimentJs = baseExpJs;
  agentJs["Time Sequence Length"] = 1;
  ASSERT_NO_THROW(a->setConfiguration(agentJs));

  agentJs = baseOptJs;
  experimentJs = baseExpJs;
  agentJs.erase("Learning Rate");
  ASSERT_ANY_THROW(a->setConfiguration(agentJs));

  agentJs = baseOptJs;
  experimentJs = baseExpJs;
  agentJs["Learning Rate"] = "Not a Number";
  ASSERT_ANY_THROW(a->setConfiguration(agentJs));

  agentJs = baseOptJs;
  experimentJs = baseExpJs;
  agentJs["Learning Rate"] = 1.0;
  ASSERT_NO_THROW(a->setConfiguration(agentJs));
 
  agentJs = baseOptJs;
  experimentJs = baseExpJs;
  agentJs.erase("Importance Weight Truncation Level");
  ASSERT_ANY_THROW(a->setConfiguration(agentJs));

  agentJs = baseOptJs;
  experimentJs = baseExpJs;
  agentJs["Importance Weight Truncation Level"] = "Not a Number";
  ASSERT_ANY_THROW(a->setConfiguration(agentJs));

  agentJs = baseOptJs;
  experimentJs = baseExpJs;
  agentJs["Importance Weight Truncation Level"] = 1.0;
  ASSERT_NO_THROW(a->setConfiguration(agentJs));

  agentJs = baseOptJs;
  experimentJs = baseExpJs;
  agentJs["L2 Regularization"].erase("Enabled");
  ASSERT_ANY_THROW(a->setConfiguration(agentJs));

  agentJs = baseOptJs;
  experimentJs = baseExpJs;
  agentJs["L2 Regularization"]["Enabled"] = "Not a Number";
  ASSERT_ANY_THROW(a->setConfiguration(agentJs));

  agentJs = baseOptJs;
  experimentJs = baseExpJs;
  agentJs["L2 Regularization"]["Enabled"] = true;
  ASSERT_NO_THROW(a->setConfiguration(agentJs));

  agentJs = baseOptJs;
  experimentJs = baseExpJs;
  agentJs["L2 Regularization"].erase("Importance");
  ASSERT_ANY_THROW(a->setConfiguration(agentJs));

  agentJs = baseOptJs;
  experimentJs = baseExpJs;
  agentJs["L2 Regularization"]["Importance"] = "Not a Number";
  ASSERT_ANY_THROW(a->setConfiguration(agentJs));

  agentJs = baseOptJs;
  experimentJs = baseExpJs;
  agentJs["L2 Regularization"]["Importance"] = 1.0;
  ASSERT_NO_THROW(a->setConfiguration(agentJs));

  agentJs = baseOptJs;
  experimentJs = baseExpJs;
  agentJs["Neural Network"].erase("Hidden Layers");
  ASSERT_ANY_THROW(a->setConfiguration(agentJs));

  agentJs = baseOptJs;
  experimentJs = baseExpJs;
  agentJs["Neural Network"]["Hidden Layers"] = knlohmann::json();
  ASSERT_NO_THROW(a->setConfiguration(agentJs));

  agentJs = baseOptJs;
  experimentJs = baseExpJs;
  agentJs["Neural Network"].erase("Optimizer");
  ASSERT_ANY_THROW(a->setConfiguration(agentJs));

  agentJs = baseOptJs;
  experimentJs = baseExpJs;
  agentJs["Neural Network"]["Optimizer"] = 1.0;
  ASSERT_ANY_THROW(a->setConfiguration(agentJs));

  agentJs = baseOptJs;
  experimentJs = baseExpJs;
  agentJs["Neural Network"]["Optimizer"] = "Adam";
  ASSERT_NO_THROW(a->setConfiguration(agentJs));

  agentJs = baseOptJs;
  experimentJs = baseExpJs;
  agentJs["Neural Network"].erase("Engine");
  ASSERT_ANY_THROW(a->setConfiguration(agentJs));

  agentJs = baseOptJs;
  experimentJs = baseExpJs;
  agentJs["Neural Network"]["Engine"] = 1.0;
  ASSERT_ANY_THROW(a->setConfiguration(agentJs));

  agentJs = baseOptJs;
  experimentJs = baseExpJs;
  agentJs["Neural Network"]["Engine"] = "Adam";
  ASSERT_NO_THROW(a->setConfiguration(agentJs));

  agentJs = baseOptJs;
  experimentJs = baseExpJs;
  agentJs.erase("Discount Factor");
  ASSERT_ANY_THROW(a->setConfiguration(agentJs));

  agentJs = baseOptJs;
  experimentJs = baseExpJs;
  agentJs["Discount Factor"] = "Not a Number";
  ASSERT_ANY_THROW(a->setConfiguration(agentJs));

  agentJs = baseOptJs;
  experimentJs = baseExpJs;
  agentJs["Discount Factor"] = 0.99;
  ASSERT_NO_THROW(a->setConfiguration(agentJs));

  agentJs = baseOptJs;
  experimentJs = baseExpJs;
  agentJs["Experience Replay"].erase("Serialize");
  ASSERT_ANY_THROW(a->setConfiguration(agentJs));

  agentJs = baseOptJs;
  experimentJs = baseExpJs;
  agentJs["Experience Replay"]["Serialize"] = "Not a Number";
  ASSERT_ANY_THROW(a->setConfiguration(agentJs));

  agentJs = baseOptJs;
  experimentJs = baseExpJs;
  agentJs["Experience Replay"]["Serialize"] = true;
  ASSERT_NO_THROW(a->setConfiguration(agentJs));

  agentJs = baseOptJs;
  experimentJs = baseExpJs;
  agentJs["Experience Replay"].erase("Start Size");
  ASSERT_ANY_THROW(a->setConfiguration(agentJs));

  agentJs = baseOptJs;
  experimentJs = baseExpJs;
  agentJs["Experience Replay"]["Start Size"] = "Not a Number";
  ASSERT_ANY_THROW(a->setConfiguration(agentJs));

  agentJs = baseOptJs;
  experimentJs = baseExpJs;
  agentJs["Experience Replay"]["Start Size"] = 1;
  ASSERT_NO_THROW(a->setConfiguration(agentJs));

  agentJs = baseOptJs;
  experimentJs = baseExpJs;
  agentJs["Experience Replay"].erase("Maximum Size");
  ASSERT_ANY_THROW(a->setConfiguration(agentJs));

  agentJs = baseOptJs;
  experimentJs = baseExpJs;
  agentJs["Experience Replay"]["Maximum Size"] = "Not a Number";
  ASSERT_ANY_THROW(a->setConfiguration(agentJs));

  agentJs = baseOptJs;
  experimentJs = baseExpJs;
  agentJs["Experience Replay"]["Maximum Size"] = 1;
  ASSERT_NO_THROW(a->setConfiguration(agentJs));

  agentJs = baseOptJs;
  experimentJs = baseExpJs;
  agentJs["Experience Replay"]["Off Policy"].erase("Cutoff Scale");
  ASSERT_ANY_THROW(a->setConfiguration(agentJs));

  agentJs = baseOptJs;
  experimentJs = baseExpJs;
  agentJs["Experience Replay"]["Off Policy"]["Cutoff Scale"] = "Not a Number";
  ASSERT_ANY_THROW(a->setConfiguration(agentJs));

  agentJs = baseOptJs;
  experimentJs = baseExpJs;
  agentJs["Experience Replay"]["Off Policy"]["Cutoff Scale"] = 1.0;
  ASSERT_NO_THROW(a->setConfiguration(agentJs));

  agentJs = baseOptJs;
  experimentJs = baseExpJs;
  agentJs["Experience Replay"]["Off Policy"].erase("Target");
  ASSERT_ANY_THROW(a->setConfiguration(agentJs));

  agentJs = baseOptJs;
  experimentJs = baseExpJs;
  agentJs["Experience Replay"]["Off Policy"]["Target"] = "Not a Number";
  ASSERT_ANY_THROW(a->setConfiguration(agentJs));

  agentJs = baseOptJs;
  experimentJs = baseExpJs;
  agentJs["Experience Replay"]["Off Policy"]["Target"] = 1.0;
  ASSERT_NO_THROW(a->setConfiguration(agentJs));

  agentJs = baseOptJs;
  experimentJs = baseExpJs;
  agentJs["Experience Replay"]["Off Policy"].erase("Annealing Rate");
  ASSERT_ANY_THROW(a->setConfiguration(agentJs));

  agentJs = baseOptJs;
  experimentJs = baseExpJs;
  agentJs["Experience Replay"]["Off Policy"]["Annealing Rate"] = "Not a Number";
  ASSERT_ANY_THROW(a->setConfiguration(agentJs));

  agentJs = baseOptJs;
  experimentJs = baseExpJs;
  agentJs["Experience Replay"]["Off Policy"]["Annealing Rate"] = 1.0;
  ASSERT_NO_THROW(a->setConfiguration(agentJs));

  agentJs = baseOptJs;
  experimentJs = baseExpJs;
  agentJs["Experience Replay"]["Off Policy"].erase("REFER Beta");
  ASSERT_ANY_THROW(a->setConfiguration(agentJs));

  agentJs = baseOptJs;
  experimentJs = baseExpJs;
  agentJs["Experience Replay"]["Off Policy"]["REFER Beta"] = "Not a Number";
  ASSERT_ANY_THROW(a->setConfiguration(agentJs));

  agentJs = baseOptJs;
  experimentJs = baseExpJs;
  agentJs["Experience Replay"]["Off Policy"]["REFER Beta"] = 1.0;
  ASSERT_NO_THROW(a->setConfiguration(agentJs));

  agentJs = baseOptJs;
  experimentJs = baseExpJs;
  agentJs.erase("Experiences Between Policy Updates");
  ASSERT_ANY_THROW(a->setConfiguration(agentJs));

  agentJs = baseOptJs;
  experimentJs = baseExpJs;
  agentJs["Experiences Between Policy Updates"] = "Not a Number";
  ASSERT_ANY_THROW(a->setConfiguration(agentJs));

  agentJs = baseOptJs;
  experimentJs = baseExpJs;
  agentJs["Experiences Between Policy Updates"] = 1;
  ASSERT_NO_THROW(a->setConfiguration(agentJs));

  agentJs = baseOptJs;
  experimentJs = baseExpJs;
  agentJs["State Rescaling"].erase("Enabled");
  ASSERT_ANY_THROW(a->setConfiguration(agentJs));

  agentJs = baseOptJs;
  experimentJs = baseExpJs;
  agentJs["State Rescaling"]["Enabled"] = "Not a Number";
  ASSERT_ANY_THROW(a->setConfiguration(agentJs));

  agentJs = baseOptJs;
  experimentJs = baseExpJs;
  agentJs["State Rescaling"]["Enabled"] = true;
  ASSERT_NO_THROW(a->setConfiguration(agentJs));

  agentJs = baseOptJs;
  experimentJs = baseExpJs;
  agentJs["Reward"]["Rescaling"].erase("Enabled");
  ASSERT_ANY_THROW(a->setConfiguration(agentJs));

  agentJs = baseOptJs;
  experimentJs = baseExpJs;
  agentJs["Reward"]["Rescaling"]["Enabled"] = "Not a Number";
  ASSERT_ANY_THROW(a->setConfiguration(agentJs));

  agentJs = baseOptJs;
  experimentJs = baseExpJs;
  agentJs["Reward"]["Rescaling"]["Enabled"] = false;
  ASSERT_NO_THROW(a->setConfiguration(agentJs));

  agentJs = baseOptJs;
  experimentJs = baseExpJs;
  agentJs["Testing"].erase("Sample Ids");
  ASSERT_ANY_THROW(a->setConfiguration(agentJs));

  agentJs = baseOptJs;
  experimentJs = baseExpJs;
  agentJs["Testing"]["Sample Ids"] = "Not a Number";
  ASSERT_ANY_THROW(a->setConfiguration(agentJs));

  agentJs = baseOptJs;
  experimentJs = baseExpJs;
  agentJs["Testing"]["Sample Ids"] = std::vector<size_t>({0});
  ASSERT_NO_THROW(a->setConfiguration(agentJs));

  agentJs = baseOptJs;
  experimentJs = baseExpJs;
  agentJs["Testing"].erase("Current Policy");
  ASSERT_ANY_THROW(a->setConfiguration(agentJs));

  agentJs = baseOptJs;
  experimentJs = baseExpJs;
  agentJs["Testing"]["Current Policy"] = std::vector<size_t>({0});
  ASSERT_NO_THROW(a->setConfiguration(agentJs));

  agentJs = baseOptJs;
  experimentJs = baseExpJs;
  agentJs["Training"]["Environment Id History"] = "Not a Number";
  ASSERT_ANY_THROW(a->setConfiguration(agentJs));

  agentJs = baseOptJs;
  experimentJs = baseExpJs;
  agentJs["Training"]["Environment Id History"] = std::vector<size_t>({1});
  ASSERT_NO_THROW(a->setConfiguration(agentJs));

  agentJs = baseOptJs;
  experimentJs = baseExpJs;
  agentJs["Training"].erase("Average Depth");
  ASSERT_ANY_THROW(a->setConfiguration(agentJs));

  agentJs = baseOptJs;
  experimentJs = baseExpJs;
  agentJs["Training"]["Average Depth"] = "Not a Number";
  ASSERT_ANY_THROW(a->setConfiguration(agentJs));

  agentJs = baseOptJs;
  experimentJs = baseExpJs;
  agentJs["Training"]["Average Depth"] = 2;
  ASSERT_NO_THROW(a->setConfiguration(agentJs));

  agentJs = baseOptJs;
  experimentJs = baseExpJs;
  agentJs["Training"].erase("Current Policy");
  ASSERT_NO_THROW(a->setConfiguration(agentJs));

  agentJs = baseOptJs;
  experimentJs = baseExpJs;
  agentJs["Training"]["Current Policy"] = std::vector<size_t>({0});
  ASSERT_NO_THROW(a->setConfiguration(agentJs));

  agentJs = baseOptJs;
  experimentJs = baseExpJs;
  agentJs["Training"].erase("Best Policy");
  ASSERT_NO_THROW(a->setConfiguration(agentJs));

  agentJs = baseOptJs;
  experimentJs = baseExpJs;
  agentJs["Training"]["Best Policy"] = std::vector<size_t>({0});
  ASSERT_NO_THROW(a->setConfiguration(agentJs));

  agentJs = baseOptJs;
  experimentJs = baseExpJs;
  agentJs["Termination Criteria"].erase("Max Experiences");
  ASSERT_ANY_THROW(a->setConfiguration(agentJs));

  agentJs = baseOptJs;
  experimentJs = baseExpJs;
  agentJs["Termination Criteria"]["Max Experiences"] = "Not a Number";
  ASSERT_ANY_THROW(a->setConfiguration(agentJs));

  agentJs = baseOptJs;
  experimentJs = baseExpJs;
  agentJs["Termination Criteria"]["Max Experiences"] = 200;
  ASSERT_NO_THROW(a->setConfiguration(agentJs));

  agentJs = baseOptJs;
  experimentJs = baseExpJs;
  agentJs["Termination Criteria"].erase("Max Episodes");
  ASSERT_ANY_THROW(a->setConfiguration(agentJs));

  agentJs = baseOptJs;
  experimentJs = baseExpJs;
  agentJs["Termination Criteria"]["Max Episodes"] = "Not a Number";
  ASSERT_ANY_THROW(a->setConfiguration(agentJs));

  agentJs = baseOptJs;
  experimentJs = baseExpJs;
  agentJs["Termination Criteria"]["Max Episodes"] = 200;
  ASSERT_NO_THROW(a->setConfiguration(agentJs));

  agentJs = baseOptJs;
  experimentJs = baseExpJs;
  agentJs["Termination Criteria"].erase("Max Policy Updates");
  ASSERT_ANY_THROW(a->setConfiguration(agentJs));

  agentJs = baseOptJs;
  experimentJs = baseExpJs;
  agentJs["Termination Criteria"]["Max Policy Updates"] = "Not a Number";
  ASSERT_ANY_THROW(a->setConfiguration(agentJs));

  agentJs = baseOptJs;
  experimentJs = baseExpJs;
  agentJs["Termination Criteria"]["Max Policy Updates"]  = 200.0;
  ASSERT_NO_THROW(a->setConfiguration(agentJs));

  // Testing termination criteria
  e._currentGeneration = 2;

  // Control check
  ASSERT_FALSE(a->checkTermination());

  // Checking max episodes termination
  a->_mode = "Training";
  a->_maxEpisodes = 10;
  a->_currentEpisode = 20;
  ASSERT_TRUE(a->checkTermination());
  a->_currentEpisode = 5;
  ASSERT_FALSE(a->checkTermination());

  // Checking max experiences termination
  a->_mode = "Training";
  a->_maxExperiences = 10;
  a->_experienceCount = 20;
  ASSERT_TRUE(a->checkTermination());
  a->_experienceCount = 5;
  ASSERT_FALSE(a->checkTermination());

  // Checking curPolicy update termination
  a->_mode = "Training";
  a->_maxPolicyUpdates   = 10;
  a->_policyUpdateCount   = 20;
  ASSERT_TRUE(a->checkTermination());
  a->_policyUpdateCount   = 5;
  ASSERT_FALSE(a->checkTermination());
 }

 //// Continous Agent

 TEST(a, continuousAgent)
  {
   // Creating base experiment
   Experiment e;
   e._logger = new Logger("Detailed", stdout);
   auto& experimentJs = e._js.getJson();
   experimentJs["Variables"][0]["Name"] = "X";
   Variable v;
   e._variables.push_back(&v);

   // Creating optimizer configuration Json
   knlohmann::json agentJs;

   // Configuring Problem
   e["Problem"]["Type"] = "Reinforcement Learning / Continuous";
   reinforcementLearning::Continuous* pC;
   knlohmann::json problemRefJs;
   problemRefJs["Type"] = "Reinforcement Learning / Continuous";
   problemRefJs["Environment Function"] = [](Sample &s){};

   e["Variables"][0]["Name"] = "State0";
   e["Variables"][1]["Name"] = "Action0";
   e["Variables"][1]["Type"] = "Action";
   e["Variables"][1]["Initial Exploration Noise"] = 0.45;
   e["Variables"][1]["Lower Bound"] = 0.00;
   e["Variables"][1]["Upper Bound"] = 1.00;

   Variable vState;
   vState._name = "State0";
   vState._type = "State";

   Variable vAction;
   vAction._name = "Action0";
   vAction._type = "Action";
   vAction._initialExplorationNoise = 0.45;
   vAction._lowerBound = 0.00;
   vAction._upperBound = 1.00;

   e._variables.push_back(&vState);
   e._variables.push_back(&vAction);

   ASSERT_NO_THROW(pC = dynamic_cast<reinforcementLearning::Continuous *>(Module::getModule(problemRefJs, &e)));
   e._problem = pC;
   ASSERT_NO_THROW(pC->initialize());

   // Using a neural network solver (deep learning) for inference

   agentJs["Type"] = "Agent / Continuous / VRACER";
   agentJs["Mode"] = "Training";
   agentJs["Episodes Per Generation"] = 10;
   agentJs["Experiences Between Policy Updates"] = 1;
   agentJs["Discount Factor"] = 0.99;
   agentJs["Learning Rate"] = 0.0001;
   agentJs["Mini Batch"]["Size"] = 32;
   agentJs["Experience Replay"]["Start Size"] = 1000;
   agentJs["Experience Replay"]["Maximum Size"] = 10000;

   /// Configuring the neural network and its hidden layers

   agentJs["Neural Network"]["Engine"] = "OneDNN";
   agentJs["Neural Network"]["Optimizer"] = "Adam";

   agentJs["Neural Network"]["Hidden Layers"][0]["Type"] = "Layer/Linear";
   agentJs["Neural Network"]["Hidden Layers"][0]["Output Channels"] = 16;

   // Creating module
   VRACER* a;
   ASSERT_NO_THROW(a = dynamic_cast<VRACER *>(Module::getModule(agentJs, &e)));

   // Defaults should be applied without a problem
   ASSERT_NO_THROW(a->applyModuleDefaults(agentJs));

   // Covering variable functions (no effect)
   ASSERT_NO_THROW(a->applyVariableDefaults());

   // Creating normal generator
   knlohmann::json normalDistroJs;
   normalDistroJs["Type"] = "Univariate/Normal";
   normalDistroJs["Mean"] = 0.0;
   normalDistroJs["Standard Deviation"] = 1.0;
   a->_normalGenerator = dynamic_cast<korali::distribution::univariate::Normal*>(korali::Module::getModule(normalDistroJs, &e));
   a->_normalGenerator->applyVariableDefaults();
   a->_normalGenerator->applyModuleDefaults(normalDistroJs);
   a->_normalGenerator->setConfiguration(normalDistroJs);

   // Creating uniform generator
   knlohmann::json uniformDistroJs;
   uniformDistroJs["Type"] = "Univariate/Uniform";
   uniformDistroJs["Minimum"] = -1.0;
   uniformDistroJs["Maximum"] = +1.0;
   a->_uniformGenerator = dynamic_cast<korali::distribution::univariate::Uniform*>(korali::Module::getModule(uniformDistroJs, &e));
   a->_uniformGenerator->applyVariableDefaults();
   a->_uniformGenerator->applyModuleDefaults(uniformDistroJs);
   a->_uniformGenerator->setConfiguration(uniformDistroJs);

   // Backup the correct base configuration
   auto baseOptJs = agentJs;
   auto baseExpJs = experimentJs;

   // Testing distribution corner cases
   policy_t curPolicy;
   policy_t prevPolicy;
   curPolicy.distributionParameters = std::vector<float>({0.0, 1.0});
   prevPolicy.distributionParameters = std::vector<float>({0.0, 0.5});
   curPolicy.unboundedAction = std::vector<float>({0.1f});
   prevPolicy.unboundedAction = std::vector<float>({0.1f});
   auto testAction = std::vector<float>({0.1f});

   a->_policyDistribution = "Normal";
   ASSERT_NO_THROW(a->agent::Continuous::initializeAgent());
   ASSERT_NO_THROW(a->generateTrainingAction(curPolicy));
   ASSERT_NO_THROW(a->generateTestingAction(curPolicy));
   ASSERT_NO_THROW(a->calculateImportanceWeight(testAction, curPolicy, prevPolicy));
   ASSERT_NO_THROW(a->calculateImportanceWeightGradient(testAction, curPolicy, prevPolicy));
   ASSERT_NO_THROW(a->calculateKLDivergenceGradient(curPolicy, prevPolicy));

   a->_policyDistribution = "Clipped Normal";
   a->_actionLowerBounds = std::vector<float>({0.0});
   a->_actionUpperBounds = std::vector<float>({1.0});
   ASSERT_NO_THROW(a->agent::Continuous::initializeAgent());
   ASSERT_NO_THROW(a->generateTrainingAction(curPolicy));
   ASSERT_NO_THROW(a->generateTestingAction(curPolicy));

   a->_policyDistribution = "Truncated Normal";
   ASSERT_NO_THROW(a->agent::Continuous::initializeAgent());
   ASSERT_NO_THROW(a->generateTrainingAction(curPolicy));
   ASSERT_NO_THROW(a->generateTestingAction(curPolicy));

   a->_policyDistribution = "Beta";
   a->_actionLowerBounds = std::vector<float>({-std::numeric_limits<float>::infinity()});
   a->_actionUpperBounds = std::vector<float>({1.0f});
   ASSERT_ANY_THROW(a->agent::Continuous::initializeAgent());

   a->_actionLowerBounds = std::vector<float>({0.0f});
   a->_actionUpperBounds = std::vector<float>({+std::numeric_limits<float>::infinity()});
   ASSERT_ANY_THROW(a->agent::Continuous::initializeAgent());

   a->_actionLowerBounds = std::vector<float>({-1.0f});
   a->_actionUpperBounds = std::vector<float>({1.0f});
   ASSERT_NO_THROW(a->agent::Continuous::initializeAgent());
   ASSERT_NO_THROW(a->generateTrainingAction(curPolicy));
   curPolicy.distributionParameters = std::vector<float>({0.5, 0.2236});
   ASSERT_NO_THROW(a->generateTestingAction(curPolicy));

   ASSERT_NO_THROW(a->generateTestingAction(curPolicy));
   ASSERT_NO_THROW(a->calculateImportanceWeight(testAction, curPolicy, prevPolicy));
   ASSERT_NO_THROW(a->calculateImportanceWeightGradient(testAction, curPolicy, prevPolicy));
   ASSERT_NO_THROW(a->calculateKLDivergenceGradient(curPolicy, prevPolicy));

   pC->_actionVectorIndexes[0] = 1;
   e._variables[1]->_initialExplorationNoise = -1.0f;
   ASSERT_ANY_THROW(a->agent::Continuous::initializeAgent());
   e._variables[1]->_initialExplorationNoise = 0.45f;

   a->_policyDistribution = "Squashed Normal";
   ASSERT_NO_THROW(a->agent::Continuous::initializeAgent());

   a->_actionLowerBounds = std::vector<float>({-std::numeric_limits<float>::infinity()});
   a->_actionUpperBounds = std::vector<float>({1.0f});
   ASSERT_ANY_THROW(a->agent::Continuous::initializeAgent());

   a->_actionLowerBounds = std::vector<float>({0.0f});
   a->_actionUpperBounds = std::vector<float>({+std::numeric_limits<float>::infinity()});
   ASSERT_ANY_THROW(a->agent::Continuous::initializeAgent());

   a->_actionLowerBounds = std::vector<float>({0.0f});
   a->_actionUpperBounds = std::vector<float>({1.0f});
   ASSERT_NO_THROW(a->agent::Continuous::initializeAgent());
   ASSERT_NO_THROW(a->generateTrainingAction(curPolicy));
   ASSERT_NO_THROW(a->generateTestingAction(curPolicy));
   ASSERT_NO_THROW(a->calculateImportanceWeight(testAction, curPolicy, prevPolicy));
   ASSERT_NO_THROW(a->calculateImportanceWeightGradient(testAction, curPolicy, prevPolicy));
   ASSERT_NO_THROW(a->calculateKLDivergenceGradient(curPolicy, prevPolicy));

   pC->_actionVectorIndexes[0] = 1;
   e._variables[1]->_initialExplorationNoise = -1.0f;
   ASSERT_ANY_THROW(a->agent::Continuous::initializeAgent());
   e._variables[1]->_initialExplorationNoise = 0.45f;

   // Testing optional parameters
   agentJs = baseOptJs;
   experimentJs = baseExpJs;
   agentJs["Action Shifts"] = "Not a Number";
   ASSERT_ANY_THROW(a->setConfiguration(agentJs));

   agentJs = baseOptJs;
   experimentJs = baseExpJs;
   agentJs["Action Shifts"] = std::vector<float>({0.0});
   ASSERT_NO_THROW(a->setConfiguration(agentJs));

   agentJs = baseOptJs;
   experimentJs = baseExpJs;
   agentJs["Action Scales"] = "Not a Number";
   ASSERT_ANY_THROW(a->setConfiguration(agentJs));

   agentJs = baseOptJs;
   experimentJs = baseExpJs;
   agentJs["Action Scales"] = std::vector<float>({0.0});
   ASSERT_NO_THROW(a->setConfiguration(agentJs));

   agentJs = baseOptJs;
   experimentJs = baseExpJs;
   agentJs["Policy"]["Parameter Count"] = "Not a Number";
   ASSERT_ANY_THROW(a->setConfiguration(agentJs));

   agentJs = baseOptJs;
   experimentJs = baseExpJs;
   agentJs["Policy"]["Parameter Count"] = 1;
   ASSERT_NO_THROW(a->setConfiguration(agentJs));

   agentJs = baseOptJs;
   experimentJs = baseExpJs;
   agentJs["Policy"]["Parameter Transformation Masks"] = 1.0;
   ASSERT_ANY_THROW(a->setConfiguration(agentJs));

   agentJs = baseOptJs;
   experimentJs = baseExpJs;
   agentJs["Policy"]["Parameter Transformation Masks"] = std::vector<std::string>({""});
   ASSERT_NO_THROW(a->setConfiguration(agentJs));

   agentJs = baseOptJs;
   experimentJs = baseExpJs;
   agentJs["Policy"]["Parameter Scaling"] = 1.0;
   ASSERT_ANY_THROW(a->setConfiguration(agentJs));

   agentJs = baseOptJs;
   experimentJs = baseExpJs;
   agentJs["Policy"]["Parameter Scaling"] = std::vector<float>({0.0});
   ASSERT_NO_THROW(a->setConfiguration(agentJs));

   agentJs = baseOptJs;
   experimentJs = baseExpJs;
   agentJs["Policy"]["Parameter Shifting"] = 1.0;
   ASSERT_ANY_THROW(a->setConfiguration(agentJs));

   agentJs = baseOptJs;
   experimentJs = baseExpJs;
   agentJs["Policy"]["Parameter Shifting"] = std::vector<float>({0.0});
   ASSERT_NO_THROW(a->setConfiguration(agentJs));

   // Testing mandatory parameters

   agentJs = baseOptJs;
   experimentJs = baseExpJs;
   agentJs["Policy"].erase("Distribution");
   ASSERT_ANY_THROW(a->setConfiguration(agentJs));

   agentJs = baseOptJs;
   experimentJs = baseExpJs;
   agentJs["Policy"]["Distribution"] = 1.0;
   ASSERT_ANY_THROW(a->setConfiguration(agentJs));

   agentJs = baseOptJs;
   experimentJs = baseExpJs;
   agentJs["Policy"]["Distribution"] = "Unknown";
   ASSERT_ANY_THROW(a->setConfiguration(agentJs));

   agentJs = baseOptJs;
   experimentJs = baseExpJs;
   agentJs["Policy"]["Distribution"] = "Normal";
   ASSERT_NO_THROW(a->setConfiguration(agentJs));
  }

 //// Continous Agent

 TEST(a, VRACER)
  {
   // Creating base experiment
   Experiment e;
   e._logger = new Logger("Detailed", stdout);
   auto& experimentJs = e._js.getJson();
   experimentJs["Variables"][0]["Name"] = "X";
   Variable v;
   e._variables.push_back(&v);

   // Creating optimizer configuration Json
   knlohmann::json agentJs;

   // Configuring Problem
   e["Problem"]["Type"] = "Reinforcement Learning / Continuous";
   reinforcementLearning::Continuous* pC;
   knlohmann::json problemRefJs;
   problemRefJs["Type"] = "Reinforcement Learning / Continuous";
   problemRefJs["Environment Function"] = 0;

   e["Variables"][0]["Name"] = "State0";
   e["Variables"][1]["Name"] = "Action0";
   e["Variables"][1]["Type"] = "Action";
   e["Variables"][1]["Initial Exploration Noise"] = 0.45;
   e["Variables"][1]["Lower Bound"] = 0.00;
   e["Variables"][1]["Upper Bound"] = 1.00;

   Variable vState;
   vState._name = "State0";
   vState._type = "State";

   Variable vAction;
   vAction._name = "Action0";
   vAction._type = "Action";
   vAction._initialExplorationNoise = 0.45;
   vAction._lowerBound = 0.00;
   vAction._upperBound = 1.00;

   e._variables.push_back(&vState);
   e._variables.push_back(&vAction);

   ASSERT_NO_THROW(pC = dynamic_cast<reinforcementLearning::Continuous *>(Module::getModule(problemRefJs, &e)));
   e._problem = pC;
   ASSERT_NO_THROW(pC->initialize());

   // Using a neural network solver (deep learning) for inference

   agentJs["Type"] = "Agent / Continuous / VRACER";
   agentJs["Mode"] = "Training";
   agentJs["Episodes Per Generation"] = 10;
   agentJs["Experiences Between Policy Updates"] = 1;
   agentJs["Discount Factor"] = 0.99;
   agentJs["Learning Rate"] = 0.0001;
   agentJs["Mini Batch"]["Size"] = 32;
   agentJs["Experience Replay"]["Start Size"] = 1000;
   agentJs["Experience Replay"]["Maximum Size"] = 10000;

   /// Configuring the neural network and its hidden layers

   agentJs["Neural Network"]["Engine"] = "OneDNN";
   agentJs["Neural Network"]["Optimizer"] = "Adam";

   agentJs["Neural Network"]["Hidden Layers"][0]["Type"] = "Layer/Linear";
   agentJs["Neural Network"]["Hidden Layers"][0]["Output Channels"] = 16;

   // Creating module
   VRACER* a;
   ASSERT_NO_THROW(a = dynamic_cast<VRACER *>(Module::getModule(agentJs, &e)));

   // Defaults should be applied without a problem
   ASSERT_NO_THROW(a->applyModuleDefaults(agentJs));

   // Covering variable functions (no effect)
   ASSERT_NO_THROW(a->applyVariableDefaults());

   // Backup the correct base configuration
   auto baseOptJs = agentJs;
   auto baseExpJs = experimentJs;

   // Testing optional parameters
   agentJs = baseOptJs;
   experimentJs = baseExpJs;
   agentJs["Statistics"]["Average Action Sigmas"] = "Not a Number";
   ASSERT_ANY_THROW(a->setConfiguration(agentJs));

   agentJs = baseOptJs;
   experimentJs = baseExpJs;
   agentJs["Statistics"]["Average Action Sigmas"] = std::vector<float>({0.0});
   ASSERT_NO_THROW(a->setConfiguration(agentJs));

   agentJs = baseOptJs;
   experimentJs = baseExpJs;
   e["Variables"][0].erase("Initial Exploration Noise");
   ASSERT_ANY_THROW(a->setConfiguration(agentJs));

   agentJs = baseOptJs;
   experimentJs = baseExpJs;
   e["Variables"][0]["Initial Exploration Noise"] = "Not a Number";
   ASSERT_ANY_THROW(a->setConfiguration(agentJs));

   agentJs = baseOptJs;
   experimentJs = baseExpJs;
   e["Variables"][0]["Initial Exploration Noise"] = 0.0f;
   ASSERT_NO_THROW(a->setConfiguration(agentJs));
   ASSERT_NO_THROW(a->initializeAgent());
   e._solver = a;

   Sample s;
   auto curPolicy = a->getPolicy();
   s["Policy Hyperparameters"] = a->getPolicy();
   s["Sample Id"] = 0;
   s["State Rescaling"]["Means"] = std::vector<float>({0.0});
   s["State Rescaling"]["Standard Deviations"] = std::vector<float>({1.0});

   // Evaluation function
   _functionVector.resize(1);
   std::function<void(korali::Sample&)> modelFc = [](Sample& s)
   {
     s["Termination"] = "Unknown";
   };
   _functionVector[0] = &modelFc;

   // Creating conduit
   knlohmann::json conduitJs;
   conduitJs["Type"] = "Sequential";
   ASSERT_NO_THROW(_conduit = dynamic_cast<Sequential *>(Module::getModule(conduitJs, NULL)));

   _launchId = 0;
   __envFunctionId = 0;
   __currentSample = &s;
   ASSERT_ANY_THROW(__environmentWrapper());

   modelFc = [](Sample& s)
   {
     s["Termination"] = "Non Terminal";
   };
   ASSERT_ANY_THROW(__environmentWrapper());

   modelFc = [](Sample& s)
   {
     s["Termination"] = "Terminal";
   };
   s._workerThread = co_active();
   ASSERT_ANY_THROW(__environmentWrapper());

   _envThread = co_active();
   s["State"] = std::vector<float>({std::numeric_limits<float>::infinity()});
   ASSERT_ANY_THROW(pC->runEnvironment(s));
  }

 //// Discrete Agent

 TEST(a, discreteAgent)
  {
   // Creating base experiment
   Experiment e;
   e._logger = new Logger("Detailed", stdout);
   auto& experimentJs = e._js.getJson();
   experimentJs["Variables"][0]["Name"] = "X";
   Variable v;
   e._variables.push_back(&v);

   // Creating optimizer configuration Json
   knlohmann::json agentJs;

   // Configuring Problem
   e["Problem"]["Type"] = "Reinforcement Learning / Discrete";
   reinforcementLearning::Discrete* pD;
   knlohmann::json problemRefJs;
   problemRefJs["Type"] = "Reinforcement Learning / Discrete";
   problemRefJs["Environment Function"] = [](Sample &s){};

   problemRefJs["Possible Actions"] = std::vector<std::vector<float>>({ { -10.0 }, { 10.0 } });
   e["Variables"][0]["Name"] = "State0";
   e["Variables"][1]["Name"] = "Action0";
   e["Variables"][1]["Type"] = "Action";

   Variable vState;
   vState._name = "State0";
   vState._type = "State";

   Variable vAction;
   vAction._name = "Action0";
   vAction._type = "Action";

   e._variables.push_back(&vState);
   e._variables.push_back(&vAction);

   ASSERT_NO_THROW(pD = dynamic_cast<reinforcementLearning::Discrete *>(Module::getModule(problemRefJs, &e)));
   e._problem = pD;
   pD->_possibleActions = std::vector<std::vector<float>>({ { -10.0 }, { 10.0 } });
   pD->initialize();
   ASSERT_NO_THROW(pD->initialize());

   // Using a neural network solver (deep learning) for inference

   agentJs["Type"] = "Agent / Discrete / dVRACER";
   agentJs["Mode"] = "Training";
   agentJs["Episodes Per Generation"] = 10;
   agentJs["Experiences Between Policy Updates"] = 1;
   agentJs["Discount Factor"] = 0.99;
   agentJs["Learning Rate"] = 0.0001;
   agentJs["Importance Weight Truncation Level"] = 0.0001;
   agentJs["Mini Batch"]["Size"] = 32;
   agentJs["Experience Replay"]["Start Size"] = 1000;
   agentJs["Experience Replay"]["Maximum Size"] = 10000;

   /// Configuring the neural network and its hidden layers

   agentJs["Neural Network"]["Engine"] = "OneDNN";
   agentJs["Neural Network"]["Optimizer"] = "Adam";

   agentJs["Neural Network"]["Hidden Layers"][0]["Type"] = "Layer/Linear";
   agentJs["Neural Network"]["Hidden Layers"][0]["Output Channels"] = 16;

   // Creating module
   dVRACER* a;
   ASSERT_NO_THROW(a = dynamic_cast<dVRACER *>(Module::getModule(agentJs, &e)));

   // Defaults should be applied without a problem
   ASSERT_NO_THROW(a->applyModuleDefaults(agentJs));

   // Covering variable functions (no effect)
   ASSERT_NO_THROW(a->applyVariableDefaults());

   // Backup the correct base configuration
   auto baseOptJs = agentJs;
   auto baseExpJs = experimentJs;

   // Testing distribution corner cases
   policy_t curPolicy;
   policy_t prevPolicy;
   curPolicy.distributionParameters = std::vector<float>({0.2, 0.8, 1.0}); // Q values andbeta
   curPolicy.actionProbabilities = std::vector<float>({0.2, 0.8}); // Probability distribution of possible actions
   curPolicy.actionIndex = 0;
   prevPolicy.distributionParameters = std::vector<float>({0.5, 0.5, 1.0}); // Q values andbeta
   prevPolicy.actionProbabilities = std::vector<float>({0.5, 0.5}); // Probability distribution of possible actions
   prevPolicy.actionIndex = 0;
   auto testAction = std::vector<float>({-10.0f});

   ASSERT_NO_THROW(a->agent::Discrete::initializeAgent());
   ASSERT_NO_THROW(a->calculateImportanceWeight(testAction, curPolicy, prevPolicy));
   ASSERT_NO_THROW(a->calculateImportanceWeightGradient(curPolicy, prevPolicy));
   ASSERT_NO_THROW(a->calculateKLDivergenceGradient(curPolicy, prevPolicy));
  }

 TEST(a, dVRACER)
   {
    // Creating base experiment
    Experiment e;
    e._logger = new Logger("Detailed", stdout);
    auto& experimentJs = e._js.getJson();
    experimentJs["Variables"][0]["Name"] = "X";
    Variable v;
    e._variables.push_back(&v);

    // Creating optimizer configuration Json
    knlohmann::json agentJs;

    // Configuring Problem
    e["Problem"]["Type"] = "Reinforcement Learning / Discrete";
    reinforcementLearning::Discrete* pD;
    knlohmann::json problemRefJs;
    problemRefJs["Type"] = "Reinforcement Learning / Discrete";
    problemRefJs["Environment Function"] = [](Sample &s){};
    problemRefJs["Possible Actions"] = std::vector<std::vector<float>>({ { -10.0 }, { 10.0 } });

    e["Variables"][0]["Name"] = "State0";
    e["Variables"][1]["Name"] = "Action0";
    e["Variables"][1]["Type"] = "Action";

    Variable vState;
    vState._name = "State0";
    vState._type = "State";

    Variable vAction;
    vAction._name = "Action0";
    vAction._type = "Action";

    e._variables.push_back(&vState);
    e._variables.push_back(&vAction);

    ASSERT_NO_THROW(pD = dynamic_cast<reinforcementLearning::Discrete *>(Module::getModule(problemRefJs, &e)));
    e._problem = pD;
    pD->_possibleActions = std::vector<std::vector<float>>({ { -10.0 }, { 10.0 } });
    ASSERT_NO_THROW(pD->initialize());

    // Using a neural network solver (deep learning) for inference

    agentJs["Type"] = "Agent / Discrete / dVRACER";
    agentJs["Mode"] = "Training";
    agentJs["Episodes Per Generation"] = 10;
    agentJs["Experiences Between Policy Updates"] = 1;
    agentJs["Discount Factor"] = 0.99;
    agentJs["Learning Rate"] = 0.0001;
    agentJs["Mini Batch"]["Size"] = 32;
    agentJs["Experience Replay"]["Start Size"] = 1000;
    agentJs["Experience Replay"]["Maximum Size"] = 10000;

    /// Configuring the neural network and its hidden layers

    agentJs["Neural Network"]["Engine"] = "OneDNN";
    agentJs["Neural Network"]["Optimizer"] = "Adam";

    agentJs["Neural Network"]["Hidden Layers"][0]["Type"] = "Layer/Linear";
    agentJs["Neural Network"]["Hidden Layers"][0]["Output Channels"] = 16;

    // Creating module
    dVRACER* a;
    ASSERT_NO_THROW(a = dynamic_cast<dVRACER *>(Module::getModule(agentJs, &e)));

    // Defaults should be applied without a problem
    ASSERT_NO_THROW(a->applyModuleDefaults(agentJs));

    // Covering variable functions (no effect)
    ASSERT_NO_THROW(a->applyVariableDefaults());

    // Backup the correct base configuration
    auto baseOptJs = agentJs;
    auto baseExpJs = experimentJs;

    // Testing optional parameters
    agentJs = baseOptJs;
    experimentJs = baseExpJs;
    agentJs["Statistics"]["Average Action Sigmas"] = "Not a Number";
    ASSERT_ANY_THROW(a->setConfiguration(agentJs));

    agentJs = baseOptJs;
    experimentJs = baseExpJs;
    agentJs["Statistics"]["Average Action Sigmas"] = std::vector<float>({0.0});
    ASSERT_NO_THROW(a->setConfiguration(agentJs));

    agentJs = baseOptJs;
    experimentJs = baseExpJs;
    e["Variables"][0].erase("Initial Exploration Noise");
    ASSERT_ANY_THROW(a->setConfiguration(agentJs));

    agentJs = baseOptJs;
    experimentJs = baseExpJs;
    e["Variables"][0]["Initial Exploration Noise"] = "Not a Number";
    ASSERT_ANY_THROW(a->setConfiguration(agentJs));

    agentJs = baseOptJs;
    experimentJs = baseExpJs;
    e["Variables"][0]["Initial Exploration Noise"] = 0.0f;
    ASSERT_NO_THROW(a->setConfiguration(agentJs));
   }

} // namespace
