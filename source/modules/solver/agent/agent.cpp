#include "auxiliar/fs.hpp"
#include "engine.hpp"
#include "modules/solver/agent/agent.hpp"
#include "sample/sample.hpp"
#include <chrono>

namespace korali
{
namespace solver
{
;

#pragma omp declare reduction(vec_int_plus        \
                              : std::vector <int> \
                              : std::transform(omp_out.begin(), omp_out.end(), omp_in.begin(), omp_out.begin(), std::plus <int>())) initializer(omp_priv = decltype(omp_orig)(omp_orig.size()))

void Agent::initialize()
{
  _variableCount = _k->_variables.size();

  // Getting problem pointer
  _problem = dynamic_cast<problem::ReinforcementLearning *>(_k->_problem);

  // Formatting reward history for each agent
  _trainingRewardHistory.resize(_problem->_agentsPerEnvironment);

  // Allocating and obtaining action bounds information
  _actionLowerBounds.resize(_problem->_actionVectorSize);
  _actionUpperBounds.resize(_problem->_actionVectorSize);

  for (size_t i = 0; i < _problem->_actionVectorSize; i++)
  {
    auto varIdx = _problem->_actionVectorIndexes[i];
    _actionLowerBounds[i] = _k->_variables[varIdx]->_lowerBound;
    _actionUpperBounds[i] = _k->_variables[varIdx]->_upperBound;

    if (_actionUpperBounds[i] - _actionLowerBounds[i] <= 0.0) KORALI_LOG_ERROR("Upper (%f) and Lower Bound (%f) of action variable %lu invalid.\n", _actionUpperBounds[i], _actionLowerBounds[i], i);
  }

  if (_episodesPerGeneration < 1)
    KORALI_LOG_ERROR("Episodes Per Generation must be larger equal 1 (is %zu)", _episodesPerGeneration);

  // Initializing selected policy
  initializeAgent();

  // Initializing random seed for the shuffle operation
  mt = new std::mt19937(rd());
  mt->seed(_k->_randomSeed++);

  // If not set, using heurisitc for maximum size
  if (_experienceReplayMaximumSize == 0)
    _experienceReplayMaximumSize = std::pow(2, 14) * std::sqrt(_problem->_stateVectorSize + _problem->_actionVectorSize);

  // If not set, filling ER before learning
  if (_experienceReplayStartSize == 0)
    _experienceReplayStartSize = _experienceReplayMaximumSize;

  // Initialize current beta for all agents
  _experienceReplayOffPolicyREFERCurrentBeta = std::vector<float>(_problem->_agentsPerEnvironment, _experienceReplayOffPolicyREFERBeta);

  //  Pre-allocating space for the experience replay memory
  _stateVector.resize(_experienceReplayMaximumSize);
  _actionVector.resize(_experienceReplayMaximumSize);
  _retraceValueVector.resize(_experienceReplayMaximumSize);
  _rewardVector.resize(_experienceReplayMaximumSize);
  _stateValueVector.resize(_experienceReplayMaximumSize);
  _importanceWeightVector.resize(_experienceReplayMaximumSize);
  _truncatedImportanceWeightVector.resize(_experienceReplayMaximumSize);
  _productImportanceWeightVector.resize(_experienceReplayMaximumSize);
  _truncatedStateValueVector.resize(_experienceReplayMaximumSize);
  _truncatedStateVector.resize(_experienceReplayMaximumSize);
  _terminationVector.resize(_experienceReplayMaximumSize);
  _expPolicyVector.resize(_experienceReplayMaximumSize);
  _curPolicyVector.resize(_experienceReplayMaximumSize);
  _isOnPolicyVector.resize(_experienceReplayMaximumSize);
  _episodePosVector.resize(_experienceReplayMaximumSize);
  _episodeIdVector.resize(_experienceReplayMaximumSize);

  //  Pre-allocating space for state time sequence
  _stateTimeSequence.resize(_problem->_agentsPerEnvironment);
  for (size_t agentId = 0; agentId < _problem->_agentsPerEnvironment; ++agentId)
    _stateTimeSequence[agentId].resize(_timeSequenceLength);

  /*********************************************************************
   * If initial generation, set initial agent configuration
   *********************************************************************/
  if (_k->_currentGeneration == 0)
  {
    _currentEpisode = 0;
    _policyUpdateCount = 0;
    _experienceCount = 0;

    // Initializing training and episode statistics //TODO go through all
    _testingAverageReward = -korali::Inf;
    _testingBestReward = -korali::Inf;
    _testingWorstReward = -korali::Inf;
    _testingBestAverageReward = -korali::Inf;
    _testingBestEpisodeId = 0;
    _trainingBestReward.resize(_problem->_agentsPerEnvironment, -korali::Inf);
    _trainingBestEpisodeId.resize(_problem->_agentsPerEnvironment, 0);
    _trainingAverageReward.resize(_problem->_agentsPerEnvironment, -korali::Inf);

    /* Initializing REFER information */

    // If cutoff scale is not defined, use a heuristic value [defaults to 4.0]
    if (_experienceReplayOffPolicyCutoffScale < 0.0f)
      KORALI_LOG_ERROR("Experience Replay Cutoff Scale must be larger 0.0");
    _experienceReplayOffPolicyCount.resize(_problem->_agentsPerEnvironment, 0);
    _experienceReplayOffPolicyRatio.resize(_problem->_agentsPerEnvironment, 0.0f);
    _currentLearningRate = _learningRate;

    _experienceReplayOffPolicyCurrentCutoff = _experienceReplayOffPolicyCutoffScale;

    // Rescaling information
    _stateRescalingMeans = std::vector<std::vector<float>>(_problem->_agentsPerEnvironment, std::vector<float>(_problem->_stateVectorSize, 0.0f));
    _stateRescalingSigmas = std::vector<std::vector<float>>(_problem->_agentsPerEnvironment, std::vector<float>(_problem->_stateVectorSize, 1.0f));
    _rewardRescalingMeans = std::vector<float>(_problem->_agentsPerEnvironment, 0.);
    _rewardRescalingSigmas = std::vector<float>(_problem->_agentsPerEnvironment, 1.);
    _rewardRescalingSumSquaredRewards = std::vector<float>(_problem->_agentsPerEnvironment, 0.);

    // Getting agent's initial policy
    _trainingCurrentPolicies = getAgentPolicy();
  }

  // Setting current agent's training state
  setAgentPolicy(_trainingCurrentPolicies["Policy Hyperparameters"]);

  // If this continues a previous training run, deserialize previous input experience replay. Only for the root (engine) rank
  if ( _k->_currentGeneration > 0 )
    if (_mode == "Training")
      if (_k->_engine->_conduit != NULL)
        deserializeExperienceReplay();

  // Initializing session-wise profiling timers
  _sessionRunningTime = 0.0;
  _sessionSerializationTime = 0.0;
  _sessionAgentComputationTime = 0.0;
  _sessionAgentCommunicationTime = 0.0;
  _sessionAgentPolicyEvaluationTime = 0.0;
  _sessionPolicyUpdateTime = 0.0;
  _sessionAgentAttendingTime = 0.0;

  // Initializing session-specific counters
  _sessionExperienceCount = 0;
  _sessionEpisodeCount = 0;
  _sessionGeneration = 1;
  _sessionPolicyUpdateCount = 0;

  // Calculating how many more experiences do we need in this session to reach the starting size
  _sessionExperiencesUntilStartSize = _stateVector.size() > _experienceReplayStartSize ? 0 : _experienceReplayStartSize - _stateVector.size();

  if (_mode == "Training")
  {
    // Creating storate for _agents and their status
    _agents.resize(_concurrentEnvironments);
    _isAgentRunning.resize(_concurrentEnvironments, false);

    // In case the agent was tested before, remove _testingCurrentPolicies
    _testingCurrentPolicies.clear();
  }

  if (_mode == "Testing")
  {
    // Fixing termination criteria for testing mode
    _maxGenerations = _k->_currentGeneration + 1;

    // Setting testing policy to best testing hyperparameters if not custom-set by the user
    if (_testingCurrentPolicies.empty())
    {
      // Checking if testing policies have been generated
      if (_testingBestPolicies.empty())
      {
        _k->_logger->logInfo("Minimal", "Using current training policy for testing.\n");
        _testingCurrentPolicies = _trainingCurrentPolicies;
      }
      else
      {
        _k->_logger->logInfo("Minimal", "Using best testing policy for test-run.\n");
        _testingCurrentPolicies = _testingBestPolicies;
      }
    }

    // Checking if there's testing samples defined
    if (_testingSampleIds.size() == 0)
      KORALI_LOG_ERROR("For testing, you need to indicate the sample ids to run in the ['Testing']['Sample Ids'] field.\n");

    // Prepare storage for rewards from tested samples
    _testingReward.resize(_testingSampleIds.size());
  }
}

void Agent::runGeneration()
{
  if (_mode == "Training") trainingGeneration();
  if (_mode == "Testing") testingGeneration();
}

void Agent::trainingGeneration()
{
  auto beginTime = std::chrono::steady_clock::now(); // Profiling

  // Setting generation-specific timers
  _generationRunningTime = 0.0;
  _generationSerializationTime = 0.0;
  _generationAgentComputationTime = 0.0;
  _generationAgentCommunicationTime = 0.0;
  _generationAgentPolicyEvaluationTime = 0.0;
  _generationPolicyUpdateTime = 0.0;
  _generationAgentAttendingTime = 0.0;

  // Running until all _agents have finished
  while (_sessionEpisodeCount < _episodesPerGeneration * _sessionGeneration)
  {
    // Launching (or re-launching) agents
    for (size_t agentId = 0; agentId < _concurrentEnvironments; agentId++)
      if (_isAgentRunning[agentId] == false)
      {
        _agents[agentId]["Sample Id"] = _currentEpisode++;
        _agents[agentId]["Module"] = "Problem";
        _agents[agentId]["Operation"] = "Run Training Episode";
        for (size_t p = 0; p < _problem->_policiesPerEnvironment; p++)
          _agents[agentId]["Policy Hyperparameters"][p] = _trainingCurrentPolicies["Policy Hyperparameters"][p];
        _agents[agentId]["State Rescaling"]["Means"] = _stateRescalingMeans;
        _agents[agentId]["State Rescaling"]["Standard Deviations"] = _stateRescalingSigmas;

        KORALI_START(_agents[agentId]);

        _isAgentRunning[agentId] = true;
      }

    // Listening to _agents for incoming experiences
    KORALI_LISTEN(_agents);

    // Attending to running agents, checking if any experience has been received
    for (size_t agentId = 0; agentId < _concurrentEnvironments; agentId++)
      if (_isAgentRunning[agentId] == true)
        attendAgent(agentId);

    // Perform optimization steps on the critic/policy, if reached the minimum replay memory size
    if (_experienceCount >= _experienceReplayStartSize)
    {
        
      if (_policyUpdateCount == 0)
      {
        // If we accumulated enough experiences, we rescale the states once
        if (_stateRescalingEnabled == true)
          rescaleStates();
     
        // If we accumulated enough experiences, we init the reward rescaling once
        if (_rewardRescalingEnabled == true)
          initRewardRescaling();
      }


      // If we accumulated enough experiences between updates in this session, update now
      while (_sessionExperienceCount > (_experiencesBetweenPolicyUpdates * _sessionPolicyUpdateCount + _sessionExperiencesUntilStartSize))
      {
        auto beginTime = std::chrono::steady_clock::now(); // Profiling

        // Calling the algorithm specific policy training algorithm
        trainPolicy();

        auto endTime = std::chrono::steady_clock::now();                                                                  // Profiling
        _sessionPolicyUpdateTime += std::chrono::duration_cast<std::chrono::nanoseconds>(endTime - beginTime).count();    // Profiling
        _generationPolicyUpdateTime += std::chrono::duration_cast<std::chrono::nanoseconds>(endTime - beginTime).count(); // Profiling

        // Increasing policy update counters
        _policyUpdateCount++;
        _sessionPolicyUpdateCount++;

        // Updating the off policy cutoff
        _experienceReplayOffPolicyCurrentCutoff = _experienceReplayOffPolicyCutoffScale / (1.0f + _experienceReplayOffPolicyAnnealingRate * (float)_policyUpdateCount);

        for (size_t d = 0; d < _problem->_agentsPerEnvironment; d++)
        {
          // Updating REFER learning rate and beta parameters
          _currentLearningRate = _learningRate / (1.0f + _experienceReplayOffPolicyAnnealingRate * (float)_policyUpdateCount);
          if (_experienceReplayOffPolicyRatio[d] > _experienceReplayOffPolicyTarget)
            _experienceReplayOffPolicyREFERCurrentBeta[d] = (1.0f - _currentLearningRate) * _experienceReplayOffPolicyREFERCurrentBeta[d];
          else
            _experienceReplayOffPolicyREFERCurrentBeta[d] = (1.0f - _currentLearningRate) * _experienceReplayOffPolicyREFERCurrentBeta[d] + _currentLearningRate;
        }
      }

      // Getting new policy hyperparameters (for agents to generate actions)
      _trainingCurrentPolicies = getAgentPolicy();
    }
  }

  // Now serializing experience replay database
  if (_experienceReplaySerialize == true)
    if (_k->_fileOutputEnabled)
      if (_k->_fileOutputFrequency > 0)
        if (_k->_currentGeneration % _k->_fileOutputFrequency == 0)
          serializeExperienceReplay();

  // Measuring generation time
  auto endTime = std::chrono::steady_clock::now();                                                             // Profiling
  _sessionRunningTime += std::chrono::duration_cast<std::chrono::nanoseconds>(endTime - beginTime).count();    // Profiling
  _generationRunningTime += std::chrono::duration_cast<std::chrono::nanoseconds>(endTime - beginTime).count(); // Profiling

  /*********************************************************************
   * Updating statistics/bookkeeping
   *********************************************************************/

  // Updating average cumulative reward statistics
  _trainingAverageReward = std::vector<float>(_problem->_agentsPerEnvironment, 0.0f);
  for (size_t d = 0; d < _problem->_agentsPerEnvironment; d++)
  {
    ssize_t startEpisodeId = _trainingRewardHistory[d].size() - _trainingAverageDepth;
    ssize_t endEpisodeId = _trainingRewardHistory[d].size() - 1;
    if (startEpisodeId < 0) startEpisodeId = 0;
    for (ssize_t e = startEpisodeId; e <= endEpisodeId; e++)
      _trainingAverageReward[d] += _trainingRewardHistory[d][e];
    _trainingAverageReward[d] /= (float)(endEpisodeId - startEpisodeId + 1);
  }

  // Increasing session's generation count
  _sessionGeneration++;
}

void Agent::testingGeneration()
{
  // Allocating testing agents
  std::vector<Sample> testingAgents(_testingSampleIds.size());

  // Launching  agents
  for (size_t sampleId = 0; sampleId < _testingSampleIds.size(); sampleId++)
  {
    testingAgents[sampleId]["Sample Id"] = _testingSampleIds[sampleId];
    testingAgents[sampleId]["Module"] = "Problem";
    testingAgents[sampleId]["Operation"] = "Run Testing Episode";
    for (size_t p = 0; p < _problem->_policiesPerEnvironment; p++)
      testingAgents[sampleId]["Policy Hyperparameters"][p] = _testingCurrentPolicies["Policy Hyperparameters"][p];
    testingAgents[sampleId]["State Rescaling"]["Means"] = _stateRescalingMeans;
    testingAgents[sampleId]["State Rescaling"]["Standard Deviations"] = _stateRescalingSigmas;

    KORALI_START(testingAgents[sampleId]);
  }

  KORALI_WAITALL(testingAgents);

  for (size_t sampleId = 0; sampleId < _testingSampleIds.size(); sampleId++)
    _testingReward[sampleId] = testingAgents[sampleId]["Testing Reward"].get<float>();
}

void Agent::rescaleStates()
{
  // Calculation of state moments
  std::vector<std::vector<float>> sumStates(_problem->_agentsPerEnvironment, std::vector<float>(_problem->_stateVectorSize, 0.0f));
  std::vector<std::vector<float>> squaredSumStates(_problem->_agentsPerEnvironment, std::vector<float>(_problem->_stateVectorSize, 0.0f));

  for (size_t i = 0; i < _stateVector.size(); ++i)
    for (size_t j = 0; j < _problem->_agentsPerEnvironment; ++j)
      for (size_t d = 0; d < _problem->_stateVectorSize; ++d)
      {
        sumStates[j][d] += _stateVector[i][j][d];
        squaredSumStates[j][d] += _stateVector[i][j][d] * _stateVector[i][j][d];
      }

  _k->_logger->logInfo("Detailed", " + Using State Normalization N(Mean, Sigma):\n");

  for (size_t j = 0; j < _problem->_agentsPerEnvironment; ++j)
    for (size_t d = 0; d < _problem->_stateVectorSize; ++d)
    {
      _stateRescalingMeans[j][d] = sumStates[j][d] / (float)_stateVector.size();
      if (std::isfinite(_stateRescalingMeans[j][d]) == false) _stateRescalingMeans[j][d] = 0.0f;

      _stateRescalingSigmas[j][d] = std::sqrt(squaredSumStates[j][d] / (float)_stateVector.size() - _stateRescalingMeans[j][d] * _stateRescalingMeans[j][d]);
      if (std::isfinite(_stateRescalingSigmas[j][d]) == false) _stateRescalingSigmas[j][d] = 1.0f;
      if (_stateRescalingSigmas[j][d] <= 1e-9) _stateRescalingSigmas[j][d] = 1.0f;

      _k->_logger->logInfo("Detailed", " + State [%zu]: N(%f, %f)\n", d, _stateRescalingMeans[j][d], _stateRescalingSigmas[j][d]);
    }

  // Actual rescaling of initial states
  for (size_t i = 0; i < _stateVector.size(); ++i)
    for (size_t j = 0; j < _problem->_agentsPerEnvironment; ++j)
      for (size_t d = 0; d < _problem->_stateVectorSize; ++d)
        _stateVector[i][j][d] = (_stateVector[i][j][d] - _stateRescalingMeans[j][d]) / _stateRescalingSigmas[j][d];
}

void Agent::initRewardRescaling()
{



}


void Agent::attendAgent(size_t agentId)
{
  auto beginTime = std::chrono::steady_clock::now(); // Profiling

  // Storage for the incoming message
  knlohmann::json message;

  // Retrieving the experience, if any has arrived for the current agent.
  if (_agents[agentId].retrievePendingMessage(message))
  {
    // Getting episode Id
    size_t episodeId = message["Sample Id"];
    message["Episodes"]["Sample Id"] = episodeId;

    // If agent requested new policy, send the new hyperparameters
    if (message["Action"] == "Request New Policy")
    {
      KORALI_SEND_MSG_TO_SAMPLE(_agents[agentId], _trainingCurrentPolicies["Policy Hyperparameters"]);
    }

    // Process episode(s) incoming from the agent(s)
    if (message["Action"] == "Send Episodes")
    {
      
      // Process every episode received and its experiences (add them to replay memory)
      const bool error = (message["Error"].get<size_t>() == 1);
      if (error == 0)
        processEpisode(message["Episodes"]);

      // Waiting for the agent to come back with all the information
      KORALI_WAIT(_agents[agentId]);
 
      if (error == 0)
      {
          // Getting the training reward of the latest episodes
          _trainingLastReward = _agents[agentId]["Training Rewards"].get<std::vector<float>>();

          // Keeping training statistics. Updating if exceeded best training policy so far.
          for (size_t d = 0; d < _problem->_agentsPerEnvironment; d++)
          {
            if (_trainingLastReward[d] > _trainingBestReward[d])
            {
              _trainingBestReward[d] = _trainingLastReward[d];
              _trainingBestEpisodeId[d] = episodeId;
            }
            _trainingRewardHistory[d].push_back(_trainingLastReward[d]);
          }
          // Storing bookkeeping information
          _trainingExperienceHistory.push_back(message["Episodes"]["Experiences"].size());

          // If the policy has exceeded the threshold during training, we gather its statistics
          if (_agents[agentId]["Tested Policy"] == true)
          {
            _testingCandidateCount++;
            _testingBestReward = _agents[agentId]["Best Testing Reward"].get<float>();
            _testingWorstReward = _agents[agentId]["Worst Testing Reward"].get<float>();
            _testingAverageReward = _agents[agentId]["Average Testing Reward"].get<float>();
            _testingAverageRewardHistory.push_back(_testingAverageReward);

            // If the average testing reward is better than the previous best, replace it
            // and store hyperparameters as best so far.
            if (_testingAverageReward > _testingBestAverageReward)
            {
              _testingBestAverageReward = _testingAverageReward;
              _testingBestEpisodeId = episodeId;
              for (size_t d = 0; d < _problem->_policiesPerEnvironment; ++d)
                _testingBestPolicies["Policy Hyperparameters"][d] = _agents[agentId]["Policy Hyperparameters"][d];
            }
          }
      }

      // Obtaining profiling information
      _sessionAgentComputationTime += _agents[agentId]["Computation Time"].get<double>();
      _sessionAgentCommunicationTime += _agents[agentId]["Communication Time"].get<double>();
      _sessionAgentPolicyEvaluationTime += _agents[agentId]["Policy Evaluation Time"].get<double>();
      _generationAgentComputationTime += _agents[agentId]["Computation Time"].get<double>();
      _generationAgentCommunicationTime += _agents[agentId]["Communication Time"].get<double>();
      _generationAgentPolicyEvaluationTime += _agents[agentId]["Policy Evaluation Time"].get<double>();

      // Set agent as finished
      _isAgentRunning[agentId] = false;

      // Increasing session episode count
      _sessionEpisodeCount++;
    }
  }

  auto endTime = std::chrono::steady_clock::now();                                                                    // Profiling
  _sessionAgentAttendingTime += std::chrono::duration_cast<std::chrono::nanoseconds>(endTime - beginTime).count();    // Profiling
  _generationAgentAttendingTime += std::chrono::duration_cast<std::chrono::nanoseconds>(endTime - beginTime).count(); // Profiling
}

void Agent::processEpisode(knlohmann::json &episode)
{
  /*********************************************************************
   * Adding episode's experiences into the replay memory
   *********************************************************************/
  size_t episodeId = episode["Sample Id"];
          
  // Increasing total experience counters
  _experienceCount += episode["Experiences"].size();
  _sessionExperienceCount += episode["Experiences"].size();

  // Storage for the episode's cumulative reward
  std::vector<float> cumulativeReward(_problem->_agentsPerEnvironment, 0.0f);

  // Go over experiences in episode
  const size_t episodeExperienceCount = episode["Experiences"].size();
  for (size_t expId = 0; expId < episodeExperienceCount; expId++)
  {
    // Put state to replay memory
    _stateVector.add(episode["Experiences"][expId]["State"].get<std::vector<std::vector<float>>>());

    // Get action and put it to replay memory
    _actionVector.add(episode["Experiences"][expId]["Action"].get<std::vector<std::vector<float>>>());

    // Get reward
    std::vector<float> reward = episode["Experiences"][expId]["Reward"].get<std::vector<float>>();

    // For cooporative multi-agent model rewards are averaged
    if (_multiAgentRelationship == "Cooperation")
    {
      float avgReward = std::accumulate(reward.begin(), reward.end(), 0.);
      avgReward /= _problem->_agentsPerEnvironment;
      reward = std::vector<float>(_problem->_agentsPerEnvironment, avgReward);
    }

    // Update reward rescaling moments
    if (_rewardRescalingEnabled)
    {
      if (_rewardVector.size() >= _experienceReplayMaximumSize)
      {
        for (size_t a = 0; a < _problem->_agentsPerEnvironment; a++)
          _rewardRescalingSumSquaredRewards[a] -= (_rewardVector[0][a] - _rewardRescalingMeans[a]) * (_rewardVector[0][a] - _rewardRescalingMeans[a]);
      }
      for (size_t a = 0; a < _problem->_agentsPerEnvironment; a++)
      {
        _rewardRescalingSumSquaredRewards[a] += (reward[a] - _rewardRescalingMeans[a]) * (reward[a]-_rewardRescalingMeans[a]);
      }
    }

    // Put reward to replay memory
    _rewardVector.add(reward);

    // Keeping statistics
    for (size_t d = 0; d < _problem->_agentsPerEnvironment; d++)
      cumulativeReward[d] += reward[d];

    // Checking and adding experience termination status and truncated state to replay memory
    termination_t termination;
    std::vector<std::vector<float>> truncatedState;
    std::vector<float> truncatedStateValue;

    if (episode["Experiences"][expId]["Termination"] == "Non Terminal") termination = e_nonTerminal;
    if (episode["Experiences"][expId]["Termination"] == "Terminal") termination = e_terminal;
    if (episode["Experiences"][expId]["Termination"] == "Truncated")
    {
      termination = e_truncated;
      truncatedState = episode["Experiences"][expId]["Truncated State"].get<std::vector<std::vector<float>>>();
    }

    _terminationVector.add(termination);
    _truncatedStateVector.add(truncatedState);
    _truncatedStateValueVector.add(truncatedStateValue);

    // Getting policy information and state value
    std::vector<policy_t> expPolicy(_problem->_agentsPerEnvironment);
    std::vector<float> stateValue(_problem->_agentsPerEnvironment);

    if (isDefined(episode["Experiences"][expId], "Policy", "State Value"))
    {
      stateValue = episode["Experiences"][expId]["Policy"]["State Value"].get<std::vector<float>>();
      for (size_t d = 0; d < _problem->_agentsPerEnvironment; d++)
      {
        expPolicy[d].stateValue = stateValue[d];
      }
    }
    else
    {
      KORALI_LOG_ERROR("Policy has not produced state value for the current experience.\n");
    }
    _stateValueVector.add(stateValue);

    /* Story policy information for continuous action spaces */
    if (isDefined(episode["Experiences"][expId], "Policy", "Distribution Parameters"))
    {
      const auto distParams = episode["Experiences"][expId]["Policy"]["Distribution Parameters"].get<std::vector<std::vector<float>>>();
      for (size_t d = 0; d < _problem->_agentsPerEnvironment; d++)
        expPolicy[d].distributionParameters = distParams[d];
    }

    if (isDefined(episode["Experiences"][expId], "Policy", "Unbounded Action"))
    {
      const auto unboundedAc = episode["Experiences"][expId]["Policy"]["Unbounded Action"].get<std::vector<std::vector<float>>>();
      for (size_t d = 0; d < _problem->_agentsPerEnvironment; d++)
        expPolicy[d].unboundedAction = unboundedAc[d];
    }

    /* Story policy information for discrete action spaces */
    if (isDefined(episode["Experiences"][expId], "Policy", "Action Index"))
    {
      const auto actIdx = episode["Experiences"][expId]["Policy"]["Action Index"].get<std::vector<size_t>>();
      for (size_t d = 0; d < _problem->_agentsPerEnvironment; d++)
        expPolicy[d].actionIndex = actIdx[d];
    }

    if (isDefined(episode["Experiences"][expId], "Policy", "Action Probabilities"))
    {
      const auto actProb = episode["Experiences"][expId]["Policy"]["Action Probabilities"].get<std::vector<std::vector<float>>>();
      for (size_t d = 0; d < _problem->_agentsPerEnvironment; d++)
        expPolicy[d].actionProbabilities = actProb[d];
    }

    if (isDefined(episode["Experiences"][expId], "Policy", "Available Actions"))
    {
      const auto availAct = episode["Experiences"][expId]["Policy"]["Available Actions"].get<std::vector<std::vector<size_t>>>();
      for (size_t d = 0; d < _problem->_agentsPerEnvironment; d++)
      {
        expPolicy[d].availableActions = availAct[d];
        if (std::accumulate(expPolicy[d].availableActions.begin(), expPolicy[d].availableActions.end(), 0) == 0)
          KORALI_LOG_ERROR("State with experience id %zu for agent %zu detected with no available actions.", expId, d);
      }
    }

    // Storing policy information in replay memory
    _expPolicyVector.add(expPolicy);
    _curPolicyVector.add(expPolicy);

    // Storing Episode information in replay memory
    _episodeIdVector.add(episodeId);
    _episodePosVector.add(expId);

    // Adding placeholder for retrace value
    _retraceValueVector.add(std::vector<float>(_problem->_agentsPerEnvironment, 0.0f));

    // If outgoing experience is off policy, subtract off policy counter
    if (_isOnPolicyVector.size() == _experienceReplayMaximumSize)
    {
      const auto &onPolicyVector = _isOnPolicyVector[0];

      size_t count = 1;
      // Consider all observation for the off-policy statistics
      if (_problem->_policiesPerEnvironment == 1)
        count = std::count(onPolicyVector.begin(), onPolicyVector.end(), false);

      // Update offPolicyCount
      for (size_t d = 0; d < _problem->_agentsPerEnvironment; d++)
        if ((onPolicyVector[d] == false) || (_problem->_policiesPerEnvironment == 1))
        {
          // Safety check for overflow
          if (_experienceReplayOffPolicyCount[d] < count)
          {
            KORALI_LOG_ERROR("Agent %ld: _experienceReplayOffPolicyCount=%ld smaller than decrement %ld.\n", d, _experienceReplayOffPolicyCount[d], count);
            // count = _experienceReplayOffPolicyCount[d];
          }

          // Update off-policy count
          _experienceReplayOffPolicyCount[d] -= count;
        }
    }

    // Adding new experience's on policiness (by default is true when adding it to the ER)
    _isOnPolicyVector.add(std::vector<char>(_problem->_agentsPerEnvironment, true));

    // Initialize experience's importance weight (1.0 because its freshly produced)
    _importanceWeightVector.add(std::vector<float>(_problem->_agentsPerEnvironment, 1.0f));
    _truncatedImportanceWeightVector.add(std::vector<float>(_problem->_agentsPerEnvironment, 1.0f));
    _productImportanceWeightVector.add(1.0f);
  }

  /*********************************************************************
   * Computing initial retrace value for the newly added experiences
   *********************************************************************/

  // Getting position of the final experience of the episode in the replay memory
  ssize_t endId = (ssize_t)_stateVector.size() - 1;

  // Getting the starting ID of the initial experience of the episode in the replay memory
  ssize_t startId = endId - episodeExperienceCount + 1;

  // Storage for the retrace value
  std::vector<float> retV(_problem->_agentsPerEnvironment, 0.0f);

  // If it was a truncated episode, add the value function for the terminal state to retV
  if (_terminationVector[endId] == e_truncated)
  {
    for (size_t d = 0; d < _problem->_agentsPerEnvironment; d++)
    {
      // Get truncated state
      auto expTruncatedStateSequence = getTruncatedStateSequence(endId, d);

      // Forward tuncated state. Take policy d if there is multiple policies, otherwise policy 0
      std::vector<policy_t> truncatedPolicy;
      if (_problem->_policiesPerEnvironment == 1)
        retV[d] += calculateStateValue(expTruncatedStateSequence);
      else
        retV[d] += calculateStateValue(expTruncatedStateSequence, d);

      // Get value of trucated state
      if (std::isfinite(retV[d]) == false)
        KORALI_LOG_ERROR("Calculated state value for truncated state returned an invalid value: %f\n", retV[d]);
    }

    // For cooporative multi-agent model truncated state-values are averaged
    if (_multiAgentRelationship == "Cooperation")
    {
      float avgRetV = std::accumulate(retV.begin(), retV.end(), 0.);
      avgRetV /= _problem->_agentsPerEnvironment;
      retV = std::vector<float>(_problem->_agentsPerEnvironment, avgRetV);
    }

    // The value of the truncated state equals initial retrace Value
    _truncatedStateValueVector[endId] = retV;
  }

  // Now going backwards, setting the retrace value of every experience
  for (ssize_t expId = endId; expId >= startId; expId--)
  {
    for (size_t a = 0; a < _problem->_agentsPerEnvironment; a++)
    {
      // Calculating retrace value. Importance weight is 1.0f because the policy is current.
      retV[a] = getScaledReward(a, _rewardVector[expId][a]) + _discountFactor * retV[a];
      _retraceValueVector[expId][a] = retV[a];
    }
  }

  // Update reward rescaling sigma
  if (_rewardRescalingEnabled)
    for (size_t a = 0; a < _problem->_agentsPerEnvironment; a++)
    {
    _rewardRescalingSigmas[a] = std::sqrt(_rewardRescalingSumSquaredRewards[a] / _rewardVector.size() );
    }
}

std::vector<std::pair<size_t, size_t>> Agent::generateMiniBatch()
{
  // Allocating storage for mini batch experiecne indexes
  std::vector<std::pair<size_t, size_t>> miniBatch(_effectiveMinibatchSize);

  // Get number of agents
  const size_t numAgents = _problem->_agentsPerEnvironment;

  // Fill minibatch
  for (size_t b = 0; b < _miniBatchSize; b++)
  {
    // Producing random (uniform) number for the selection of the experience
    float x = _uniformGenerator->getRandomNumber();

    // Selecting experience
    size_t expId = std::floor(x * (float)(_stateVector.size() - 1));

    for (size_t a = 0; a < numAgents; a++)
    {
      // Setting experience
      miniBatch[b * numAgents + a].first = expId;
      miniBatch[b * numAgents + a].second = a;

      // Sample agentId
      if (_multiAgentSampling == "Experience")
      {
        // Producing random (uniform) number for the selection of the experience
        float ax = _uniformGenerator->getRandomNumber();

        // Selecting agent
        miniBatch[b * numAgents + a].second = std::floor(ax * (float)(numAgents - 1));
      }

      // Sample both
      if (_multiAgentSampling == "Baseline")
      {
        // Producing random (uniform) number for the selection of the experience
        float ex = _uniformGenerator->getRandomNumber();
        float ax = _uniformGenerator->getRandomNumber();

        // Selecting experience
        miniBatch[b * numAgents + a].first = std::floor(ex * (float)(_stateVector.size() - 1));

        // Selecting agent
        miniBatch[b * numAgents + a].second = std::floor(ax * (float)(numAgents - 1));
      }
    }
  }

  // Sorting minibatch: first by expId, second by agentId
  // to quickly detect duplicates when updating metadata
  std::sort(miniBatch.begin(), miniBatch.end(), [numAgents](const std::pair<size_t, size_t> &exp0, const std::pair<size_t, size_t> &exp1) -> bool {
    return exp0.first * numAgents + exp0.second < exp1.first * numAgents + exp1.second;
  });

  // Returning generated minibatch
  return miniBatch;
}

std::vector<std::vector<std::vector<float>>> Agent::getMiniBatchStateSequence(const std::vector<std::pair<size_t, size_t>> &miniBatch)
{
  // Allocating state sequence vector
  std::vector<std::vector<std::vector<float>>> stateSequence(_effectiveMinibatchSize);

#pragma omp parallel for
  for (size_t b = 0; b < _effectiveMinibatchSize; b++)
  {
    // Getting current expId and agentId
    const size_t expId = miniBatch[b].first;
    const size_t agentId = miniBatch[b].second;

    // Getting starting expId
    const size_t startId = getTimeSequenceStartExpId(expId);

    // Calculating time sequence length
    const size_t T = expId - startId + 1;

    // Resizing state sequence vector to the correct time sequence length
    stateSequence[b].resize(T);
    for (size_t t = 0; t < T; t++)
    {
      // Now adding states
      const size_t sequenceId = startId + t;
      stateSequence[b][t].reserve(_problem->_stateVectorSize);
      stateSequence[b][t].insert(stateSequence[b][t].begin(), _stateVector[sequenceId][agentId].begin(), _stateVector[sequenceId][agentId].end());
    }
  }

  return stateSequence;
}

void Agent::updateExperienceMetadata(const std::vector<std::pair<size_t, size_t>> &miniBatch, const std::vector<policy_t> &policyData)
{
  const size_t miniBatchSize = miniBatch.size();
  const size_t numAgents = _problem->_agentsPerEnvironment;

  /* Creating a selection of unique expIds, agentIds
   * Important: this assumes the minibatch ids are sorted.
   */

  // Create Buffers
  std::vector<size_t> updateBatch;
  std::vector<std::pair<size_t, size_t>> updateMinibatch;
  std::vector<policy_t> updatePolicyData;

  // Fill updateMinibatch and updatePolicyData
  size_t b = 0;
  while (b < miniBatchSize)
  {
    // Add new unique combination
    updateBatch.push_back(b);
    updateMinibatch.push_back(miniBatch[b]);
    updatePolicyData.push_back(policyData[b]);

    size_t a = 1;

    // Iterate over experiences with same expId
    while ((miniBatch[b + a].first == miniBatch[b + a - 1].first) && (b + a < miniBatchSize))
    {
      // Add unique experiences from agents
      if (miniBatch[b + a].second != miniBatch[b + a - 1].second)
      {
        updateMinibatch.push_back(miniBatch[b + a]);
        updatePolicyData.push_back(policyData[b + a]);
      }
      a++;
    }

    // Increment batch counter by the number of same expIds
    b += a;
  }

  // Container to compute offpolicy count difference in minibatch
  std::vector<int> offPolicyCountDelta(numAgents, 0);

#pragma omp parallel for reduction(vec_int_plus \
                                   : offPolicyCountDelta)
  for (size_t i = 0; i < updateMinibatch.size(); i++)
  {
    // Get current expId and agentId
    const size_t expId = updateMinibatch[i].first;
    const size_t agentId = updateMinibatch[i].second;

    // Get and set current policy
    const auto &curPolicy = updatePolicyData[i];
    _curPolicyVector[expId][agentId] = curPolicy;

    // Get state value
    _stateValueVector[expId][agentId] = curPolicy.stateValue;
    if (std::isfinite(curPolicy.stateValue) == false)
      KORALI_LOG_ERROR("Calculated state value returned an invalid value: %f\n", curPolicy.stateValue);

    // Get action and policy for this experience
    const auto &expAction = _actionVector[expId][agentId];
    const auto &expPolicy = _expPolicyVector[expId][agentId];

    // Compute importance weight
    const float importanceWeight = calculateImportanceWeight(expAction, curPolicy, expPolicy);
    if (std::isfinite(importanceWeight) == false)
      KORALI_LOG_ERROR("Calculated value of importanceWeight returned an invalid value: %f\n", importanceWeight);

    // Set importance weight and truncated importance weight
    _importanceWeightVector[expId][agentId] = importanceWeight;
    _truncatedImportanceWeightVector[expId][agentId] = std::min(_importanceWeightTruncationLevel, importanceWeight);

    // Keep track of off-policyness (in principle only necessary for agentId==policyId)
    if (not _multiAgentCorrelation)
    {
      // Checking if experience is on policy
      const bool isOnPolicy = (importanceWeight > (1.0f / _experienceReplayOffPolicyCurrentCutoff)) && (importanceWeight < _experienceReplayOffPolicyCurrentCutoff);

      // Updating off policy count if a change is detected
      if (_isOnPolicyVector[expId][agentId] == true && isOnPolicy == false)
        offPolicyCountDelta[agentId]++;

      if (_isOnPolicyVector[expId][agentId] == false && isOnPolicy == true)
        offPolicyCountDelta[agentId]--;

      // Write to onPolicy vector
      _isOnPolicyVector[expId][agentId] = isOnPolicy;
    }

    // Update truncated state value
    if (_terminationVector[expId] == e_truncated)
    {
      // Get truncated state
      auto expTruncatedStateSequence = getTruncatedStateSequence(expId, agentId);

      // Forward tuncated state
      // TODO: other policy for exp-sharing in multi-policy case??
      float truncatedStateValue;
      if (_problem->_policiesPerEnvironment == 1)
        truncatedStateValue = calculateStateValue(expTruncatedStateSequence);
      else
        truncatedStateValue = calculateStateValue(expTruncatedStateSequence, agentId);

      // Check value of trucated state
      if (std::isfinite(truncatedStateValue) == false)
        KORALI_LOG_ERROR("Calculated state value for truncated state returned an invalid value: %f\n", truncatedStateValue);

      // Write truncated state value
      _truncatedStateValueVector[expId][agentId] = truncatedStateValue;
    }
  }

  /* Now taking care of advanced correlation features for MARL */

  if (_multiAgentCorrelation)
  {
#pragma omp parallel for reduction(vec_int_plus \
                                   : offPolicyCountDelta)
    for (size_t i = 0; i < updateBatch.size(); i++)
    {
      const size_t batchId = updateBatch[i];
      const size_t expId = miniBatch[batchId].first;

      // Load importance weight for expId
      const auto &importanceWeight = _importanceWeightVector[expId];

      // Compute product of importance weights
      float logProdImportanceWeight = 0.0f;
      for (size_t d = 0; d < numAgents; d++)
      {
        // Numerical safety [ ln(10)=2.30258509299 ]
        if (importanceWeight[d] == 0)
          logProdImportanceWeight += 2.30258509299 * std::numeric_limits<float>::min_exponent10;
        else
          logProdImportanceWeight += std::log(importanceWeight[d]);
      }

      // Compute cut-off in log-space
      const float logCutOff = (float)_problem->_agentsPerEnvironment * std::log(_experienceReplayOffPolicyCurrentCutoff);

      // Check whether experience is onPolicy
      const bool onPolicy = (logProdImportanceWeight > (-1. * logCutOff)) && (logProdImportanceWeight < logCutOff);

      // Load isOnPolicy
      auto &isOnPolicy = _isOnPolicyVector[expId];

      // Write to prodImportanceWeight vector
      _productImportanceWeightVector[expId] = std::exp(logProdImportanceWeight);

      // Record change of on-policyness
      for (size_t d = 0; d < numAgents; d++)
      {
        if (isOnPolicy[d] == true && onPolicy == false)
          offPolicyCountDelta[d]++;

        if (isOnPolicy[d] == false && onPolicy == true)
          offPolicyCountDelta[d]--;
      }

      // Overwrite onPolicyVector
      std::fill(isOnPolicy.begin(), isOnPolicy.end(), onPolicy);
    }
  }

  // Average state values for cooperative MA
  if (_multiAgentRelationship == "Cooperation")
  {
#pragma omp parallel for
    for (size_t i = 0; i < updateBatch.size(); i++)
    {
      const size_t batchId = updateBatch[i];
      const size_t expId = miniBatch[batchId].first;

      // Load state-value
      auto &stateValue = _stateValueVector[expId];

      // Average state-value
      float averageStateValue = 0.0f;
      for (size_t d = 0; d < numAgents; d++)
        averageStateValue += stateValue[d];
      averageStateValue /= _problem->_agentsPerEnvironment;

      // Overwrite state value with average
      std::fill(stateValue.begin(), stateValue.end(), averageStateValue);

      // Same for truncated state-value
      if (_terminationVector[expId] == e_truncated)
      {
        // Load truncated state value
        auto &truncatedStateValue = _truncatedStateValueVector[expId];

        // Average truncated state value
        float averageTruncatedStateValue = std::accumulate(truncatedStateValue.begin(), truncatedStateValue.end(), 0.);
        averageTruncatedStateValue /= _problem->_agentsPerEnvironment;

        // Overwrite truncated state value with average
        std::fill(truncatedStateValue.begin(), truncatedStateValue.end(), averageTruncatedStateValue);
      }
    }
  }

  /* Taking care of off-policy statistics */

  if (_problem->_policiesPerEnvironment == 1)
  {
    // Consider all observation for the off-policy statistics
    int sumOffPolicyCountDelta = std::accumulate(offPolicyCountDelta.begin(), offPolicyCountDelta.end(), 0.);
    offPolicyCountDelta = std::vector<int>(_problem->_agentsPerEnvironment, sumOffPolicyCountDelta);
  }

  // Updating the off policy count and ratio
  for (size_t d = 0; d < numAgents; d++)
  {
    // Safety check for overflow
    if ((int)_experienceReplayOffPolicyCount[d] < -offPolicyCountDelta[d])
    {
      KORALI_LOG_ERROR("Agent %ld: offPolicyCountDelta=%d bigger than _experienceReplayOffPolicyCount=%ld.\n", d, offPolicyCountDelta[d], _experienceReplayOffPolicyCount[d]);
      // offPolicyCountDelta[d] = -(int)_experienceReplayOffPolicyCount[d];
    }

    // Update off policy count
    _experienceReplayOffPolicyCount[d] += offPolicyCountDelta[d];
    _experienceReplayOffPolicyRatio[d] = (float)_experienceReplayOffPolicyCount[d] / (float)_isOnPolicyVector.size();

    // Normalize off policy Ratio
    if (_problem->_policiesPerEnvironment == 1)
      _experienceReplayOffPolicyRatio[d] /= (float)(_problem->_agentsPerEnvironment);
  }

  /* Update Retrace value */

  // Now filtering experiences from the same episode
  std::vector<size_t> retraceMiniBatch;

  // Adding last experience from the sorted minibatch
  retraceMiniBatch.push_back(miniBatch[updateBatch.back()].first);

  // Adding experiences so long as they do not repeat episodes
  for (ssize_t i = updateBatch.size() - 2; i >= 0; i--)
  {
    size_t currExpId = miniBatch[updateBatch[i]].first;
    size_t nextExpId = miniBatch[updateBatch[i + 1]].first;
    size_t curEpisode = _episodeIdVector[currExpId];
    size_t nextEpisode = _episodeIdVector[nextExpId];
    if (curEpisode != nextEpisode) retraceMiniBatch.push_back(currExpId);
  }

// Calculating retrace value for the oldest experiences of unique episodes
#pragma omp parallel for schedule(guided, 1)
  for (size_t i = 0; i < retraceMiniBatch.size(); i++)
  {
    // Determine start of the episode
    ssize_t endId = retraceMiniBatch[i];
    ssize_t startId = endId - _episodePosVector[endId];

    // If start of episode has been discarded, take earliest one
    if (startId < 0) startId = 0;

    // Storage for the retrace value
    std::vector<float> retV(_problem->_agentsPerEnvironment, 0.0f);

    // For truncated episode, set truncated state value function
    if (_terminationVector[endId] == e_truncated)
      retV = _truncatedStateValueVector[endId];

    // If non-terminal state, set next retrace value
    if (_terminationVector[endId] == e_nonTerminal)
      retV = _retraceValueVector[endId + 1];

    // Now iterating backwards and compute retrace value
    for (ssize_t curId = endId; curId >= startId; curId--)
    {
      // Load truncated importance weight
      std::vector<float> truncatedImportanceWeights = _truncatedImportanceWeightVector[curId];

      // Handle multi-agent correlation
      if (_multiAgentCorrelation)
      {
        // Truncate product of importance weight
        const float truncatedProdImportanceWeight = std::min(_importanceWeightTruncationLevel, _productImportanceWeightVector[curId]);

        // Overwrite truncated importance weights
        std::fill(truncatedImportanceWeights.begin(), truncatedImportanceWeights.end(), truncatedProdImportanceWeight);
      }

      // Load state value
      const auto &stateValue = _stateValueVector[curId];

      // Updated Retrace value
      for (size_t a = 0; a < _problem->_agentsPerEnvironment; a++)
      {
        // Getting current reward, action, and state
        const float curReward = getScaledReward(a, _rewardVector[curId][a]);

        // Apply recursion
        retV[a] = stateValue[a] + truncatedImportanceWeights[a] * (curReward + _discountFactor * retV[a] - stateValue[a]);
      }

      // Store retrace value
      _retraceValueVector[curId] = retV;
    }
  }
}

size_t Agent::getTimeSequenceStartExpId(size_t expId)
{
  const size_t episodePos = _episodePosVector[expId];

  // Determine actual length of time sequence
  const size_t lookBack = std::min(_timeSequenceLength - 1, episodePos);

  if (lookBack > expId)
    // Return start of buffer if expId is part of a cut episode
    return 0;
  else
    // Return time sequence start
    return expId - lookBack;
}

void Agent::resetTimeSequence()
{
  for (size_t agentId = 0; agentId < _problem->_agentsPerEnvironment; ++agentId)
    _stateTimeSequence[agentId].clear();
}

std::vector<std::vector<float>> Agent::getTruncatedStateSequence(size_t expId, size_t agentId)
{
  // Getting starting expId
  size_t startId = getTimeSequenceStartExpId(expId);

  // Creating storage for the time sequence
  std::vector<std::vector<float>> timeSequence;

  // Now adding states, except for the initial one
  for (size_t e = startId + 1; e <= expId; e++)
    timeSequence.push_back(_stateVector[e][agentId]);

  // Lastly, adding truncated state
  timeSequence.push_back(_truncatedStateVector[expId][agentId]);

  return timeSequence;
}

void Agent::finalize()
{
  if (_mode != "Training") return;

  if (_experienceReplaySerialize == true)
    if (_k->_fileOutputEnabled)
      serializeExperienceReplay();

  _k->_logger->logInfo("Normal", "Waiting for pending agents to finish...\n");

  // Waiting for pending agents to finish
  bool agentsRemain = true;
  do
  {
    agentsRemain = false;
    for (size_t agentId = 0; agentId < _concurrentEnvironments; agentId++)
      if (_isAgentRunning[agentId] == true)
      {
        attendAgent(agentId);
        agentsRemain = true;
      }

    if (agentsRemain) KORALI_LISTEN(_agents);
  } while (agentsRemain == true);
}

void Agent::serializeExperienceReplay()
{
  _k->_logger->logInfo("Detailed", "Serializing Agent's Training State...\n");
  auto beginTime = std::chrono::steady_clock::now(); // Profiling

  // Creating JSON storage variable
  knlohmann::json stateJson;

  // Serializing agent's database into the JSON storage
  for (size_t i = 0; i < _stateVector.size(); i++)
  {
    stateJson["Experience Replay"][i]["Episode Id"] = _episodeIdVector[i];
    stateJson["Experience Replay"][i]["Episode Pos"] = _episodePosVector[i];
    stateJson["Experience Replay"][i]["State"] = _stateVector[i];
    stateJson["Experience Replay"][i]["Action"] = _actionVector[i];
    stateJson["Experience Replay"][i]["Reward"] = _rewardVector[i];
    stateJson["Experience Replay"][i]["State Value"] = _stateValueVector[i];
    stateJson["Experience Replay"][i]["Retrace Value"] = _retraceValueVector[i];
    stateJson["Experience Replay"][i]["Importance Weight"] = _importanceWeightVector[i];
    stateJson["Experience Replay"][i]["Truncated Importance Weight"] = _truncatedImportanceWeightVector[i];
    stateJson["Experience Replay"][i]["Product Importance Weight"] = _productImportanceWeightVector[i];
    stateJson["Experience Replay"][i]["Is On Policy"] = _isOnPolicyVector[i];
    stateJson["Experience Replay"][i]["Truncated State"] = _truncatedStateVector[i];
    stateJson["Experience Replay"][i]["Truncated State Value"] = _truncatedStateValueVector[i];
    stateJson["Experience Replay"][i]["Termination"] = _terminationVector[i];

    std::vector<float> expStateValue(_problem->_agentsPerEnvironment, 0.0f);
    std::vector<std::vector<float>> expDistributionParameter(_problem->_agentsPerEnvironment, std::vector<float>(_expPolicyVector[0][0].distributionParameters.size()));
    std::vector<size_t> expActionIdx(_problem->_agentsPerEnvironment, 0);
    std::vector<std::vector<float>> expUnboundedAct(_problem->_agentsPerEnvironment, std::vector<float>(_expPolicyVector[0][0].unboundedAction.size()));
    std::vector<std::vector<float>> expActProb(_problem->_agentsPerEnvironment, std::vector<float>(_expPolicyVector[0][0].actionProbabilities.size()));
    std::vector<std::vector<size_t>> expAvailAct(_problem->_agentsPerEnvironment, std::vector<size_t>(_expPolicyVector[0][0].availableActions.size()));

    std::vector<float> curStateValue(_problem->_agentsPerEnvironment, 0.0f);
    std::vector<std::vector<float>> curDistributionParameter(_problem->_agentsPerEnvironment, std::vector<float>(_curPolicyVector[0][0].distributionParameters.size()));
    std::vector<size_t> curActionIdx(_problem->_agentsPerEnvironment, 0);
    std::vector<std::vector<float>> curUnboundedAct(_problem->_agentsPerEnvironment, std::vector<float>(_curPolicyVector[0][0].unboundedAction.size()));
    std::vector<std::vector<float>> curActProb(_problem->_agentsPerEnvironment, std::vector<float>(_curPolicyVector[0][0].actionProbabilities.size()));
    std::vector<std::vector<size_t>> curAvailAct(_problem->_agentsPerEnvironment, std::vector<size_t>(_curPolicyVector[0][0].availableActions.size()));

    for (size_t j = 0; j < _problem->_agentsPerEnvironment; j++)
    {
      expStateValue[j] = _expPolicyVector[i][j].stateValue;
      expDistributionParameter[j] = _expPolicyVector[i][j].distributionParameters;
      expActionIdx[j] = _expPolicyVector[i][j].actionIndex;
      expUnboundedAct[j] = _expPolicyVector[i][j].unboundedAction;
      expActProb[j] = _expPolicyVector[i][j].actionProbabilities;
      expAvailAct[j] = _expPolicyVector[i][j].availableActions;

      curStateValue[j] = _curPolicyVector[i][j].stateValue;
      curDistributionParameter[j] = _curPolicyVector[i][j].distributionParameters;
      curActionIdx[j] = _curPolicyVector[i][j].actionIndex;
      curUnboundedAct[j] = _curPolicyVector[i][j].unboundedAction;
      curActProb[j] = _curPolicyVector[i][j].actionProbabilities;
      curAvailAct[j] = _curPolicyVector[i][j].availableActions;
    }
    stateJson["Experience Replay"][i]["Experience Policy"]["State Value"] = expStateValue;
    stateJson["Experience Replay"][i]["Experience Policy"]["Distribution Parameters"] = expDistributionParameter;
    stateJson["Experience Replay"][i]["Experience Policy"]["Action Index"] = expActionIdx;
    stateJson["Experience Replay"][i]["Experience Policy"]["Unbounded Action"] = expUnboundedAct;
    stateJson["Experience Replay"][i]["Experience Policy"]["Action Probabilities"] = expActProb;
    stateJson["Experience Replay"][i]["Experience Policy"]["Available Actions"] = expAvailAct;

    stateJson["Experience Replay"][i]["Current Policy"]["State Value"] = curStateValue;
    stateJson["Experience Replay"][i]["Current Policy"]["Distribution Parameters"] = curDistributionParameter;
    stateJson["Experience Replay"][i]["Current Policy"]["Action Index"] = curActionIdx;
    stateJson["Experience Replay"][i]["Current Policy"]["Unbounded Action"] = curUnboundedAct;
    stateJson["Experience Replay"][i]["Current Policy"]["Action Probabilities"] = curActProb;
    stateJson["Experience Replay"][i]["Current Policy"]["Available Actions"] = curAvailAct;
  }

  // If results directory doesn't exist, create it
  if (!dirExists(_k->_fileOutputPath)) mkdir(_k->_fileOutputPath);

  // Resolving file path
  std::string statePath = _k->_fileOutputPath + "/state.json";

  // Storing database to file
  _k->_logger->logInfo("Detailed", "Saving json..\n");
  if (saveJsonToFile(statePath.c_str(), stateJson) != 0)
    KORALI_LOG_ERROR("Could not serialize training state into file %s\n", statePath.c_str());
  _k->_logger->logInfo("Detailed", "Agent's Training State serialized\n");

  auto endTime = std::chrono::steady_clock::now();                                                                   // Profiling
  _sessionSerializationTime += std::chrono::duration_cast<std::chrono::nanoseconds>(endTime - beginTime).count();    // Profiling
  _generationSerializationTime += std::chrono::duration_cast<std::chrono::nanoseconds>(endTime - beginTime).count(); // Profiling
}

void Agent::deserializeExperienceReplay()
{
  auto beginTime = std::chrono::steady_clock::now(); // Profiling

  // Creating JSON storage variable
  knlohmann::json stateJson;

  // Resolving file path
  std::string statePath = _k->_fileOutputPath + "/state.json";

  // Loading database from file
  _k->_logger->logInfo("Normal", "Loading previous run training state from file %s...\n", statePath.c_str());
  if (loadJsonFromFile(stateJson, statePath.c_str()) == false)
    KORALI_LOG_ERROR("Trying to resume training or test policy but could not find or deserialize agent's state from from file %s...\n", statePath.c_str());

  // Clearing existing database
  _stateVector.clear();
  _actionVector.clear();
  _retraceValueVector.clear();
  _rewardVector.clear();
  _stateValueVector.clear();
  _importanceWeightVector.clear();
  _truncatedImportanceWeightVector.clear();
  _truncatedStateValueVector.clear();
  _productImportanceWeightVector.clear();
  _truncatedStateVector.clear();
  _terminationVector.clear();
  _expPolicyVector.clear();
  _curPolicyVector.clear();
  _isOnPolicyVector.clear();
  _priorityVector.clear();
  _probabilityVector.clear();
  _episodePosVector.clear();
  _episodeIdVector.clear();

  // Deserializing database from JSON to the agent's state
  for (size_t i = 0; i < stateJson["Experience Replay"].size(); i++)
  {
    _episodeIdVector.add(stateJson["Experience Replay"][i]["Episode Id"].get<size_t>());
    _episodePosVector.add(stateJson["Experience Replay"][i]["Episode Pos"].get<size_t>());
    _stateVector.add(stateJson["Experience Replay"][i]["State"].get<std::vector<std::vector<float>>>());
    _actionVector.add(stateJson["Experience Replay"][i]["Action"].get<std::vector<std::vector<float>>>());
    _rewardVector.add(stateJson["Experience Replay"][i]["Reward"].get<std::vector<float>>());
    _stateValueVector.add(stateJson["Experience Replay"][i]["State Value"].get<std::vector<float>>());
    _retraceValueVector.add(stateJson["Experience Replay"][i]["Retrace Value"].get<std::vector<float>>());
    _importanceWeightVector.add(stateJson["Experience Replay"][i]["Importance Weight"].get<std::vector<float>>());
    _truncatedImportanceWeightVector.add(stateJson["Experience Replay"][i]["Truncated Importance Weight"].get<std::vector<float>>());
    _productImportanceWeightVector.add(stateJson["Experience Replay"][i]["Product Importance Weight"].get<float>());
    _isOnPolicyVector.add(stateJson["Experience Replay"][i]["Is On Policy"].get<std::vector<char>>());
    _truncatedStateVector.add(stateJson["Experience Replay"][i]["Truncated State"].get<std::vector<std::vector<float>>>());
    _truncatedStateValueVector.add(stateJson["Experience Replay"][i]["Truncated State Value"].get<std::vector<float>>());
    _terminationVector.add(stateJson["Experience Replay"][i]["Termination"].get<termination_t>());

    std::vector<policy_t> expPolicy(_problem->_agentsPerEnvironment);
    std::vector<policy_t> curPolicy(_problem->_agentsPerEnvironment);
    for (size_t d = 0; d < _problem->_agentsPerEnvironment; d++)
    {
      expPolicy[d].stateValue = stateJson["Experience Replay"][i]["Experience Policy"]["State Value"][d].get<float>();
      expPolicy[d].distributionParameters = stateJson["Experience Replay"][i]["Experience Policy"]["Distribution Parameters"][d].get<std::vector<float>>();
      expPolicy[d].unboundedAction = stateJson["Experience Replay"][i]["Experience Policy"]["Unbounded Action"][d].get<std::vector<float>>();
      expPolicy[d].actionIndex = stateJson["Experience Replay"][i]["Experience Policy"]["Action Index"][d].get<size_t>();
      expPolicy[d].actionProbabilities = stateJson["Experience Replay"][i]["Experience Policy"]["Action Probabilities"][d].get<std::vector<float>>();
      expPolicy[d].availableActions = stateJson["Experience Replay"][i]["Experience Policy"]["Available Actions"][d].get<std::vector<size_t>>();

      curPolicy[d].stateValue = stateJson["Experience Replay"][i]["Current Policy"]["State Value"][d].get<float>();
      curPolicy[d].distributionParameters = stateJson["Experience Replay"][i]["Current Policy"]["Distribution Parameters"][d].get<std::vector<float>>();
      curPolicy[d].actionIndex = stateJson["Experience Replay"][i]["Current Policy"]["Action Index"][d].get<size_t>();
      curPolicy[d].unboundedAction = stateJson["Experience Replay"][i]["Current Policy"]["Unbounded Action"][d].get<std::vector<float>>();
      curPolicy[d].actionProbabilities = stateJson["Experience Replay"][i]["Current Policy"]["Action Probabilities"][d].get<std::vector<float>>();
      curPolicy[d].availableActions = stateJson["Experience Replay"][i]["Current Policy"]["Available Actions"][d].get<std::vector<size_t>>();
    }
    _expPolicyVector.add(expPolicy);
    _curPolicyVector.add(curPolicy);
  }

  auto endTime = std::chrono::steady_clock::now();                                                                         // Profiling
  double deserializationTime = std::chrono::duration_cast<std::chrono::nanoseconds>(endTime - beginTime).count() / 1.0e+9; // Profiling
  _k->_logger->logInfo("Normal", "Took %fs to deserialize training state.\n", deserializationTime);
}

void Agent::printGenerationAfter()
{
  if (_mode == "Training")
  {
    _k->_logger->logInfo("Normal", "Experience Replay Statistics:\n");
    _k->_logger->logInfo("Normal", " + Experience Memory Size:      %lu/%lu\n", _stateVector.size(), _experienceReplayMaximumSize);
    if (_maxEpisodes > 0)
      _k->_logger->logInfo("Normal", " + Total Episodes Count:        %lu/%lu\n", _currentEpisode, _maxEpisodes);
    else
      _k->_logger->logInfo("Normal", " + Total Episodes Count:        %lu\n", _currentEpisode);

    if (_maxExperiences > 0)
      _k->_logger->logInfo("Normal", " + Total Experience Count:      %lu/%lu\n", _experienceCount, _maxExperiences);
    else
      _k->_logger->logInfo("Normal", " + Total Experience Count:      %lu\n", _experienceCount);

    _k->_logger->logInfo("Normal", "Training Statistics:\n");
    if (_maxPolicyUpdates > 0)
      _k->_logger->logInfo("Normal", " + Policy Update Count:         %lu/%lu\n", _policyUpdateCount, _maxPolicyUpdates);
    else
      _k->_logger->logInfo("Normal", " + Policy Update Count:         %lu\n", _policyUpdateCount);

    size_t numPolicies = _problem->_policiesPerEnvironment;
    for (size_t d = 0; d < numPolicies; d++)
    {
      _k->_logger->logInfo("Normal", "Off-Policy Statistics for policy %lu: \n", d);
      _k->_logger->logInfo("Normal", " + Count (Ratio/Target):        %lu/%lu (%.3f/%.3f)\n", numPolicies > 1 ? _experienceReplayOffPolicyCount[d] : _experienceReplayOffPolicyCount[d] / _problem->_agentsPerEnvironment, _stateVector.size(), _experienceReplayOffPolicyRatio[d], _experienceReplayOffPolicyTarget);
      _k->_logger->logInfo("Normal", " + Importance Weight Cutoff:    [%.3f, %.3f]\n", 1.0f / _experienceReplayOffPolicyCurrentCutoff, _experienceReplayOffPolicyCurrentCutoff);
      _k->_logger->logInfo("Normal", " + REFER Beta Factor:           %f\n", _experienceReplayOffPolicyREFERCurrentBeta[d]);

      _k->_logger->logInfo("Normal", " + Latest Reward for agent %lu:               %f\n", d, _trainingLastReward[d]);
      _k->_logger->logInfo("Normal", " + %lu-Episode Average Reward for agent %lu:  %f\n", _trainingAverageDepth, d, _trainingAverageReward[d]);
      _k->_logger->logInfo("Normal", " + Best Reward for agent %lu:                 %f (%lu)\n", d, _trainingBestReward[d], _trainingBestEpisodeId[d]);
    }

    if (_rewardRescalingEnabled)
     for (size_t a = 0; a < _problem->_agentsPerEnvironment; a++)
      _k->_logger->logInfo("Normal", " + Reward Rescaling: N(%.3e, %.3e)\n", _rewardRescalingMeans[a], _rewardRescalingSigmas[a]);

    if (_testingBestEpisodeId > 0)
    {
      _k->_logger->logInfo("Normal", "Testing Statistics:\n");
      _k->_logger->logInfo("Normal", " + Best Average Reward: %f (%lu)\n", _testingBestAverageReward, _testingBestEpisodeId);
      _k->_logger->logInfo("Normal", " + Latest Average (Worst / Best) Reward: %f (%f / %f)\n", _testingAverageReward, _testingWorstReward, _testingBestReward);
    }

    if( _policyUpdateCount != 0 )
    {
      printAgentInformation();
      _k->_logger->logInfo("Normal", " + Current Learning Rate:           %.3e\n", _currentLearningRate);
    }

    if (_stateRescalingEnabled)
      _k->_logger->logInfo("Normal", " + Using State Rescaling\n");

    _k->_logger->logInfo("Detailed", "Profiling Information:                  [Generation] - [Session]\n");
    _k->_logger->logInfo("Detailed", " + Experience Serialization Time:       [%5.3fs] - [%3.3fs]\n", _generationSerializationTime / 1.0e+9, _sessionSerializationTime / 1.0e+9);
    _k->_logger->logInfo("Detailed", " + Agent Attending Time:                [%5.3fs] - [%3.3fs]\n", _generationAgentAttendingTime / 1.0e+9, _sessionAgentAttendingTime / 1.0e+9);
    _k->_logger->logInfo("Detailed", " + Avg Agent Computation Time:          [%5.3fs] - [%3.3fs]\n", _generationAgentComputationTime / 1.0e+9, _sessionAgentComputationTime / 1.0e+9);
    _k->_logger->logInfo("Detailed", " + Avg Agent Communication/Wait Time:   [%5.3fs] - [%3.3fs]\n", _generationAgentCommunicationTime / 1.0e+9, _sessionAgentCommunicationTime / 1.0e+9);
    _k->_logger->logInfo("Detailed", " + Avg Agent Policy Evaluation Time:    [%5.3fs] - [%3.3fs]\n", _generationAgentPolicyEvaluationTime / 1.0e+9, _sessionAgentPolicyEvaluationTime / 1.0e+9);
    _k->_logger->logInfo("Detailed", " + Policy Update Time:                  [%5.3fs] - [%3.3fs]\n", _generationPolicyUpdateTime / 1.0e+9, _sessionPolicyUpdateTime / 1.0e+9);
    _k->_logger->logInfo("Detailed", " + Running Time:                        [%5.3fs] - [%3.3fs]\n", _generationRunningTime / 1.0e+9, _sessionRunningTime / 1.0e+9);
    _k->_logger->logInfo("Detailed", " + [I/O] Result File Saving Time:        %5.3fs\n", _k->_resultSavingTime / 1.0e+9);
  }

  if (_mode == "Testing")
  {
    _k->_logger->logInfo("Normal", "Testing Results:\n");
    for (size_t sampleId = 0; sampleId < _testingSampleIds.size(); sampleId++)
    {
      _k->_logger->logInfo("Normal", " + Sample %lu:\n", _testingSampleIds[sampleId]);
      _k->_logger->logInfo("Normal", "   + (Average) Cumulative Reward:           %f\n", _testingReward[sampleId]);
    }
  }
}

void Agent::setConfiguration(knlohmann::json& js) 
{
 if (isDefined(js, "Results"))  eraseValue(js, "Results");

 if (isDefined(js, "Policy", "Parameter Count"))
 {
 try { _policyParameterCount = js["Policy"]["Parameter Count"].get<size_t>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ agent ] \n + Key:    ['Policy']['Parameter Count']\n%s", e.what()); } 
   eraseValue(js, "Policy", "Parameter Count");
 }

 if (isDefined(js, "Action Lower Bounds"))
 {
 try { _actionLowerBounds = js["Action Lower Bounds"].get<std::vector<float>>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ agent ] \n + Key:    ['Action Lower Bounds']\n%s", e.what()); } 
   eraseValue(js, "Action Lower Bounds");
 }

 if (isDefined(js, "Action Upper Bounds"))
 {
 try { _actionUpperBounds = js["Action Upper Bounds"].get<std::vector<float>>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ agent ] \n + Key:    ['Action Upper Bounds']\n%s", e.what()); } 
   eraseValue(js, "Action Upper Bounds");
 }

 if (isDefined(js, "Current Episode"))
 {
 try { _currentEpisode = js["Current Episode"].get<size_t>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ agent ] \n + Key:    ['Current Episode']\n%s", e.what()); } 
   eraseValue(js, "Current Episode");
 }

 if (isDefined(js, "Training", "Reward History"))
 {
 try { _trainingRewardHistory = js["Training"]["Reward History"].get<std::vector<std::vector<float>>>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ agent ] \n + Key:    ['Training']['Reward History']\n%s", e.what()); } 
   eraseValue(js, "Training", "Reward History");
 }

 if (isDefined(js, "Training", "Experience History"))
 {
 try { _trainingExperienceHistory = js["Training"]["Experience History"].get<std::vector<size_t>>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ agent ] \n + Key:    ['Training']['Experience History']\n%s", e.what()); } 
   eraseValue(js, "Training", "Experience History");
 }

 if (isDefined(js, "Testing", "Average Reward History"))
 {
 try { _testingAverageRewardHistory = js["Testing"]["Average Reward History"].get<std::vector<float>>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ agent ] \n + Key:    ['Testing']['Average Reward History']\n%s", e.what()); } 
   eraseValue(js, "Testing", "Average Reward History");
 }

 if (isDefined(js, "Training", "Average Reward"))
 {
 try { _trainingAverageReward = js["Training"]["Average Reward"].get<std::vector<float>>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ agent ] \n + Key:    ['Training']['Average Reward']\n%s", e.what()); } 
   eraseValue(js, "Training", "Average Reward");
 }

 if (isDefined(js, "Training", "Last Reward"))
 {
 try { _trainingLastReward = js["Training"]["Last Reward"].get<std::vector<float>>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ agent ] \n + Key:    ['Training']['Last Reward']\n%s", e.what()); } 
   eraseValue(js, "Training", "Last Reward");
 }

 if (isDefined(js, "Training", "Best Reward"))
 {
 try { _trainingBestReward = js["Training"]["Best Reward"].get<std::vector<float>>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ agent ] \n + Key:    ['Training']['Best Reward']\n%s", e.what()); } 
   eraseValue(js, "Training", "Best Reward");
 }

 if (isDefined(js, "Training", "Best Episode Id"))
 {
 try { _trainingBestEpisodeId = js["Training"]["Best Episode Id"].get<std::vector<size_t>>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ agent ] \n + Key:    ['Training']['Best Episode Id']\n%s", e.what()); } 
   eraseValue(js, "Training", "Best Episode Id");
 }

 if (isDefined(js, "Training", "Current Policies"))
 {
 _trainingCurrentPolicies = js["Training"]["Current Policies"].get<knlohmann::json>();

   eraseValue(js, "Training", "Current Policies");
 }

 if (isDefined(js, "Training", "Best Policies"))
 {
 _trainingBestPolicies = js["Training"]["Best Policies"].get<knlohmann::json>();

   eraseValue(js, "Training", "Best Policies");
 }

 if (isDefined(js, "Testing", "Reward"))
 {
 try { _testingReward = js["Testing"]["Reward"].get<std::vector<float>>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ agent ] \n + Key:    ['Testing']['Reward']\n%s", e.what()); } 
   eraseValue(js, "Testing", "Reward");
 }

 if (isDefined(js, "Testing", "Best Reward"))
 {
 try { _testingBestReward = js["Testing"]["Best Reward"].get<float>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ agent ] \n + Key:    ['Testing']['Best Reward']\n%s", e.what()); } 
   eraseValue(js, "Testing", "Best Reward");
 }

 if (isDefined(js, "Testing", "Worst Reward"))
 {
 try { _testingWorstReward = js["Testing"]["Worst Reward"].get<float>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ agent ] \n + Key:    ['Testing']['Worst Reward']\n%s", e.what()); } 
   eraseValue(js, "Testing", "Worst Reward");
 }

 if (isDefined(js, "Testing", "Best Episode Id"))
 {
 try { _testingBestEpisodeId = js["Testing"]["Best Episode Id"].get<size_t>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ agent ] \n + Key:    ['Testing']['Best Episode Id']\n%s", e.what()); } 
   eraseValue(js, "Testing", "Best Episode Id");
 }

 if (isDefined(js, "Testing", "Candidate Count"))
 {
 try { _testingCandidateCount = js["Testing"]["Candidate Count"].get<size_t>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ agent ] \n + Key:    ['Testing']['Candidate Count']\n%s", e.what()); } 
   eraseValue(js, "Testing", "Candidate Count");
 }

 if (isDefined(js, "Testing", "Average Reward"))
 {
 try { _testingAverageReward = js["Testing"]["Average Reward"].get<float>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ agent ] \n + Key:    ['Testing']['Average Reward']\n%s", e.what()); } 
   eraseValue(js, "Testing", "Average Reward");
 }

 if (isDefined(js, "Testing", "Best Average Reward"))
 {
 try { _testingBestAverageReward = js["Testing"]["Best Average Reward"].get<float>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ agent ] \n + Key:    ['Testing']['Best Average Reward']\n%s", e.what()); } 
   eraseValue(js, "Testing", "Best Average Reward");
 }

 if (isDefined(js, "Testing", "Best Policies"))
 {
 _testingBestPolicies = js["Testing"]["Best Policies"].get<knlohmann::json>();

   eraseValue(js, "Testing", "Best Policies");
 }

 if (isDefined(js, "Experience Replay", "Off Policy", "Count"))
 {
 try { _experienceReplayOffPolicyCount = js["Experience Replay"]["Off Policy"]["Count"].get<std::vector<size_t>>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ agent ] \n + Key:    ['Experience Replay']['Off Policy']['Count']\n%s", e.what()); } 
   eraseValue(js, "Experience Replay", "Off Policy", "Count");
 }

 if (isDefined(js, "Experience Replay", "Off Policy", "Ratio"))
 {
 try { _experienceReplayOffPolicyRatio = js["Experience Replay"]["Off Policy"]["Ratio"].get<std::vector<float>>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ agent ] \n + Key:    ['Experience Replay']['Off Policy']['Ratio']\n%s", e.what()); } 
   eraseValue(js, "Experience Replay", "Off Policy", "Ratio");
 }

 if (isDefined(js, "Experience Replay", "Off Policy", "Current Cutoff"))
 {
 try { _experienceReplayOffPolicyCurrentCutoff = js["Experience Replay"]["Off Policy"]["Current Cutoff"].get<float>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ agent ] \n + Key:    ['Experience Replay']['Off Policy']['Current Cutoff']\n%s", e.what()); } 
   eraseValue(js, "Experience Replay", "Off Policy", "Current Cutoff");
 }

 if (isDefined(js, "Experience Replay", "Off Policy", "REFER Current Beta"))
 {
 try { _experienceReplayOffPolicyREFERCurrentBeta = js["Experience Replay"]["Off Policy"]["REFER Current Beta"].get<std::vector<float>>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ agent ] \n + Key:    ['Experience Replay']['Off Policy']['REFER Current Beta']\n%s", e.what()); } 
   eraseValue(js, "Experience Replay", "Off Policy", "REFER Current Beta");
 }

 if (isDefined(js, "Current Learning Rate"))
 {
 try { _currentLearningRate = js["Current Learning Rate"].get<float>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ agent ] \n + Key:    ['Current Learning Rate']\n%s", e.what()); } 
   eraseValue(js, "Current Learning Rate");
 }

 if (isDefined(js, "Policy Update Count"))
 {
 try { _policyUpdateCount = js["Policy Update Count"].get<size_t>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ agent ] \n + Key:    ['Policy Update Count']\n%s", e.what()); } 
   eraseValue(js, "Policy Update Count");
 }

 if (isDefined(js, "Uniform Generator"))
 {
 _uniformGenerator = dynamic_cast<korali::distribution::univariate::Uniform*>(korali::Module::getModule(js["Uniform Generator"], _k));
 _uniformGenerator->applyVariableDefaults();
 _uniformGenerator->applyModuleDefaults(js["Uniform Generator"]);
 _uniformGenerator->setConfiguration(js["Uniform Generator"]);
   eraseValue(js, "Uniform Generator");
 }

 if (isDefined(js, "Experience Count"))
 {
 try { _experienceCount = js["Experience Count"].get<size_t>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ agent ] \n + Key:    ['Experience Count']\n%s", e.what()); } 
   eraseValue(js, "Experience Count");
 }

 if (isDefined(js, "Reward", "Rescaling", "Means"))
 {
 try { _rewardRescalingMeans = js["Reward"]["Rescaling"]["Means"].get<std::vector<float>>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ agent ] \n + Key:    ['Reward']['Rescaling']['Means']\n%s", e.what()); } 
   eraseValue(js, "Reward", "Rescaling", "Means");
 }

 if (isDefined(js, "Reward", "Rescaling", "Sigmas"))
 {
 try { _rewardRescalingSigmas = js["Reward"]["Rescaling"]["Sigmas"].get<std::vector<float>>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ agent ] \n + Key:    ['Reward']['Rescaling']['Sigmas']\n%s", e.what()); } 
   eraseValue(js, "Reward", "Rescaling", "Sigmas");
 }

 if (isDefined(js, "Reward", "Rescaling", "Sum Squared Rewards"))
 {
 try { _rewardRescalingSumSquaredRewards = js["Reward"]["Rescaling"]["Sum Squared Rewards"].get<std::vector<float>>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ agent ] \n + Key:    ['Reward']['Rescaling']['Sum Squared Rewards']\n%s", e.what()); } 
   eraseValue(js, "Reward", "Rescaling", "Sum Squared Rewards");
 }

 if (isDefined(js, "State Rescaling", "Means"))
 {
 try { _stateRescalingMeans = js["State Rescaling"]["Means"].get<std::vector<std::vector<float>>>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ agent ] \n + Key:    ['State Rescaling']['Means']\n%s", e.what()); } 
   eraseValue(js, "State Rescaling", "Means");
 }

 if (isDefined(js, "State Rescaling", "Sigmas"))
 {
 try { _stateRescalingSigmas = js["State Rescaling"]["Sigmas"].get<std::vector<std::vector<float>>>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ agent ] \n + Key:    ['State Rescaling']['Sigmas']\n%s", e.what()); } 
   eraseValue(js, "State Rescaling", "Sigmas");
 }

 if (isDefined(js, "Effective Minibatch Size"))
 {
 try { _effectiveMinibatchSize = js["Effective Minibatch Size"].get<size_t>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ agent ] \n + Key:    ['Effective Minibatch Size']\n%s", e.what()); } 
   eraseValue(js, "Effective Minibatch Size");
 }

 if (isDefined(js, "Mode"))
 {
 try { _mode = js["Mode"].get<std::string>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ agent ] \n + Key:    ['Mode']\n%s", e.what()); } 
{
 bool validOption = false; 
 if (_mode == "Training") validOption = true; 
 if (_mode == "Testing") validOption = true; 
 if (validOption == false) KORALI_LOG_ERROR(" + Unrecognized value (%s) provided for mandatory setting: ['Mode'] required by agent.\n", _mode.c_str()); 
}
   eraseValue(js, "Mode");
 }
  else   KORALI_LOG_ERROR(" + No value provided for mandatory setting: ['Mode'] required by agent.\n"); 

 if (isDefined(js, "Testing", "Sample Ids"))
 {
 try { _testingSampleIds = js["Testing"]["Sample Ids"].get<std::vector<size_t>>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ agent ] \n + Key:    ['Testing']['Sample Ids']\n%s", e.what()); } 
   eraseValue(js, "Testing", "Sample Ids");
 }
  else   KORALI_LOG_ERROR(" + No value provided for mandatory setting: ['Testing']['Sample Ids'] required by agent.\n"); 

 if (isDefined(js, "Testing", "Current Policies"))
 {
 _testingCurrentPolicies = js["Testing"]["Current Policies"].get<knlohmann::json>();

   eraseValue(js, "Testing", "Current Policies");
 }
  else   KORALI_LOG_ERROR(" + No value provided for mandatory setting: ['Testing']['Current Policies'] required by agent.\n"); 

 if (isDefined(js, "Training", "Average Depth"))
 {
 try { _trainingAverageDepth = js["Training"]["Average Depth"].get<size_t>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ agent ] \n + Key:    ['Training']['Average Depth']\n%s", e.what()); } 
   eraseValue(js, "Training", "Average Depth");
 }
  else   KORALI_LOG_ERROR(" + No value provided for mandatory setting: ['Training']['Average Depth'] required by agent.\n"); 

 if (isDefined(js, "Concurrent Environments"))
 {
 try { _concurrentEnvironments = js["Concurrent Environments"].get<size_t>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ agent ] \n + Key:    ['Concurrent Environments']\n%s", e.what()); } 
   eraseValue(js, "Concurrent Environments");
 }
  else   KORALI_LOG_ERROR(" + No value provided for mandatory setting: ['Concurrent Environments'] required by agent.\n"); 

 if (isDefined(js, "Episodes Per Generation"))
 {
 try { _episodesPerGeneration = js["Episodes Per Generation"].get<size_t>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ agent ] \n + Key:    ['Episodes Per Generation']\n%s", e.what()); } 
   eraseValue(js, "Episodes Per Generation");
 }
  else   KORALI_LOG_ERROR(" + No value provided for mandatory setting: ['Episodes Per Generation'] required by agent.\n"); 

 if (isDefined(js, "Mini Batch", "Size"))
 {
 try { _miniBatchSize = js["Mini Batch"]["Size"].get<size_t>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ agent ] \n + Key:    ['Mini Batch']['Size']\n%s", e.what()); } 
   eraseValue(js, "Mini Batch", "Size");
 }
  else   KORALI_LOG_ERROR(" + No value provided for mandatory setting: ['Mini Batch']['Size'] required by agent.\n"); 

 if (isDefined(js, "Mini Batch", "Strategy"))
 {
 try { _miniBatchStrategy = js["Mini Batch"]["Strategy"].get<std::string>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ agent ] \n + Key:    ['Mini Batch']['Strategy']\n%s", e.what()); } 
{
 bool validOption = false; 
 if (_miniBatchStrategy == "Uniform") validOption = true; 
 if (validOption == false) KORALI_LOG_ERROR(" + Unrecognized value (%s) provided for mandatory setting: ['Mini Batch']['Strategy'] required by agent.\n", _miniBatchStrategy.c_str()); 
}
   eraseValue(js, "Mini Batch", "Strategy");
 }
  else   KORALI_LOG_ERROR(" + No value provided for mandatory setting: ['Mini Batch']['Strategy'] required by agent.\n"); 

 if (isDefined(js, "Time Sequence Length"))
 {
 try { _timeSequenceLength = js["Time Sequence Length"].get<size_t>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ agent ] \n + Key:    ['Time Sequence Length']\n%s", e.what()); } 
   eraseValue(js, "Time Sequence Length");
 }
  else   KORALI_LOG_ERROR(" + No value provided for mandatory setting: ['Time Sequence Length'] required by agent.\n"); 

 if (isDefined(js, "Learning Rate"))
 {
 try { _learningRate = js["Learning Rate"].get<float>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ agent ] \n + Key:    ['Learning Rate']\n%s", e.what()); } 
   eraseValue(js, "Learning Rate");
 }
  else   KORALI_LOG_ERROR(" + No value provided for mandatory setting: ['Learning Rate'] required by agent.\n"); 

 if (isDefined(js, "L2 Regularization", "Enabled"))
 {
 try { _l2RegularizationEnabled = js["L2 Regularization"]["Enabled"].get<int>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ agent ] \n + Key:    ['L2 Regularization']['Enabled']\n%s", e.what()); } 
   eraseValue(js, "L2 Regularization", "Enabled");
 }
  else   KORALI_LOG_ERROR(" + No value provided for mandatory setting: ['L2 Regularization']['Enabled'] required by agent.\n"); 

 if (isDefined(js, "L2 Regularization", "Importance"))
 {
 try { _l2RegularizationImportance = js["L2 Regularization"]["Importance"].get<float>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ agent ] \n + Key:    ['L2 Regularization']['Importance']\n%s", e.what()); } 
   eraseValue(js, "L2 Regularization", "Importance");
 }
  else   KORALI_LOG_ERROR(" + No value provided for mandatory setting: ['L2 Regularization']['Importance'] required by agent.\n"); 

 if (isDefined(js, "Neural Network", "Hidden Layers"))
 {
 _neuralNetworkHiddenLayers = js["Neural Network"]["Hidden Layers"].get<knlohmann::json>();

   eraseValue(js, "Neural Network", "Hidden Layers");
 }
  else   KORALI_LOG_ERROR(" + No value provided for mandatory setting: ['Neural Network']['Hidden Layers'] required by agent.\n"); 

 if (isDefined(js, "Neural Network", "Optimizer"))
 {
 try { _neuralNetworkOptimizer = js["Neural Network"]["Optimizer"].get<std::string>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ agent ] \n + Key:    ['Neural Network']['Optimizer']\n%s", e.what()); } 
   eraseValue(js, "Neural Network", "Optimizer");
 }
  else   KORALI_LOG_ERROR(" + No value provided for mandatory setting: ['Neural Network']['Optimizer'] required by agent.\n"); 

 if (isDefined(js, "Neural Network", "Engine"))
 {
 try { _neuralNetworkEngine = js["Neural Network"]["Engine"].get<std::string>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ agent ] \n + Key:    ['Neural Network']['Engine']\n%s", e.what()); } 
   eraseValue(js, "Neural Network", "Engine");
 }
  else   KORALI_LOG_ERROR(" + No value provided for mandatory setting: ['Neural Network']['Engine'] required by agent.\n"); 

 if (isDefined(js, "Discount Factor"))
 {
 try { _discountFactor = js["Discount Factor"].get<float>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ agent ] \n + Key:    ['Discount Factor']\n%s", e.what()); } 
   eraseValue(js, "Discount Factor");
 }
  else   KORALI_LOG_ERROR(" + No value provided for mandatory setting: ['Discount Factor'] required by agent.\n"); 

 if (isDefined(js, "Importance Weight Truncation Level"))
 {
 try { _importanceWeightTruncationLevel = js["Importance Weight Truncation Level"].get<float>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ agent ] \n + Key:    ['Importance Weight Truncation Level']\n%s", e.what()); } 
   eraseValue(js, "Importance Weight Truncation Level");
 }
  else   KORALI_LOG_ERROR(" + No value provided for mandatory setting: ['Importance Weight Truncation Level'] required by agent.\n"); 

 if (isDefined(js, "Experience Replay", "Serialize"))
 {
 try { _experienceReplaySerialize = js["Experience Replay"]["Serialize"].get<int>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ agent ] \n + Key:    ['Experience Replay']['Serialize']\n%s", e.what()); } 
   eraseValue(js, "Experience Replay", "Serialize");
 }
  else   KORALI_LOG_ERROR(" + No value provided for mandatory setting: ['Experience Replay']['Serialize'] required by agent.\n"); 

 if (isDefined(js, "Experience Replay", "Start Size"))
 {
 try { _experienceReplayStartSize = js["Experience Replay"]["Start Size"].get<size_t>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ agent ] \n + Key:    ['Experience Replay']['Start Size']\n%s", e.what()); } 
   eraseValue(js, "Experience Replay", "Start Size");
 }
  else   KORALI_LOG_ERROR(" + No value provided for mandatory setting: ['Experience Replay']['Start Size'] required by agent.\n"); 

 if (isDefined(js, "Experience Replay", "Maximum Size"))
 {
 try { _experienceReplayMaximumSize = js["Experience Replay"]["Maximum Size"].get<size_t>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ agent ] \n + Key:    ['Experience Replay']['Maximum Size']\n%s", e.what()); } 
   eraseValue(js, "Experience Replay", "Maximum Size");
 }
  else   KORALI_LOG_ERROR(" + No value provided for mandatory setting: ['Experience Replay']['Maximum Size'] required by agent.\n"); 

 if (isDefined(js, "Experience Replay", "Off Policy", "Cutoff Scale"))
 {
 try { _experienceReplayOffPolicyCutoffScale = js["Experience Replay"]["Off Policy"]["Cutoff Scale"].get<float>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ agent ] \n + Key:    ['Experience Replay']['Off Policy']['Cutoff Scale']\n%s", e.what()); } 
   eraseValue(js, "Experience Replay", "Off Policy", "Cutoff Scale");
 }
  else   KORALI_LOG_ERROR(" + No value provided for mandatory setting: ['Experience Replay']['Off Policy']['Cutoff Scale'] required by agent.\n"); 

 if (isDefined(js, "Experience Replay", "Off Policy", "Target"))
 {
 try { _experienceReplayOffPolicyTarget = js["Experience Replay"]["Off Policy"]["Target"].get<float>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ agent ] \n + Key:    ['Experience Replay']['Off Policy']['Target']\n%s", e.what()); } 
   eraseValue(js, "Experience Replay", "Off Policy", "Target");
 }
  else   KORALI_LOG_ERROR(" + No value provided for mandatory setting: ['Experience Replay']['Off Policy']['Target'] required by agent.\n"); 

 if (isDefined(js, "Experience Replay", "Off Policy", "Annealing Rate"))
 {
 try { _experienceReplayOffPolicyAnnealingRate = js["Experience Replay"]["Off Policy"]["Annealing Rate"].get<float>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ agent ] \n + Key:    ['Experience Replay']['Off Policy']['Annealing Rate']\n%s", e.what()); } 
   eraseValue(js, "Experience Replay", "Off Policy", "Annealing Rate");
 }
  else   KORALI_LOG_ERROR(" + No value provided for mandatory setting: ['Experience Replay']['Off Policy']['Annealing Rate'] required by agent.\n"); 

 if (isDefined(js, "Experience Replay", "Off Policy", "REFER Beta"))
 {
 try { _experienceReplayOffPolicyREFERBeta = js["Experience Replay"]["Off Policy"]["REFER Beta"].get<float>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ agent ] \n + Key:    ['Experience Replay']['Off Policy']['REFER Beta']\n%s", e.what()); } 
   eraseValue(js, "Experience Replay", "Off Policy", "REFER Beta");
 }
  else   KORALI_LOG_ERROR(" + No value provided for mandatory setting: ['Experience Replay']['Off Policy']['REFER Beta'] required by agent.\n"); 

 if (isDefined(js, "Experiences Between Policy Updates"))
 {
 try { _experiencesBetweenPolicyUpdates = js["Experiences Between Policy Updates"].get<float>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ agent ] \n + Key:    ['Experiences Between Policy Updates']\n%s", e.what()); } 
   eraseValue(js, "Experiences Between Policy Updates");
 }
  else   KORALI_LOG_ERROR(" + No value provided for mandatory setting: ['Experiences Between Policy Updates'] required by agent.\n"); 

 if (isDefined(js, "State Rescaling", "Enabled"))
 {
 try { _stateRescalingEnabled = js["State Rescaling"]["Enabled"].get<int>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ agent ] \n + Key:    ['State Rescaling']['Enabled']\n%s", e.what()); } 
   eraseValue(js, "State Rescaling", "Enabled");
 }
  else   KORALI_LOG_ERROR(" + No value provided for mandatory setting: ['State Rescaling']['Enabled'] required by agent.\n"); 

 if (isDefined(js, "Reward", "Rescaling", "Enabled"))
 {
 try { _rewardRescalingEnabled = js["Reward"]["Rescaling"]["Enabled"].get<int>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ agent ] \n + Key:    ['Reward']['Rescaling']['Enabled']\n%s", e.what()); } 
   eraseValue(js, "Reward", "Rescaling", "Enabled");
 }
  else   KORALI_LOG_ERROR(" + No value provided for mandatory setting: ['Reward']['Rescaling']['Enabled'] required by agent.\n"); 

 if (isDefined(js, "Multi Agent Relationship"))
 {
 try { _multiAgentRelationship = js["Multi Agent Relationship"].get<std::string>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ agent ] \n + Key:    ['Multi Agent Relationship']\n%s", e.what()); } 
{
 bool validOption = false; 
 if (_multiAgentRelationship == "Individual") validOption = true; 
 if (_multiAgentRelationship == "Cooperation") validOption = true; 
 if (validOption == false) KORALI_LOG_ERROR(" + Unrecognized value (%s) provided for mandatory setting: ['Multi Agent Relationship'] required by agent.\n", _multiAgentRelationship.c_str()); 
}
   eraseValue(js, "Multi Agent Relationship");
 }
  else   KORALI_LOG_ERROR(" + No value provided for mandatory setting: ['Multi Agent Relationship'] required by agent.\n"); 

 if (isDefined(js, "Multi Agent Correlation"))
 {
 try { _multiAgentCorrelation = js["Multi Agent Correlation"].get<int>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ agent ] \n + Key:    ['Multi Agent Correlation']\n%s", e.what()); } 
   eraseValue(js, "Multi Agent Correlation");
 }
  else   KORALI_LOG_ERROR(" + No value provided for mandatory setting: ['Multi Agent Correlation'] required by agent.\n"); 

 if (isDefined(js, "Multi Agent Sampling"))
 {
 try { _multiAgentSampling = js["Multi Agent Sampling"].get<std::string>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ agent ] \n + Key:    ['Multi Agent Sampling']\n%s", e.what()); } 
{
 bool validOption = false; 
 if (_multiAgentSampling == "Tuple") validOption = true; 
 if (_multiAgentSampling == "Baseline") validOption = true; 
 if (_multiAgentSampling == "Experience") validOption = true; 
 if (validOption == false) KORALI_LOG_ERROR(" + Unrecognized value (%s) provided for mandatory setting: ['Multi Agent Sampling'] required by agent.\n", _multiAgentSampling.c_str()); 
}
   eraseValue(js, "Multi Agent Sampling");
 }
  else   KORALI_LOG_ERROR(" + No value provided for mandatory setting: ['Multi Agent Sampling'] required by agent.\n"); 

 if (isDefined(js, "Termination Criteria", "Max Episodes"))
 {
 try { _maxEpisodes = js["Termination Criteria"]["Max Episodes"].get<size_t>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ agent ] \n + Key:    ['Termination Criteria']['Max Episodes']\n%s", e.what()); } 
   eraseValue(js, "Termination Criteria", "Max Episodes");
 }
  else   KORALI_LOG_ERROR(" + No value provided for mandatory setting: ['Termination Criteria']['Max Episodes'] required by agent.\n"); 

 if (isDefined(js, "Termination Criteria", "Max Experiences"))
 {
 try { _maxExperiences = js["Termination Criteria"]["Max Experiences"].get<size_t>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ agent ] \n + Key:    ['Termination Criteria']['Max Experiences']\n%s", e.what()); } 
   eraseValue(js, "Termination Criteria", "Max Experiences");
 }
  else   KORALI_LOG_ERROR(" + No value provided for mandatory setting: ['Termination Criteria']['Max Experiences'] required by agent.\n"); 

 if (isDefined(js, "Termination Criteria", "Max Policy Updates"))
 {
 try { _maxPolicyUpdates = js["Termination Criteria"]["Max Policy Updates"].get<size_t>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ agent ] \n + Key:    ['Termination Criteria']['Max Policy Updates']\n%s", e.what()); } 
   eraseValue(js, "Termination Criteria", "Max Policy Updates");
 }
  else   KORALI_LOG_ERROR(" + No value provided for mandatory setting: ['Termination Criteria']['Max Policy Updates'] required by agent.\n"); 

 if (isDefined(_k->_js.getJson(), "Variables"))
 for (size_t i = 0; i < _k->_js["Variables"].size(); i++) { 
 } 
 Solver::setConfiguration(js);
 _type = "agent";
 if(isDefined(js, "Type")) eraseValue(js, "Type");
 if(isEmpty(js) == false) KORALI_LOG_ERROR(" + Unrecognized settings for Korali module: agent: \n%s\n", js.dump(2).c_str());
} 

void Agent::getConfiguration(knlohmann::json& js) 
{

 js["Type"] = _type;
   js["Mode"] = _mode;
   js["Testing"]["Sample Ids"] = _testingSampleIds;
   js["Testing"]["Current Policies"] = _testingCurrentPolicies;
   js["Training"]["Average Depth"] = _trainingAverageDepth;
   js["Concurrent Environments"] = _concurrentEnvironments;
   js["Episodes Per Generation"] = _episodesPerGeneration;
   js["Mini Batch"]["Size"] = _miniBatchSize;
   js["Mini Batch"]["Strategy"] = _miniBatchStrategy;
   js["Time Sequence Length"] = _timeSequenceLength;
   js["Learning Rate"] = _learningRate;
   js["L2 Regularization"]["Enabled"] = _l2RegularizationEnabled;
   js["L2 Regularization"]["Importance"] = _l2RegularizationImportance;
   js["Neural Network"]["Hidden Layers"] = _neuralNetworkHiddenLayers;
   js["Neural Network"]["Optimizer"] = _neuralNetworkOptimizer;
   js["Neural Network"]["Engine"] = _neuralNetworkEngine;
   js["Discount Factor"] = _discountFactor;
   js["Importance Weight Truncation Level"] = _importanceWeightTruncationLevel;
   js["Experience Replay"]["Serialize"] = _experienceReplaySerialize;
   js["Experience Replay"]["Start Size"] = _experienceReplayStartSize;
   js["Experience Replay"]["Maximum Size"] = _experienceReplayMaximumSize;
   js["Experience Replay"]["Off Policy"]["Cutoff Scale"] = _experienceReplayOffPolicyCutoffScale;
   js["Experience Replay"]["Off Policy"]["Target"] = _experienceReplayOffPolicyTarget;
   js["Experience Replay"]["Off Policy"]["Annealing Rate"] = _experienceReplayOffPolicyAnnealingRate;
   js["Experience Replay"]["Off Policy"]["REFER Beta"] = _experienceReplayOffPolicyREFERBeta;
   js["Experiences Between Policy Updates"] = _experiencesBetweenPolicyUpdates;
   js["State Rescaling"]["Enabled"] = _stateRescalingEnabled;
   js["Reward"]["Rescaling"]["Enabled"] = _rewardRescalingEnabled;
   js["Multi Agent Relationship"] = _multiAgentRelationship;
   js["Multi Agent Correlation"] = _multiAgentCorrelation;
   js["Multi Agent Sampling"] = _multiAgentSampling;
   js["Termination Criteria"]["Max Episodes"] = _maxEpisodes;
   js["Termination Criteria"]["Max Experiences"] = _maxExperiences;
   js["Termination Criteria"]["Max Policy Updates"] = _maxPolicyUpdates;
   js["Policy"]["Parameter Count"] = _policyParameterCount;
   js["Action Lower Bounds"] = _actionLowerBounds;
   js["Action Upper Bounds"] = _actionUpperBounds;
   js["Current Episode"] = _currentEpisode;
   js["Training"]["Reward History"] = _trainingRewardHistory;
   js["Training"]["Experience History"] = _trainingExperienceHistory;
   js["Testing"]["Average Reward History"] = _testingAverageRewardHistory;
   js["Training"]["Average Reward"] = _trainingAverageReward;
   js["Training"]["Last Reward"] = _trainingLastReward;
   js["Training"]["Best Reward"] = _trainingBestReward;
   js["Training"]["Best Episode Id"] = _trainingBestEpisodeId;
   js["Training"]["Current Policies"] = _trainingCurrentPolicies;
   js["Training"]["Best Policies"] = _trainingBestPolicies;
   js["Testing"]["Reward"] = _testingReward;
   js["Testing"]["Best Reward"] = _testingBestReward;
   js["Testing"]["Worst Reward"] = _testingWorstReward;
   js["Testing"]["Best Episode Id"] = _testingBestEpisodeId;
   js["Testing"]["Candidate Count"] = _testingCandidateCount;
   js["Testing"]["Average Reward"] = _testingAverageReward;
   js["Testing"]["Best Average Reward"] = _testingBestAverageReward;
   js["Testing"]["Best Policies"] = _testingBestPolicies;
   js["Experience Replay"]["Off Policy"]["Count"] = _experienceReplayOffPolicyCount;
   js["Experience Replay"]["Off Policy"]["Ratio"] = _experienceReplayOffPolicyRatio;
   js["Experience Replay"]["Off Policy"]["Current Cutoff"] = _experienceReplayOffPolicyCurrentCutoff;
   js["Experience Replay"]["Off Policy"]["REFER Current Beta"] = _experienceReplayOffPolicyREFERCurrentBeta;
   js["Current Learning Rate"] = _currentLearningRate;
   js["Policy Update Count"] = _policyUpdateCount;
 if(_uniformGenerator != NULL) _uniformGenerator->getConfiguration(js["Uniform Generator"]);
   js["Experience Count"] = _experienceCount;
   js["Reward"]["Rescaling"]["Means"] = _rewardRescalingMeans;
   js["Reward"]["Rescaling"]["Sigmas"] = _rewardRescalingSigmas;
   js["Reward"]["Rescaling"]["Sum Squared Rewards"] = _rewardRescalingSumSquaredRewards;
   js["State Rescaling"]["Means"] = _stateRescalingMeans;
   js["State Rescaling"]["Sigmas"] = _stateRescalingSigmas;
   js["Effective Minibatch Size"] = _effectiveMinibatchSize;
 for (size_t i = 0; i <  _k->_variables.size(); i++) { 
 } 
 Solver::getConfiguration(js);
} 

void Agent::applyModuleDefaults(knlohmann::json& js) 
{

 std::string defaultString = "{\"Episodes Per Generation\": 1, \"Concurrent Environments\": 1, \"Discount Factor\": 0.995, \"Time Sequence Length\": 1, \"Importance Weight Truncation Level\": 1.0, \"Multi Agent Relationship\": \"Individual\", \"Multi Agent Correlation\": false, \"Multi Agent Sampling\": \"Tuple\", \"State Rescaling\": {\"Enabled\": false}, \"Reward\": {\"Rescaling\": {\"Enabled\": false}}, \"Mini Batch\": {\"Strategy\": \"Uniform\", \"Size\": 256}, \"L2 Regularization\": {\"Enabled\": false, \"Importance\": 0.0001}, \"Training\": {\"Average Depth\": 100, \"Current Policies\": {}, \"Best Policies\": {}}, \"Testing\": {\"Sample Ids\": [], \"Current Policies\": {}, \"Best Policies\": {}}, \"Termination Criteria\": {\"Max Episodes\": 0, \"Max Experiences\": 0, \"Max Policy Updates\": 0}, \"Experience Replay\": {\"Serialize\": true, \"Off Policy\": {\"Cutoff Scale\": 4.0, \"Target\": 0.1, \"REFER Beta\": 0.3, \"Annealing Rate\": 0.0}}, \"Uniform Generator\": {\"Type\": \"Univariate/Uniform\", \"Minimum\": 0.0, \"Maximum\": 1.0}}";
 knlohmann::json defaultJs = knlohmann::json::parse(defaultString);
 mergeJson(js, defaultJs); 
 Solver::applyModuleDefaults(js);
} 

void Agent::applyVariableDefaults() 
{

 std::string defaultString = "{}";
 knlohmann::json defaultJs = knlohmann::json::parse(defaultString);
 if (isDefined(_k->_js.getJson(), "Variables"))
  for (size_t i = 0; i < _k->_js["Variables"].size(); i++) 
   mergeJson(_k->_js["Variables"][i], defaultJs); 
 Solver::applyVariableDefaults();
} 

bool Agent::checkTermination()
{
 bool hasFinished = false;

 if ((_mode == "Training") && (_maxEpisodes > 0) && (_currentEpisode >= _maxEpisodes))
 {
  _terminationCriteria.push_back("agent['Max Episodes'] = " + std::to_string(_maxEpisodes) + ".");
  hasFinished = true;
 }

 if ((_mode == "Training") && (_maxExperiences > 0) && (_experienceCount >= _maxExperiences))
 {
  _terminationCriteria.push_back("agent['Max Experiences'] = " + std::to_string(_maxExperiences) + ".");
  hasFinished = true;
 }

 if ((_mode == "Training") && (_maxPolicyUpdates > 0) && (_policyUpdateCount >= _maxPolicyUpdates))
 {
  _terminationCriteria.push_back("agent['Max Policy Updates'] = " + std::to_string(_maxPolicyUpdates) + ".");
  hasFinished = true;
 }

 hasFinished = hasFinished || Solver::checkTermination();
 return hasFinished;
}

;

} //solver
} //korali
;
