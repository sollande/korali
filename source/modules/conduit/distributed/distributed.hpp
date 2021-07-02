/** \namespace conduit
* @brief Namespace declaration for modules of type: conduit.
*/

/** \file
* @brief Header file for module: Distributed.
*/

/** \dir conduit/distributed
* @brief Contains code, documentation, and scripts for module: Distributed.
*/


#ifndef _KORALI_CONDUIT_DISTRIBUTED_
#define _KORALI_CONDUIT_DISTRIBUTED_


#include "config.hpp"
#include "modules/conduit/conduit.hpp"
#include <map>
#include <queue>
#include <vector>

#ifdef _KORALI_USE_MPI
  #include "mpi.h"
#endif

namespace korali
{
// Defining fallback type for MPI_Comm in case Korali wasn't compiled with compatibility with MPI
#ifndef _KORALI_USE_MPI

  /**
 * @brief Dummy communicator storage for the current Korali Worker
 */
  typedef long int MPI_Comm;

#endif

/**
* @brief Communicator storage for the current Korali Worker
*/
extern MPI_Comm __KoraliGlobalMPIComm;

/**
* @brief Communicator storage for the current Korali Worker
*/
extern MPI_Comm __koraliWorkerMPIComm;

#ifdef _KORALI_USE_MPI

/**
  * @brief Sets global MPI communicator
  * @param comm The MPI communicator to use
  * @return The return code of MPI_Comm_Dup
  */
int setKoraliMPIComm(const MPI_Comm &comm);

/**
   * @brief Returns MPI communicator for the current Korali Worker
   * @return A pointer to the MPI Communicator
   */
extern void *getWorkerMPIComm();

#else
}

/**
   * @brief Error handler for when MPI is not defined
   * @param ... accepts any parameters since it will fail anyway
   * @return Error code -1
   */
int setKoraliMPIComm(...);

/**
  * @brief Error handler for when MPI is not defined
  * @return A NULL pointer
  */
extern void *getWorkerMPIComm();

#endif

__startNamespace__;

/**
* @brief Class declaration for module: Distributed.
*/
class Distributed : public Conduit
{
  public: 
  /**
  * @brief Specifies the number of MPI ranks per Korali worker (k).
  */
   int _ranksPerWorker;
  
 
  /**
  * @brief Obtains the entire current state and configuration of the module.
  * @param js JSON object onto which to save the serialized state of the module.
  */
  void getConfiguration(knlohmann::json& js) override;
  /**
  * @brief Sets the entire state and configuration of the module, given a JSON object.
  * @param js JSON object from which to deserialize the state of the module.
  */
  void setConfiguration(knlohmann::json& js) override;
  /**
  * @brief Applies the module's default configuration upon its creation.
  * @param js JSON object containing user configuration. The defaults will not override any currently defined settings.
  */
  void applyModuleDefaults(knlohmann::json& js) override;
  /**
  * @brief Applies the module's default variable configuration to each variable in the Experiment upon creation.
  */
  void applyVariableDefaults() override;
  

  /**
  * @brief ID of the current rank.
  */
  int _rankId;

  /**
   * @brief Total number of ranks in execution
   */
  int _rankCount;

  /**
  * @brief Number of Korali Teams in execution
  */
  int _workerCount;

  /**
  * @brief Signals whether the worker has been assigned a team
  */
  int _workerIdSet;

  /**
 * @brief Local ID the rank within its Korali Worker
 */
  int _localRankId;

  /**
  * @brief Storage that contains the rank teams for each worker
  */
  std::vector<std::vector<int>> _workerTeams;

  /**
  * @brief Map that indicates to which worker does the current rank correspond to
  */
  std::vector<int> _rankToWorkerMap;

  /**
   * @brief Checks whether the number of MPI workers satisfies the requirement
   */
  void checkRankCount();

  void initServer() override;
  void initialize() override;
  void terminateServer() override;

  void stackEngine(Engine *engine) override;
  void popEngine() override;
  void listenWorkers() override;
  void broadcastMessageToWorkers(knlohmann::json &message) override;
  void sendMessageToEngine(knlohmann::json &message) override;
  knlohmann::json recvMessageFromEngine() override;
  void sendMessageToSample(Sample &sample, knlohmann::json &message) override;
  size_t getProcessId() override;

  /**
  * @brief Determines which rank is the root.
  * @return The rank id of the root rank.
  */
  int getRootRank();
  bool isRoot() override;
};

__endNamespace__;

  #endif // _KORALI_CONDUIT_DISTRIBUTED_

