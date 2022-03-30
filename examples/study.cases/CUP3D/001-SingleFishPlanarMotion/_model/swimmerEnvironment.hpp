//  Korali environment for CubismUP-3D
//  Copyright (c) 2020 CSE-Lab, ETH Zurich, Switzerland.

#include "korali.hpp"
#include <algorithm>
#include <random>
#include <filesystem>
#include "obstacles/StefanFish.h"
#include "Simulation.h"
#include "utils/BufferedLogger.h"
#include <Cubism/ArgumentParser.h>

#define NACTIONS 1

using namespace cubismup3d;

// command line arguments are read in Korali application
extern int _argc;
extern char **_argv;

void runEnvironment(korali::Sample &s);
void setInitialConditions(StefanFish *agent, size_t agentId, const bool isTraining, int rank, MPI_Comm comm);
bool isTerminal(StefanFish *agent);
double getReward(StefanFish *agent);
