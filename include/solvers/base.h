#ifndef _BASESOLVER_H_
#define _BASESOLVER_H_

#include <stdlib.h>
#include "problems/base.h"
#include <functional>
#include <map>

enum verbosity { korali_minimal = 0, korali_normal = 1, korali_detailed = 2 };

namespace Korali::Conduit {
    class Base;
}

namespace Korali::Solver
{

class Base {
 public:
 // Korali Runtime Variables
 size_t _currentGeneration;
 size_t _maxGens;                  // Max number of Conduit Generations
 size_t _sampleCount;
 size_t  N; // Parameter Count

 Korali::Conduit::Base* _conduit;
 Korali::Problem::Base* _problem;

 // Verbosity Settings
 size_t _reportFrequency;
 int _verbosity;

 Base(Korali::Problem::Base* problem);
 void setPopulationSize(int size) { _sampleCount = size; }
 void setMaxGenerations(int maxGens) { _maxGens = maxGens; }
 void setReportFrequency(size_t reportFrequency) { _reportFrequency = reportFrequency; }
 void setReportVerbosity(int verbosity) { _verbosity = verbosity; }

 void run();
 virtual void runSolver() = 0;
 virtual void processSample(size_t sampleId, double fitness) = 0;
};

} // namespace Korali

#endif // _BASESOLVER_H_
