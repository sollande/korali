#include "solvers/base.h"
#include "conduits/base.h"
#include "conduits/upcxx.h"
#include "conduits/single.h"

Korali::Solver::Base::Base(Korali::Problem::Base* problem)
{
  _problem = problem;
  _verbose = false;
  _sampleCount = 1000;
	_maxGens = 200;

	N = _problem->_parameterCount;
}

void Korali::Solver::Base::run()
{
  _problem->initializeParameters();

  std::string conduitString = getenv("KORALI_CONDUIT");

  bool recognized = false;
  auto printSyntax = [](){fprintf(stderr, "[Korali] Use $export KORALI_CONDUIT={single,smp,upcxx,mpi} to select a conduit.\n");};

  if (conduitString == "")
  {
			fprintf(stderr, "[Korali] Error: No sampling conduit was selected.\n");
			printSyntax();
			fprintf(stderr, "[Korali] Defaulting to 'single'.\n");
			conduitString = "single";
  }

  if (conduitString == "single") { _conduit = new Korali::Conduit::Single(this); recognized = true; }
  if (conduitString == "upcxx")  { _conduit = new Korali::Conduit::UPCXX(this);  recognized = true; }

  if (recognized == false)
  {
			fprintf(stderr, "[Korali] Error: Unrecognized conduit '%s' selected.\n", conduitString.c_str());
			printSyntax();
			exit(-1);
  }

  _conduit->initialize();
}
