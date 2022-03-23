#include "fAdagrad.hpp"
#include <cmath>
#include <cstdlib>
#include <stdio.h>
#include <stdexcept>

namespace korali
{
fAdagrad::fAdagrad(size_t nVars) : fAdam(nVars)
{
  _s.resize(nVars);

  reset();
}

void fAdagrad::reset()
{
  _currentGeneration = 1;
  _modelEvaluationCount = 0;

  for (size_t i = 0; i < _nVars; i++)
    if (std::isfinite(_initialValues[i]) == false)
    {
      fprintf(stderr, "Initial Value of variable \'%lu\' not defined (no defaults can be calculated).\n", i);
      throw std::runtime_error("Bad Inputs for Optimizer.");
    }

#pragma omp parallel for simd
  for (size_t i = 0; i < _nVars; i++)
  {
    _currentValue[i] = _initialValues[i];
    _s[i] = 0.0f;
  }

  _bestEvaluation = +std::numeric_limits<float>::infinity();
}

void fAdagrad::processResult(std::vector<float> &gradient)
{
  if (gradient.size() != _nVars)
  {
    fprintf(stderr, "Size of sample's gradient evaluations vector (%lu) is different from the number of problem variables defined (%lu).\n", gradient.size(), _nVars);
    throw std::runtime_error("Bad Inputs for Optimizer.");
  }

#pragma omp parallel for simd
  for (size_t i = 0; i < _nVars; i++)
  {
    _s[i] = _s[i] + (gradient[i] * gradient[i]);
    _currentValue[i] += (_eta / std::sqrt(_s[i] + _epsilon)) * gradient[i];
  }

  _modelEvaluationCount++;
}

bool fAdagrad::checkTermination()
{
  if (_currentGeneration >= _maxGenerations) return true;

  return false;
}

void fAdagrad::printInfo()
{
  printf("x = [ ");
  for (size_t k = 0; k < _nVars; k++) printf(" %.5le  ", _currentValue[k]);
  printf(" ]\n");

  printf("F(X) = %le \n", _currentEvaluation);

  printf("DF(X) = [ ");
  for (size_t k = 0; k < _nVars; k++) printf(" %.5le  ", _gradient[k]);
  printf(" ]\n");
}

} // namespace korali
