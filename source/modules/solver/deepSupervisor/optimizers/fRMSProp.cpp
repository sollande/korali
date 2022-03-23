#include "fRMSProp.hpp"
#include <cmath>
#include <cstdlib>
#include <stdio.h>
#include <stdexcept>

namespace korali
{
fRMSProp::fRMSProp(size_t nVars)
{
 // Variable Parameters
 _currentGeneration = 1;
 _nVars = nVars;
 _initialValues.resize(_nVars, 0.0);
 _currentValue.resize(_nVars, 0.0);
 _gradient.resize(_nVars, 0.0);
 _modelEvaluationCount = 0;

  // Defaults
  _eta = 0.001f;
  _epsilon = 1e-08f;
  _decay = 0.999;

  _r.resize(nVars);
  _v.resize(nVars);
  reset();
}

void fRMSProp::reset()
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
    _r[i] = 0.0f;
    _v[i] = 0.0f;
  }

  _bestEvaluation = +std::numeric_limits<float>::infinity();
}

void fRMSProp::processResult(std::vector<float> &gradient)
{
  if (gradient.size() != _nVars)
  {
    fprintf(stderr, "Size of sample's gradient evaluations vector (%lu) is different from the number of problem variables defined (%lu).\n", gradient.size(), _nVars);
    throw std::runtime_error("Bad Inputs for Optimizer.");
  }

#pragma omp parallel for simd
  for (size_t i = 0; i < _nVars; i++)
  {
    _r[i] = (1.0f - _decay) * (gradient[i] * gradient[i]) + _decay * _r[i];
    _v[i] = (_eta / (std::sqrt(_r[i]) + _epsilon)) * -gradient[i];
    _currentValue[i] = _currentValue[i] - _v[i];
  }

  _modelEvaluationCount++;
}

bool fRMSProp::checkTermination()
{
  if (_currentGeneration >= _maxGenerations) return true;

  return false;
}

void fRMSProp::printInfo()
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
