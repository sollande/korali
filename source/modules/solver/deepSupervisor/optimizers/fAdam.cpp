#include "fAdam.hpp"
#include <cmath>
#include <cstdlib>
#include <stdio.h>
#include <stdexcept>

namespace korali
{
fAdam::fAdam(size_t nVars)
{
  // Variable Parameters
  _currentGeneration = 1;
  _nVars = nVars;
  _initialValues.resize(_nVars, 0.0);
  _currentValue.resize(_nVars, 0.0);
  _gradient.resize(_nVars, 0.0);
  _modelEvaluationCount = 0;

  // Variable Parameters
  _firstMoment.resize(_nVars, 0.0);
  _secondMoment.resize(_nVars, 0.0);

  // Defaults
  _beta1 = 0.9f;
  _beta2 = 0.999f;
  _beta1Pow = 1.0f;
  _beta2Pow = 1.0f;
  _eta = 0.001f;
  _epsilon = 1e-08f;

  // Termination Criteria
  _maxGenerations = 10000000;
  _minGradientNorm = 1e-16f;
  _maxGradientNorm = 1e+16f;

  reset();
}

void fAdam::reset()
{
  _currentGeneration = 1;
  _modelEvaluationCount = 0;
  _beta1Pow = 1.f;
  _beta2Pow = 1.f;

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
    _firstMoment[i] = 0.0f;
    _secondMoment[i] = 0.0f;
  }

  _bestEvaluation = +std::numeric_limits<float>::infinity();
}

void fAdam::processResult(std::vector<float> &gradient)
{
  _modelEvaluationCount++;

  // Calculate powers of beta1 & beta2
  _beta1Pow*=_beta1;
  _beta2Pow*=_beta2;

  if (gradient.size() != _nVars)
  {
    fprintf(stderr, "Size of sample's gradient evaluations vector (%lu) is different from the number of problem variables defined (%lu).\n", gradient.size(), _nVars);
    throw std::runtime_error("Bad Inputs for Optimizer.");
  }

  const float firstCentralMomentFactor = 1.0f / (1.0f - _beta1Pow);
  const float secondCentralMomentFactor = 1.0f / (1.0f - _beta2Pow);
  const float notBeta1 = 1.0f - _beta1;
  const float notBeta2 = 1.0f - _beta2;

  // update first and second moment estimators and parameters
#pragma omp parallel for simd
  for (size_t i = 0; i < _nVars; i++)
  {
    _firstMoment[i] = _beta1 * _firstMoment[i] - notBeta1 * gradient[i];
    _secondMoment[i] = _beta2 * _secondMoment[i] + notBeta2 * gradient[i] * gradient[i];
    _currentValue[i] -= _eta / (std::sqrt(_secondMoment[i] * secondCentralMomentFactor) + _epsilon) * _firstMoment[i] * firstCentralMomentFactor;
  }
}

bool fAdam::checkTermination()
{
  if (_currentGeneration >= _maxGenerations) return true;

  return false;
}

void fAdam::printInfo()
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
