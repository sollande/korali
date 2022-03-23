#include "fAdaBelief.hpp"
#include <cmath>
#include <cstdlib>
#include <stdio.h>
#include <stdexcept>

namespace korali
{
fAdaBelief::fAdaBelief(size_t nVars) : fAdam(nVars)
{
  _secondCentralMoment.resize(nVars);

  reset();
}

void fAdaBelief::reset()
{
  fAdam::reset();

#pragma omp parallel for simd
  for (size_t i = 0; i < _nVars; i++)
    _secondCentralMoment[i] = 0.0f;
}

void fAdaBelief::processResult(std::vector<float> &gradient)
{
  _modelEvaluationCount++;

  if (gradient.size() != _nVars)
  {
    fprintf(stderr, "Size of sample's gradient evaluations vector (%lu) is different from the number of problem variables defined (%lu).\n", _gradient.size(), _nVars);
    throw std::runtime_error("Bad Inputs for Optimizer.");
  }

  const float secondCentralMomentFactor = 1.0f / (1.0f - std::pow(_beta2, (float)_modelEvaluationCount));
  const float firstCentralMomentFactor = 1.0f / (1.0f - std::pow(_beta1, (float)_modelEvaluationCount));
  const float notBeta1 = 1.0f - _beta1;
  const float notBeta2 = 1.0f - _beta2;

// update first and second moment estimators and bias corrected versions
#pragma omp parallel for simd
  for (size_t i = 0; i < _nVars; i++)
  {
    _firstMoment[i] = _beta1 * _firstMoment[i] - notBeta1 * gradient[i];

    const float biasCorrectedFirstMoment = _firstMoment[i] * firstCentralMomentFactor;
    const float secondMomentGradientDiff = gradient[i] + _firstMoment[i];
    _secondCentralMoment[i] = _beta2 * _secondCentralMoment[i] + notBeta2 * secondMomentGradientDiff * secondMomentGradientDiff;

    const float biasCorrectedSecondCentralMoment = _secondCentralMoment[i] * secondCentralMomentFactor;
    _currentValue[i] -= _eta / (std::sqrt(biasCorrectedSecondCentralMoment) + _epsilon) * biasCorrectedFirstMoment;
  }
}

} // namespace korali
