#include "gtest/gtest.h"
#include "korali.hpp"
#include "modules/solver/deepSupervisor/optimizers/fCMAES.hpp"
#include "modules/solver/deepSupervisor/optimizers/fAdaBelief.hpp"
#include "modules/solver/deepSupervisor/optimizers/fAdagrad.hpp"
#include "modules/solver/deepSupervisor/optimizers/fAdam.hpp"
#include "modules/solver/deepSupervisor/optimizers/fMadGrad.hpp"
#include "modules/solver/deepSupervisor/optimizers/fRMSProp.hpp"
#include <limits>

namespace
{

 float model(const std::vector<float>& x)
 {
  float result = 0.0f;
  for (size_t i = 0; i < x.size(); i++)
   result += -(x[i]+1)*(x[i]+1) + 2;
  return result;
 }

 std::vector<float> model_gradient(const std::vector<float>& x)
 {
  std::vector<float> grad(x.size());
  for (size_t i = 0; i < x.size(); i++)
   grad[i] = -2*(x[i]+1);
  return grad;
 }

 const size_t N = 4;
 const size_t popSize = 128;
 const size_t muSize = 32;
 const size_t seed = 1337;
 const float targetPrecision = 0.0001f;
 const size_t maxGenerations = 30;

 using namespace korali;

 TEST(fCMAES, fullCovariance)
 {
  fCMAES c(N, popSize, muSize);
  ASSERT_NO_THROW(c.setSeed(seed));

  c._initialCumulativeCovariance = 0.9;
  c._maxGenerations = maxGenerations;
  for (size_t i = 0; i < N; i++)
  {
   c._lowerBounds[i] = -10.0f;
   c._upperBounds[i] = +10.0f;
   c._initialMeans[i] = 0.0f;
   c._initialStandardDeviations[i] = 5.0f;
  }

  ASSERT_NO_THROW(c.reset());

  std::vector<float> candidateEvaluations(popSize);

  while (c.checkTermination() == false)
  {
   ASSERT_NO_THROW(c.prepareGeneration());

   for (size_t i = 0; i < popSize; i++)
    candidateEvaluations[i] = model(c._samplePopulation[i]);

   ASSERT_NO_THROW(c.updateDistribution(candidateEvaluations));
   c._currentGeneration++;
  }

  c.printInfo();
  for (size_t i = 0; i < N; i++)
   ASSERT_NEAR(c._bestEverVariables[i], -1.0f, 0.001);
  ASSERT_NEAR(c._bestEverValue, 8.0f, targetPrecision);
 }

 TEST(fCMAES, diagonalCovariance)
 {
  fCMAES c(N, popSize, muSize);
  ASSERT_NO_THROW(c.setSeed(seed));

  c._maxGenerations = maxGenerations;
  c._isDiagonal = 1;
  c._isSigmaBounded = 1;

  for (size_t i = 0; i < N; i++)
  {
   c._lowerBounds[i] = -10.0f;
   c._upperBounds[i] = +10.0f;
   c._initialMeans[i] = 0.0f;
   c._initialStandardDeviations[i] = 5.0f;
  }

  ASSERT_NO_THROW(c.reset());

  std::vector<float> candidateEvaluations(popSize);

  while (c.checkTermination() == false)
  {
   ASSERT_NO_THROW(c.prepareGeneration());

   for (size_t i = 0; i < popSize; i++)
    candidateEvaluations[i] = model(c._samplePopulation[i]);

   ASSERT_NO_THROW(c.updateDistribution(candidateEvaluations));
   c._currentGeneration++;
  }

  c.printInfo();
  for (size_t i = 0; i < N; i++)
   ASSERT_NEAR(c._bestEverVariables[i], -1.0f, 0.001);
  ASSERT_NEAR(c._bestEverValue, 8.0f, targetPrecision);
 }

 TEST(fCMAES, equalMu)
 {
  fCMAES c(N, popSize, muSize);
  ASSERT_NO_THROW(c.setSeed(seed));

  c._maxGenerations = maxGenerations;
  c._muType == "Equal";

  for (size_t i = 0; i < N; i++)
  {
   c._lowerBounds[i] = -10.0f;
   c._upperBounds[i] = +10.0f;
   c._initialMeans[i] = 0.0f;
   c._initialStandardDeviations[i] = 5.0f;
  }

  ASSERT_NO_THROW(c.reset());

  std::vector<float> candidateEvaluations(popSize);

  while (c.checkTermination() == false)
  {
   ASSERT_NO_THROW(c.prepareGeneration());

   for (size_t i = 0; i < popSize; i++)
    candidateEvaluations[i] = model(c._samplePopulation[i]);

   ASSERT_NO_THROW(c.updateDistribution(candidateEvaluations));
   c._currentGeneration++;
  }

  c.printInfo();
  for (size_t i = 0; i < N; i++)
   ASSERT_NEAR(c._bestEverVariables[i], -1.0f, 0.001);
  ASSERT_NEAR(c._bestEverValue, 8.0f, targetPrecision);
 }

 TEST(fCMAES, logMu)
 {
  fCMAES c(N, popSize, muSize);
  ASSERT_NO_THROW(c.setSeed(seed));

  c._maxGenerations = maxGenerations;
  c._muType == "Logarithmic";

  for (size_t i = 0; i < N; i++)
  {
   c._lowerBounds[i] = -10.0f;
   c._upperBounds[i] = +10.0f;
   c._initialMeans[i] = 0.0f;
   c._initialStandardDeviations[i] = 5.0f;
  }

  ASSERT_NO_THROW(c.reset());

  std::vector<float> candidateEvaluations(popSize);

  while (c.checkTermination() == false)
  {
   ASSERT_NO_THROW(c.prepareGeneration());

   for (size_t i = 0; i < popSize; i++)
    candidateEvaluations[i] = model(c._samplePopulation[i]);

   ASSERT_NO_THROW(c.updateDistribution(candidateEvaluations));
   c._currentGeneration++;
  }

  c.printInfo();
  for (size_t i = 0; i < N; i++)
   ASSERT_NEAR(c._bestEverVariables[i], -1.0f, 0.001);
  ASSERT_NEAR(c._bestEverValue, 8.0f, targetPrecision);
 }

 TEST(fCMAES, failBadMu)
 {
  fCMAES c(N, popSize, muSize);
  c._muType = "Undefined";
  ASSERT_ANY_THROW(c.initMuWeights(muSize));
 }

 TEST(fCMAES, failNonFiniteValue)
 {
  fCMAES c(N, popSize, muSize);
  ASSERT_NO_THROW(c.setSeed(seed));

  std::vector<float> M(N*N, std::numeric_limits<float>::infinity());
  M[0] = 0.0f;

  ASSERT_NO_THROW(c.updateEigensystem(M));
 }


 /////////////////// AdaBelief

 TEST(fGradientBasedOptimizer, baseDestructor)
 {
  fGradientBasedOptimizer* c = new fAdaBelief(N);
  delete c;
 }

 // In Korali, optimizers maximize, so concave functions are candidates for correct execution
 TEST(fAdaBelief, concaveFunction)
 {
  fAdaBelief c(N);

  c._maxGenerations = 5000;
  for (size_t i = 0; i < N; i++)
   c._initialValues[i] = 0.0f;

  ASSERT_NO_THROW(c.reset());

  while (c.checkTermination() == false)
  {
   auto result = model(c._currentValue);
   auto gradient = model_gradient(c._currentValue);

   ASSERT_NO_THROW(c.processResult(gradient));
   c._currentGeneration++;
  }

  c.printInfo();
  for (size_t i = 0; i < N; i++)
   ASSERT_NEAR(c._currentValue[i], -1.0f, 0.001);

  auto result = model(c._currentValue);
  ASSERT_NEAR(result, 8.0f, targetPrecision);
 }

 TEST(fAdaBelief, testFailBadInputs)
 {
  fAdaBelief c(N);
  c._initialValues = std::vector<float>(N, std::numeric_limits<float>::infinity());
  ASSERT_ANY_THROW(c.reset());
 }

 TEST(fAdaBelief, testFailBadGradientSize)
 {
  fAdaBelief c(N);
  std::vector<float> gradient;
  ASSERT_ANY_THROW(c.processResult(gradient));
 }

 /////////////////// Adagrad

 // In Korali, optimizers maximize, so concave functions are candidates for correct execution
 TEST(fAdagrad, concaveFunction)
 {
  fAdagrad c(N);

  c._maxGenerations = 15000;
  c._eta = 0.01f;
  for (size_t i = 0; i < N; i++)
   c._initialValues[i] = 0.0f;

  ASSERT_NO_THROW(c.reset());

  while (c.checkTermination() == false)
  {
   auto result = model(c._currentValue);
   auto gradient = model_gradient(c._currentValue);

   ASSERT_NO_THROW(c.processResult(gradient));
   c._currentGeneration++;
  }

  c.printInfo();
  for (size_t i = 0; i < N; i++)
   ASSERT_NEAR(c._currentValue[i], -1.0f, 0.01);

  auto result = model(c._currentValue);
  ASSERT_NEAR(result, 8.0f, targetPrecision);
 }

 TEST(fAdagrad, testFailBadInputs)
 {
  fAdagrad c(N);
  c._initialValues = std::vector<float>(N, std::numeric_limits<float>::infinity());
  ASSERT_ANY_THROW(c.reset());
 }

 TEST(fAdagrad, testFailBadGradientSize)
 {
  fAdagrad c(N);
  std::vector<float> gradient;
  ASSERT_ANY_THROW(c.processResult(gradient));
 }

 /////////////////// Adam

 // In Korali, optimizers maximize, so concave functions are candidates for correct execution
 TEST(fAdam, concaveFunction)
 {
  fAdam c(N);

  c._maxGenerations = 5000;
  for (size_t i = 0; i < N; i++)
   c._initialValues[i] = 0.0f;

  ASSERT_NO_THROW(c.reset());

  while (c.checkTermination() == false)
  {
   auto result = model(c._currentValue);
   auto gradient = model_gradient(c._currentValue);

   ASSERT_NO_THROW(c.processResult(gradient));
   c._currentGeneration++;
  }

  c.printInfo();
  for (size_t i = 0; i < N; i++)
   ASSERT_NEAR(c._currentValue[i], -1.0f, 0.001);

  auto result = model(c._currentValue);
  ASSERT_NEAR(result, 8.0f, targetPrecision);
 }

 TEST(fAdam, testFailBadInputs)
 {
  fAdam c(N);
  c._initialValues = std::vector<float>(N, std::numeric_limits<float>::infinity());
  ASSERT_ANY_THROW(c.reset());
 }

 TEST(fAdam, testFailBadGradientSize)
 {
  fAdam c(N);
  std::vector<float> gradient;
  ASSERT_ANY_THROW(c.processResult(gradient));
 }

 /////////////////// MadGrad

 // In Korali, optimizers maximize, so concave functions are candidates for correct execution
 TEST(fMadGrad, concaveFunction)
 {
  fMadGrad c(N);

  c._maxGenerations = 5000;
  for (size_t i = 0; i < N; i++)
   c._initialValues[i] = 0.0f;

  ASSERT_NO_THROW(c.reset());

  while (c.checkTermination() == false)
  {
   auto result = model(c._currentValue);
   auto gradient = model_gradient(c._currentValue);

   ASSERT_NO_THROW(c.processResult(gradient));
   c._currentGeneration++;
  }

  c.printInfo();
  for (size_t i = 0; i < N; i++)
   ASSERT_NEAR(c._currentValue[i], -1.0f, 0.001);

  auto result = model(c._currentValue);
  ASSERT_NEAR(result, 8.0f, targetPrecision);
 }

 TEST(fMadGrad, testFailBadInputs)
 {
  fMadGrad c(N);
  c._initialValues = std::vector<float>(N, std::numeric_limits<float>::infinity());
  ASSERT_ANY_THROW(c.reset());
 }

 TEST(fMadGrad, testFailBadGradientSize)
 {
  fMadGrad c(N);
  std::vector<float> gradient;
  ASSERT_ANY_THROW(c.processResult(gradient));
 }

 /////////////////// RMSProp

 // In Korali, optimizers maximize, so concave functions are candidates for correct execution
 TEST(fRMSProp, concaveFunction)
 {
  fRMSProp c(N);

  c._maxGenerations = 5000;
  for (size_t i = 0; i < N; i++)
   c._initialValues[i] = 0.0f;

  ASSERT_NO_THROW(c.reset());

  while (c.checkTermination() == false)
  {
   auto result = model(c._currentValue);
   auto gradient = model_gradient(c._currentValue);

   ASSERT_NO_THROW(c.processResult(gradient));
   c._currentGeneration++;
  }

  c.printInfo();
  for (size_t i = 0; i < N; i++)
   ASSERT_NEAR(c._currentValue[i], -1.0f, 0.001);

  auto result = model(c._currentValue);
  ASSERT_NEAR(result, 8.0f, targetPrecision);
 }

 TEST(fRMSProp, testFailBadInputs)
 {
  fRMSProp c(N);
  c._initialValues = std::vector<float>(N, std::numeric_limits<float>::infinity());
  ASSERT_ANY_THROW(c.reset());
 }

 TEST(fRMSProp, testFailBadGradientSize)
 {
  fRMSProp c(N);
  std::vector<float> gradient;
  ASSERT_ANY_THROW(c.processResult(gradient));
 }

} // namespace
