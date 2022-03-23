/**************************************************************
 * A single-precision fast version of MADGRAD for Learning
 **************************************************************/

#ifndef _KORALI_FAST_MADGRAD_HPP_
#define _KORALI_FAST_MADGRAD_HPP_

#include "fGradientBasedOptimizer.hpp"

namespace korali
{
/**
* @brief Class declaration for module: MADGRAD.
*/
class fMadGrad : public fGradientBasedOptimizer
{
  public:
  /**
 * @brief Default constructor for the optimizer
 * @param nVars Variable-space dimensionality
 */
  fMadGrad(size_t nVars);

  /**
  * @brief Gradient Component
  */
  std::vector<float> _s;

  /**
  * @brief Squared Gradient Component
  */
  std::vector<float> _v;

  /**
  * @brief Update rule
  */
  std::vector<float> _z;

  /**
   * @brief Safety addition on divide
   */
  float _epsilon;

  /**
   * @brief Update momentum
   */
  float _momentum;

  virtual bool checkTermination() override;
  virtual void processResult(std::vector<float> &gradient) override;
  virtual void reset() override;
  virtual void printInfo() override;
};

} // namespace korali

#endif // _KORALI_FAST_MADGRAD_HPP_
