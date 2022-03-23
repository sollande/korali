/**************************************************************
 * A single-precision fast version of MADGRAD for Learning
 **************************************************************/

#ifndef _KORALI_FAST_ADAGRAD_HPP_
#define _KORALI_FAST_ADAGRAD_HPP_

#include "fAdam.hpp"

namespace korali
{
/**
* @brief Class declaration for module: MADGRAD.
*/
class fAdagrad : public fAdam
{
  public:
  /**
 * @brief Default constructor for the optimizer
 * @param nVars Variable-space dimensionality
 */
  fAdagrad(size_t nVars);

  /**
  * @brief Squared Gradient Component
  */
  std::vector<float> _s;

  virtual bool checkTermination() override;
  virtual void processResult(std::vector<float> &gradient) override;
  virtual void reset() override;
  virtual void printInfo() override;
};

} // namespace korali

#endif // _KORALI_FAST_ADAGRAD_HPP_
