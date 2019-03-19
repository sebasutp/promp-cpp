
#include <robotics/utils/math.hpp>
#include <algorithm>
#include <armadillo>

using namespace std;

namespace robotics {

  /**
   * @brief Computes the logarithm of the sum of exponentials in a numerically stable way
   */
  double log_sum_exp(const std::vector<double>& x) {
    double maxl = *max_element(x.begin(), x.end());
    double rest_sum = 0.0;
    for (const double& xi : x) {
      rest_sum += exp(xi - maxl);
    }
    return maxl + log(rest_sum);
  }
};

