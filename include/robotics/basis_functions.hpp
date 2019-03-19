#ifndef ROBOTICS_BASIS_FUNCTIONS_H
#define ROBOTICS_BASIS_FUNCTIONS_H

#include <armadillo>
#include <json.hpp>
#include <memory>
#include <vector>

namespace robotics {

  /**
   * @brief Base abstract class for Basis function representations for ProMPs
   * @since version 0.0.1
   * This class should not be instantiated directly. Inherit from this class if you want to build
   * new basis functions
   */
  class ScalarBasisFun {
    public:
      /**
       * @brief Evaluate the basis functions on the given scalar t
       * @since version 0.0.1
       */
      virtual arma::vec eval(double t) const = 0;

      /**
       * @brief Basis function derivatives with respect to the scalar input t
       * @since version 0.0.1
       * Compute n_th order the derivative of the basis functions with respect to t and
       * evaluates this derivative on the given value of t.
       * @param[in] t Value for the scalar input where the derivative is evaluated
       * @param[in] order Order of the derivative to compute. It is recommended to
       * implement at least up to second order derivatives.
       */
      virtual arma::vec deriv(double time, unsigned int order) const = 0;

      /**
       * @brief Number of basis functions
       * @since version 0.0.1
       * The number of dimensions on the vectors returned in eval and deriv methods. This
       * also correspond to the number of basis functions.
       */
      virtual unsigned int dim() const = 0;

      virtual nlohmann::json to_stream() const = 0;

      /**
       * @brief Virtual distructor
       */
      virtual ~ScalarBasisFun() = default;
  };

  /**
   * Represents a set of scalar radial basis functions. If \f$c_i\f$ represent
   * the center of the basis function i, the method eval returns a vector y where
   * each component \f$y_i\f$ is computed as
   * \f[ y_i = \exp(-\frac{(t - c_i)^2}{2\sigma^2}) \f]
   * where \f$\sigma\f$ corresponds to the parameter sigma passed in the constructor
   * and \f$c_i\f$ is taken from each component the vector of centers passed in the
   * constructor as well.  
   * @brief Gaussian Kernel Basis functions for ProMPs
   * @since version 0.0.1
   */
  class ScalarGaussBasis : public ScalarBasisFun {
    public:
      ScalarGaussBasis();
      ScalarGaussBasis(const arma::vec& centers, double sigma);
      ScalarGaussBasis(const ScalarGaussBasis& b);
      ScalarGaussBasis(ScalarGaussBasis&& b);
      ScalarGaussBasis& operator=(const ScalarGaussBasis& b);
      ScalarGaussBasis& operator=(ScalarGaussBasis&& b);
      ~ScalarGaussBasis();

      arma::vec eval(double t) const;
      arma::vec deriv(double t, unsigned int order) const;
      unsigned int dim() const;
      nlohmann::json to_stream() const;
    private:
      class Impl;
      std::unique_ptr<Impl> _impl;
  };

  /**
   * @brief Polynomial Basis functions for ProMPs
   * @since version 0.0.1
   */
  class ScalarPolyBasis : public ScalarBasisFun {
    public:
      ScalarPolyBasis();
      ScalarPolyBasis(unsigned int order);
      ScalarPolyBasis(const ScalarPolyBasis& b); //copy constructor
      ScalarPolyBasis(ScalarPolyBasis&& b);
      ScalarPolyBasis& operator=(const ScalarPolyBasis& b);
      ScalarPolyBasis& operator=(ScalarPolyBasis&& b);
      ~ScalarPolyBasis();

      arma::vec eval(double t) const;
      arma::vec deriv(double t, unsigned int order) const;
      unsigned int dim() const;
      nlohmann::json to_stream() const;
    private:
      class Impl;
      std::unique_ptr<Impl> _impl;
  };

  /**
   * @brief Combined Basis functions for ProMPs
   * @since version 0.1.1
   */
  class ScalarCombBasis : public ScalarBasisFun {
    public:
      ScalarCombBasis(const std::vector<std::shared_ptr<ScalarBasisFun>>& basis);
      ~ScalarCombBasis();

      arma::vec eval(double t) const;
      arma::vec deriv(double t, unsigned int order) const;
      unsigned int dim() const;
      nlohmann::json to_stream() const;
    private:
      class Impl;
      std::unique_ptr<Impl> _impl;      
  };
};

#endif
