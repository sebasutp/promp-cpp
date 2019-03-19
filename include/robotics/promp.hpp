#ifndef ROBOTICS_PROMP_H
#define ROBOTICS_PROMP_H

#include <armadillo>
#include <memory>
#include <robotics/utils/random.hpp>

/*!
@brief namespace for the robotics library
@since version 0.0.1
*/
namespace robotics {

  /**
   * @brief Basic Probabilistic Movement Primitive class
   * This class models a Generic and simple probabilistic movement primitive that can be used for any
   * application. For instance this class can be used to model robot joints independently or dependent.
   * It can also be used with any kind of basis function. Use this class only if the task specific
   * classes are not appropriate for your problem, since those classes are easier to use.
   */ 
  class ProMP {
    public:
      ProMP();
      ProMP(const arma::vec& mu_w, const arma::mat& Sigma_w, const arma::mat& Sigma_y);
      ProMP(const ProMP& b); //copy constructor
      ProMP(ProMP&& b);
      ProMP& operator=(const ProMP& b);
      ProMP& operator=(ProMP&& b);
      ~ProMP();

      ProMP condition(const arma::mat& phi_t, const random::NormalDist& q_dist) const;
      ProMP condition_no_Sigma_y(const arma::mat& phi_t, const random::NormalDist& q_dist) const;
      ProMP condition_multiple_obs(const std::vector<arma::mat>& phi, const arma::mat& obs) const;
      random::NormalDist joint_dist(const arma::mat& position_phi_t) const;

      double log_lh(const std::vector<arma::mat>& phi, const arma::mat& obs) const;

      const arma::mat& get_Sigma_w() const;
      const arma::mat& get_Sigma_y() const;
      const arma::vec& get_mu_w() const;
      void set_Sigma_w(const arma::mat& Sigma_w);
      void set_Sigma_y(const arma::mat& Sigma_y);
      void set_mu_w(const arma::vec& mu_w);
    private:
      class Impl;
      std::unique_ptr<Impl> _impl;
  };

};

#endif
