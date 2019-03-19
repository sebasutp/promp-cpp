#ifndef ROBOTICS_FULL_PROMP_H
#define ROBOTICS_FULL_PROMP_H

#include <armadillo>
#include <memory>
#include <vector>
#include <robotics/basis_functions.hpp>
#include <robotics/promp.hpp>
#include <robotics/trajectory.hpp>
#include <robotics/utils/random.hpp>

namespace robotics {

  /**
   * @brief Probabilistic Movement Primitive class
   * @since version 0.0.1
   * This class models a probabilistic movement primitive as defined in Alex's paper. The joint 
   * dependencies are also modeled as in that paper, but the conditioning is done differently.
   */ 
  class FullProMP {
    public:
      FullProMP();
      FullProMP(std::shared_ptr<const ScalarBasisFun> kernel, const ProMP& model, 
          unsigned int num_joints);
      FullProMP(const FullProMP& b); //copy constructor
      FullProMP(FullProMP&& b);
      FullProMP& operator=(const FullProMP& b);
      FullProMP& operator=(FullProMP&& b);
      ~FullProMP();

      FullProMP condition_pos(double z, const random::NormalDist& q) const;
      FullProMP condition_current_state(double t, double T, const arma::vec& pos_t, 
          const arma::vec& vel_t) const;
      FullProMP condition_current_position(double t, double T, const arma::vec& pos_t) const;
      FullProMP condition_multiple_obs(const arma::vec& z, const arma::mat& obs) const;

      random::NormalDist joint_dist(double z) const;
      random::NormalDist joint_dist(double z, bool use_pos, bool use_vel, bool use_acc) const;
      TrajectoryStep mean_traj_step(double t, double T) const;
      double log_lh(const arma::vec& z, const arma::mat& obs) const;

      arma::mat get_phi_t(double z) const;
      const ProMP& get_model() const;
      std::shared_ptr<const ScalarBasisFun> get_kernel() const;
      unsigned int get_num_joints() const;
      void set_model(const ProMP& model);
      void set_kernel(std::shared_ptr<const ScalarBasisFun> kernel);
      void set_num_joints(unsigned int num_joints);
    private:
      class Impl;
      std::unique_ptr<Impl> _impl;
  };

};

#endif
