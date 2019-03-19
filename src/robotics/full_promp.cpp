
#include "robotics/full_promp.hpp"
#include "robotics/utils.hpp"

using namespace std;
using namespace arma;

namespace robotics {

  /**
   * Implementation class for the ProMPs. This class is for internal use in the library only.
   */
  class FullProMP::Impl {
    public:
      ProMP model;
      shared_ptr<const ScalarBasisFun> kernel;
      unsigned int num_joints;

      Impl() {
      }

      Impl(const shared_ptr<const ScalarBasisFun>& kernel, const ProMP& model, unsigned int num_joints) {
        this->model = model;
        this->kernel = kernel;
        this->num_joints = num_joints;
      }

      Impl(const Impl& x) = default;

      ~Impl() = default;

      /**
       * Returns the basis function matrix for time t out of a ProMP of time T. 
       */ 
      mat get_Phi_t_old(double t, double T, bool pos = true, bool vel = true, bool acc = false) const {
        double z = t / T;
        double vel_fac = 1.0 / T;
        unsigned int sub_rows = 0;
        unsigned int n_cols = num_joints*kernel->dim();
        if (pos) sub_rows++;
        if (vel) sub_rows++;
        if (acc) sub_rows++;
        int n_rows = num_joints*sub_rows;
        mat ans(n_rows, n_cols, fill::zeros);
        for (unsigned int i=0; i<num_joints; i++) {
          double mul_fac = 1.0;
          for (unsigned int j=0; j<sub_rows; j++) {
            vec basis = kernel->eval(z);
            for (unsigned int k=0; k<kernel->dim(); k++) {
              ans(i*sub_rows + j, i*kernel->dim() + k) = mul_fac*basis[k];
            }
            mul_fac *= vel_fac;
          }
        }
        return ans;
      }

      mat hvec_block_diag(const vector<vec>& v) const {
        unsigned int nrows = v.size(), ncols = 0;
        for (const auto& e : v) ncols += e.n_elem;
        mat ans(nrows, ncols, fill::zeros);
        unsigned int o_cols = 0, o_rows = 0;
        for (const auto& e : v) {
          for (unsigned int i=0; i<e.n_elem; i++)
            ans(o_rows, o_cols+i) = e[i];
          o_cols += e.n_elem;
          o_rows++;
        }
        return ans;
      }

      mat get_Phi_t(double t, double T, bool pos=true, bool vel = true, bool acc = false) const {
        double z = t / T;
        double vel_fac = 1.0 / T;
        vector<vec> pos_t, vel_t, acc_t;
        for (unsigned int i=0; i<num_joints; i++) {
          if (pos) pos_t.push_back( kernel->eval(z) );
          if (vel) vel_t.push_back( kernel->deriv(z, 1) * vel_fac );
          if (acc) acc_t.push_back( kernel->deriv(z, 2) * (vel_fac*vel_fac) );
        }
        vector<mat> ans;
        if (pos) ans.push_back( hvec_block_diag(pos_t) );
        if (vel) ans.push_back( hvec_block_diag(vel_t) );
        if (acc) ans.push_back( hvec_block_diag(acc_t) );
        return mat_concat_vertical(ans);
      }

      /**
       * We assume that the matrix Sigma_y in the model was computed using both position and velocity.
       * Following the convention of the paper, where y = [q, q_dot]. That is, first goes the
       * position and then the velocity.
       */ 
      unique_ptr<Impl> condition_current_state(double t, double T, const vec& pos_t, 
          const vec& vel_t, const mat& Sigma_pos, const mat& Sigma_vel) const {
        mat phi_t = get_Phi_t(t, T, true, true, false);
        vec state_t = join_cols(pos_t, vel_t);
        mat cov_t = block_diag({Sigma_pos, Sigma_vel});
        //ProMP cond = model.condition(phi_t, random::NormalDist{state_t, cov_t});
        ProMP cond = model.condition_no_Sigma_y(phi_t, random::NormalDist{state_t, cov_t});
        return unique_ptr<Impl>(new Impl(kernel, cond, num_joints));
      }

      unique_ptr<Impl> condition_pos(double z, const random::NormalDist& q) const {
        mat phi_t = get_Phi_t(z, 1.0, true, false, false);
        auto c_promp = model.condition_no_Sigma_y(phi_t, q);
        return unique_ptr<Impl>(new Impl{kernel, c_promp, num_joints});
      }

      /**
       * Here the y vector is also assumed to contain first all q values, then all qd values and
       * finally all the qdd values.
       */
      TrajectoryStep particular_traj_step(double t, double T, const vec& w) const {
        TrajectoryStep ans {vec(num_joints), vec(num_joints), vec(num_joints), t};
        mat phi_t = get_Phi_t(t, T, true, true, true);
        vec y = phi_t*w;
        ans.q = y.subvec(0,num_joints-1);
        ans.qd = y.subvec(num_joints, 2*num_joints-1);
        ans.qdd = y.subvec(2*num_joints, 3*num_joints-1);
        return ans;
      }

      TrajectoryStep mean_traj_step(double t, double T) const {
        return particular_traj_step(t, T, model.get_mu_w());
      }

      random::NormalDist joint_dist(double z, bool use_pos, bool use_vel, bool use_acc) const {
        //The actual execution time doesn't matter to compute the position kernel values
        mat position_phi_t = get_Phi_t(z, 1.0, use_pos, use_vel, use_acc);
        return model.joint_dist(position_phi_t);
      }

      double log_lh(const arma::vec& z, const arma::mat& obs) const {
        vector<mat> phi; 
        for (unsigned int i=0; i<obs.n_rows; i++) {
          phi.push_back( get_Phi_t(z[i],1,true,false,false) );
        }
        return model.log_lh(phi, obs);
      }

      unique_ptr<Impl> condition_multiple_obs(const arma::vec& z, const arma::mat& obs) const {
        vector<mat> phi; 
        for (unsigned int i=0; i<obs.n_rows; i++) {
          phi.push_back( get_Phi_t(z[i],1,true,false,false) );
        }
        auto nmodel = model.condition_multiple_obs(phi, obs);
        return unique_ptr<Impl>(new Impl(kernel, nmodel, num_joints));
      }

  };


  /**
   * Default constructor. Do not use this constructor unless you know what you are doing. This
   * constructor does not initialize the memory and using any method on an object created with
   * this constructor will result in undefined behaviour.
   */
  FullProMP::FullProMP() {
    _impl = nullptr;
  }

  FullProMP::FullProMP(std::shared_ptr<const ScalarBasisFun> kernel, const ProMP& model, 
          unsigned int num_joints) {
    _impl = unique_ptr<Impl>(new Impl(kernel, model, num_joints));
  }
      
  FullProMP::FullProMP(const FullProMP& b) {
    if (b._impl) {
      _impl = unique_ptr<Impl>(new Impl(*b._impl));
    }
  }

  FullProMP& FullProMP::operator=(const FullProMP& b) {
    if (this != &b && b._impl) {
      _impl = unique_ptr<Impl>(new Impl(*b._impl));
    }
    return *this;
  }

  double FullProMP::log_lh(const arma::vec& z, const arma::mat& obs) const {
    return _impl->log_lh(z, obs);
  }

  FullProMP FullProMP::condition_multiple_obs(const arma::vec& z, const arma::mat& obs) const {
    FullProMP ans;
    ans._impl = _impl->condition_multiple_obs(z, obs);
    return ans;
  }

  FullProMP::FullProMP(FullProMP&& b) = default;
  FullProMP& FullProMP::operator=(FullProMP&& b) = default;
  FullProMP::~FullProMP() = default;

  /**
   * Conditions the ProMP in the current velocity and position states. This method assumes that
   * the matrix Sigma_y of the underlying ProMP was trained with both position and velocity. First
   * comes the noise of the position for every degree of freedom and then the noise of the velocity
   * also for every degree of freedom.
   * @param t Current time (Value between 0 and T)
   * @param T Total execution time of the ProMP
   * @param pos_t Current position (at time t)
   * @params vel_t Current velocity (at time t)
   */ 
  FullProMP FullProMP::condition_current_state(double t, double T, const arma::vec& pos_t, 
      const arma::vec& vel_t) const {
    unsigned int d = _impl->num_joints;
    const mat zero = 1e-6*eye<mat>(d,d);
    FullProMP ans;
    ans._impl = _impl->condition_current_state(t, T, pos_t, vel_t, zero, zero);
    return ans;
  }

  /**
   * Conditions the ProMP in the current position. Using the same assumptions as
   * FullProMP::condition_current_state.
   * @param t Current time (Value between 0 and T)
   * @param T Total execution time of the ProMP
   * @param pos_t Current position (at time t)
   * @params vel_t Current velocity (at time t)
   */ 
  FullProMP FullProMP::condition_current_position(double t, double T, const arma::vec& pos_t) const {
    unsigned int d = _impl->num_joints;
    const mat zero = 1e-6*eye<mat>(d,d);
    FullProMP ans;
    ans._impl = _impl->condition_pos(t/T, random::NormalDist{pos_t, zero});
    return ans;
  }


  FullProMP FullProMP::condition_pos(double z, const random::NormalDist& q) const {
    FullProMP ans;
    ans._impl = _impl->condition_pos(z, q);
    return ans;
  }

  /**
   * @brief Returns the distribution of the joint positions at a given desired point in time.
   * @param[in] z Represents the phase variable between 0 and 1 related to time
   */
  random::NormalDist FullProMP::joint_dist(double z) const {
    return _impl->joint_dist(z, true, false, false);
  }

  /**
   * @brief Returns the distribution of the joint positions, velocities or accelerations
   * at a given desired point in time.
   * @param[in] z Represents the phase variable between 0 and 1 related to time
   */
  random::NormalDist FullProMP::joint_dist(double z, bool use_pos, bool use_vel, bool use_acc) const {
    return _impl->joint_dist(z, use_pos, use_vel, use_acc);
  }

  /**
   * Returns a trajectory step for the time t on a ProMP of total execution time T. For this method
   * the desired trajectory to follow is the mean of the ProMP. 
   * @param t Time where the trajectory step is desired (Value between 0 and T)
   * @param T Total execution time of the ProMP
   */ 
  TrajectoryStep FullProMP::mean_traj_step(double t, double T) const {
    return _impl->mean_traj_step(t, T);
  }

  /**
   * @brief Returns the matrix Phi (basis function matrix) evaluated at z.
   * @param[in] z Represents the phase variable between 0 and 1 related to time.
   */
  mat FullProMP::get_phi_t(double z) const {
    return _impl->get_Phi_t(z, 1.0, true, false, false);
  }

  const ProMP& FullProMP::get_model() const {
    return _impl->model;
  }

  shared_ptr<const ScalarBasisFun> FullProMP::get_kernel() const {
    return _impl->kernel;
  }

  unsigned int FullProMP::get_num_joints() const {
    return _impl->num_joints;
  }

  void FullProMP::set_model(const ProMP& model) {
    _impl->model = model;
  }
      
  void FullProMP::set_kernel(shared_ptr<const ScalarBasisFun> kernel) {
    _impl->kernel = kernel;
  }

};
