
#include "robotics/promp.hpp"

using namespace std;
using namespace arma;

namespace robotics {

  /**
   * Implementation class for the ProMPs. This class is for internal use in the library only.
   */
  class ProMP::Impl {
    public:
      arma::vec mu_w;
      arma::mat Sigma_w;
      arma::mat Sigma_y;

      Impl() {
      }

      Impl(const vec& mu_w, const mat& Sigma_w, const mat& Sigma_y) {
        this->mu_w = mu_w;
        this->Sigma_w = Sigma_w;
        this->Sigma_y = Sigma_y;
      }

      Impl(const Impl& x) = default;

      ~Impl() = default;

      unique_ptr<Impl> condition(const mat& phi_t, const random::NormalDist& q_dist) const {
        const vec& mu_q = q_dist.mean();
        const mat& Sigma_q = q_dist.cov();
        mat inv_Sig_w = this->Sigma_w.i();
        mat inv_Sig_y = this->Sigma_y.i();
        mat Sw = inv(inv_Sig_w + phi_t.t()*inv_Sig_y*phi_t);
        mat A = Sw*phi_t.t()*inv_Sig_y;
        vec b = Sw*inv_Sig_w*this->mu_w;
        vec mu_w = A*mu_q + b;
        mat Sigma_w = Sw + A*Sigma_q*A.t();
        return unique_ptr<Impl>(new Impl(mu_w, Sigma_w, this->Sigma_y));
      }

      unique_ptr<Impl> condition_no_Sigma_y(const mat& phi_t, const random::NormalDist& q_dist) const {
        const vec& mu_q = q_dist.mean();
        const mat& Sigma_q = q_dist.cov();
        int d = phi_t.n_rows;
        mat tmp1 = this->Sigma_w*phi_t.t();
        mat tmp2 = inv_sympd(phi_t*this->Sigma_w*phi_t.t());
        mat tmp3 = tmp1*tmp2;
        vec mu_w = this->mu_w + tmp3*(mu_q - phi_t*this->mu_w);
        mat tmp4 = eye<mat>(d,d) - Sigma_q*tmp2;
        mat Sigma_w = this->Sigma_w - tmp3*tmp4*tmp1.t();
        return unique_ptr<Impl>(new Impl(mu_w, Sigma_w, this->Sigma_y));
      }

      unique_ptr<Impl> condition_multiple_obs(const vector<mat>& phi, const mat& obs) {
        unsigned int Tn = phi.size();
        mat invSy = inv(Sigma_y);
        mat invSw = inv(Sigma_w);

        vec sum_mean = invSw*mu_w;
        mat sum_cov = invSw;
        for (unsigned int t=0; t<Tn; t++) {
          const mat& phi_nt = phi[t];
          mat tmp1 = phi_nt.t()*invSy;
          sum_mean += tmp1*obs.col(t);
          sum_cov += tmp1*phi_nt;
        }
        mat Sw = inv(sum_cov);
        Sw = (Sw + Sw.t()) / 2.0; //Force symmetry
        vec wn = Sw*sum_mean;
        return unique_ptr<Impl>(new Impl(wn, Sw, Sigma_y));
      }

      random::NormalDist joint_dist(const mat& phi_t) const {
        vec res_mean = phi_t*this->mu_w;
        mat res_cov = phi_t*this->Sigma_w*phi_t.t();
        //Sigma_y is ignored because we do not want the measurement noise to be on the distribution
        return random::NormalDist{res_mean, res_cov};
      }

      double log_lh(const std::vector<arma::mat>& phi, const arma::mat& obs) const {
        double ans=0.0;
        vec mu(mu_w);
        mat Sw(Sigma_w), Sy(Sigma_y);
        for (unsigned int i = 0; i<phi.size(); i++) {
          mat S = phi[i]*Sw*phi[i].t() + Sy;
          random::NormalDist dist = random::NormalDist(phi[i]*mu, S);
          ans += log_normal_density(dist, obs.col(i));

          // Using Kalman Update for efficiency
          mat K = Sw*phi[i].t()*inv(S);
          mu = mu + K*(obs.col(i) - phi[i]*mu);
          Sw = Sw - K*S*K.t();
        }
        return ans;
      }
  };

  /**
   * Default constructor. Do not use this constructor unless you know what you are doing. This
   * constructor does not initialize the memory and using any method on an object created with
   * this constructor will result in undefined behaviour.
   */
  ProMP::ProMP() {
    _impl = nullptr;
  }

  //Default operators required here for Pimpl idiom to work
  ProMP::ProMP(ProMP&& b) = default;
  ProMP& ProMP::operator=(ProMP&& b) = default;
  ProMP::~ProMP() = default;

  ProMP::ProMP(const arma::vec& mu_w, const arma::mat& Sigma_w, const arma::mat& Sigma_y) {
    _impl = unique_ptr<Impl>(new Impl(mu_w, Sigma_w, Sigma_y));
  }

  ProMP::ProMP(const ProMP& b) {
    if (b._impl) {
      _impl = unique_ptr<Impl>(new Impl(*b._impl));
    }
  }

  ProMP& ProMP::operator=(const ProMP& b) {
    if (this != &b && b._impl) {
      _impl = unique_ptr<Impl>(new Impl(*b._impl));
    }
    return *this;
  }

  const arma::mat& ProMP::get_Sigma_w() const {
    return _impl->Sigma_w;
  }

  const arma::mat& ProMP::get_Sigma_y() const {
    return _impl->Sigma_y;
  }

  const arma::vec& ProMP::get_mu_w() const {
    return _impl->mu_w;
  }

  void ProMP::set_Sigma_w(const arma::mat& Sigma_w) {
    _impl->Sigma_w = Sigma_w;
  }

  void ProMP::set_Sigma_y(const arma::mat& Sigma_y) {
    _impl->Sigma_y = Sigma_y;
  }

  void ProMP::set_mu_w(const arma::vec& mu_w) {
    _impl->mu_w = mu_w;
  }

  /**
   * Returns a new ProMP that corresponds to the current ProMP conditioned on a Joint Space 
   * Gaussian distribution with mean mu_q and covariance Sigma_q. The parameter phi_t corresponds
   * to the basis functions evaluated in the time where the conditioning is desired.
   */ 
  ProMP ProMP::condition(const arma::mat& phi_t, const random::NormalDist& q_dist) const {
    ProMP ans;
    ans._impl = _impl->condition(phi_t, q_dist);
    return ans;
  }

  /**
   * Returns a new ProMP that corresponds to the current ProMP conditioned on a Joint Space 
   * Gaussian distribution with mean mu_q and covariance Sigma_q. The parameter phi_t corresponds
   * to the basis functions evaluated in the time where the conditioning is desired. For this method
   * the sensor noise is taken to zero.
   */ 
  ProMP ProMP::condition_no_Sigma_y(const arma::mat& phi_t, const random::NormalDist& q_dist) const {
    ProMP ans;
    ans._impl = _impl->condition_no_Sigma_y(phi_t, q_dist);
    return ans;
  }

  /**
   * When the variable w is marginalized in the ProMP for a particular value of phi_t the resulting
   * distribution is Gaussian. This method returns the resulting distribution.
   * @brief Returns a (Gaussian) probability distribution for the joint position at a given time
   * @param[in] position_phi_t Matrix of basis functions for the desired time
   */
  random::NormalDist ProMP::joint_dist(const arma::mat& position_phi_t) const {
    return _impl->joint_dist(position_phi_t);
  }

  double ProMP::log_lh(const std::vector<arma::mat>& phi, const arma::mat& obs) const {
    return _impl->log_lh(phi, obs);
  }

  ProMP ProMP::condition_multiple_obs(const std::vector<arma::mat>& phi, const arma::mat& obs) const {
    ProMP ans;
    ans._impl = _impl->condition_multiple_obs(phi, obs);
    return ans;
  }

};
