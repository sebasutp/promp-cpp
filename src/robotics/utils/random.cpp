
#include <robotics/utils/random.hpp>

using namespace std;
using namespace arma;

namespace robotics {

  namespace random {

    /**
     * @brief Represents a Multivariate Normal (Gaussian) distribution
     */
    class NormalDist::Impl {
      private:
        arma::vec _mean; //<! Mean vector
        arma::mat _cov; //<! Covariance matrix
        mutable arma::mat _icov; //<! Inverse of the covariance matrix
        mutable arma::mat _chol_cov; //<! Cholesky decomposition of the covariance matrix
        mutable double _log_det_cov; //<! Log of the determinant of the covariance matrix
        mutable unsigned int lazy_state = 0; //<! Bit-mask of what is already computed

        enum class Lazy {
          icov = 1<<8,
          chol_cov = 1<<9,
          log_det_cov = 1<<12
        };
      public:
        Impl(const arma::vec& mean, const arma::mat& cov) : _mean(mean), _cov(cov) {          
        }

        ~Impl() = default;

        Impl(const Impl& x) = default;

        const arma::vec& mean() const {
          return _mean;
        }

        const arma::mat& cov() const {
          return _cov;
        }

        void set_mean(const arma::vec& mean) {
          _mean = mean;
        }

        void set_cov(const arma::mat& cov) {
          _cov = cov;
          lazy_state = 0;
        }

        const arma::mat& inv_cov() const {
          if (!(lazy_state & static_cast<int>(Lazy::icov))) {
            _icov = inv_sympd(_cov);
            lazy_state |= static_cast<int>(Lazy::icov);
          }
          return _icov;
        }

        const arma::mat& chol_cov() const {
          if (!(lazy_state & static_cast<int>(Lazy::chol_cov))) {
            _chol_cov = arma::chol(_cov);
            lazy_state |= static_cast<int>(Lazy::chol_cov);
          }
          return _chol_cov;
        }

        double log_det_cov() const {
          if (!(lazy_state & static_cast<int>(Lazy::log_det_cov))) {
            _log_det_cov = 0.0;
            const mat& U = chol_cov();
            for (unsigned int i=0; i<U.n_rows; i++) 
              _log_det_cov += log(U(i,i));
            _log_det_cov *= 2;
            lazy_state |= static_cast<int>(Lazy::log_det_cov);
          }
          return _log_det_cov;
        }
    };

    NormalDist::NormalDist(const arma::vec& mean, const arma::mat& cov) {
      _impl = unique_ptr<Impl>(new Impl(mean, cov));
    }
        
    NormalDist::NormalDist(const NormalDist& b) {
      _impl = unique_ptr<Impl>(new Impl(*b._impl));
    }

    NormalDist::NormalDist(NormalDist&& b) = default;
        
    NormalDist::~NormalDist() = default;

    NormalDist& NormalDist::operator=(const NormalDist& b) {
      if (this != &b) {
        _impl = unique_ptr<Impl>(new Impl(*b._impl));
      }
      return *this;
    }

    NormalDist& NormalDist::operator=(NormalDist&& b) = default;

    const arma::vec& NormalDist::mean() const {
      return _impl->mean();
    }

    const arma::mat& NormalDist::cov() const {
      return _impl->cov();
    } 

    void NormalDist::set_mean(const arma::vec& mean) {
      _impl->set_mean(mean);
    }

    void NormalDist::set_cov(const arma::mat& cov) {
      _impl->set_cov(cov);
    }

    const arma::mat& NormalDist::inv_cov() const {
      return _impl->inv_cov();
    }

    const arma::mat& NormalDist::chol_cov() const {
      return _impl->chol_cov();
    }

    double NormalDist::log_det_cov() const {
      return _impl->log_det_cov();
    }

    NormalDist mle_multivariate_normal(const std::vector<arma::vec>& samples) {
      if (samples.size() == 0) 
        throw std::logic_error("Invalid argument: Maximum likelihood estimate requires at least D samples");
      unsigned int D = samples[0].n_elem;
      vec sum(D, fill::zeros);
      mat cov(D, D, fill::zeros);
      for (const auto& x : samples) {
        sum += x;
        cov += x*x.t();
      }
      vec mean = (1.0/samples.size())*sum;
      cov = (1.0/samples.size())*cov - mean*mean.t();
      return {mean, cov};
    }

    /**
     * Computes \f$ \log{\int{p_1(x)p_2(x)dx}} \f$ for two Gaussian distributions
     * @brief Returns the log of the overlap of two Gaussian distributions of the same dimensionality
     */
    double log_normal_overlap(const random::NormalDist& dist1, const random::NormalDist& dist2) {
      if (dist1.mean().n_elem != dist2.mean().n_elem)
        throw std::logic_error("The two Gaussian distributions must have the same dimensionality to compute the overlap");
      
      unsigned int D = dist1.mean().n_elem; //Dimension of the random variable
      double tmp = D*log(2*acos(-1)); //D*log(2*pi)
      mat lam_hat = dist1.inv_cov() + dist2.inv_cov();
      vec nab_hat = dist1.inv_cov()*dist1.mean() + dist2.inv_cov()*dist2.mean();
      double log_det_lam_hat, sign;
      log_det(log_det_lam_hat, sign, lam_hat);
      double c_hat = -0.5*(tmp - log_det_lam_hat + arma::dot( nab_hat, lam_hat.i()*nab_hat ));
      double c_1 = -0.5*(tmp + dist1.log_det_cov() + 
          arma::dot(dist1.mean(), dist1.inv_cov()*dist1.mean()));
      double c_2 = -0.5*(tmp + dist2.log_det_cov() +
          arma::dot(dist2.mean(), dist2.inv_cov()*dist2.mean()));
      return c_1 + c_2 - c_hat;
    }

    /**
     * @brief Computes the log of the density function of a Gaussian distribution evaluated in x
     */
    double log_normal_density(const random::NormalDist& dist, const arma::vec& x) {
      if (dist.mean().n_elem != x.n_elem)
        throw std::logic_error("Dimensionality mismatch trying to compute the log_normal_density");
      
      unsigned int D = x.n_elem; //Dimension of the random variable

      static const double PI = acos(-1);
      double log_norm_const = -0.5*(D*log(2*PI) + dist.log_det_cov());
      vec diff = x - dist.mean();
      double norm_exp = -0.5*dot(diff, dist.inv_cov()*diff);

      return log_norm_const + norm_exp;
    }


  };
};
