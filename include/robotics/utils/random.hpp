#ifndef ROBOTICS_UTILS_RANDOM_H
#define ROBOTICS_UTILS_RANDOM_H

#include <armadillo>
#include <vector>
#include <memory>
#include <random>

namespace robotics {

  /**
   * A set of methods to produce random numbers that can be used in conjunction with the standard
   * library for random numbers. It also contains a set of definitions of probability distributions.
   * @brief Random number generation and probability distribution utilities
   */ 
  namespace random {

    /**
     * @brief Represents a Multivariate Normal (Gaussian) distribution
     */
    class NormalDist {
      private:
        class Impl;
        std::unique_ptr<Impl> _impl;
      public:
        NormalDist(const arma::vec& mean, const arma::mat& cov);
        NormalDist(const NormalDist& b);
        NormalDist(NormalDist&& b);
        ~NormalDist();

        NormalDist& operator=(const NormalDist& b);
        NormalDist& operator=(NormalDist&& b);

        const arma::vec& mean() const; //!< Mean of a normal distribution
        const arma::mat& cov() const; //!< Covariance matrix of a normal distribution
        const arma::mat& inv_cov() const; //!< Inverse of the covariance matrix
        const arma::mat& chol_cov() const; //!< Cholesky decomposition R such that R.t()*R = covariance
        double log_det_cov() const; //!< Log of the determinant of the covariance matrix

        void set_mean(const arma::vec& mean); //!< Sets the mean of the normal distribution
        void set_cov(const arma::mat& cov); //!< Sets the covariance matrix of the normal distribution
    };

    /**
     * @brief Estimates a Gaussian distribution from samples using maximum likelihood estimation.
     */
    NormalDist mle_multivariate_normal(const std::vector<arma::vec>& samples);

    double log_normal_overlap(const random::NormalDist& dist1, const random::NormalDist& dist2);
    double log_normal_density(const random::NormalDist& dist, const arma::vec& x);

    /**
     * Produces a uniform random permutation of K numbers out of N numbers.
     * @b Complexity Linear with respect to N
     */
    template<class UniformGenerator, class IntType = unsigned int>
      std::vector<IntType> sample_permutation(UniformGenerator& g, unsigned int N, unsigned int K) {
        if (K>N) K=N;
        std::vector<IntType> ans(N);
        for (unsigned int i=0; i<N; i++) {
          ans[i] = i;
        }
        for (unsigned int i=0; i<K; i++) {
          std::uniform_int_distribution<IntType> uniform(0,N-i-1);
          std::swap(ans[i], ans[i+uniform(g)]);
        }
        ans.erase(ans.begin()+K, ans.end());
        return ans;
      }

    /**
     * Generates N samples from a Bernoulli distribution. The samples are boolean values
     * with probability p of being true and (1-p) of being false
     * @brief Generates N samples from a Bernoulli distribution.
     */
    template<class UniformGenerator>
      std::vector<bool> sample_bernoulli(UniformGenerator& g, double p, unsigned int N) {
        std::vector<bool> ans;
        std::bernoulli_distribution dist(p);
        for (unsigned int i=0; i<N; i++) {
          ans.push_back(dist(g));
        }
        return ans;
      }

    /**
     * @brief Generates N samples from a Gaussian distribution with the given mean and standard deviation.
     */
    template<class UniformGenerator>
      arma::vec sample_normal(UniformGenerator &g, double mean, double std, unsigned int N) {
        arma::vec ans(N);
        std::normal_distribution<> gauss(mean, std);
        for (unsigned int i=0; i<N; i++) {
          ans[i] = gauss(g);
        }
        return ans;
      }

    /**
     * @brief Generates N samples from a given multivariate Gaussian distribution.
     */
    template<class UniformGenerator>
      std::vector<arma::vec> sample_multivariate_normal(UniformGenerator &g, const NormalDist& normal, unsigned int N) {
        std::vector<arma::vec> ans;
        std::normal_distribution<> gauss(0,1);
        unsigned int D = normal.mean().n_elem;
        for (unsigned int i=0; i<N; i++) {
          arma::vec x(D);
          for (unsigned int j=0; j<D; j++) x[j] = gauss(g);
          arma::vec y = normal.chol_cov()*x + normal.mean();
          ans.push_back(std::move(y));
        }
        return ans;
      }
  };
};

#endif
