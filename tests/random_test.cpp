
#include <robotics/utils.hpp>
#include <armadillo>
#define BOOST_TEST_DYN_LINK
#include <boost/test/unit_test.hpp>
#include <random>
#include <vector>

using namespace std;
using namespace arma;
using namespace robotics;
using json = nlohmann::json;

BOOST_AUTO_TEST_CASE( multivariate_normal_test ) {
  unsigned int num_samples = 10000; //Precise to 2 figures because error decrease with sqrt(N)
  unsigned int seed = 0;
  mt19937 gen(seed);
  arma::vec mean {-1,1,2};
  arma::mat cov;
  cov << 2 << 0.5 << -1 << endr <<
    0.5 << 3 << 0 << endr <<
    -1 << 0.0 << 4.0 << endr;
  vector<vec> samples = random::sample_multivariate_normal(gen, {mean,cov}, num_samples);
  random::NormalDist ans = random::mle_multivariate_normal(samples);
  BOOST_CHECK_MESSAGE(arma::norm(ans.mean() - mean) < cov.n_elem*1e-2, 
      "MLE estimate of the mean " << ans.mean() << " with samples differ from real mean " << mean);
  BOOST_CHECK_MESSAGE(arma::norm(ans.cov() - cov) < cov.n_elem*1e-1, 
      "MLE estimate of the covariance " << ans.cov() << " with samples differ from real covariance " << cov);
}

BOOST_AUTO_TEST_CASE( normal_overlap_test ) {
  unsigned int num_samples = 10000; //Precise to 2 figures because error decrease with sqrt(N)
  unsigned int seed = 0;
  mt19937 gen(seed);
  random::NormalDist dist1{ vec{0,0}, 4*eye(2,2) };
  random::NormalDist dist2{ vec{1,0}, eye(2,2) };

  double monte_carlo_overlap = 0.0;
  vector<vec> samples = random::sample_multivariate_normal(gen, dist2, num_samples);
  for (unsigned int i=0; i<num_samples; i++) {
    monte_carlo_overlap += exp(log_normal_density(dist1, samples[i]));
  }
  monte_carlo_overlap /= num_samples;

  double analytical_overlap = exp(log_normal_overlap(dist1, dist2));
  cout << "Dist overlap: mc=" << monte_carlo_overlap << " analytic=" << analytical_overlap << endl;
  BOOST_CHECK_MESSAGE( fabs(monte_carlo_overlap - analytical_overlap) < 1e-3,
      "Analytic and Montecarlo estimate of the overlap is different");
}
