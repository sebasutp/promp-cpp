
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

BOOST_AUTO_TEST_CASE( log_sum_exp_test ) {
  vector<double> t1{-10000, -9000, -8000, -9500, -8900};
  vector<double> t2{-10000, 9000, 8000, 9500, 8900, -1000};
  vector<double> t3{-10000, -5000, 10000, 10001, 10002, 10005, 10003, 10004};

  BOOST_CHECK(fabs(log_sum_exp(t1) - (-8000)) < 1e-6);
  BOOST_CHECK(fabs(log_sum_exp(t2) - 9500) < 1e-6);
  BOOST_CHECK(log_sum_exp(t3) > (10005+1e-6) && log_sum_exp(t3)<(10006-1e-6));
}
