
#include "robotics.hpp"
#include <armadillo>
#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE Utils_Test
#include <boost/test/unit_test.hpp>
#include <json.hpp>
#include <fstream>
#include <string>
#include <unordered_map>
#include <cmath>
#include <memory>

#define EPS 1e-8

using namespace robotics;
using namespace arma;
using namespace std;
using json = nlohmann::json;

BOOST_AUTO_TEST_CASE( TestUncPoly ) {
  FullProMP unconstrained = create_promp_poly(7,3);
  vec zeros(7, fill::zeros);
  vec ones(7, fill::ones);
  FullProMP constrained = unconstrained.condition_current_state(0,1,zeros,
      zeros).condition_current_state(1,1,ones,zeros);
  auto first = constrained.mean_traj_step(0,1);
  auto last = constrained.mean_traj_step(1,1);
  BOOST_CHECK( norm(first.q - zeros) < 1e-6 );
  BOOST_CHECK( norm(first.qd - zeros) < 1e-6 );
  BOOST_CHECK( norm(last.q - ones) < 1e-6 );
  BOOST_CHECK( norm(last.qd - zeros) < 1e-6 );
}

BOOST_AUTO_TEST_CASE( TestFreezeProMP ) {
  FullProMP freeze = freeze_promp(7);
  vec q{0.1,0.2,0.3,0.4,0.5,0.6,0.7};
  auto constrained = freeze.condition_current_position(0,1,q);
  for (double t=0; t<=1.0; t+=0.1) {
    auto tmp = constrained.mean_traj_step(t,1.0);
    BOOST_CHECK( norm(tmp.q - q) < 1e-6 );
    BOOST_CHECK( norm(tmp.qd) < 1e-6 );
    BOOST_CHECK( norm(tmp.qdd) < 1e-6 );
  }
}
