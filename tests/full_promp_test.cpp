
#include "robotics/full_promp.hpp"
#include <armadillo>
#define BOOST_TEST_DYN_LINK
#include <boost/test/unit_test.hpp>
#include <json.hpp>
#include <fstream>
#include <string>
#include <unordered_map>
#include <cmath>
#include <memory>
#include "robotics/utils.hpp"
#include "robotics/json_factories.hpp"

#define EPS 1e-8

using namespace robotics;
using namespace arma;
using namespace std;
using json = nlohmann::json;

vec from_q {0.0}, from_qdot{0.0}, to_q{1.0}, to_qdot{0.0};
double T=0.5;
double qdev = 0.0001, qddev = 0.0005;
unsigned int num_joints = 1;

FullProMP loose_poly_promp_1() {
  unsigned int poly_order = 3;
  mat Sigma_y = diagmat(vec {qdev*qdev,qddev*qddev});
  ProMP model(zeros<vec>(poly_order+1), 100*eye<mat>(poly_order+1,poly_order+1), Sigma_y);
  shared_ptr<ScalarBasisFun> kernel{ new ScalarPolyBasis(3) };
  FullProMP poly(kernel, model, num_joints);
  return poly;
}

FullProMP loose_poly_promp_2() {
  ifstream in("tests/loose_poly_promp_1d.txt");
  json obj;
  in >> obj;
  return json2full_promp(obj);
}

FullProMP cond_poly_promp(const FullProMP& poly) {
  auto npromp = poly.condition_current_state(0, T, from_q, from_qdot); 
  npromp = npromp.condition_current_state(T, T, to_q, to_qdot);
  return npromp;
}

BOOST_AUTO_TEST_CASE( FullPolyProMP1 ) {
  double nstd = 5.0;
  auto poly = loose_poly_promp_1();
  auto npromp = cond_poly_promp(poly);
  auto first = npromp.mean_traj_step(0,T);
  auto last = npromp.mean_traj_step(T,T);

  for (unsigned int i=0; i<num_joints; i++) {
    BOOST_CHECK( fabs(first.q[i]-from_q[i]) < nstd*qdev );
    BOOST_CHECK( fabs(first.qd[i] - from_qdot[i]) < nstd*qddev );
    BOOST_CHECK( fabs(last.q[i] - to_q[i]) < nstd*qdev );
    BOOST_CHECK( fabs(last.qd[i] - to_qdot[i]) < nstd*qddev );
  }
}

BOOST_AUTO_TEST_CASE( FullPolyProMP2 ) {
  double nstd = 5.0;
  auto poly = loose_poly_promp_2();
  auto npromp = cond_poly_promp(poly);
  auto first = npromp.mean_traj_step(0,T);
  auto last = npromp.mean_traj_step(T,T);

  for (unsigned int i=0; i<num_joints; i++) {
    BOOST_CHECK( fabs(first.q[i]-from_q[i]) < nstd*qdev );
    BOOST_CHECK( fabs(first.qd[i] - from_qdot[i]) < nstd*qddev );
    BOOST_CHECK( fabs(last.q[i] - to_q[i]) < nstd*qdev );
    BOOST_CHECK( fabs(last.qd[i] - to_qdot[i]) < nstd*qddev );
  }
}

BOOST_AUTO_TEST_CASE( FullPolyProMP3 ) {
  double nstd = 5.0;
  auto poly = go_to_promp(to_q, to_qdot, T);
  auto npromp = poly.condition_current_state(0,T,from_q,from_qdot); 
  auto first = npromp.mean_traj_step(0,T);
  auto last = npromp.mean_traj_step(T,T);

  for (unsigned int i=0; i<num_joints; i++) {
    BOOST_CHECK( fabs(first.q[i]-from_q[i]) < nstd*qdev );
    BOOST_CHECK( fabs(first.qd[i] - from_qdot[i]) < nstd*qddev );
    BOOST_CHECK( fabs(last.q[i] - to_q[i]) < nstd*qdev );
    BOOST_CHECK( fabs(last.qd[i] - to_qdot[i]) < nstd*qddev );
  }
}

BOOST_AUTO_TEST_CASE( GoToProMP_NDoF ) {
  unsigned int ndof = 7;
  vec ndof_to_q(ndof,fill::ones), ndof_zeros(ndof,fill::zeros);
  double nstd = 5.0;
  auto poly = go_to_promp(ndof_to_q, ndof_zeros, 1.0);
  auto npromp = poly.condition_current_state(0,1.0,ndof_zeros,ndof_zeros);
  vec mu_w = npromp.get_model().get_mu_w();
  for (unsigned int i=0; i<ndof; i++) {
    BOOST_CHECK( fabs(mu_w[4*i + 0]) < 1e-3 );
    BOOST_CHECK( fabs(mu_w[4*i + 1]) < 1e-3 );
    BOOST_CHECK( fabs(mu_w[4*i + 2] - 3) < 1e-3 );
    BOOST_CHECK( fabs(mu_w[4*i + 3] + 2) < 1e-3 );
  }

  auto first = npromp.mean_traj_step(0.0,1.0);
  auto last = npromp.mean_traj_step(1.0,1.0);
  for (unsigned int i=0; i<ndof; i++) {
    BOOST_CHECK( fabs(first.q[i]) < nstd*qdev );
    BOOST_CHECK( fabs(first.qd[i]) < nstd*qddev );
    BOOST_CHECK( fabs(last.q[i] - ndof_to_q[i]) < nstd*qdev );
    BOOST_CHECK( fabs(last.qd[i]) < nstd*qddev );
  }
}

BOOST_AUTO_TEST_CASE( LoadProMPJSON_test ) {
  cout << "Loading ProMP from file..." << endl;
  auto promp = load_full_promp("tests/promp.json");
  cout << "Loading marginal distribution from file..." << endl;
  ifstream in("tests/promp_marg.json");
  json marg;
  in >> marg;
  cout << "Checking produced marginal is equal to the one loaded from file..." << endl;
  cout << marg["time"].size() << endl;
  for (unsigned int i=0; i<marg["time"].size(); i++) {
    double z = marg["time"][i];
    vec mean = json2vec(marg["means"][i]);
    mat cov = json2mat(marg["covs"][i]);

    auto x = promp.joint_dist(z, true, true, false);
    BOOST_REQUIRE_EQUAL(mean.n_elem, x.mean().n_elem);
    BOOST_REQUIRE_EQUAL(cov.n_rows, x.cov().n_rows);
    BOOST_REQUIRE_EQUAL(cov.n_cols, x.cov().n_cols);
    unsigned int D = mean.n_elem;
    for (unsigned int r=0; r<D; r++) {
      BOOST_CHECK_CLOSE(mean[r], x.mean()[r], 0.1);
      for (unsigned int c=0; c<D; c++) {
        BOOST_CHECK_CLOSE(cov(r,c), x.cov()(r,c), 0.1);
      }
    }
  }
  cout << "All tests of the marginal were run, quitting test case" << endl;
}
