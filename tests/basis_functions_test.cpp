
#include "robotics/basis_functions.hpp"
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

#define EPS 1e-8

using namespace robotics;
using namespace arma;
using namespace std;
using json = nlohmann::json;

BOOST_AUTO_TEST_CASE( ScalarPolyBasisTest ) {
  unsigned int poly_order = 3;
  ScalarPolyBasis poly(poly_order); //create 3rd order polynomial basis functions [1  x  x**2  x**3]
  BOOST_CHECK( norm(poly.eval(0) - vec{1.0,0,0,0}) < EPS );
  BOOST_CHECK( norm(poly.eval(1.0) - vec{1,1,1,1}) < EPS );
  BOOST_CHECK( norm(poly.eval(0.5) - vec{1,0.5,0.25,0.125}) < EPS );

  BOOST_CHECK( norm(poly.deriv(0.0, 1) - vec{0,1.0,0,0}) < EPS );
  BOOST_CHECK( norm(poly.deriv(1.0, 1) - vec{0,1,2,3}) < EPS );
  BOOST_CHECK( norm(poly.deriv(0.5, 1) - vec{0,1,2*0.5,3*0.25}) < EPS );

  BOOST_CHECK( norm(poly.deriv(0.0, 2) - vec{0,0,2,0}) < EPS );
  BOOST_CHECK( norm(poly.deriv(1.0, 2) - vec{0,0,2,6}) < EPS );
  BOOST_CHECK( norm(poly.deriv(0.5, 2) - vec{0,0,2,6*0.5}) < EPS );
  BOOST_CHECK( poly.dim() == (poly_order+1) );
}

BOOST_AUTO_TEST_CASE( ScalarGaussBasisTest ) {
  ifstream in("tests/gaussian_kernel_test.txt");
  json json;
  in >> json;
  for (auto &elem : json) {
    vec phi_t = json2vec(elem["phi_t"]);
    vec der1 = json2vec(elem["der1"]);
    vec der2 = json2vec(elem["der2"]);
    double sigma = elem["sigma"];
    double t = elem["t"];
    vec centers = json2vec(elem["centers"]);
    ScalarGaussBasis gauss_basis(centers, sigma);
    BOOST_CHECK( norm(gauss_basis.eval(t) - phi_t) < EPS );
    BOOST_CHECK( norm(gauss_basis.deriv(t,1) - der1) < EPS );
    BOOST_CHECK( norm(gauss_basis.deriv(t,2) - der2) < EPS );
  }
}

BOOST_AUTO_TEST_CASE( ScalarCombBasisTest ) {
  auto rbf = shared_ptr<ScalarGaussBasis>( new ScalarGaussBasis({0.25,0.5,0.75},0.25) );
  auto poly = make_shared<ScalarPolyBasis>(1);
  auto comb = shared_ptr<ScalarCombBasis>( new ScalarCombBasis({rbf, poly}) );

  for (double z=0; z<=1; z+=0.1) {
    vec v1 = comb->eval(z);
    vec v2 = comb->eval(z + 1e-5);
    vec num_diff = (v2 - v1) / (1e-5);
    vec an_diff = comb->deriv(z, 1);

    for (unsigned int d=0; d<5; d++) {
      cout << "d: " << d << " -> " << an_diff[d] << " ?= " << num_diff[d] << endl;
      if (fabs(an_diff[d]) > 1e-8)
        BOOST_CHECK_CLOSE(an_diff[d], num_diff[d], 1);
      else
        BOOST_CHECK(fabs(num_diff[d]) < 1e-3);
    }
    BOOST_CHECK(fabs(an_diff[3]) < 1e-6);
    BOOST_CHECK_CLOSE(1.0, an_diff[4], 0.01);
  }
}
