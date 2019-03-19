
#include <robotics.hpp>
#include <armadillo>
#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE ProMP_Test
#include <boost/test/unit_test.hpp>
#include <json.hpp>
#include <fstream>
#include <string>
#include <unordered_map>

using namespace robotics;
using namespace arma;
using namespace std;
using json = nlohmann::json;

#define EPS 1e-8

BOOST_AUTO_TEST_CASE( contructor_and_getters ) {
  vec mu_w = {1,2};
  mat Sig_w(2,2,fill::eye);
  mat Sig_y(1,1,fill::zeros); 
  Sig_y << 0.5 << endr;
  ProMP promp(mu_w, Sig_w, Sig_y);
  BOOST_CHECK( norm(promp.get_mu_w() - mu_w) < 1e-6 );
  BOOST_CHECK( norm(promp.get_Sigma_w() - Sig_w) < 1e-6 );
  BOOST_CHECK( norm(promp.get_Sigma_y() - Sig_y) < 1e-6 );
}

BOOST_AUTO_TEST_CASE( copy_and_setters ) {
  vec mu_w = {1,2};
  mat Sig_w(2,2,fill::eye);
  mat Sig_y(1,1,fill::zeros); 
  Sig_y << 0.5 << endr;
  ProMP promp(mu_w, Sig_w, Sig_y);
  ProMP promp2 = promp;
  vec mu_w2 = {-1,-1};
  promp2.set_mu_w(mu_w2);
  BOOST_CHECK( norm(promp.get_mu_w() - mu_w) < EPS );
  BOOST_CHECK( norm(promp2.get_mu_w() - mu_w2) < EPS );
  BOOST_CHECK( norm(promp2.get_Sigma_w() - Sig_w) < EPS );
  BOOST_CHECK( norm(promp2.get_Sigma_y() - Sig_y) < EPS );
}

BOOST_AUTO_TEST_CASE( condition1 ) {
  ifstream in("tests/test_simple_promp_in.txt");
  ifstream out("tests/test_simple_promp_out.txt");
  unordered_map<int, ProMP> outputs;
  json j_in;
  json j_out;
  in >> j_in;
  out >> j_out;
  for (auto &elem : j_in) {
    mat Sigma_y = json2mat(elem["Sigma_y"]);
    mat Sigma_w = json2mat(elem["Sigma_w"]);
    vec mu_w = json2vec(elem["mu_w"]);
    ProMP simple(mu_w, Sigma_w, Sigma_y);
    mat phi_t = json2mat(elem["phi_t"]);
    vec mu_q = json2vec(elem["mu_q"]);
    mat Sigma_q = json2mat(elem["Sigma_q"]);
    ProMP cond = simple.condition(phi_t, random::NormalDist{mu_q, Sigma_q});
    outputs[elem["case_num"]] = cond;
  }
  for (auto &elem : j_out) {
    mat Sigma_y = json2mat(elem["Sigma_y"]);
    mat Sigma_w = json2mat(elem["Sigma_w"]);
    vec mu_w = json2vec(elem["mu_w"]);
    int case_num = elem["case_num"];
    BOOST_CHECK( norm(mu_w - outputs[case_num].get_mu_w()) < EPS );
    BOOST_CHECK( norm( Sigma_w - outputs[case_num].get_Sigma_w()) < EPS );
    BOOST_CHECK( norm( Sigma_y - outputs[case_num].get_Sigma_y()) < EPS );
  }
}

BOOST_AUTO_TEST_CASE( tojson ) {
  vec v = linspace<vec>(0, 8, 9);
  mat m = reshape(v, 3, 3);
  json json_v = vec2json(v);
  json json_m = mat2json(m);
  vec read_v = json2vec(json_v);
  mat read_m = json2mat(json_m);
  BOOST_CHECK( norm(read_v - v) < EPS);
  BOOST_CHECK( norm(vectorise(read_m - m)) < EPS);
}
