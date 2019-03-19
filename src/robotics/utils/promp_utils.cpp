
#include "robotics/utils/promp_utils.hpp"
#include "robotics/full_promp.hpp"
#include "robotics/basis_functions.hpp"
#include <memory>

using namespace std;
using namespace arma;

namespace robotics {

  /**
   * Use this method to construct an unconstrained polynomial trajectory for a robot with a number
   * of degrees of freedom equal to the one given as parameter.
   * @brief Create a generic (Unconstrained) polynomial trajectory representation as a ProMP
   */
  FullProMP create_promp_poly(unsigned int ndof, unsigned int poly_deg) {
    shared_ptr<ScalarBasisFun> kernel{ new ScalarPolyBasis(poly_deg) };
    //now create a prior equivalent to regression with regularization
    double params_std = 30; //the final polynomial parameters should not be much bigger than this
    double sensor_noise = 1e-10; //assume a very small sensor noise
    unsigned int ndim = ndof*(poly_deg+1);
    vec mu_w(ndim, fill::zeros); //zero prior mean
    mat Sigma_w = eye<mat>(ndim,ndim) * (params_std*params_std);
    mat Sigma_y = eye<mat>(2*ndof, 2*ndof) * sensor_noise;
    ProMP model(mu_w, Sigma_w, Sigma_y);
    return FullProMP(kernel, model, ndof);
  }
  
  /**
   * @brief Return a ProMP that takes the robot from the current state to the desired state
   * @param[in] q Desired joint configuration
   * @param[in] qd Desired joint velocity
   * @param[in] T Desired execution time
   */
  FullProMP go_to_promp(const arma::vec& q, const arma::vec& qd, double T) {
    unsigned int dof = q.n_elem;
    unsigned int poly_order = 3;
    FullProMP unc_poly = create_promp_poly(dof, poly_order);
    return unc_poly.condition_current_state(T,T,q,qd);
  }

  /**
   * @brief Returns a constant ProMP that will mantain the same joint output
   */
  FullProMP freeze_promp(unsigned int ndof) {
    shared_ptr<ScalarBasisFun> kernel{ new ScalarPolyBasis(0) };
    //now create a prior equivalent to regression with regularization
    vec mu_w(ndof, fill::zeros); //zero prior mean
    mat Sigma_w = eye<mat>(ndof,ndof) * 25;
    mat Sigma_y = eye<mat>(ndof, ndof) * 1e-6;
    ProMP model(mu_w, Sigma_w, Sigma_y);
    return FullProMP(kernel, model, ndof);
  }
};
