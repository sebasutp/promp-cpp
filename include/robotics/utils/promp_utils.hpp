#ifndef ROBOTICS_UTILS_PROMP_HPP
#define ROBOTICS_UTILS_PROMP_HPP

/**
 * @file
 */

#include <armadillo>
#include "robotics/full_promp.hpp"

namespace robotics {

  FullProMP create_promp_poly(unsigned int ndof, unsigned int poly_deg);
  FullProMP go_to_promp(const arma::vec& q, const arma::vec& qd, double T);
  FullProMP freeze_promp(unsigned int ndof);
};

#endif
