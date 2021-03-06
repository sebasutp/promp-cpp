#include <iostream>
#include <armadillo>
#include <memory>
#include <robotics.hpp>

using namespace std;
using namespace arma;
using namespace robotics;

int main() {
  // Creating a basic ProMP with the desired parameters
  vec mu_w {0, 0, 0, 0};
  mat Sigma_w = 100*eye<mat>(4,4);
  mat Sigma_y = 0.0001*eye<mat>(2,2);
  ProMP promp(mu_w, Sigma_w, Sigma_y);

  //Setting a third order polynomial basis function for the ProMP
  shared_ptr<ScalarBasisFun> kernel{ new ScalarPolyBasis(3) };
  //And create a new ProMP for a robot with one degree of freedom
  FullProMP poly(kernel, promp, 1);

  //If we want the ProMP to take us from 0 to 1 in 0.5 seconds, we use conditioning
  double T=0.5;
  vec init_pos{0}, init_vel{0}, final_pos{1}, final_vel{0};
  poly = poly.condition_current_state(0, T, init_pos, init_vel);
  poly = poly.condition_current_state(T, T, final_pos, final_vel);

  //Now print the mean trajectory of the conditioned ProMP with time intervals of 0.1 seconds
  for (double t=0.0; t<=T; t+=0.1) {
    TrajectoryStep step = poly.mean_traj_step(t, T);
    cout << "Time: " << t <<  " Position: " << step.q[0] << " Velocity: " << step.qd[0] << endl;
  }

}
