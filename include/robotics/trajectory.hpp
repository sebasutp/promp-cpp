#ifndef ROBOTICS_TRAJECTORY_H
#define ROBOTICS_TRAJECTORY_H

#include <armadillo>
#include <memory>
#include <vector>

namespace robotics {

  /**
   * @brief Represent a trajectory in a robot trajectory in joint space
   * @since version 0.0.1
   * This struct provides representation for a trajectory step in joint space.
   */
  struct TrajectoryStep {
    arma::vec q; //!< vector of joint angles
    arma::vec qd; //!< vector of joint angular velocities
    arma::vec qdd; //!< vector of joint anglular accelerations
    double time; //!< time stamp of the trajectory
  };

};

#endif
