Robotics C++ Librares
=====================

This repository contains a collection of C++ libraries (Some with C API support) for robotics and
applications. The folder `lib_robotics` contains implementations of generic utilities for robotics
applications like movement primitives, that can be applied to any robotic task. The folder
`table_tennis` contains implementation of utilities related with robot table tennis. All these
libraries use the same name space in C++, named `robotics`.

## Code Example

The first example shows how to create a Probabilistic Movement Primitive (ProMP) for a one degree of freedom
robot. The prior parameters are set to zero mean and a diagonal covariance matrix (with standard deviation
of 10, variance 100). The noise covariance is set to a diagonal matrix as well with standard deviation 0.001.
For the basis functions a third order polynomial is used. To see the ProMP working, the example uses 
conditioning to make the polynomial go from zero to one in half a second (With zero initial and final
velocities). At the end the mean trajectory is printed.

```
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
```

The output of this example is something like:

```
Time: 0 Position: 0.000013 Velocity: 0.000004
Time: 0.1 Position: 0.104011 Velocity: 1.91995
Time: 0.2 Position: 0.352004 Velocity: 2.87992
Time: 0.3 Position: 0.647996 Velocity: 2.87992
Time: 0.4 Position: 0.89599 Velocity: 1.91995
Time: 0.5 Position: 0.999987 Velocity: 0.000002
```

The library has also a C API for some of the functionality. The following example show how to predict
the trajectory of the ball using the C API from the table tennis library. In this example the ball
trajectory is modeled using a Kalman Filter, whose parameters are loaded from a text file.

```
#include <stdio.h>
#include <robotics/table_tennis/table_tennis.h>

int main() {
  char opt[10];
  /* Allocate the resources and load configurations */
  void *table = rob_tt_load_table("models/table.txt");
  void *ball_model = rob_tt_load_ball_model(table, "models/trained_kf.txt");
  double deltaT = 0.016; /* The model in the file trained_kf.txt was trained with this deltaT */
  double obs[3], position[3];
  void *ball_state;
  double time = 0.0;

  printf("Now the models are loaded. Suppose we see a new ball trajectory (Initial time = %.2lf s)\n", time);
  ball_state = rob_tt_new_ball_traj(ball_model);
  while(1) {
    printf("Insert the 3D location of the ball observation at t = %.3lf s \
(Each coordinate separated by space)\n", time);
    scanf("%lf %lf %lf", &obs[0], &obs[1], &obs[2]);

    printf("The Mahalanobis distance of the given observation to the current estimate is %.4lf\n",
        rob_tt_ball_mah_dist(ball_state, ball_model, obs));
    printf("Do you accept this ball as correct (No means outlier)? (Y/N) ");
    scanf("%s", opt);
    /* Take anything different from No as yes. In case Yes add ball observation to the state */
    if (opt[0] != 'N' && opt[0]!='n') 
      rob_tt_ball_observe(ball_state, ball_model, obs);
    /* Now compute the ball estimated (filtered) position */
    rob_tt_ball_position(position, ball_state, ball_model);
    printf("The current ball filtered position (Position estimate) is (%.3lf, %.3lf, %.3lf)\n", position[0], 
        position[1], position[2]);

    printf("Do you want to exit the demo? (Y/N) ");
    scanf("%s", opt);
    /* Take anything different from yes as no */
    if (opt[0] == 'Y' || opt[0]=='y') 
      break;

    /* Advancing time by deltaT */
    rob_tt_ball_advance_traj(ball_state, ball_model, deltaT);
    time += deltaT;
  }
  /* Now, make sure you release all the resources you allocated */
  rob_tt_free_ball_model(ball_model);
  rob_tt_free_table(table);
}
```

## Motivation

This project was created with the aim of making the coding efforts for robotic projects reusable.

## Installation

The first step to install this library is to install the pre-requisits. The easiest way to install these
pre-requisits is to use your OS package manager, but make sure that the versions on your OS package manager
are recent. Below I provide the links to the official mantainers of the required packages, in case you
decide to install these packages from the source (For instance, if your OS package versions are too old).

1. [Armadillo](http://arma.sourceforge.net/)
2. [Boost](http://www.boost.org/)
3. [CMake](https://cmake.org/)
4. [NLOpt](http://ab-initio.mit.edu/wiki/index.php/NLopt)
5. [ZMQ](http://zeromq.org/)
6. [Protobuff] (https://developers.google.com/protocol-buffers/)

And to install a Python extension (Recommended), you will also need:

10. [Python C API](https://www.python.org)
11. [SWIG](http://www.swig.org/)
12. [Armanpy](https://sourceforge.net/projects/armanpy) 

In Ubuntu 16.04, all this packages (except the 6 and 12) can be installed with a single command:

```
sudo apt-get install libarmadillo-dev libboost-dev libboost-test-dev cmake libnlopt-dev libzmq3-dev \
  python-dev swig libboost-log-dev libboost-program-options-dev
```

After the pre-requisites are installed, download the source of this library, compile and install. You need
CMake to be able to compile and install this library. 

**Important**: First install ``lib_robotics`` and then install ``table_tennis``. To install follow the
standard procedure to compile packages with CMake, for instance, in Linux you would type:

```
mkdir build
cd build
cmake ..
make
make test #optional (Tests that the compiled library passes the test cases)
make install
```

To do `make install` you probably need root permisions. You can also install the library in your home
folder if you do not have root permisions, by changing the CMake installation prefix.

If you do not want to build the python extensions, run CMake with the option PYLIB in off:

```
cmake .. -DPYLIB=Off
```

## API Reference

To compile de API documentation you require Doxygen. Simply run the ``doxygen`` command on each subproject
folder and find the documentation in the doc folder.

## Contributors

Feel free to report bugs or to add new functionality to the libraries. If you want to contribute to the
code please make sure you use the ``Pimpl`` [design pattern](https://en.wikibooks.org/wiki/C%2B%2B_Programming/Idioms),
and always write test code for your API. So far I coded all my tests using the [Boost test library](
http://www.boost.org/doc/libs/1_60_0/libs/test/doc/html/index.html), and then add the test code file to the
CMake list file for tests.

I used also DOxygen to document my API. Not all the code has been documented but I am always trying to improve
that. 

## License

I don't know yet what licence can be used for this code. Once I know I will update this file.
