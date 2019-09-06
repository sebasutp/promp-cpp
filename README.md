ProMPs for C++
==============


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

## Installation

The first step to install this library is to install the pre-requisites. The easiest way to install these
pre-requisites is to use your OS package manager, but make sure that the versions on your OS package manager
are recent. Below I provide the links to the official mantainers of the required packages, in case you
decide to install these packages from the source (For instance, if your OS package versions are too old).

1. [Armadillo](http://arma.sourceforge.net/)
2. [Boost](http://www.boost.org/)
3. [CMake](https://cmake.org/)
3. [JSON](https://github.com/nlohmann/json)

These can be installed with the following command:

```
sudo apt-get install libarmadillo-dev libboost-dev libboost-test-dev cmake nlohmann-json-dev
```

After the pre-requisites are installed, download the source of this library, compile and install. You need
CMake to be able to compile and install this library. 

To install follow the standard procedure to compile packages with CMake, for instance, in Linux you would type:

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

Publications
------------

The implementations provided in this repository are based on the following publications:

1) [Adaptation and Robust Learning of Probabilistic Movement Primitives](https://arxiv.org/pdf/1808.10648.pdf)
2) [Using probabilistic movement primitives for striking movements, IEEE RAS International 
Conference on Humanoid Robots, 2016](https://ieeexplore.ieee.org/abstract/document/7803322/)

Please refer to these papers to understand our implementation, get general information about
probabilistic movement primitives and see the evaluation of the implemented methods in real
robotic platforms. We also have a [Python implementation](https://github.com/sebasutp/promp) of
these methods, including the code to train the Probabilistic Movement Primitives.
