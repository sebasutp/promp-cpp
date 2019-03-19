/*!
@mainpage
These pages contain the API documentation of RobCPP. A library for Robotics developed
at the Max Planck Institute for intelligent systems that runs on Modern C++.

@section Introduction
This library depends on Armadillo (http://arma.sourceforge.net/) and Boost
(http://www.boost.org/). Make sure that these two libraries are installed before
you use this library. This library also uses the JSON library
(http://nlohmann.github.io/json/), but you should not need to install it since 
its source was copied to this project under the header file include/json.hpp.

The goal of this particular library component is to provide generic utilities
for robot control and movement primitive representation. The code in this library
was designed thinking on good software practices and efficiency, but real time
issues like avoid memory allocation were not considered. If you require real time
you should create a separate thread for the real time part and to use this code.

@section Installation
Before installing this library, you need to install CMake (https://cmake.org/), 
Armadillo (http://arma.sourceforge.net/) and Boost (http://www.boost.org/). To compile
the library follow the usual CMake installation procedure, in a Linux machine it should
be something like:

\code{.sh}
mkdir build
cd build
cmake ..
make
\endcode

The previous commands will compile the library and examples in the build subdirectory. To
run the tests do

\code{.sh}
make test
\endcode

And to install the library in your machine run as administrator

\code{.sh}
make install
\endcode

@section Example Code

The following example shows a inverse dynamics control example for a simulated system that
works reasonably well. It also uses Probabilistic Movement Primitives (ProMPs) to represent
the desired trajectories to execute.

\include inv_dyn_control.cpp

@author [Sebastian Gomez-Gonzalez](http://ei.is.tuebingen.mpg.de/person/sgomez)
@see http://git.ias.informatik.tu-darmstadt.de/sgomez/robcpp to download the source code

@version 0.1.1
*/


#ifndef LIB_ROBOTICS
#define LIB_ROBOTICS

#include "robotics/promp.hpp"
#include "robotics/utils.hpp"
#include "robotics/basis_functions.hpp"
#include "robotics/full_promp.hpp"
#include "robotics/json_factories.hpp"

#endif
