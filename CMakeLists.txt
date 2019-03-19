cmake_minimum_required(VERSION 3.5)
project(robotics)
set(CMAKE_CXX_STANDARD 11)
find_package(Armadillo REQUIRED)

include_directories(include)
include_directories(${Boost_INCLUDE_DIR})

add_library(robotics SHARED
  src/robotics/basis_functions.cpp
  src/robotics/full_promp.cpp
  src/robotics/json_factories.cpp
  src/robotics/promp.cpp
  src/robotics/utils.cpp
  src/robotics/utils/promp_utils.cpp
  src/robotics/utils/chrono.cpp
  src/robotics/utils/random.cpp
  src/robotics/utils/math.cpp
  )
target_link_libraries(robotics
  ${ARMADILLO_LIBRARIES}
  )

add_subdirectory(tests)
enable_testing ()
add_test (NAME ProMP_Test COMMAND promp_test WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
add_test (NAME Utils_Test COMMAND utils_test WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})

add_subdirectory(examples)

install(TARGETS robotics DESTINATION lib)
install(FILES include/robotics.hpp DESTINATION include)
install(DIRECTORY include/robotics DESTINATION include)
