find_package(Boost COMPONENTS unit_test_framework REQUIRED)
add_executable(promp_test
  promp_test.cpp
  basis_functions_test.cpp
  full_promp_test.cpp
  )
target_link_libraries(promp_test
  ${Boost_UNIT_TEST_FRAMEWORK_LIBRARY}
  robotics
  )

add_executable(utils_test
  promp_utils_test.cpp
  random_test.cpp
  utils_math_test.cpp
  )
target_link_libraries(utils_test
  ${Boost_UNIT_TEST_FRAMEWORK_LIBRARY}
  robotics
  )
