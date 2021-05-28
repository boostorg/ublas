cmake_minimum_required(VERSION 3.12)

project("Boost.uBLAS.TensorTest" 
	DESCRIPTION "Boost.uBLAS.TensorTest is part of the Boost C++ Libraries. It is directed towards testing Boost.uBLAS.Tensor various operations within Boost.uBLAS"
        HOMEPAGE_URL "https://github.com/boostorg/ublas")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include_directories(SYSTEM ${BOOST_HEADERS_DIR})
include_directories(${UBLAS_HEADER_DIR})

add_executable(
	test_tensor 
	  test_tensor.cpp
          test_strides.cpp
          test_expression.cpp
          test_operators_comparison.cpp 
          test_operators_arithmetic.cpp
          test_multiplication.cpp
          test_multi_index_utility.cpp
          test_multi_index.cpp
          test_extents.cpp
          test_expression_evaluation.cpp
          test_einstein_notation.cpp
          test_algorithms.cpp
          test_tensor_matrix_vector.cpp
          test_functions.cpp) 

add_executable(
	test_static_tensor
	  test_static_tensor.cpp
          test_static_extents.cpp
          test_static_strides.cpp
          test_static_operators_arithmetic.cpp
          test_static_operators_comparison.cpp
          test_static_expression_evaluation.cpp
          test_static_tensor_matrix_vector.cpp
          test_static_functions.cpp)

add_executable(
	test_fixed_rank_tensor
	  test_fixed_rank_tensor.cpp
          test_fixed_rank_extents.cpp
          test_fixed_rank_strides.cpp
          test_fixed_rank_operators_arithmetic.cpp
          test_fixed_rank_operators_comparison.cpp
          test_fixed_rank_expression_evaluation.cpp
          test_fixed_rank_tensor_matrix_vector.cpp
          test_fixed_rank_functions.cpp)

find_package(Boost COMPONENTS unit_test_framework REQUIRED)

target_link_libraries(test_tensor ${Boost_UNIT_TEST_FRAMEWORK_LIBRARY})
target_link_libraries(test_static_tensor ${Boost_UNIT_TEST_FRAMEWORK_LIBRARY})
target_link_libraries(test_fixed_rank_tensor ${Boost_UNIT_TEST_FRAMEWORK_LIBRARY})

enable_testing()

add_test(NAME TensorTests     COMMAND test_tensor)
add_test(NAME TensorFixedRank COMMAND test_fixed_rank_tensor)
add_test(NAME TensorStatic    COMMAND test_static_tensor)
