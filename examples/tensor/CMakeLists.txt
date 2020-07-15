cmake_minimum_required(VERSION 3.12)

project("Boost.uBLAS.Examples" 
	DESCRIPTION "Boost.uBLAS.Examples is part of the Boost C++ Libraries. It is directed towards showing different use cases and examples to get started with various operations within Boost.uBLAS"
        HOMEPAGE_URL "https://github.com/boostorg/ublas")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include_directories(SYSTEM ${BOOST_HEADERS_DIR})
include_directories(${UBLAS_HEADER_DIR})

# Add a new example here. Use spaces and respect my beautiful alignment.

add_executable(construction_access          construction_access.cpp)
add_executable(simple_expressions           simple_expressions.cpp)
add_executable(dynamic_prod_expressions     dynamic_prod_expressions.cpp)
add_executable(fixed_rank_prod_expressions  fixed_rank_prod_expressions.cpp)
add_executable(static_prod_expressions      static_prod_expressions.cpp)
add_executable(fixed_rank_tensor            fixed_rank_tensor.cpp)
add_executable(static_tensor                static_tensor.cpp)
add_executable(einstein_notation            einstein_notation.cpp)
add_executable(tensor_construction          tensor_construction.cpp)

