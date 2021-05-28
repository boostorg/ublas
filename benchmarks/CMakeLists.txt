cmake_minimum_required(VERSION 3.12)

project("Boost.uBLAS.Benchmark" 
	DESCRIPTION "Boost.uBLAS.Benchmark is part of the Boost C++ Libraries. It is directed towards benchmarking various operations within Boost.uBLAS"
        HOMEPAGE_URL "https://github.com/boostorg/ublas")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include_directories(SYSTEM ${BOOST_HEADERS_DIR})
include_directories(${UBLAS_HEADER_DIR})

# If you decide to use any other Boost library that requires linkage
# you should add them to the list of components in the below line.
# To link call `link_libraries(<name>)`. This library will be linked to all targets.
# To link to specific target, use `target_link_libraries(<name_of_target> <lib>)` after
# you have created the target with `add_executable(<name_of_target> <sources>)`

find_package(Boost COMPONENTS program_options REQUIRED)
link_libraries(${Boost_PROGRAM_OPTIONS_LIBRARY})

# Add new benchmarks here.

add_executable(add        add.cpp)
add_executable(mm_prod    mm_prod.cpp)
add_executable(mv_prod    mv_prod.cpp)
add_executable(inner_prod inner_prod.cpp)
add_executable(outer_prod outer_prod.cpp)

# Add a reference to benchamark here.
add_executable(add_ref        reference/add.cpp)
add_executable(mm_prod_ref    reference/mm_prod.cpp)
add_executable(mv_prod_ref    reference/mv_prod.cpp)
add_executable(inner_prod_ref reference/inner_prod.cpp)
add_executable(outer_prod_ref reference/outer_prod.cpp)

