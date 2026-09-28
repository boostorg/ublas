//
//  Copyright (c) 2018, Cem Bassoy, cem.bassoy@gmail.com
//  Copyright (c) 2019, Amit Singh, amitsingh19975@gmail.com
//
//  Distributed under the Boost Software License, Version 1.0. (See
//  accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)
//
//  The authors gratefully acknowledge the support of Google
//


#include <boost/numeric/ublas/tensor.hpp>
#include <vector>
#include <iostream>
#include <type_traits>

void instantiate_tensor_dynamic()
{
  namespace ublas = boost::numeric::ublas;
  using value   = float;
  using layout  = ublas::layout::first_order; // storage format
  using tensor  = boost::numeric::ublas::tensor_dynamic<value,layout>;
  constexpr auto ones = ublas::ones<value,layout>{};


  try {
    // tensor is resizable has dynamic dimensions
  	// elements are stored contiguously in memory using the 1st-format (column-major)

    tensor t1  = ones(3,4,2);
    std::cout << "t1 = " << t1 << std::endl;

    tensor t2  = 2 * ones(3,4,2);
    std::cout << "t2 = " << t2 << std::endl;

    tensor t3  = 3*t2 + t1;
    std::cout << "t3 = " << t3 << std::endl;


  } catch (const std::exception& e) {
    std::cerr << "Cought exception " << e.what();
    std::cerr << "in the instantiate_tensor_dynamic function of instantiate-tensor." << std::endl;
    throw;
  }
}


void instantiate_tensor_dynamics_with_static_order()
{
  namespace ublas = boost::numeric::ublas;
  using value   = float;
  using layout  = boost::numeric::ublas::layout::first_order; // storage format
  using tensor  = boost::numeric::ublas::tensor_static_rank<value,3u, layout>;
  constexpr auto ones = ublas::ones_static_rank<value,layout>{};

  try {
    // tensor type has static order and dynamic dimensions
    // elements are stored contiguously in memory using the 1st-format (column-major)

    auto t1  = ones(3,4,2);
    std::cout << "t1 = " << t1 << std::endl;

    tensor t2  = 2*ones(3,4,2);
    std::cout << "t2 = " << t2 << std::endl;

    tensor t3  = 3*t2 + t1;
    std::cout << "t3 = " << t3 << std::endl;


  } catch (const std::exception& e) {
    std::cerr << "Cought exception " << e.what();
    std::cerr << "in the instantiate_tensor_dynamic function of instantiate-tensor." << std::endl;
    throw;
  }
}


void instantiate_tensor_static()
{

  namespace ublas = boost::numeric::ublas;
  using value   = float;
  using layout  = ublas::layout::first_order; // storage format
  using shape   = ublas::extents<3,4,2>;
  using tensor  = ublas::tensor_static<value,shape,layout>;
  constexpr auto ones = ublas::ones_static<value,layout>{};

  try {
    // tensor type has static order and static dimensions
    // elements are stored contiguously in memory using the 1st-format (column-major)

    auto t1  = tensor{};
    std::cout << "t1 = " << t1 << std::endl;

    tensor t2  = 2 * ones(shape{});
    std::cout << "t2 = " << t2 << std::endl;

    tensor t3  = 3*t2 + t1;
    std::cout << "t3 = " << t3 << std::endl;


  } catch (const std::exception& e) {
    std::cerr << "Cought exception " << e.what();
    std::cerr << "in the instantiate_tensor_dynamic function of instantiate-tensor." << std::endl;
    throw;
  }
}


int main() 
{
  try{
    instantiate_tensor_dynamic();
    instantiate_tensor_dynamics_with_static_order();
    instantiate_tensor_static();
  } catch (const std::exception& e) {
    std::cerr << "Cought exception " << e.what();
    std::cerr << "in the main function of instantiate-tensor." << std::endl;
  }
}
