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

void instantiate_dynamic_tensor()
{

    using value_t   = float;
    using format_t  = boost::numeric::ublas::layout::first_order; // storage format
    using tensor_t  = boost::numeric::ublas::dynamic_tensor<value_t,format_t>;
    using shape_t   = typename tensor_t::extents_type;
    
  	// tensor type has dynamic order and dimensions
  	// elements are stored contiguously in memory using the 1st-format (column-major)

    auto t1  = tensor_t{3,4,2};
    std::cout << "t1 = " << t1 << std::endl;

    auto t2  = tensor_t(shape_t{3,4,2},2.0F);
    std::cout << "t2 = " << t2 << std::endl;

    auto t3  = tensor_t(t2);
    std::cout << "t3 = " << t3 << std::endl;

}


void instantiate_dynamic_tensors_with_static_order()
{

  constexpr auto order = 3U;
  using value_t   = float;
  using format_t  = boost::numeric::ublas::layout::first_order; // storage format
  using tensor_t  = boost::numeric::ublas::fixed_rank_tensor<value_t, order, format_t>;
  using shape_t   = typename tensor_t::extents_type;

  // tensor type has static order and dynamic dimensions
  // elements are stored contiguously in memory using the 1st-format (column-major)

  auto t1  = tensor_t{3,4,2};
  std::cout << "t1 = " << t1 << std::endl;

  auto t2  = tensor_t(shape_t{3,4,2},2.0F);
  std::cout << "t2 = " << t2 << std::endl;

  auto t3  = tensor_t(t2);
  std::cout << "t3 = " << t3 << std::endl;

}


void instantiate_static_tensor()
{

  using value_t   = float;
  using format_t  = boost::numeric::ublas::layout::first_order; // storage format
  using shape_t   = boost::numeric::ublas::static_extents<3U,4U,2U>;
  using tensor_t  = boost::numeric::ublas::static_tensor<value_t, shape_t, format_t>;

  // tensor type has static order and static dimensions
  // elements are stored contiguously in memory using the 1st-format (column-major)

  auto t1  = tensor_t{};
  std::cout << "t1 = " << t1 << std::endl;

  auto t2  = tensor_t(2.0F);
  std::cout << "t2 = " << t2 << std::endl;

  auto t3  = tensor_t(t2);
  std::cout << "t3 = " << t3 << std::endl;

}


int main() 
{
  instantiate_dynamic_tensor();
  instantiate_dynamic_tensors_with_static_order();
  instantiate_static_tensor();
}
