//
//  Copyright (c) 2018, Cem Bassoy, cem.bassoy@gmail.com
//  Copyright (c) 2020, Amit Singh, amitsingh19975@gmail.com
//
//  Distributed under the Boost Software License, Version 1.0. (See
//  accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)
//
//

#ifndef BOOST_NUMERIC_UBLAS_TENSOR_NORM_HPP
#define BOOST_NUMERIC_UBLAS_TENSOR_NORM_HPP

#include <cmath>
#include <stdexcept>
#include <type_traits>


#include "../traits/basic_type_traits.hpp"

namespace boost::numeric::ublas
{
template<typename tensor_engine>
class tensor_core;
} // namespace boost::numeric::ublas

namespace boost::numeric::ublas
{

/**
     *
     * @brief Computes the frobenius nor of a tensor
     *
     * @note Calls accumulate on the tensor.
     *
     * implements
     * k = sqrt( sum_(i1,...,ip) A(i1,...,ip)^2 )
     *
     * @tparam V the data type of tensor
     * @tparam F the format of tensor storage
     * @tparam A the array_type of tensor
     * @param a the tensor whose norm is expected of rank p.
     * @return the frobenius norm of a tensor.
     */
template <typename TE>
inline auto norm(tensor_core< TE > const &a)
{
  using value_type = typename tensor_core< TE >::value_type;

  if (a.empty()) {
    throw std::runtime_error("Error in boost::numeric::ublas::norm: tensors should not be empty.");
  }

  return std::sqrt(accumulate(a.order(), a.extents().data(), a.data(), a.strides().data(), value_type{},
                              [](auto const &l, auto const &r) { return l + r * r; }));
}

} // namespace boost::numeric::ublas

#endif   // BOOST_NUMERIC_UBLAS_TENSOR_NORM_HPP
