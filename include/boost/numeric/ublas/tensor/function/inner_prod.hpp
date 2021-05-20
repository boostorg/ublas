//
//  Copyright (c) 2018, Cem Bassoy, cem.bassoy@gmail.com
//  Copyright (c) 2020, Amit Singh, amitsingh19975@gmail.com
//
//  Distributed under the Boost Software License, Version 1.0. (See
//  accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)
//
//

#ifndef BOOST_NUMERIC_UBLAS_TENSOR_INNER_HPP
#define BOOST_NUMERIC_UBLAS_TENSOR_INNER_HPP

#include <stdexcept>
#include <type_traits>

#include "../extents.hpp"
#include "../multiplication.hpp"


namespace boost::numeric::ublas
{
template<typename tensor_engine>
class tensor_core;
} // namespace boost::numeric::ublas

namespace boost::numeric::ublas
{

/** @brief Computes the inner product of two tensors     *
     * Implements c = sum(A[i1,i2,...,ip] * B[i1,i2,...,jp])
     *
     * @note calls inner function
     *
     * @param[in] a tensor object A
     * @param[in] b tensor object B
     *
     * @returns a value type.
    */
template <typename TE1, typename TE2>
inline decltype(auto) inner_prod(tensor_core< TE1 > const &a, tensor_core< TE2 > const &b)
{
  using value_type = typename tensor_core< TE1 >::value_type;

  static_assert(
    std::is_same_v<value_type, typename tensor_core< TE2 >::value_type>,
    "error in boost::numeric::ublas::inner_prod(tensor_core< TE1 > const&, tensor_core< TensorEngine2 > const&): "
    "Both the tensor should have the same value_type"
    );

  if (a.rank() != b.rank())
    throw std::length_error("error in boost::numeric::ublas::inner_prod: Rank of both the tensors must be the same.");

  if (a.empty() || b.empty())
            throw std::length_error("error in boost::numeric::ublas::inner_prod: Tensors should not be empty.");

  //if (a.extents() != b.extents())
  if (::operator!=(a.extents(),b.extents()))
    throw std::length_error("error in boost::numeric::ublas::inner_prod: Tensor extents should be the same.");

  return inner(a.rank(), a.extents().data(),
               a.data(), a.strides().data(),
               b.data(), b.strides().data(), value_type{0});
}

} // namespace boost::numeric::ublas

#endif   // BOOST_NUMERIC_UBLAS_TENSOR_PROD_DYNAMIC_HPP
