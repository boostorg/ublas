//
//  Copyright (c) 2021, Cem Bassoy, cem.bassoy@gmail.com
//  Copyright (c) 2021, Amit Singh, amitsingh19975@gmail.com
//
//  Distributed under the Boost Software License, Version 1.0. (See
//  accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)
//
//

#ifndef BOOST_NUMERIC_UBLAS_TENSOR_OUTER_HPP
#define BOOST_NUMERIC_UBLAS_TENSOR_OUTER_HPP

#include <stdexcept>
#include <type_traits>

#include "../detail/extents_functions.hpp"
#include "../multiplication.hpp"
#include "../traits/basic_type_traits.hpp"
//#include "../traits/basic_type_extents.hpp"
#include "../tags.hpp"


namespace boost::numeric::ublas
{
template<typename tensor_engine>
class tensor_core;
} // namespace boost::numeric::ublas

namespace boost::numeric::ublas
{

/** @brief Computes the outer product of two tensors
     *
     * Implements C[i1,...,ip,j1,...,jq] = A[i1,i2,...,ip] * B[j1,j2,...,jq]
     *
     * @note calls outer function
     *
     * @param[in] a tensor object A
     * @param[in] b tensor object B
     *
     * @returns tensor object C with the same storage format F and allocator type A1
    */
template <typename TE1,
          typename TE2,
          typename E1 = typename tensor_core< TE1 >::extents_type,
          typename E2 = typename tensor_core< TE2 >::extents_type,
          std::enable_if_t<
            (is_dynamic_rank_v<E1> || is_dynamic_rank_v<E2>) &&
            !(is_static_v<E1> && is_static_v<E2>)
              ,void>* = nullptr
          >
inline decltype(auto) outer_prod(
  tensor_core< TE1 > const &a,
  tensor_core< TE2 > const &b)
{
  using tensor_type1   = tensor_core< TE1 >;
  using value_type1    = typename tensor_type1::value_type;
  using tensor_type2   = tensor_core< TE2 >;
  using value_type2    = typename tensor_type2::value_type;

  using return_type   = std::conditional_t < is_dynamic_rank_v<E1>, tensor_type1, tensor_type2 >;
  using value_type    = typename return_type::value_type;
  using extents_type  = typename return_type::extents_type;

  static_assert( std::is_same_v<value_type1, value_type2> );

  if (a.empty() || b.empty()){
    throw std::runtime_error("Error in boost::numeric::ublas::outer_prod: tensors should not be empty.");
  }

  auto const& na = a.extents();
  auto const& nb = b.extents();
  auto nc_base = typename extents_type::base_type(ublas::size(na)+ublas::size(nb));
  auto nci = std::copy(ublas::begin(na),ublas::end(na), std::begin(nc_base));
  std::copy(ublas::begin(nb),ublas::end(nb), nci);
  auto nc = extents_type(nc_base);

  auto c = return_type( nc, value_type{} );

  outer(c.data(), c.rank(), data(c.extents()), c.strides().data(),
        a.data(), a.rank(), data(na), a.strides().data(),
        b.data(), b.rank(), data(nb), b.strides().data());

  return c;
}

} // namespace boost::numeric::ublas

#endif   // BOOST_NUMERIC_UBLAS_TENSOR_PROD_DYNAMIC_HPP
