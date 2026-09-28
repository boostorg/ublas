//
//  Copyright (c) 2021, Cem Bassoy, cem.bassoy@gmail.com
//  Copyright (c) 2021, Amit Singh, amitsingh19975@gmail.com
//
//  Distributed under the Boost Software License, Version 1.0. (See
//  accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)
//
//

#ifndef BOOST_NUMERIC_UBLAS_TENSOR_TRANS_HPP
#define BOOST_NUMERIC_UBLAS_TENSOR_TRANS_HPP

#include <stdexcept>
#include <type_traits>
#include <vector>

#include "../extents.hpp"
#include "../traits/basic_type_traits.hpp"
#include "../multiplication.hpp"


namespace boost::numeric::ublas
{
template<typename tensor_engine>
class tensor_core;
} // namespace boost::numeric::ublas

namespace boost::numeric::ublas
{

/** @brief Transposes a tensor according to a permutation tuple
     *
     * Implements C[tau[i1],tau[i2]...,tau[ip]] = A[i1,i2,...,ip]
     *
     * @note calls trans function
     *
     * @param[in] a    tensor object of rank p
     * @param[in] tau  one-based permutation tuple of length p
     * @returns        a transposed tensor object with the same storage format F and allocator type A
    */
template <typename TensorEngine,typename PermuType = std::vector<std::size_t> >
inline decltype(auto) trans(tensor_core< TensorEngine > const &a, PermuType const &tau)
{

  using tensor_type   = tensor_core< TensorEngine >;
  using extents_type  = typename tensor_type::extents_type;

  static_assert( is_dynamic_v< extents_type > );

  auto const p = a.rank();
  auto const &na = a.extents();
  typename extents_type::base_type nc;

  if constexpr( is_dynamic_rank_v<extents_type> ){
    nc.resize(p);
  }

  for (auto i = 0u; i < p; ++i){
    nc.at(tau.at(i) - 1) = na.at(i);
  }

  auto c = tensor_type( extents_type( std::move(nc) ) );

  if (a.empty()){
    return c;
  }

  trans(a.rank(), a.extents().data(), tau.data(),
        c.data(), c.strides().data(),
        a.data(), a.strides().data());

  return c;
}

} // namespace boost::numeric::ublas

#endif   // BOOST_NUMERIC_UBLAS_TENSOR_PROD_DYNAMIC_HPP
