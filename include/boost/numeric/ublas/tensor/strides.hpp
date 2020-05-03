//
//  Copyright (c) 2018-2020, Cem Bassoy, cem.bassoy@gmail.com
//  Copyright (c) 2019-2020, Amit Singh, amitsingh19975@gmail.com
//
//  Distributed under the Boost Software License, Version 1.0. (See
//  accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)
//
//  The authors gratefully acknowledge the support of
//  Google
//
/// \file strides.hpp Definition for the basic_strides template class

#ifndef _BOOST_UBLAS_TENSOR_STRIDES_HPP_
#define _BOOST_UBLAS_TENSOR_STRIDES_HPP_

#include <boost/numeric/ublas/tensor/dynamic_strides.hpp>
#include <boost/numeric/ublas/tensor/fixed_rank_strides.hpp>
#include <boost/numeric/ublas/tensor/static_strides.hpp>

namespace boost::numeric::ublas{

  template <class LStrides, class RStrides, 
    std::enable_if_t<
      is_strides_v<LStrides> && is_strides_v<RStrides>
    , int> = 0 
  >
  [[nodiscard]] inline
  constexpr bool operator==(LStrides const& lhs, RStrides const& rhs) noexcept{
    static_assert( std::is_same_v<typename LStrides::value_type, typename RStrides::value_type>, 
      "boost::numeric::ublas::operator==(LStrides,RStrides) : LHS value type should be same as RHS value type");

    return lhs.size() == rhs.size() && std::equal(lhs.begin(), lhs.end(), rhs.begin());
  }

  template <class LStrides, class RStrides, 
    std::enable_if_t<
      is_strides_v<LStrides> && is_strides_v<RStrides>
    , int> = 0 
  >
  [[nodiscard]] inline
  constexpr bool operator!=(LStrides const& lhs, RStrides const& rhs) noexcept{
    static_assert( std::is_same_v<typename LStrides::value_type, typename RStrides::value_type>, 
      "boost::numeric::ublas::operator!=(LStrides,RStrides) : LHS value type should be same as RHS value type");
    return !( lhs == rhs );
  }
  
} // namespace boost::numeric::ublas


namespace boost::numeric::ublas::detail {

  /** @brief Returns relative memory index with respect to a multi-index
   *
   * @code auto j = access(std::vector{3,4,5}, strides{shape{4,2,3},first_order}); @endcode
   *
   * @param[in] i multi-index of length p
   * @param[in] w stride vector of length p
   * @returns relative memory location depending on \c i and \c w
  */
  template<class Stride, class size_type = typename Stride::size_type >
  [[nodiscard]] inline
  constexpr auto access(std::vector<size_type> const& i, Stride const& w)
  {
    static_assert( is_strides_v<Stride>, 
      "boost::numeric::ublas::detail::access() : invalid type, type should be a strides");

    const auto p = i.size();
    size_type sum = 0u;
    for(auto r = 0u; r < p; ++r)
      sum += i[r]*w[r];
    return sum;
  }

  /** @brief Returns relative memory index with respect to a multi-index
   *
   * @code auto j = access(0, strides{shape{4,2,3},first_order}, 2,3,4); @endcode
   *
   * @param[in] i   first element of the partial multi-index
   * @param[in] is  the following elements of the partial multi-index
   * @param[in] sum the current relative memory index
   * @returns relative memory location depending on \c i and \c w
  */
  template<std::size_t r, class Stride, class ... size_types>
  [[nodiscard]]
  constexpr auto access(std::size_t sum, Stride const& w, std::size_t i, size_types ... is)
  { 
    static_assert( is_strides_v<Stride>, 
      "boost::numeric::ublas::detail::access() : invalid type, type should be a strides");
    sum += i*w[r];
    if constexpr (sizeof...(is) == 0)
      return sum;
    else
      return detail::access<r+1>(sum,w,std::forward<size_types>(is)...);
  }

} // namespace boost::numeric::ublas::detail

#endif
