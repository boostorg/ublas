//
//  Copyright (c) 2018-2020, Cem Bassoy, cem.bassoy@gmail.com
//  Copyright (c) 2019-2020, Amit Singh, amitsingh19975@gmail.com
//
//  Distributed under the Boost Software License, Version 1.0. (See
//  accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)
//
//  The authors gratefully acknowledge the support of
//  Google and Fraunhofer IOSB, Ettlingen, Germany
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
      "boost::numeric::ublas::operator==(LStrides,RStrides) : LHS value type should be the same as the RHS value type");

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
      "boost::numeric::ublas::operator!=(LStrides,RStrides) : LHS value type should be the same as the RHS value type");
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
  template<class Stride>
  [[nodiscard]] inline
  constexpr auto access(std::vector<typename Stride::value_type> const& i, Stride const& w)
  {
    static_assert( is_strides_v<Stride>, 
      "boost::numeric::ublas::detail::access() : invalid type, the type should be a strides");
    
    using value_type = typename Stride::value_type;
    return std::inner_product(i.begin(), i.end(), w.begin(), value_type{});
  }

  /** @brief Returns relative memory index with respect to a multi-index
   *
   * @code auto j = access(strides{shape{4,2,3},first_order}, 2,3,4); @endcode
   *
   * @param[in] is  the elements of the partial multi-index
   * @param[in] sum the current relative memory index
   * @returns relative memory location depending on \c i and \c w
  */
  template<class Stride, class ... Indices>
  [[nodiscard]] inline
  constexpr auto access(Stride const& w, Indices ... is)
  { 
    static_assert( is_strides_v<Stride>, 
      "boost::numeric::ublas::detail::access() : invalid type, the type should be a strides");
    
    if constexpr( is_static_rank_v<Stride> ){
      static_assert( Stride::_size >= sizeof...(is), 
        "boost::numeric::ublas::detail::access() : number of indices exceeds the size of the stride");
    }

    using value_type = typename Stride::value_type;
    std::array<value_type, sizeof...(is)> i = {is...};
    return std::inner_product(i.begin(), i.end(), w.begin(), value_type{});
  }

} // namespace boost::numeric::ublas::detail

#endif
