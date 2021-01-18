//
//  Copyright (c) 2021, Cem Bassoy, cem.bassoy@gmail.com
//
//  Distributed under the Boost Software License, Version 1.0. (See
//  accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)
//
//  The authors gratefully acknowledge the support of
//  Google and Fraunhofer IOSB, Ettlingen, Germany
//


#ifndef _BOOST_NUMERIC_UBLAS_TENSOR_DETAIL_STRIDES_FUNCTIONS_HPP_
#define _BOOST_NUMERIC_UBLAS_TENSOR_DETAIL_STRIDES_FUNCTIONS_HPP_

#include <type_traits>
//#include <functional>
//#include <numeric>
//#include <boost/numeric/ublas/tensor/detail/extents_functions.hpp>
#include <algorithm>



namespace boost::numeric::ublas
{

template<class derived_type>
struct extents_base;

template<class derived_type>
struct strides_base;

//template<class derived_type_extents, class derived_type_strides>
//inline constexpr
//  void compute_strides(
//    extents_base<derived_type_extents> const& extents,
//    strides_base<derived_type_strides> & strides)
//{
//  if(extents().size() != strides().size() || extents().size() == 0U)
//    return;

//  std::fill(strides().begin(), strides().end(), 1U);

//  if( is_vector(extents) || is_scalar(extents) )
//    return;

//  using layout_type = typename derived_type_strides::layout_type;
//  if constexpr (std::is_same<layout_type,layout::first_order>::value ) {
//    std::transform(extents().begin(), extents().end() - 1, strides().begin(), strides().begin() + 1, std::multiplies<>{});
//  } else {
//    std::transform(extents().rbegin(), extents().rend() - 1, strides().rbegin(), strides().rbegin() + 1, std::multiplies<>{});
//  }
//}



template <class L, class R>
[[nodiscard]] inline constexpr
  bool operator==( strides_base<L> const& lhs,  strides_base<R> const& rhs )
{
  return lhs().size() == rhs().size() && std::equal( lhs().begin(), lhs().end(), rhs().begin() );
}

template <class L, class R>
[[nodiscard]] inline constexpr
  bool operator!=( strides_base<L> const& lhs, strides_base<R> const& rhs )
{
  return !( lhs == rhs) ;
}


} // namespace boost::numeric::ublas

#endif // _BOOST_NUMERIC_UBLAS_TENSOR_DETAIL_STRIDES_FUNCTIONS_HPP_
