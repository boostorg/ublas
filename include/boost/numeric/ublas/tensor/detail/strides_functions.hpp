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


#ifndef BOOST_NUMERIC_UBLAS_TENSOR_DETAIL_STRIDES_FUNCTIONS_HPP
#define BOOST_NUMERIC_UBLAS_TENSOR_DETAIL_STRIDES_FUNCTIONS_HPP

#include <algorithm>
#include <type_traits>

#include "../layout.hpp"


namespace boost::numeric::ublas
{

template<class derived_type>
struct extents_base;

template<class derived_type>
struct strides_base;

namespace detail{
  
  template<typename Extents, typename Strides>
  constexpr void compute_strides_helper(Extents&& in, Strides&& out, layout::first_order) noexcept{
    std::transform(in.begin(),  in.end() - 1,  out.begin(),  out.begin()  + 1, std::multiplies<>{});
  }

  template<typename Extents, typename Strides>
  constexpr void compute_strides_helper(Extents&& in, Strides&& out, layout::last_order) noexcept{
    std::transform(in.rbegin(),  in.rend() - 1,  out.rbegin(),  out.rbegin()  + 1, std::multiplies<>{});
  }

} // namespace detail


template<class derived_type_extents, class derived_type_strides>
inline constexpr
  void compute_strides(
    extents_base<derived_type_extents> const& extents,
    strides_base<derived_type_strides> & strides)
{
  if(extents().size() != strides().size() || extents().size() == 0U)
    return;

  std::fill(strides().begin(), strides().end(), 1U);

  if( is_vector(extents) || is_scalar(extents) )
    return;

  using layout_type = typename derived_type_strides::layout_type;
  detail::compute_strides_helper(extents(),strides(),layout_type{});
}



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
