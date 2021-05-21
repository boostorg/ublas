//
//  Copyright (c) 2020, Cem Bassoy, cem.bassoy@gmail.com
//
//  Distributed under the Boost Software License, Version 1.0. (See
//  accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)
//


#ifndef BOOST_NUMERIC_UBLAS_TENSOR_EXTENTS_BASE_HPP
#define BOOST_NUMERIC_UBLAS_TENSOR_EXTENTS_BASE_HPP

#include <algorithm>
//#include <concepts>
#include <type_traits>

#include "../concepts.hpp"

namespace boost::numeric::ublas {


template<class D>
struct extents_base
{

  using derived_type  = D;
  inline constexpr decltype(auto) operator()() const { return static_cast<const derived_type&>(*this); }
  inline constexpr decltype(auto) operator()()       { return static_cast<      derived_type&>(*this); }  

};

template<integral T, T ...>
class extents_core;

template<std::size_t ... es>
using extents = extents_core<std::size_t, es...>;

template <class E> struct is_extents      : std::false_type {};
template <class E> struct is_strides      : std::false_type {};
template <class E> struct is_dynamic      : std::false_type {};
template <class E> struct is_static       : std::false_type {};
template <class E> struct is_dynamic_rank : std::false_type {};
template <class E> struct is_static_rank  : std::false_type {};

template <class E> inline static constexpr bool const is_extents_v      = is_extents<E>::value;
template <class E> inline static constexpr bool const is_strides_v      = is_strides<E>::value;
template <class E> inline static constexpr bool const is_dynamic_v      = is_dynamic<E>::value;
template <class E> inline static constexpr bool const is_static_v       = is_static <E>::value;
template <class E> inline static constexpr bool const is_dynamic_rank_v = is_dynamic_rank<E>::value;
template <class E> inline static constexpr bool const is_static_rank_v  = is_static_rank<E>::value;

} // namespace boost::numeric::ublas

#endif // _BOOST_NUMERIC_UBLAS_TENSOR_EXTENTS_BASE_HPP_
