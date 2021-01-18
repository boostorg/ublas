//
//  Copyright (c) 2019, Amit Singh, amitsingh19975@gmail.com
//  Copyright (c) 2021, Cem Bassoy, cem.bassoy@gmail.com
//
//  Distributed under the Boost Software License, Version 1.0. (See
//  accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)
//

#ifndef BOOST_UBLAS_TENSOR_BASIC_TYPE_TRAITS_HPP
#define BOOST_UBLAS_TENSOR_BASIC_TYPE_TRAITS_HPP

#include <type_traits>
#include <cstddef>

namespace boost::numeric::ublas {

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

#endif
