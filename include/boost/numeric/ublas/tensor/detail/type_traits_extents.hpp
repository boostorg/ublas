//
// 	Copyright (c) 2018-2020, Cem Bassoy, cem.bassoy@gmail.com
// 	Copyright (c) 2019-2020, Amit Singh, amitsingh19975@gmail.com
//
//  Distributed under the Boost Software License, Version 1.0. (See
//  accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)
//
//  The authors gratefully acknowledge the support of
//  Google
//

#ifndef BOOST_UBLAS_TENSOR_TYPE_TRAITS_EXTENTS_HPP
#define BOOST_UBLAS_TENSOR_TYPE_TRAITS_EXTENTS_HPP

namespace boost::numeric::ublas{
  
  template <class ExtentsType, ExtentsType... E> struct basic_static_extents;
  template <class ExtentsType, std::size_t N> struct basic_fixed_rank_extents;
  template<class ExtentsType> class basic_extents;

} // namespace boost::numeric::ublas::


namespace boost::numeric::ublas::detail {

// checks if type is extents or not
template <class E>
struct is_extents : std::false_type {};

template <class T, T... E>
struct is_extents<basic_static_extents<T, E...>> : std::true_type {};

template <class T, std::size_t R>
struct is_extents<basic_fixed_rank_extents<T, R>> : std::true_type {};

template <class T> 
struct is_extents<basic_extents<T>> : std::true_type {};

template <class E>
inline static constexpr bool const is_extents_v = is_extents<E>::value;

template <class T>
struct is_dynamic<basic_extents<T>> : std::true_type {};

template <class T, std::size_t R>
struct is_dynamic<basic_fixed_rank_extents<T,R>> : std::true_type {};

template <class T, T... E>
struct is_static<basic_static_extents<T, E...>> : std::true_type {};

template <class T>
struct is_dynamic_rank< basic_extents<T> > : std::true_type {};

template <class T, std::size_t... E>
struct is_static_rank<basic_static_extents<T, E...>> : std::true_type {};

template <class T, std::size_t R>
struct is_static_rank<basic_fixed_rank_extents<T,R>> : std::true_type {};

} // namespace boost::numeric::ublas::detail


#endif