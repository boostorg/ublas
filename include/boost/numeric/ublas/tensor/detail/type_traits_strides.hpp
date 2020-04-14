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

#ifndef BOOST_UBLAS_TENSOR_TYPE_TRAITS_STRIDES_HPP
#define BOOST_UBLAS_TENSOR_TYPE_TRAITS_STRIDES_HPP

namespace boost::numeric::ublas{
  
  template <class ExtentsType, ExtentsType... E> struct basic_static_extents;
  template <class ExtentsType, std::size_t N> struct basic_fixed_rank_extents;
  template<class ExtentsType> class basic_extents;

  template <class E, class L> struct basic_static_strides;
  template<class T, std::size_t R, class L> class basic_fixed_rank_strides;
  template<class T, class L> class basic_strides;

} // namespace boost::numeric::ublas::


namespace boost::numeric::ublas::detail {

// checks if type is strides or not
template <class E>
struct is_strides : std::false_type {};

template <class E>
inline static constexpr bool const is_strides_v = is_strides<E>::value;

template <class L, class T, T... E>
struct is_strides< basic_static_strides< basic_static_extents<T, E...>, L > > : std::true_type {};

template <class L, class T, std::size_t R>
struct is_strides< basic_fixed_rank_strides< T, R, L> > : std::true_type {};

template <class L, class T> 
struct is_strides<basic_strides<T,L>> : std::true_type {};

template <class T, class L>
struct is_dynamic< basic_strides<T,L> > : std::true_type {};

template <class T, std::size_t R, class L>
struct is_dynamic< basic_fixed_rank_strides<T,R,L> > : std::true_type {};

template <class T, T... E, class L>
struct is_static< basic_static_strides< basic_static_extents<T, E...>, L > > : std::true_type {};

template <class T, class L>
struct is_dynamic_rank< basic_strides<T, L> > : std::true_type {};

template <class T, T... E, class L>
struct is_static_rank< basic_static_strides< basic_static_extents<T, E...>, L > > : std::true_type {};

template <class T, std::size_t R, class L>
struct is_static_rank< basic_fixed_rank_strides<T,R,L> > : std::true_type {};

} // namespace boost::numeric::ublas::detail


#endif