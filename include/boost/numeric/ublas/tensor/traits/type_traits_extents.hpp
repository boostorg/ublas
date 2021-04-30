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

#ifndef BOOST_UBLAS_TENSOR_TYPE_TRAITS_EXTENTS_HPP
#define BOOST_UBLAS_TENSOR_TYPE_TRAITS_EXTENTS_HPP

#include <cstddef>

namespace boost::numeric::ublas {

template<class int_type>                class basic_extents;
template<class int_type, std::size_t N> class basic_fixed_rank_extents;

namespace detail{
template <std::size_t... N> struct extents_impl;
template <>                 struct extents_impl<>  { using type = basic_extents<std::size_t>; };
template <std::size_t N>    struct extents_impl<N> { using type = basic_fixed_rank_extents<std::size_t, N>; };
} // namespace detail

template<std::size_t... E> using extents = typename detail::extents_impl<E...>::type;


} // namespace boost::numeric::ublas

#endif
