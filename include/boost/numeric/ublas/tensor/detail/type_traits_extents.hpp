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

#ifndef BOOST_UBLAS_TENSOR_TYPE_TRAITS_EXTENTS_HPP
#define BOOST_UBLAS_TENSOR_TYPE_TRAITS_EXTENTS_HPP

namespace boost::numeric::ublas {

// checks if type is extents or not
template <class E>
struct is_extents : std::false_type {};

template <class E>
inline static constexpr bool const is_extents_v = is_extents<E>::value;

namespace detail{

    template<std::size_t... N>
    struct dynamic_extents_impl;

} // detail

template<std::size_t... E>
using dynamic_extents = typename detail::dynamic_extents_impl<E...>::type;

} // namespace boost::numeric::ublas::detail

#include <boost/numeric/ublas/tensor/detail/type_traits_dynamic_extents.hpp>
#include <boost/numeric/ublas/tensor/detail/type_traits_static_extents.hpp>
#include <boost/numeric/ublas/tensor/detail/type_traits_fixed_rank_extents.hpp>

#endif
