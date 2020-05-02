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

#ifndef BOOST_UBLAS_TENSOR_TYPE_TRAITS_FIXED_RANK_EXTENTS_HPP
#define BOOST_UBLAS_TENSOR_TYPE_TRAITS_FIXED_RANK_EXTENTS_HPP

#include <boost/numeric/ublas/tensor/detail/basic_type_traits.hpp>

namespace boost::numeric::ublas{
    
template <class ExtentsType, std::size_t N> struct basic_fixed_rank_extents;

} // namespace boost::numeric::ublas

namespace boost::numeric::ublas{
    
    template <class T, std::size_t R>
    struct is_extents< basic_fixed_rank_extents<T, R> > : std::true_type {};

    template <class T, std::size_t R>
    struct is_dynamic< basic_fixed_rank_extents<T,R> > : std::true_type {};

    template <class T, std::size_t R>
    struct is_static_rank< basic_fixed_rank_extents<T,R> > : std::true_type {};

    namespace detail{

        template <std::size_t N> struct dynamic_extents_impl<N> {
            using type = basic_fixed_rank_extents<std::size_t, N>;
        };

    } // namespace detail

} // namespace boost::numeric::ublas

#endif
