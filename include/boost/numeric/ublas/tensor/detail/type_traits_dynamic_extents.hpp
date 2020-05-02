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

#ifndef BOOST_UBLAS_TENSOR_TYPE_TRAITS_DYNAMIC_EXTENTS_HPP
#define BOOST_UBLAS_TENSOR_TYPE_TRAITS_DYNAMIC_EXTENTS_HPP

#include <boost/numeric/ublas/tensor/detail/basic_type_traits.hpp>

namespace boost::numeric::ublas{
    
template<class int_type> class basic_extents;

} // namespace boost::numeric::ublas

namespace boost::numeric::ublas{
    
    template <class T> 
    struct is_extents< basic_extents<T> > : std::true_type {};

    template <class T>
    struct is_dynamic< basic_extents<T> > : std::true_type {};

    template <class T>
    struct is_dynamic_rank< basic_extents<T> > : std::true_type {};


    namespace detail{
        
        template <> struct dynamic_extents_impl<> {
            using type = basic_extents<std::size_t>;
        };

    } // namespace detail

} // namespace boost::numeric::ublas

#endif
