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

#ifndef BOOST_UBLAS_TENSOR_TYPE_TRAITS_DYNAMIC_STRIDES_HPP
#define BOOST_UBLAS_TENSOR_TYPE_TRAITS_DYNAMIC_STRIDES_HPP

#include <boost/numeric/ublas/tensor/detail/basic_type_traits.hpp>

namespace boost::numeric::ublas{
    
template<class int_type> class basic_extents;

template<class T, class L>
class basic_strides;

} // namespace boost::numeric::ublas

namespace boost::numeric::ublas{
    
    template <class L, class T> 
    struct is_strides<basic_strides<T,L>> : std::true_type {};

    template <class T, class L>
    struct is_dynamic< basic_strides<T,L> > : std::true_type {};

    template <class T, class L>
    struct is_dynamic_rank< basic_strides<T, L> > : std::true_type {};

    namespace detail{
        
        /** @brief Partial Specialization of strides for basic_extents
         *
         *
         * @tparam Layout either first_order or last_order
         *
         * @tparam T extents type
         *
         */
        template <class Layout, class T>
        struct strides_impl<basic_extents<T>, Layout>
        {
            using type = basic_strides<T, Layout>;
        };
        
    } // detail

} // namespace boost::numeric::ublas

#endif
