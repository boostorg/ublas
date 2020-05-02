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

#ifndef BOOST_UBLAS_TENSOR_TYPE_TRAITS_FIXED_RANK_STRIDES_HPP
#define BOOST_UBLAS_TENSOR_TYPE_TRAITS_FIXED_RANK_STRIDES_HPP

#include <boost/numeric/ublas/tensor/detail/basic_type_traits.hpp>

namespace boost::numeric::ublas{
    
template <class ExtentsType, std::size_t N> struct basic_fixed_rank_extents;

template<class T, std::size_t N, class L> class basic_fixed_rank_strides;

} // namespace boost::numeric::ublas

namespace boost::numeric::ublas{
    
    template <class L, class T, std::size_t R>
    struct is_strides< basic_fixed_rank_strides< T, R, L> > : std::true_type {};

    template <class T, std::size_t R, class L>
    struct is_dynamic< basic_fixed_rank_strides<T,R,L> > : std::true_type {};

    template <class T, std::size_t R, class L>
    struct is_static_rank< basic_fixed_rank_strides<T,R,L> > : std::true_type {};

    namespace detail{

        /** @brief Partial Specialization of strides for basic_fixed_rank_strides
         *
         *
         * @tparam Layout either first_order or last_order
         *
         * @tparam T extents type
         *
         */
        template <class Layout, std::size_t N, class T>
        struct strides_impl<basic_fixed_rank_extents<T,N>, Layout>
        {
            using type = basic_fixed_rank_strides<T, N, Layout>;
        };
        
    } // detail

} // namespace boost::numeric::ublas

#endif
