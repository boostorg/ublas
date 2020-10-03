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

#ifndef BOOST_UBLAS_TENSOR_TYPE_TRAITS_DYNAMIC_STRIDES_HPP
#define BOOST_UBLAS_TENSOR_TYPE_TRAITS_DYNAMIC_STRIDES_HPP

#include <boost/numeric/ublas/tensor/traits/basic_type_traits.hpp>

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

    template <class T>
    struct strides<basic_extents<T>>
    {
        template<typename Layout>
        using type = basic_strides<T, Layout>;
    };

} // namespace boost::numeric::ublas

#endif
