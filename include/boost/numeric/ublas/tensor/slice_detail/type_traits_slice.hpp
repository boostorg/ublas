//
//  Copyright (c) 2018-2020, Cem Bassoy, cem.bassoy@gmail.com
// 	Copyright (c) 2019-2020, Amit Singh, amitsingh19975@gmail.com
//
//  Distributed under the Boost Software License, Version 1.0. (See
//  accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)
//
//  The authors gratefully acknowledge the support of
//  Google
//

#ifndef _BOOST_NUMERIC_UBLAS_TENSOR_TYPE_TRAITS_SLICE_HPP_
#define _BOOST_NUMERIC_UBLAS_TENSOR_TYPE_TRAITS_SLICE_HPP_

#include <type_traits>
#include <boost/numeric/ublas/tensor/detail/basic_type_traits.hpp>

namespace boost::numeric::ublas::experimental {
    
    template<typename T, T...>
    struct basic_slice;

    template<typename T>
    struct is_slice : std::false_type{};

    template<typename T>
    inline static constexpr auto const is_slice_v = is_slice<T>::value;

} // namespace boost::numeric::ublas::span

namespace boost::numeric::ublas::experimental {
    
    template<typename T, T... Vs>
    struct is_slice< basic_slice<T, Vs...> > : std::true_type{};

} // namespace boost::numeric::ublas::span

namespace boost::numeric::ublas{
    
    template<typename T>
    struct is_dynamic< experimental::basic_slice<T> > : std::true_type{};
    
    template<typename T, T s, T... Vs>
    struct is_static< experimental::basic_slice<T, s, Vs...> > : std::true_type{};

} // namespace boost::numeric::ublas


#endif
