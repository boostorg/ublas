//
//  Copyright (c) 2019, Amit Singh, amitsingh19975@gmail.com
//  Copyright (c) 2021, Cem Bassoy, cem.bassoy@gmail.com
//
//  Distributed under the Boost Software License, Version 1.0. (See
//  accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)
//

#ifndef BOOST_UBLAS_TENSOR_BASIC_TYPE_TRAITS_HPP
#define BOOST_UBLAS_TENSOR_BASIC_TYPE_TRAITS_HPP

#include <type_traits>
#include <cstddef>
#include <array>
#include <complex>

namespace boost::numeric::ublas {


template<typename T>
struct is_complex : std::false_type{};

template<typename T>
struct is_complex< std::complex<T> > : std::true_type{};

template<typename T>
inline static constexpr bool is_complex_v = is_complex<T>::value;

} // namespace boost::numeric::ublas

#endif
