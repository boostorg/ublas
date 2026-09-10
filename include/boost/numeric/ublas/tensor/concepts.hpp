//
//  Copyright (c) 2021, Cem Bassoy, cem.bassoy@gmail.com
//  Copyright (c) 2021, Amit Singh, amitsingh19975@gmail.com
//
//  Distributed under the Boost Software License, Version 1.0. (See
//  accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)
//
//  The authors gratefully acknowledge the support of
//  Google and Fraunhofer IOSB, Ettlingen, Germany
//

#ifndef BOOST_UBLAS_TENSOR_CONCEPTS_HPP
#define BOOST_UBLAS_TENSOR_CONCEPTS_HPP

#include <type_traits>

namespace boost::numeric::ublas{

template<typename T>
concept integral = std::is_integral_v<T>;

template<typename T>
concept signed_integral = integral<T> && std::is_signed_v<T>;

template<typename T>
concept unsigned_integral = integral<T> && !signed_integral<T>;

template<typename T>
concept floating_point = std::is_floating_point_v<T>;

} // namespace boost::numeric::ublas

#endif // BOOST_UBLAS_TENSOR_CONCEPTS_BASIC_HPP
