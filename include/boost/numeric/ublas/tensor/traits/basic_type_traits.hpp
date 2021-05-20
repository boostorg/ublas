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




///// To check if the type is the std::array or not.
///// Can be extented by providing specialization.
///// Point to Remember: C-Style arrays are not supported.
//template<class T> struct is_bounded_array : std::false_type{};
//template<class T, std::size_t N> struct is_bounded_array<std::array<T,N>> : std::true_type{};
///// Gives the extent of rank one std::array.
///// Similar to is_bounded_array, it can also be
///// extented using specialization.
///// Custom Type should have similar APIs to
///// std::array.
///// Point to Remember: C-Style arrays are not supported.
//template<class T> struct extent_of_rank_one_array;
//template<class T, std::size_t N>
//struct extent_of_rank_one_array<std::array<T,N>> : std::integral_constant<std::size_t,N>{};

//template<class T>
//inline static constexpr
//  bool is_bounded_array_v = is_bounded_array<T>::value;
//template<class T>
//inline static constexpr
//  std::size_t extent_of_rank_one_array_v = extent_of_rank_one_array<T>::value;

} // namespace boost::numeric::ublas

#endif
