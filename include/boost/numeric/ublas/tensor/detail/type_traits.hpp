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

#ifndef BOOST_UBLAS_TENSOR_TYPE_TRAITS_HPP
#define BOOST_UBLAS_TENSOR_TYPE_TRAITS_HPP

#include <type_traits>
#include <cstddef>

namespace boost::numeric::ublas {
  
/** @brief Checks if the extents or strides is dynamic
 *
 * @tparam E of type basic_extents or basic_static_extents
 *
 */
template <class E> struct is_dynamic : std::false_type {};

template <class E> 
inline static constexpr bool const is_dynamic_v = is_dynamic<E>::value;

/** @brief Checks if the extents or strides is static
 *
 * @tparam E of type basic_extents or basic_static_extents
 *
 */
template <class E> struct is_static : std::false_type {};

template <class E> 
inline static constexpr bool const is_static_v = is_static<E>::value;

/** @brief Checks if the extents or strides has dynamic rank
 *
 * @tparam E of type basic_extents or basic_static_extents
 *
 */
template <class E> 
struct is_dynamic_rank : std::false_type {};

template <class E> 
inline static constexpr bool const is_dynamic_rank_v = is_dynamic_rank<E>::value;

/** @brief Checks if the extents or strides has static rank
 *
 * @tparam E of type basic_extents or basic_static_extents
 *
 */
template <class E> 
struct is_static_rank : std::false_type {};

template <class E> 
inline static constexpr bool const is_static_rank_v = is_static_rank<E>::value;

} // namespace boost::numeric::ublas::detail

#include <boost/numeric/ublas/tensor/detail/type_traits_extents.hpp>
#include <boost/numeric/ublas/tensor/detail/type_traits_strides.hpp>
#include <boost/numeric/ublas/tensor/detail/type_traits_tensor.hpp>
#include <boost/numeric/ublas/tensor/detail/storage_traits.hpp>

#endif
