//
// 	Copyright (c) 2018-2020, Cem Bassoy, cem.bassoy@gmail.com
// 	Copyright (c) 2019-2020, Amit Singh, amitsingh19975@gmail.com
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

namespace boost::numeric::ublas::detail {
  
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

/** @brief Checks if type has member function resize with
 * prototype resize(size_type) and return type can be 
 * anything and type should have empty constructor.
 * @code 
 * 
 * struct test1{
 *  test1(){...}
 *  void resize(size_t){...}
 *  ...
 * };
 * 
 * struct test2{
 *  test2(){...}
 *  ...
 * };
 * 
 * struct test3{
 *  test2(int){...}
 *  void resize(size_t){...}
 * };
 * 
 * auto v = std::vector<int>{}; 
 * auto vec_val = is_resizable_v<decltype(v)>; // vec_val == true;
 * 
 * auto a = std::array<int,10>{}; 
 * auto arr_val = is_resizable_v<decltype(a)>; // arr_val == false;
 * 
 * auto test1_val = is_resizable_v<test1>; // arr_val == true;
 * auto test2_val = is_resizable_v<test2>; // arr_val == false;
 * auto test3_val = is_resizable_v<test3>; // arr_val == false bacause test3 does not have empty constuctor.
 * 
 * @endcode
*/
template<typename T, typename = void>
struct is_resizable : std::false_type{};

template<typename T>
struct is_resizable< T, std::void_t< decltype(T{}.resize(0)) > >: std::true_type{};

template<typename T>
inline static constexpr bool const is_resizable_v = is_resizable<T>::value;

} // namespace boost::numeric::ublas::detail

#include <boost/numeric/ublas/tensor/detail/type_traits_extents.hpp>
#include <boost/numeric/ublas/tensor/detail/type_traits_strides.hpp>
#include <boost/numeric/ublas/tensor/detail/storage_traits.hpp>

#endif