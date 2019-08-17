//
//  Copyright (c) 2018-2019, Cem Bassoy, cem.bassoy@gmail.com
//
//  Distributed under the Boost Software License, Version 1.0. (See
//  accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)
//
//  The authors gratefully acknowledge the support of
//  Fraunhofer IOSB, Ettlingen, Germany
//

#ifndef BOOST_UBLAS_TENSOR_META_FUNCTIONS_HPP
#define BOOST_UBLAS_TENSOR_META_FUNCTIONS_HPP

#include "fwd.hpp"
#include <array>
#include <type_traits>
#include <vector>

namespace boost::numeric::ublas::detail
{

template <class E>
struct is_extents_impl : std::integral_constant<bool, false>
{
};

template <class T, ptrdiff_t R, ptrdiff_t... E>
struct is_extents_impl<basic_static_extents<T, R, E...>> : std::true_type
{
};

template <class T>
struct is_extents_impl<basic_extents<T>> : std::true_type
{
};

template <class E>
struct is_extents
{
  static constexpr bool value =
      is_extents_impl<typename std::decay<E>::type>::value;
};

template <class E>
struct is_static_extents_impl : std::integral_constant<bool, false>
{
};

template <class T, ptrdiff_t R, ptrdiff_t... E>
struct is_static_extents_impl<basic_static_extents<T, R, E...>>
    : std::integral_constant<bool, true>
{
};

template <class E>
struct is_static_extents
{
  static constexpr bool value =
      is_static_extents_impl<typename std::decay<E>::type>::value;
};

/** @brief type trait for checks if basic_extents_impl or not
 *
 * @tparam E of any type
 *
 **/
template <class E>
struct is_basic_extents_impl_impl : std::integral_constant<bool, false>
{
};

/** @brief is_extents_impl specialization
 *
 * @tparam R of ptrdiff_t type
 * @tparam S of basic_shape type
 *
 **/
template <ptrdiff_t R, class S>
struct is_basic_extents_impl_impl<basic_extents_impl<R, S>>
    : std::integral_constant<bool, true>
{
};

template <class E>
struct is_basic_extents_impl
{
  static constexpr bool value =
      is_basic_extents_impl_impl<typename std::decay<E>::type>::value;
};

/** @brief Checks if the extents is dynamic
 *
 * @tparam E of type basic_extents or basic_static_extents
 *
 */
template <class E>
struct is_dynamic : std::integral_constant<bool, false>
{
};

/** @brief Partial Specialization of is_dynamic_extents with basic_extens
 *
 * @tparam T of any integer type
 *
 */
template <class T>
struct is_dynamic<basic_extents<T>> : std::integral_constant<bool, true>
{
};

/** @brief Partial Specialization of is_static with basic_static_extents
 *
 * @tparam R rank of ptrdiff_t
 *
 * @tparam E parameter pack of extents
 *
 */
template <class T, ptrdiff_t R, ptrdiff_t... E>
struct is_dynamic<basic_static_extents<T, R, E...>>
{
  static constexpr bool value =
      basic_static_extents<T, R, E...>::DynamicRank != 0;
};

/** @brief Checks if the extents is static
 *
 * @tparam E of type basic_extents or basic_static_extents
 *
 */
template <class E>
struct is_static
{
  static constexpr bool value = !is_dynamic<E>::value;
};

template <>
struct product_helper_impl<>
{
  static constexpr ptrdiff_t value = 1;
};

template <ptrdiff_t E, ptrdiff_t... R>
struct product_helper_impl<E, R...>
{
  static constexpr ptrdiff_t value =
      E != -1 ? E * product_helper_impl<R...>::value : 0;
};

/** @brief removes the const and refernece
 *
 * @tparam T any type
 *
 **/
template <class T>
struct remove_cvref
{
  using type = std::remove_cv_t<std::remove_reference_t<T>>;
};

// empty struct for tagging valid iterator
struct iterator_tag
{
};

// empty struct for tagging invalid iterator
struct invalid_iterator_tag
{
};

/** @brief checks and gives back the appropriate tag
 *
 * @tparam I type of iterator
 *
 **/
template <class I>
using iterator_tag_t = std::conditional_t<
    std::is_same<typename std::iterator_traits<I>::iterator_category,
                 std::output_iterator_tag>::value,
    invalid_iterator_tag,
    std::conditional_t<
        std::numeric_limits<typename remove_cvref<
            typename std::iterator_traits<I>::reference>::type>::is_integer,
        iterator_tag, invalid_iterator_tag>>;

/** @brief checks if given type is iterator or not
 *
 * @tparam T any type
 *
 **/
template <class T, class = void>
struct is_iterator
{
  static constexpr bool value = false;
};

/** @brief Partial specialization for is_iterator
 *
 * @tparam T any type
 *
 **/
template <class T>
struct is_iterator<
    T, typename std::enable_if_t<!std::is_same<
           typename std::iterator_traits<T>::value_type, void>::value>>
{
  static constexpr bool value = true;
};

template <ptrdiff_t R>
struct dynamic_extents_impl<R>
{
  using type = basic_static_extents<std::size_t, R>;
};

template <>
struct dynamic_extents_impl<>
{
  using type = basic_extents<std::size_t>;
};

template <typename T>
struct is_stl_array : std::false_type
{
};
template <typename T, std::size_t N>
struct is_stl_array<std::array<T, N>> : std::true_type
{
};

} // namespace boost::numeric::ublas::detail

namespace boost::numeric::ublas
{

template <typename T, ptrdiff_t S, ptrdiff_t... E>
struct detail::product_helper<basic_static_extents<T, S, E...>>
{
  static constexpr T value = detail::product_helper_impl<E...>::value;
};
template <typename V, typename F, typename A>
struct tensor_mode_result
{
  using type = std::conditional_t<
      storage::detail::is_tensor_storage<A>::value,
      tensor<V, basic_extents<std::size_t>, F, A>,
      tensor<V, basic_extents<std::size_t>, F, std::vector<V>>>;
};

template <typename V, typename F, typename A>
using tensor_mode_result_t = typename tensor_mode_result<V, F, A>::type;

} // namespace boost::numeric::ublas

namespace boost::numeric::ublas::storage::detail
{

template <typename T>
struct is_tensor_storage
{
  static constexpr bool value = std::is_base_of<tensor_storage, T>::value;
};

template <typename T>
struct is_sparse_storage
{
  static constexpr bool value = std::is_base_of<sparse_storage, T>::value;
};

template <typename T>
struct is_band_storage
{
  static constexpr bool value = std::is_base_of<band_storage, T>::value;
};

template <typename T>
struct is_dense_storage
{
  static constexpr bool value = std::is_base_of<dense_storage, T>::value;
};

} // namespace boost::numeric::ublas::storage::detail

namespace boost::numeric::ublas::span::detail
{

template <typename T>
struct is_slice_impl : std::false_type
{
};

template <typename T, ptrdiff_t... Ts>
struct is_slice_impl<basic_slice<T, Ts...>> : std::true_type
{
};

template <typename T>
struct is_dynamic_slice : std::false_type
{
};

template <typename T>
struct is_static_slice : std::false_type
{
};

template <typename T>
struct is_dynamic_slice<basic_slice<T>> : std::true_type
{
};

template <typename T, ptrdiff_t V, ptrdiff_t... Vs>
struct is_static_slice<basic_slice<T, V, Vs...>> : std::true_type
{
};

template <typename... Ts>
struct is_slice;

template <>
struct is_slice<> : std::true_type
{
};

template <typename T, typename... Ts>
struct is_slice<T, Ts...>
{
  static constexpr bool value = is_slice_impl<T>::value && is_slice<Ts...>::value;
};

} // namespace boost::numeric::ublas::span::detail

#endif