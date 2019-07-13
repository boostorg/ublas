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

#ifndef BOOST_NUMERIC_UBLAS_TENSOR_FWD_HPP
#define BOOST_NUMERIC_UBLAS_TENSOR_FWD_HPP

#include <iostream>

namespace boost::numeric::ublas {

template <class T, ptrdiff_t... E> struct basic_static_extents;

template <class T> class basic_extents;

template <class E, typename> constexpr bool valid(E const &e);

template <class E, typename> constexpr bool is_scalar(E const &e);

template <class E, typename> constexpr bool is_vector(E const &e);

template <class E, typename> constexpr bool is_matrix(E const &e);

template <class E, typename> constexpr bool is_tensor(E const &e);

template <class E, typename> auto squeeze(E const &e);

template <class E, typename> auto product(E const &e);

template <class E, typename> std::string to_string(E const &e);

template <class int_type, ptrdiff_t... E> struct basic_static_extents;

template <class __int_type, class __layout> class basic_strides;

/** @brief Forward declaration of static_strides for specialization
 *
 * @code static_strides<basic_static_extents<4,1,2,3,4>, first_order> s @endcode
 * @code static_strides<basic_static_extents<4,1,2,3,4>, last_order> s @endcode
 *
 * @tparam ExtentType type of basic_static_extents
 * @tparam Layout either first_order or last_order
 *
 */
template <class ExtentType, class Layout> struct static_strides;

/** @brief Type trait for selecting static_strides or basic_stride based on the
 * type of extents
 *
 * @tparam E type of basic_extents or basic_static_extents
 *
 * @tparam Layout either first_order or last_order
 *
 */

template <class E, class Layout> struct stride_type;

template <class T, class E, class F, class A> class tensor;

template <class T, class F, class A> class matrix;

template <class T, class A> class vector;

} // namespace boost::numeric::ublas

namespace boost::numeric::ublas::detail {

/** @brief stores the extents
 *
 * tparam R of type ptrdiff_t which stands for Rank
 * tparam S of type basic_shape
 *
 */
template <ptrdiff_t R, class S> struct basic_extents_impl;

template <class E> struct is_extents_impl;

template <class E> struct is_static_extents_impl;

template <class E> struct is_static_extents;

template <class E> struct is_basic_extents_impl_impl;

template <class E> struct is_basic_extents_impl;

template <class E> struct is_dynamic;

template <class E> struct is_static;

template <class T> struct remove_cvref;

struct iterator_tag;

struct invalid_iterator_tag;

template <class T, class> struct is_iterator;

/**
 * @tparam depth type of size_t for keeping track of recursive depth
 * @tparam IndexType type of index
 * @tparam Args parameter pack of indices with different types
 * @param E type of extent
 * @param idx index of extent
 * @param args parameter pack of indices
 * @returns true if in bound or false if not
 **/
template <size_t depth = 0, class E, class IndexType, class... Args>
constexpr bool in_bounds(E const &e, IndexType const &idx, Args... args);

template <class E> constexpr bool in_bounds(E const &e);

template <ptrdiff_t... E> struct dynamic_extents_impl;

template <ptrdiff_t... E> struct shape_in_bounds;

template <ptrdiff_t... Extents> struct basic_shape;

/** @brief struct declaration which makes the range of dynamic_extents for a
 *given range
 *
 * @tparam start which is a start range
 * @tparam end which is a end of range
 *
 **/
template <ptrdiff_t start, ptrdiff_t end> struct make_dynamic_basic_shape_impl;

/** @brief struct declaration which concat two basic_shapes
 *
 * @tparam S1 type of basic_shape
 * @tparam S2 type of basic_shape
 *
 **/
template <class S1, class S2> struct concat_basic_shape;

template <ptrdiff_t start, ptrdiff_t end> struct make_dynamic_basic_shape_impl;

/** @brief struct declaration which makes the basic_shape with given rank and
 *extents
 *
 * @tparam rank which is a rank of tensor of type ptrdiff_t
 * @tparam Extents which is a parameter pack of type ptrdiff_t containing
 *extents
 *
 **/
template <ptrdiff_t rank, ptrdiff_t... Extents> struct make_basic_shape;

template <class S> struct is_basic_shape;

template <class S> struct is_dynamic_basic_shape;

template <typename T> struct is_stl_array;

template <ptrdiff_t... D> struct product_helper_impl;

template <typename E> struct product_helper;

template <typename V, typename F, typename A> struct tensor_mode_result;

// TODO: Future
// template <typename V, typename E1, typename E2, typename F, typename A>
// struct tensor_result;

} // namespace boost::numeric::ublas::detail

namespace boost::numeric::ublas::storage {

struct tensor_storage;
struct sparse_storage;
struct band_storage;
struct dense_storage;

namespace sparse_tensor {

template <typename T, typename A> struct compressed_map;
}

namespace dense_tensor {

template <typename T, typename E, typename A, typename = void>
struct default_storage;
}

namespace detail {
template <typename T> struct is_tensor_storage;
template <typename T> struct is_sparse_storage;
template <typename T> struct is_band_storage;
template <typename T> struct is_dense_storage;
} // namespace detail

} // namespace boost::numeric::ublas::storage

#endif