//
//  Copyright (c) 2021, Cem Bassoy, cem.bassoy@gmail.com
//
//  Distributed under the Boost Software License, Version 1.0. (See
//  accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)
//
//  The authors gratefully acknowledge the support of
//  Google and Fraunhofer IOSB, Ettlingen, Germany
//


#ifndef BOOST_NUMERIC_UBLAS_TENSOR_EXTENTS_STATIC_FUNCTIONS_HPP
#define BOOST_NUMERIC_UBLAS_TENSOR_EXTENTS_STATIC_FUNCTIONS_HPP


#include <algorithm>
#include <functional>
#include <numeric>

#include "extents_base.hpp"
#include "../layout.hpp"



namespace boost::numeric::ublas
{



//////////////// SIZE ///////////////

namespace detail {
template <class T>
struct size_impl_t;
template <std::size_t ... es>
struct size_impl_t<extents<es...>>
{ static constexpr auto value = sizeof ...(es); };
template <std::size_t ... es>
struct size_impl_t<std::index_sequence<es...>>
{ static constexpr auto value = sizeof ...(es); };
} // namespace detail

/** @brief Returns the size of a pure static extents type
 *
 * @code constexpr auto n = size_v<extents<4,3,2>>;
 * @note corresponds to std::tuple_size_v
 *
*/
template<class E>
constexpr inline auto size_v = detail::size_impl_t<std::decay_t<E>>::value;

//////////////// EMPTY ///////////////

namespace detail {
template <class T>
struct empty_impl_t;

template <std::size_t ... es>
struct empty_impl_t<extents<es...>>
{ static constexpr bool value = size_v<extents<es...>> == 0ul; };
} // namespace detail

/** @brief Returns if a pure static extents type is empty
 *
 * @code constexpr bool empty = empty_v<extents<4,3,2>>; // -> false
 *
*/
template<class E>
constexpr inline bool empty_v = detail::empty_impl_t<std::decay_t<E>>::value;


//////////////// GET /////////////////////

namespace detail {
template<std::size_t, std::size_t, std::size_t, class>
struct get_impl_t;

template<std::size_t j, std::size_t k, std::size_t n>
struct get_impl_t<j,k,n,extents<>>
{
  static constexpr auto value = 0;
};

template<std::size_t j, std::size_t k, std::size_t n, std::size_t e1>
struct get_impl_t<j,k,n,extents<e1>>
{
  static_assert ( j < n && k < n );
  static constexpr auto value = e1;
};

template<std::size_t j, std::size_t k, std::size_t n, std::size_t e1, std::size_t e2, std::size_t ... es>
struct get_impl_t<j,k,n,extents<e1, e2, es...>>
{
  static_assert ( k < n && j < n );
  static constexpr auto value = (j==k) ? e1 : get_impl_t<j,k+1,n,extents<e2,es...>>::value;
};
} // namespace detail

/** @brief Returns the j-th element of a pure static extents type with 0 <= j < size_v<extents>
 *
 * @code constexpr auto e_j = get_v<extents<4,3,2>,2>;
 *
*/
template<class E, std::size_t j>
constexpr inline auto get_v = detail::get_impl_t<j,0,size_v<E>,std::decay_t<E>>::value;


//////////////// CAT /////////////////////
namespace detail {
template<class EL, class ER>
struct cat_impl_t
{
  template<class, class>
  struct inner;
  template<std::size_t ... is, std::size_t ... js>
  struct inner < std::index_sequence<is...>,  std::index_sequence<js...>  >
  {
    using type = extents < get_v<EL,is>...,  get_v<ER,js>... >;
  };
  using type = typename inner <
    std::make_index_sequence<size_v<EL>>,
    std::make_index_sequence<size_v<ER>> >::type;
};
} // namespace detail

/** @brief Concatenates two static extents type
 *
 * @code using extents_t = cat<extents<4,3,2>,extents<7,6>>; // -> extents<4,3,2,7,6>
 *
 * @tparam EL left  extents<...>
 * @tparam ER right extents<...>
*/
template<class EL, class ER>
using cat_t = typename detail::cat_impl_t<std::decay_t<EL>,std::decay_t<ER>>::type;

//////////////// FOR_EACH ////////////////

namespace detail {

template<class, template <std::size_t> class>
struct for_each_impl_t;

template<template<std::size_t> class UnaryOp, typename std::size_t ... es>
struct for_each_impl_t<extents<es...>, UnaryOp >
{ using type = extents< ( UnaryOp<es>::value )... >; };

template<template<std::size_t> class UnaryOp, typename std::size_t ... is>
struct for_each_impl_t<std::index_sequence<is...>, UnaryOp >
{ using type = std::index_sequence< ( UnaryOp<is>::value )... >; };

} // namespace detail

/** @brief Applies a unary operation for each element of a given static extents type
 *
 * @code template<std::size_t e> struct add5 { static constexpr auto value = e+5; };
 * @code using extents_t = for_each<extents<4,3,2>,add5>; // -> extents<9,8,7>
 *
 * @tparam E extents<...>
*/
template<class E, template<std::size_t> typename UnaryOp>
using for_each_t = typename detail::for_each_impl_t<std::decay_t<E>, UnaryOp>::type;

//////////////// TEST ////////////////

namespace detail {

template<class, template <std::size_t> class>
struct for_each_test_impl_t;

template<template<std::size_t> class UnaryPred, typename std::size_t ... es>
struct for_each_test_impl_t<extents<es...>, UnaryPred >
{ using type = std::integer_sequence<bool, ( UnaryPred<es>::value )... >; };

template<template<std::size_t> class UnaryPred, typename std::size_t ... is>
struct for_each_test_impl_t<std::index_sequence<is...>, UnaryPred >
{ using type = std::integer_sequence<bool, ( UnaryPred<is>::value )... >; };

} // namespace detail

/** @brief Returns true if for each element of a given static extents type the unary predicate holds
 *
 * @code template<std::size_t e> struct equal5 { static constexpr bool value = e==5; };
 * @code using sequence_t = for_each<extents<4,3,2>,equal5>; // -> std::integer_sequence<false,false,false>
 *
 * @tparam E extents<...>
*/
template<class E, template<std::size_t> typename UnaryPred>
using for_each_test_t = typename detail::for_each_test_impl_t<std::decay_t<E>, UnaryPred>::type;


//////////////// SELECT INDEX SEQUENCE /////////////////

namespace detail {
template<class E, class I>
struct select_impl_t
{
  static_assert( size_v<E> >= I::size() );
  template<class>  struct inner;
  template<std::size_t ... is>
  struct inner <std::index_sequence<is...> > { using type = extents<get_v<E,is> ... >; };
  using type = typename inner<I>::type;
};
} // namespace detail

/** @brief Returns a static extents type selected from a static extents type using std::index_sequence
 *
 * @code using extents_t = select<extents<4,3,2>,std::index_sequence<0,2>>; // -> extents<4,2>
 *
 * @tparam E extents<...>
 * @tparam S std::index_sequence<...>
*/
template<class E, class S>
using select_t = typename detail::select_impl_t<std::decay_t<E>, S>::type;


//////////////// BINARY PLUS OP /////////////////

template<std::size_t i, std::size_t j>
struct plus_t { static constexpr auto value = i+j; };

template<std::size_t i, std::size_t j>
constexpr inline auto plus_v = plus_t<i,j>::value;

template<std::size_t i, std::size_t j>
struct multiply_t { static constexpr auto value = i*j;};



//////////////// SET /////////////////////

namespace detail {
template<std::size_t, std::size_t, class>
struct set_impl_t;

template<std::size_t j, std::size_t e, std::size_t ... es>
struct set_impl_t<j,e,extents<es...>>
{
  static constexpr inline auto n = size_v<extents<es...>>;
  template<std::size_t k> using plus_j1 = plus_t<k,j+1>;

  using head_indices = std::make_index_sequence<j>;
  using tail_indices = for_each_t<std::make_index_sequence<n-j-1>,plus_j1>;

  using head = select_t<extents<es...>,head_indices>;
  using tail = select_t<extents<es...>,tail_indices>;
  using type = cat_t<cat_t<head,extents<e>>,tail>;
};
} // namespace detail

/** @brief Sets the j-th element of a pure static extents type with 0 <= j < size_v<extents>
 *
 * @code using extents_t = set_t<2,5,extents<4,3,2>>; // extents<4,3,5>
 *
 * @tparam j j-th position in extents with 0 <= j < size_v<extents>
 * @tparam e value to replace the j-th element
 * @tparam E extents<k,l,...,n>
*/
template<std::size_t j, std::size_t e, class E>
using set_t = typename detail::set_impl_t<j,e,std::decay_t<E>>::type;



//////////////// REVERSE //////////////////

namespace detail {
template<class, class>
struct reverse_impl_t;

template<class E, std::size_t ... js>
struct reverse_impl_t<E, std::index_sequence<js...>>
{
  using type = extents < ( get_v<E,size_v<E>-js-1>) ...  >;
};
} // namespace detail

/** @brief Reverses static extents of a static extents type
 *
 * @code using extents_t = reverse_t<extents<4,3,2>>; // -> extents<2,3,4>
 *
 * @tparam E extents<...>
*/
template<class E>
using reverse_t = typename detail::reverse_impl_t<std::decay_t<E>, std::make_index_sequence<size_v<E>>>::type;


//////////////// REMOVE //////////////////

namespace detail{
template<std::size_t k, class E>
struct remove_element_impl_t
{
  static constexpr auto n = E::size();
  using head = select_t<E, std::make_index_sequence<k-1> >;

  template<class>
  struct tail_indices;
  template<std::size_t ... is>
  struct tail_indices<std::index_sequence<is...>>
  {
    using type = extents< (is+k+1) ... >;
  };
  using tail = select_t<E, typename tail_indices<std::make_index_sequence<n-k-1>>::type>;
  using type = cat_t< head, tail>;
};
} // namespace detail

/** @brief Removes a static extent of a static extents type
 *
 * @code using extents_t = remove<1,extents<4,3,2>>; // -> extents<4,2>
 * @note it is a special case of the select function
 *
 * @tparam k zero-based index
 * @tparam E extents<...>
*/
template<std::size_t k, class E>
using remove_element_t = typename detail::remove_element_impl_t<k,std::decay_t<E>>::type;

} //namespace boost::numeric::ublas

#endif // BOOST_NUMERIC_UBLAS_TENSOR_EXTENTS_STATIC_FUNCTIONS_HPP


