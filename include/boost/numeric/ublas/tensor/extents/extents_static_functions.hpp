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



//////////////// ACCUMULATE /////////////////////

namespace detail {

template<class, std::size_t, template<std::size_t,std::size_t> class>
struct accumulate_impl_t;

template<template<std::size_t,std::size_t> class op, std::size_t i>
struct accumulate_impl_t<extents<>, i, op>
{ static constexpr auto value = i; };

template<template<std::size_t,std::size_t> class op, std::size_t i, std::size_t e>
struct accumulate_impl_t<extents<e>, i, op>
{ static constexpr auto value = op<e,i>::value; };

template<template<std::size_t,std::size_t> class op, std::size_t i, std::size_t e, std::size_t ... es>
struct accumulate_impl_t<extents<e,es...>, i, op>
{
  using next = accumulate_impl_t<extents<es...>,i,op>;
  static constexpr auto value = op<e,next::value>::value;

};
} // namespace detail

template<class E, std::size_t I, template<std::size_t,std::size_t> class BinaryOp>
constexpr inline auto accumulate_v = detail::accumulate_impl_t<reverse_t<std::decay_t<E>>,I,BinaryOp>::value;


//////////////// Product /////////////////////

namespace detail {

template<class E>
struct product_impl_t
{
  static constexpr auto value = empty_v<E> ? 0UL : accumulate_v<E,1UL,multiply_t>;
};

} // namespace detail
template<class E>
constexpr inline auto product_v = detail::product_impl_t<std::decay_t<E>>::value;


//////////////// ALL_OF /////////////////////


namespace detail {

template<class>
struct all_of_impl_t;
template<>
struct all_of_impl_t<std::integer_sequence<bool>>
{ static constexpr bool value = true; };
template<bool e, bool ... es>
struct all_of_impl_t<std::integer_sequence<bool,e,es...>>
{ static constexpr bool value = ( e && ... && es ); };

} // namespace detail

/** @brief Returns true if all elements of Extents satisfy UnaryOp
 *
 * @code constexpr auto e_j = all_of_v<extents<4,3,2>>;
*/
template<class E, template<std::size_t> class UnaryPred>
constexpr inline bool all_of_v = detail::all_of_impl_t<for_each_test_t<std::decay_t<E>,UnaryPred>>::value;


//////////////// ALL_OF /////////////////////

namespace detail {
template<class>
struct any_of_impl_t;
template<>
struct any_of_impl_t<std::integer_sequence<bool>>
{ static constexpr bool value = true;};

template<bool e, bool ... es>
struct any_of_impl_t<std::integer_sequence<bool,e,es...>>
{ static constexpr bool value = ( e || ... || es ); };

} // namespace detail

template<class E, template<std::size_t> class UnaryOp>
constexpr inline bool any_of_v = detail::any_of_impl_t<for_each_test_t<std::decay_t<E>,UnaryOp>>::value;


//////////////// IS_VALID /////////////////////

namespace detail {

template<class>
struct is_valid_impl_t             { static constexpr bool value = false; };
template<>
struct is_valid_impl_t<extents<>>  { static constexpr bool value = true ; };

template<std::size_t e, std::size_t ... es>
struct is_valid_impl_t<extents<e,es...>>
{
  template<std::size_t n>
  struct greater_than_zero { static constexpr auto value = (n>0ul);  };

  static constexpr bool value = all_of_v<extents<e,es...>,greater_than_zero >;
};
} // namespace detail

/** @brief Returns true if extents equals ([m,n,...,l]) with m>0,n>0,...,l>0  */
template<class E>
constexpr inline bool is_valid_v = detail::is_valid_impl_t<std::decay_t<E>>::value;



//////////////// IS_SCALAR /////////////////////

namespace detail {
template<class E>
struct is_scalar_impl_t
{
  template<std::size_t n>
  struct equal_to_one { static constexpr auto value = (n == 1ul); };

  static constexpr bool value = is_valid_v<E> &&
                                !empty_v<E> &&
                                all_of_v<E,equal_to_one>;
};
} // namespace detail

/** @brief Returns true if extents equals (m,[n,...,l]) with m=1,n=1,...,l=1  */
template<class E>
constexpr inline bool is_scalar_v = detail::is_scalar_impl_t<std::decay_t<E>>::value;


//////////////// IS_VECTOR /////////////////////

namespace detail {

template<class>
struct is_vector_impl_t { static constexpr bool value = false; };
template<>
struct is_vector_impl_t<extents<>>  { static constexpr bool value = false; };

template<std::size_t e>
struct is_vector_impl_t<extents<e>> { static constexpr bool value = (e>=1); };

template<std::size_t e1, std::size_t e2, std::size_t ... es>
struct is_vector_impl_t<extents<e1,e2,es...>>
{
  template<std::size_t n> struct equal_to_one      { static constexpr auto value = (n == 1ul); };
  template<std::size_t n> struct greater_than_zero { static constexpr auto value = (n >  0ul);  };

  static constexpr bool value =
    is_valid_v <extents<e1,e2,es...>>             &&
    any_of_v   <extents<e1,e2>,greater_than_zero> &&
    any_of_v   <extents<e1,e2>,equal_to_one     > &&
    all_of_v   <extents<es...>,equal_to_one     >;
};


} // namespace detail

/** @brief Returns true if extents equals (m,[n,1,...,1]) with m>=1||n>=1 && m==1||n==1*/
template<class E>
constexpr inline bool is_vector_v = detail::is_vector_impl_t<std::decay_t<E>>::value;



//////////////// IS_MATRIX /////////////////////

namespace detail {

template<class>
struct is_matrix;

template<>
struct is_matrix<extents<>>  { static constexpr bool value = false; };

template<std::size_t e>
struct is_matrix<extents<e>> { static constexpr bool value = true; };

template<std::size_t e1, std::size_t e2, std::size_t ... es>
struct is_matrix<extents<e1,e2,es...>>
{
  template<std::size_t n> struct equal_to_one      { static constexpr auto value = (n == 1ul); };
  template<std::size_t n> struct greater_than_zero { static constexpr auto value = (n >  0ul);  };

  static constexpr bool value =
    is_valid_v <extents<e1,e2,es...>>              &&
    all_of_v   <extents<e1,e2>,greater_than_zero > &&
    all_of_v   <extents<es...>,equal_to_one      >;
};


} // namespace detail

/** @brief Returns true if (m,n,[1,...,1]) with m>=1 or n>=1 */
template<class E>
constexpr inline bool is_matrix_v = detail::is_matrix<std::decay_t<E>>::value;


//////////////// IS_TENSOR /////////////////////

namespace detail {

template<class>
struct is_tensor;

template<>
struct is_tensor<extents<>>  { static constexpr bool value = false; };

template<std::size_t e>
struct is_tensor<extents<e>> { static constexpr bool value = false; };

template<std::size_t e1, std::size_t e2, std::size_t ... es>
struct is_tensor<extents<e1,e2,es...>>
{
  template<std::size_t n>
  struct greater_than_one { static constexpr auto value = (n > 1ul);  };

  static constexpr bool value =
    is_valid_v  <extents<e1,e2,es...>>       &&
    size_v      <extents<e1,e2,es...>> > 2ul &&
    any_of_v    <extents<es...>,greater_than_one >;
};


} // namespace detail

/** @brief Returns true if extents is equal to (m,n,[1,...,1],k,[1,...,1]) with k > 1 */
template<class E>
constexpr inline bool is_tensor_v = detail::is_tensor<std::decay_t<E>>::value;


//////////////// ARRAY_CONVERSION /////////////////////

namespace detail {
template<class>
struct to_array_impl_t;

template<std::size_t ... is>
struct to_array_impl_t<std::index_sequence<is...>>
{ static constexpr auto value = std::array<std::size_t,sizeof...(is)>{is... }; };

template<std::size_t ... is>
struct to_array_impl_t<extents<is...>>
{ static constexpr auto value = std::array<std::size_t,sizeof...(is)>{is... }; };

} // namespace detail

template<class E>
constexpr inline auto to_array_v = detail::to_array_impl_t<std::decay_t<E>>::value;




namespace detail {

template<class, class, class>
struct to_strides_impl_t;

template<class E, class L, std::size_t ... is>
struct to_strides_impl_t<E, L, std::index_sequence<is...> >
{
  static_assert (is_valid_v<E>);

  static constexpr bool is_first_order = std::is_same_v<L,layout::first_order>;
  using adjusted_extents = std::conditional_t<is_first_order,E,reverse_t<E>>;

  template<std::size_t i>
  static constexpr std::size_t selected_product = product_v<select_t<adjusted_extents,std::make_index_sequence<i>>>;

  using pre_type = extents <1,( selected_product<is+1> ) ... >;
  using type = std::conditional_t<is_first_order,pre_type,reverse_t<pre_type>>;
};

} // namespace detail

template<class E, class L>
using to_strides_impl_t = typename detail::to_strides_impl_t<E,L,std::make_index_sequence<size_v<E>-1>>::type;

template<class E, class L>
constexpr inline auto to_strides_v = to_array_v<to_strides_impl_t<std::decay_t<E>,L>>;

} //namespace boost::numeric::ublas


template <
  std::size_t l1,
  std::size_t l2,
  std::size_t r1,
  std::size_t r2,
  std::size_t ... l,
  std::size_t ... r>
[[nodiscard]] inline constexpr bool operator==(
  boost::numeric::ublas::extents<l1,l2,l...> /*unused*/,
  boost::numeric::ublas::extents<r1,r2,r...> /*unused*/)
{
  return std::is_same_v<
    boost::numeric::ublas::extents<l1,l2,l...>,
    boost::numeric::ublas::extents<r1,r2,r...>>;
}

template <
  std::size_t l1,
  std::size_t l2,
  std::size_t r1,
  std::size_t r2,
  std::size_t ... l,
  std::size_t ... r>
[[nodiscard]] inline constexpr  bool operator!=(
  boost::numeric::ublas::extents<l1,l2,l...> el,
  boost::numeric::ublas::extents<r1,r2,r...> er)
{
  return !(el == er);
}

#endif // BOOST_NUMERIC_UBLAS_TENSOR_EXTENTS_STATIC_FUNCTIONS_HPP


