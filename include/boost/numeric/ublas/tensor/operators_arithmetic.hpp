//
//  Copyright (c) 2018, Cem Bassoy, cem.bassoy@gmail.com
// 	Copyright (c) 2022, Amit Singh, amitsingh19975@gmail.com
//
//  Distributed under the Boost Software License, Version 1.0. (See
//  accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)
//
//  The authors gratefully acknowledge the support of
//  Fraunhofer IOSB, Ettlingen, Germany
//

#ifndef BOOST_UBLAS_TENSOR_OPERATORS_ARITHMETIC_HPP
#define BOOST_UBLAS_TENSOR_OPERATORS_ARITHMETIC_HPP

#include "expression.hpp"
#include "expression_evaluation.hpp"
#include "multi_index_utility.hpp"
#include "functions.hpp"
#include "type_traits.hpp"

#include <type_traits>
#include <functional>
#include <algorithm>

namespace boost::numeric::ublas
{

template<class T>
class tensor_core;

template<class E>
class matrix_expression;


template<class E>
class vector_expression;

namespace detail{
  
  template<typename E>
  struct real_expression_type {
    using type = E;
  };
  
  template<typename T, typename E>
  struct real_expression_type< tensor_expression<T,E> > {
    using type = T;
  };
  
  template<typename T, typename EL, typename ER, typename OP>
  struct real_expression_type< binary_tensor_expression<T, EL, ER, OP> > {
    using type = T;
  };
  
  template<typename T, typename E, typename OP>
  struct real_expression_type< unary_tensor_expression<T, E, OP> > {
    using type = T;
  };

  template<typename E>
  using real_expression_type_t = typename real_expression_type< std::decay_t<E> >::type;

} // namespace detail


} // namespace boost::numeric::ublas


template <typename EL, typename ER>
  requires ( 
    (boost::numeric::ublas::detail::TensorExpression<EL>) &&
    (boost::numeric::ublas::detail::VectorExpression<ER> || boost::numeric::ublas::detail::MatrixExpression<ER>)
  )
inline
  constexpr auto operator*( EL&& lhs, ER&& rhs ) noexcept
{
    using tensor_type = boost::numeric::ublas::detail::real_expression_type_t<EL>;
    using value_type  = typename tensor_type::value_type;

    return boost::numeric::ublas::detail::make_binary_tensor_expression<tensor_type>(
      std::forward<EL>(lhs), std::forward<ER>(rhs), std::multiplies<value_type>{}
    );
}

template <typename EL, typename ER>
  requires ( 
    (boost::numeric::ublas::detail::TensorExpression<EL>) &&
    (boost::numeric::ublas::detail::VectorExpression<ER> || boost::numeric::ublas::detail::MatrixExpression<ER>)
  )
inline
  constexpr auto operator+( EL&& lhs, ER&& rhs ) noexcept
{
    using tensor_type = boost::numeric::ublas::detail::real_expression_type_t<EL>;
    using value_type  = typename tensor_type::value_type;

    return boost::numeric::ublas::detail::make_binary_tensor_expression<tensor_type>(
      std::forward<EL>(lhs), std::forward<ER>(rhs), std::plus<value_type>{}
    );
}

template <typename EL, typename ER>
  requires ( 
    (boost::numeric::ublas::detail::TensorExpression<EL>) &&
    (boost::numeric::ublas::detail::VectorExpression<ER> || boost::numeric::ublas::detail::MatrixExpression<ER>)
  )
inline
  constexpr auto operator-( EL&& lhs, ER&& rhs ) noexcept
{
    using tensor_type = boost::numeric::ublas::detail::real_expression_type_t<EL>;
    using value_type  = typename tensor_type::value_type;

    return boost::numeric::ublas::detail::make_binary_tensor_expression<tensor_type>(
      std::forward<EL>(lhs), std::forward<ER>(rhs), std::minus<value_type>{}
    );
}

template <typename EL, typename ER>
  requires ( 
    (boost::numeric::ublas::detail::TensorExpression<EL>) &&
    (boost::numeric::ublas::detail::VectorExpression<ER> || boost::numeric::ublas::detail::MatrixExpression<ER>)
  )
inline
  constexpr auto operator/( EL&& lhs, ER&& rhs ) noexcept
{
    using tensor_type = boost::numeric::ublas::detail::real_expression_type_t<EL>;
    using value_type  = typename tensor_type::value_type;

    return boost::numeric::ublas::detail::make_binary_tensor_expression<tensor_type>(
      std::forward<EL>(lhs), std::forward<ER>(rhs), std::divides<value_type>{}
    );
}

template <typename EL, typename ER>
  requires ( 
    (boost::numeric::ublas::detail::VectorExpression<EL> || boost::numeric::ublas::detail::MatrixExpression<EL>) &&
    (boost::numeric::ublas::detail::TensorExpression<ER>)
  )
inline
  constexpr auto operator*( EL&& lhs, ER&& rhs ) noexcept
{
    using tensor_type = boost::numeric::ublas::detail::real_expression_type_t<ER>;
    using value_type  = typename tensor_type::value_type;

    return boost::numeric::ublas::detail::make_binary_tensor_expression<tensor_type>(
      std::forward<EL>(lhs), std::forward<ER>(rhs), std::multiplies<value_type>{}
    );
}

template <typename EL, typename ER>
  requires ( 
    (boost::numeric::ublas::detail::VectorExpression<EL> || boost::numeric::ublas::detail::MatrixExpression<EL>) &&
    (boost::numeric::ublas::detail::TensorExpression<ER>)
  )
inline
  constexpr auto operator+( EL&& lhs, ER&& rhs ) noexcept
{
    using tensor_type = boost::numeric::ublas::detail::real_expression_type_t<ER>;
    using value_type  = typename tensor_type::value_type;

    return boost::numeric::ublas::detail::make_binary_tensor_expression<tensor_type>(
      std::forward<EL>(lhs), std::forward<ER>(rhs), std::plus<value_type>{}
    );
}

template <typename EL, typename ER>
  requires ( 
    (boost::numeric::ublas::detail::VectorExpression<EL> || boost::numeric::ublas::detail::MatrixExpression<EL>) &&
    (boost::numeric::ublas::detail::TensorExpression<ER>)
  )
inline
  constexpr auto operator-( EL&& lhs, ER&& rhs ) noexcept
{
    using tensor_type = boost::numeric::ublas::detail::real_expression_type_t<ER>;
    using value_type  = typename tensor_type::value_type;

    return boost::numeric::ublas::detail::make_binary_tensor_expression<tensor_type>(
      std::forward<EL>(lhs), std::forward<ER>(rhs), std::minus<value_type>{}
    );
}

template <typename EL, typename ER>
  requires ( 
    (boost::numeric::ublas::detail::VectorExpression<EL> || boost::numeric::ublas::detail::MatrixExpression<EL>) &&
    (boost::numeric::ublas::detail::TensorExpression<ER>)
  )
inline constexpr auto operator/( EL&& lhs, ER&& rhs ) noexcept
{
    using tensor_type = boost::numeric::ublas::detail::real_expression_type_t<ER>;
    using value_type  = typename tensor_type::value_type;

    return boost::numeric::ublas::detail::make_binary_tensor_expression<tensor_type>(
      std::forward<EL>(lhs), std::forward<ER>(rhs), std::divides<value_type>{}
    );
}

template<typename EL, typename ER>
  requires ( 
    (boost::numeric::ublas::detail::TensorExpression<EL>) &&
    (boost::numeric::ublas::detail::TensorExpression<ER>)
  )
inline constexpr auto operator+( EL&& lhs, ER&& rhs ) noexcept
{

  using ltensor_t = boost::numeric::ublas::detail::real_expression_type_t<EL>;
  using rtensor_t = boost::numeric::ublas::detail::real_expression_type_t<ER>;
  using lvalue_t  = typename ltensor_t::value_type;
  using rvalue_t  = typename rtensor_t::value_type;

  static_assert( std::is_same_v< lvalue_t, rvalue_t >,
                "operator+() : LHS tensor and RHS tensor should have the same value type"
                );

  return boost::numeric::ublas::detail::make_binary_tensor_expression<ltensor_t> (
    std::forward<EL>(lhs), std::forward<ER>(rhs), std::plus<lvalue_t>{}
  );
}


template<typename EL, typename ER>
  requires ( 
    (boost::numeric::ublas::detail::TensorExpression<EL>) &&
    (boost::numeric::ublas::detail::TensorExpression<ER>)
  )
inline constexpr auto operator-( EL&& lhs, ER&& rhs ) noexcept
{

  using ltensor_t = boost::numeric::ublas::detail::real_expression_type_t<EL>;
  using rtensor_t = boost::numeric::ublas::detail::real_expression_type_t<ER>;
  using lvalue_t  = typename ltensor_t::value_type;
  using rvalue_t  = typename rtensor_t::value_type;

  static_assert( std::is_same_v< lvalue_t, rvalue_t >,
                "operator-() : LHS tensor and RHS tensor should have the same value type"
                );

  return boost::numeric::ublas::detail::make_binary_tensor_expression<ltensor_t> (
    std::forward<EL>(lhs), std::forward<ER>(rhs), std::minus<lvalue_t>{}
  );
}

template<typename EL, typename ER>
  requires ( 
    (boost::numeric::ublas::detail::TensorExpression<EL>) &&
    (boost::numeric::ublas::detail::TensorExpression<ER>)
  )
inline constexpr auto operator*( EL&& lhs, ER&& rhs ) noexcept
{

  using ltensor_t = boost::numeric::ublas::detail::real_expression_type_t<EL>;
  using rtensor_t = boost::numeric::ublas::detail::real_expression_type_t<ER>;
  using lvalue_t  = typename ltensor_t::value_type;
  using rvalue_t  = typename rtensor_t::value_type;

  static_assert( std::is_same_v< lvalue_t, rvalue_t >,
                "operator*() : LHS tensor and RHS tensor should have the same value type"
                );

  return boost::numeric::ublas::detail::make_binary_tensor_expression<ltensor_t> (
    std::forward<EL>(lhs), std::forward<ER>(rhs), std::multiplies<lvalue_t>{}
  );
}

template<typename EL, typename ER>
  requires ( 
    (boost::numeric::ublas::detail::TensorExpression<EL>) &&
    (boost::numeric::ublas::detail::TensorExpression<ER>)
  )
inline constexpr auto operator/( EL&& lhs, ER&& rhs ) noexcept
{

  using ltensor_t = boost::numeric::ublas::detail::real_expression_type_t<EL>;
  using rtensor_t = boost::numeric::ublas::detail::real_expression_type_t<ER>;
  using lvalue_t  = typename ltensor_t::value_type;
  using rvalue_t  = typename rtensor_t::value_type;

  static_assert( std::is_same_v< lvalue_t, rvalue_t >,
                "operator/() : LHS tensor and RHS tensor should have the same value type"
                );

  return boost::numeric::ublas::detail::make_binary_tensor_expression<ltensor_t> (
    std::forward<EL>(lhs), std::forward<ER>(rhs), std::divides<lvalue_t>{}
  );
}


// Overloaded Arithmetic Operators with Scalars
template<typename ER>
  requires (boost::numeric::ublas::detail::TensorExpression<ER>)
inline constexpr auto operator+( 
  typename boost::numeric::ublas::detail::real_expression_type_t<ER>::value_type lhs, 
  ER&& rhs 
) noexcept {
  using tensor_type = boost::numeric::ublas::detail::real_expression_type_t<ER>;
  using value_type  = typename tensor_type::value_type;
  
  return boost::numeric::ublas::detail::make_unary_tensor_expression<tensor_type> (
    std::forward<ER>(rhs),
    [lhs](value_type const& r){ return lhs + r; }
  );
}

template<typename ER>
  requires (boost::numeric::ublas::detail::TensorExpression<ER>)
inline constexpr auto operator-( 
  typename boost::numeric::ublas::detail::real_expression_type_t<ER>::value_type lhs, 
  ER&& rhs
) noexcept {
  using tensor_type = boost::numeric::ublas::detail::real_expression_type_t<ER>;
  using value_type  = typename tensor_type::value_type;

  return boost::numeric::ublas::detail::make_unary_tensor_expression<tensor_type> (
    std::forward<ER>(rhs),
    [lhs](value_type const& r){ return lhs - r; }
  );
}

template<typename ER>
  requires (boost::numeric::ublas::detail::TensorExpression<ER>)
inline constexpr auto operator*( 
  typename boost::numeric::ublas::detail::real_expression_type_t<ER>::value_type lhs, 
  ER&& rhs
) noexcept {
  using tensor_type = boost::numeric::ublas::detail::real_expression_type_t<ER>;
  using value_type  = typename tensor_type::value_type;

  return boost::numeric::ublas::detail::make_unary_tensor_expression<tensor_type> (
    std::forward<ER>(rhs),
    [lhs](value_type const& r){ return lhs * r; }
  );
}

template<typename ER>
  requires (boost::numeric::ublas::detail::TensorExpression<ER>)
inline constexpr auto operator/( 
  typename boost::numeric::ublas::detail::real_expression_type_t<ER>::value_type lhs, 
  ER&& rhs
) noexcept {
  using tensor_type = boost::numeric::ublas::detail::real_expression_type_t<ER>;
  using value_type  = typename tensor_type::value_type;

  return boost::numeric::ublas::detail::make_unary_tensor_expression<tensor_type> (
    std::forward<ER>(rhs),
    [lhs](value_type const& r){ return lhs / r; }
  );
}

template<typename EL>
  requires (boost::numeric::ublas::detail::TensorExpression<EL>)
inline constexpr auto operator+( 
  EL&& lhs, 
  typename boost::numeric::ublas::detail::real_expression_type_t<EL>::value_type rhs 
) noexcept {
  using tensor_type = boost::numeric::ublas::detail::real_expression_type_t<EL>;
  using value_type  = typename tensor_type::value_type;
  
  return boost::numeric::ublas::detail::make_unary_tensor_expression<tensor_type> (
    std::forward<EL>(lhs),
    [rhs] (value_type const& l) { return l + rhs; } 
  );
}

template<typename EL>
  requires (boost::numeric::ublas::detail::TensorExpression<EL>)
inline constexpr auto operator-( 
  EL&& lhs, 
  typename boost::numeric::ublas::detail::real_expression_type_t<EL>::value_type rhs 
) noexcept {
  using tensor_type = boost::numeric::ublas::detail::real_expression_type_t<EL>;
  using value_type  = typename tensor_type::value_type;
  
  return boost::numeric::ublas::detail::make_unary_tensor_expression<tensor_type> (
    std::forward<EL>(lhs),
    [rhs] (value_type const& l) { return l - rhs; } 
  );
}

template<typename EL>
  requires (boost::numeric::ublas::detail::TensorExpression<EL>)
inline constexpr auto operator*( 
  EL&& lhs, 
  typename boost::numeric::ublas::detail::real_expression_type_t<EL>::value_type rhs 
) noexcept {
  using tensor_type = boost::numeric::ublas::detail::real_expression_type_t<EL>;
  using value_type  = typename tensor_type::value_type;
  
  return boost::numeric::ublas::detail::make_unary_tensor_expression<tensor_type> (
    std::forward<EL>(lhs),
    [rhs] (value_type const& l) { return l * rhs; } 
  );
}

template<typename EL>
  requires (boost::numeric::ublas::detail::TensorExpression<EL>)
inline constexpr auto operator/( 
  EL&& lhs, 
  typename boost::numeric::ublas::detail::real_expression_type_t<EL>::value_type rhs 
) noexcept {
  using tensor_type = boost::numeric::ublas::detail::real_expression_type_t<EL>;
  using value_type  = typename tensor_type::value_type;
  
  return boost::numeric::ublas::detail::make_unary_tensor_expression<tensor_type> (
    std::forward<EL>(lhs),
    [rhs] (value_type const& l) { return l / rhs; } 
  );
}

template<class T, class D>
inline constexpr auto& operator += (
  boost::numeric::ublas::tensor_core<T>& lhs,
  boost::numeric::ublas::detail::tensor_expression<boost::numeric::ublas::tensor_core<T>,D> const& expr
){
  using value_type  = typename boost::numeric::ublas::tensor_core<T>::value_type;
  boost::numeric::ublas::detail::eval(lhs, expr(), [](value_type& l, value_type const& r) { l+=r; } );
  return lhs;
}

template<class T, class D>
inline constexpr auto& operator -= (
  boost::numeric::ublas::tensor_core<T>& lhs,
  const boost::numeric::ublas::detail::tensor_expression<boost::numeric::ublas::tensor_core<T>,D> &expr
){
  using value_type  = typename boost::numeric::ublas::tensor_core<T>::value_type;
  boost::numeric::ublas::detail::eval(lhs, expr(), [](value_type& l, value_type const& r) { l-=r; } );
  return lhs;
}

template<class T, class D>
inline constexpr auto& operator *= (
  boost::numeric::ublas::tensor_core<T>& lhs,
  const boost::numeric::ublas::detail::tensor_expression<boost::numeric::ublas::tensor_core<T>,D> &expr
){
  using value_type  = typename boost::numeric::ublas::tensor_core<T>::value_type;
  boost::numeric::ublas::detail::eval(lhs, expr(), [](value_type& l, value_type const& r) { l*=r; } );
  return lhs;
}

template<class T, class D>
inline constexpr auto& operator /= (
  boost::numeric::ublas::tensor_core<T>& lhs,
  const boost::numeric::ublas::detail::tensor_expression<boost::numeric::ublas::tensor_core<T>,D> &expr
){
  using value_type  = typename boost::numeric::ublas::tensor_core<T>::value_type;
  boost::numeric::ublas::detail::eval(lhs, expr(), [](value_type& l, value_type const& r) { l/=r; } );
  return lhs;
}




template<class TensorEngine>
inline constexpr auto& operator += (
    boost::numeric::ublas::tensor_core<TensorEngine>& lhs,
    typename boost::numeric::ublas::tensor_core<TensorEngine>::value_type r
){
  using value_type  = typename boost::numeric::ublas::tensor_core<TensorEngine>::value_type;
  boost::numeric::ublas::detail::eval(lhs, [r](value_type& l) { l+=r; } );
  return lhs;
}

template<class TensorEngine>
inline constexpr auto& operator -= (
  boost::numeric::ublas::tensor_core<TensorEngine>& lhs,
  typename boost::numeric::ublas::tensor_core<TensorEngine>::value_type r
){
  using value_type  = typename boost::numeric::ublas::tensor_core<TensorEngine>::value_type;
  boost::numeric::ublas::detail::eval(lhs, [r](value_type& l) { l-=r; } );
  return lhs;
}

template<class TensorEngine>
inline constexpr auto& operator *= (
  boost::numeric::ublas::tensor_core<TensorEngine>& lhs,
  typename boost::numeric::ublas::tensor_core<TensorEngine>::value_type r
){
  using value_type  = typename boost::numeric::ublas::tensor_core<TensorEngine>::value_type;
  boost::numeric::ublas::detail::eval(lhs, [r](value_type& l) { l*=r; } );
  return lhs;
}

template<class TensorEngine>
inline constexpr auto& operator /= (
  boost::numeric::ublas::tensor_core<TensorEngine>& lhs,
  typename boost::numeric::ublas::tensor_core<TensorEngine>::value_type r
){
  using value_type  = typename boost::numeric::ublas::tensor_core<TensorEngine>::value_type;
  boost::numeric::ublas::detail::eval(lhs, [r](value_type& l) { l/=r; } );
  return lhs;
}



template<class T, class D>
inline constexpr
  auto const& operator +(boost::numeric::ublas::detail::tensor_expression<T,D> const& lhs) noexcept{
  return lhs;
}

template<typename E>
  requires boost::numeric::ublas::detail::TensorExpression<E>
inline constexpr auto operator -(E&& e) {
  using tensor_type = boost::numeric::ublas::detail::real_expression_type_t<E>;
  using value_type  = typename tensor_type::value_type;
  return boost::numeric::ublas::detail::make_unary_tensor_expression<tensor_type> (
    std::forward<E>(e), std::negate<value_type>{} 
  );
}





/** @brief Performs a tensor contraction, not an elementwise multiplication
    *
*/

template<class tensor_type_left, class tuple_type_left, class tensor_type_right, class tuple_type_right>
auto operator*(
  std::pair< tensor_type_left  const&, tuple_type_left  > lhs,
  std::pair< tensor_type_right const&, tuple_type_right > rhs)
{

  namespace ublas = boost::numeric::ublas;

  auto const& tensor_left  = lhs.first;
  auto const& tensor_right = rhs.first;

  auto multi_index_left  = lhs.second;
  auto multi_index_right = rhs.second;

  static constexpr auto num_equal_ind = ublas::number_equal_indexes<tuple_type_left, tuple_type_right>::value;

  if constexpr ( num_equal_ind == 0  ){
    return tensor_left * tensor_right;
  }
  else if constexpr ( num_equal_ind==std::tuple_size<tuple_type_left>::value && std::is_same<tuple_type_left, tuple_type_right>::value ){

    return ublas::inner_prod( tensor_left, tensor_right );
  }
  else {
    auto index_pairs = ublas::index_position_pairs(multi_index_left,multi_index_right);
    constexpr auto size = std::tuple_size_v<decltype(index_pairs)>;

    using extents_left_type  = typename tensor_type_left ::extents_type;
    using extents_right_type = typename tensor_type_right::extents_type;

    constexpr bool has_dynamic_extents = ublas::is_dynamic_rank_v<extents_left_type> || ublas::is_dynamic_rank_v<extents_right_type>;

    using index_tuple = std::conditional_t<has_dynamic_extents, std::vector<std::size_t>, std::array<std::size_t, size>>;

    auto phi_left  = index_tuple{};
    auto phi_right = index_tuple{};

    if constexpr(has_dynamic_extents) {
      phi_left .resize(size);
      phi_right.resize(size);
    }

    std::transform(index_pairs.begin(), index_pairs.end(), phi_left .begin(), [](auto a){ return a.first  +1ul; } );
    std::transform(index_pairs.begin(), index_pairs.end(), phi_right.begin(), [](auto b){ return b.second +1ul; } );

//    auto index_pairs = ublas::array_to_vector(  array_index_pairs  );
    return ublas::prod( tensor_left, tensor_right, phi_left, phi_right );
  }

}

#endif
