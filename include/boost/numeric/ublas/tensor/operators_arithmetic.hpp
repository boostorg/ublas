//
//  Copyright (c) 2018, Cem Bassoy, cem.bassoy@gmail.com
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

} // namespace boost::numeric::ublas


template <class T, class L, class R>
inline
  constexpr auto operator*(
    boost ::numeric ::ublas ::detail ::tensor_expression<T, L> const& lhs,
    boost ::numeric ::ublas ::vector_expression<R> const& rhs) noexcept
{
  return boost ::numeric ::ublas ::detail ::make_binary_tensor_expression<T>(
    lhs(), rhs(), std::multiplies<>{});
}

template <class T, class L, class R>
inline
  constexpr auto operator+(
    boost ::numeric ::ublas ::detail ::tensor_expression<T, L> const& lhs,
    boost ::numeric ::ublas ::vector_expression<R> const& rhs) noexcept
{
  return boost ::numeric ::ublas ::detail ::make_binary_tensor_expression<T>(
    lhs(), rhs(), std::plus<>{});
}

template <class T, class L, class R>
inline
  constexpr auto operator-(
    boost ::numeric ::ublas ::detail ::tensor_expression<T, L> const& lhs,
    boost ::numeric ::ublas ::vector_expression<R> const& rhs) noexcept
{
  return boost ::numeric ::ublas ::detail ::make_binary_tensor_expression<T>(
    lhs(), rhs(), std::minus<>{});
}

template <class T, class L, class R>
inline
  constexpr auto operator/(
    boost ::numeric ::ublas ::detail ::tensor_expression<T, L> const& lhs,
    boost ::numeric ::ublas ::vector_expression<R> const& rhs) noexcept
{
  return boost ::numeric ::ublas ::detail ::make_binary_tensor_expression<T>(
    lhs(), rhs(), std::divides<>{});
}


template <class T, class L, class R>
inline
  constexpr auto operator*(
    boost ::numeric ::ublas ::detail ::tensor_expression<T, L> const& lhs,
    boost ::numeric ::ublas ::matrix_expression<R> const& rhs) noexcept
{
  return boost ::numeric ::ublas ::detail ::make_binary_tensor_expression<T>(
    lhs(), rhs(), std::multiplies<>{});
}

template <class T, class L, class R>
inline
  constexpr auto operator+(
    boost ::numeric ::ublas ::detail ::tensor_expression<T, L> const& lhs,
    boost ::numeric ::ublas ::matrix_expression<R> const& rhs) noexcept
{
  return boost ::numeric ::ublas ::detail ::make_binary_tensor_expression<T>(
    lhs(), rhs(), std::plus<>{});
}

template <class T, class L, class R>
inline
  constexpr auto operator-(
    boost ::numeric ::ublas ::detail ::tensor_expression<T, L> const& lhs,
    boost ::numeric ::ublas ::matrix_expression<R> const& rhs) noexcept
{
  return boost ::numeric ::ublas ::detail ::make_binary_tensor_expression<T>(
    lhs(), rhs(), std::minus<>{});
}

template <class T, class L, class R>
inline
  constexpr auto operator/(
    boost ::numeric ::ublas ::detail ::tensor_expression<T, L> const& lhs,
    boost ::numeric ::ublas ::matrix_expression<R> const& rhs) noexcept
{
  return boost ::numeric ::ublas ::detail ::make_binary_tensor_expression<T>(
    lhs(), rhs(), std::divides<>{});
}


template <class T, class L, class R>
inline
  constexpr auto operator*(
    boost ::numeric ::ublas ::vector_expression<L> const& lhs,
    boost ::numeric ::ublas ::detail ::tensor_expression<T, R> const& rhs) noexcept
{
  return boost ::numeric ::ublas ::detail ::make_binary_tensor_expression<T>(
    lhs(), rhs(), std::multiplies<>{});
}

template <class T, class L, class R>
inline
  constexpr auto operator+(
    boost ::numeric ::ublas ::vector_expression<L> const& lhs,
    boost ::numeric ::ublas ::detail ::tensor_expression<T, R> const& rhs) noexcept
{
  return boost ::numeric ::ublas ::detail ::make_binary_tensor_expression<T>(
    lhs(), rhs(), std::plus<>{});
}

template <class T, class L, class R>
inline
  constexpr auto operator-(
    boost ::numeric ::ublas ::vector_expression<L> const& lhs,
    boost ::numeric ::ublas ::detail ::tensor_expression<T, R> const& rhs) noexcept
{
  return boost ::numeric ::ublas ::detail ::make_binary_tensor_expression<T>(
    lhs(), rhs(), std::minus<>{});
}

template <class T, class L, class R>
inline
  constexpr auto operator/(
    boost ::numeric ::ublas ::vector_expression<L> const& lhs,
    boost ::numeric ::ublas ::detail ::tensor_expression<T, R> const& rhs) noexcept
{
  return boost ::numeric ::ublas ::detail ::make_binary_tensor_expression<T>(
    lhs(), rhs(), std::divides<>{});
}


template <class T, class L, class R>
inline
  constexpr auto operator*(
    boost ::numeric ::ublas ::matrix_expression<L> const& lhs,
    boost ::numeric ::ublas ::detail ::tensor_expression<T, R> const& rhs) noexcept
{
  return boost ::numeric ::ublas ::detail ::make_binary_tensor_expression<T>(
    lhs(), rhs(), std::multiplies<>{});
}

template <class T, class L, class R>
inline
  constexpr auto operator+(
    boost ::numeric ::ublas ::matrix_expression<L> const& lhs,
    boost ::numeric ::ublas ::detail ::tensor_expression<T, R> const& rhs) noexcept
{
  return boost ::numeric ::ublas ::detail ::make_binary_tensor_expression<T>(
    lhs(), rhs(), std::plus<>{});
}

template <class T, class L, class R>
inline
  constexpr auto operator-(
    boost ::numeric ::ublas ::matrix_expression<L> const& lhs,
    boost ::numeric ::ublas ::detail ::tensor_expression<T, R> const& rhs) noexcept
{
  return boost ::numeric ::ublas ::detail ::make_binary_tensor_expression<T>(
    lhs(), rhs(), std::minus<>{});
}

template <class T, class L, class R>
inline
  constexpr auto operator/(
    boost ::numeric ::ublas ::matrix_expression<L> const& lhs,
    boost ::numeric ::ublas ::detail ::tensor_expression<T, R> const& rhs) noexcept
{
  return boost ::numeric ::ublas ::detail ::make_binary_tensor_expression<T>(
    lhs(), rhs(), std::divides<>{});
}


template<class T1, class T2, class L, class R>
inline
  constexpr auto operator+( boost::numeric::ublas::detail::tensor_expression<T1,L> const& lhs,
    boost::numeric::ublas::detail::tensor_expression<T2,R> const& rhs) 
{

  static_assert( std::is_same_v< typename T1::value_type, typename T2::value_type>,
                "operator+() : LHS tensor and RHS tensor should have the same value type"
                );

  if constexpr( !std::is_same_v<T1,T2> ){
    auto const& e = boost::numeric::ublas::detail::retrieve_extents(rhs);

    if( !boost::numeric::ublas::detail::all_extents_equal(lhs,e) ){
      throw std::runtime_error("operator+() : LHS tensor and RHS tensor should have equal extents");
    }
  }

  return boost::numeric::ublas::detail::make_binary_tensor_expression<T1> (lhs(), rhs(), [](auto const& l, auto const& r){ return l + r; });
}
template<class T1, class T2, class L, class R>
inline
  constexpr auto operator-( boost::numeric::ublas::detail::tensor_expression<T1,L> const& lhs,
    boost::numeric::ublas::detail::tensor_expression<T2,R> const& rhs) 
{

  static_assert( std::is_same_v< typename T1::value_type, typename T2::value_type>,
                "operator-() : LHS tensor and RHS tensor should have the same value type"
                );

  if constexpr( !std::is_same_v<T1,T2> ){
    auto e = boost::numeric::ublas::detail::retrieve_extents(rhs);

    if( !boost::numeric::ublas::detail::all_extents_equal(lhs,e) ){
      throw std::runtime_error("operator+() : LHS tensor and RHS tensor should have equal extents");
    }
  }

  return boost::numeric::ublas::detail::make_binary_tensor_expression<T1> (lhs(), rhs(), [](auto const& l, auto const& r){ return l - r; });
  //	return boost::numeric::ublas::detail::make_lambda<T>([&lhs,&rhs](std::size_t i){ return lhs(i) - rhs(i);});
}
template<class T1, class T2, class L, class R>
inline
  constexpr auto operator*( boost::numeric::ublas::detail::tensor_expression<T1,L> const& lhs,
    boost::numeric::ublas::detail::tensor_expression<T2,R> const& rhs) 
{

  static_assert( std::is_same_v< typename T1::value_type, typename T2::value_type>,
                "operator*() : LHS tensor and RHS tensor should have the same value type"
                );

  if constexpr( !std::is_same_v<T1,T2> ){
    auto const& e = boost::numeric::ublas::detail::retrieve_extents(rhs);

    if( !boost::numeric::ublas::detail::all_extents_equal(lhs,e) ){
      throw std::runtime_error("operator+() : LHS tensor and RHS tensor should have equal extents");
    }
  }

  return boost::numeric::ublas::detail::make_binary_tensor_expression<T1> (lhs(), rhs(), [](auto const& l, auto const& r){ return l * r; });
}
template<class T1, class T2, class L, class R>
inline
  constexpr auto operator/( boost::numeric::ublas::detail::tensor_expression<T1,L> const& lhs,
    boost::numeric::ublas::detail::tensor_expression<T2,R> const& rhs) 
{

  static_assert( std::is_same_v< typename T1::value_type, typename T2::value_type>,
                "operator/() : LHS tensor and RHS tensor should have the same value type"
                );

  if constexpr( !std::is_same_v<T1,T2> ){
    auto e = boost::numeric::ublas::detail::retrieve_extents(rhs);

    if( !boost::numeric::ublas::detail::all_extents_equal(lhs,e) ){
      throw std::runtime_error("operator+() : LHS tensor and RHS tensor should have equal extents");
    }
  }

  return boost::numeric::ublas::detail::make_binary_tensor_expression<T1> (lhs(), rhs(), std::divides<>{});
}


// Overloaded Arithmetic Operators with Scalars
template<class T, class R>
inline
  constexpr auto operator+(typename boost::numeric::ublas::tensor_core<T>::const_reference lhs,
    boost::numeric::ublas::detail::tensor_expression<boost::numeric::ublas::tensor_core<T>,R> const& rhs) noexcept
{
  using tensor_core_type = boost::numeric::ublas::tensor_core<T>;
  return boost::numeric::ublas::detail::make_unary_tensor_expression<tensor_core_type> (rhs(), [lhs](auto const& r){ return lhs + r; });
}
template<class T, class R>
inline
  constexpr auto operator-(typename boost::numeric::ublas::tensor_core<T>::const_reference lhs,
    boost::numeric::ublas::detail::tensor_expression<boost::numeric::ublas::tensor_core<T>,R> const& rhs) noexcept
{
  using tensor_core_type = boost::numeric::ublas::tensor_core<T>;
  return boost::numeric::ublas::detail::make_unary_tensor_expression<tensor_core_type> (rhs(), [lhs](auto const& r){ return lhs - r; });
}
template<class T, class R>
inline
  constexpr auto operator*(typename boost::numeric::ublas::tensor_core<T>::const_reference lhs,
    boost::numeric::ublas::detail::tensor_expression<boost::numeric::ublas::tensor_core<T>,R> const& rhs) noexcept
{
  using tensor_core_type = boost::numeric::ublas::tensor_core<T>;
  return boost::numeric::ublas::detail::make_unary_tensor_expression<tensor_core_type> (rhs(), [lhs](auto const& r){ return lhs * r; });
}
template<class T, class R>
inline
  constexpr auto operator/(typename boost::numeric::ublas::tensor_core<T>::const_reference lhs,
    boost::numeric::ublas::detail::tensor_expression<boost::numeric::ublas::tensor_core<T>,R> const& rhs) noexcept
{
  using tensor_core_type = boost::numeric::ublas::tensor_core<T>;
  return boost::numeric::ublas::detail::make_unary_tensor_expression<tensor_core_type> (rhs(), [lhs](auto const& r){ return lhs / r; });
}


template<class T, class L>
inline
  constexpr auto operator+(boost::numeric::ublas::detail::tensor_expression<boost::numeric::ublas::tensor_core<T>,L> const& lhs,
    typename boost::numeric::ublas::tensor_core<T>::const_reference rhs) noexcept
{
  using tensor_core_type = boost::numeric::ublas::tensor_core<T>;
  return boost::numeric::ublas::detail::make_unary_tensor_expression<tensor_core_type> (lhs(), [rhs] (auto const& l) { return l + rhs; } );
}
template<class T, class L>
inline
  constexpr auto operator-(boost::numeric::ublas::detail::tensor_expression<boost::numeric::ublas::tensor_core<T>,L> const& lhs,
    typename boost::numeric::ublas::tensor_core<T>::const_reference rhs) noexcept
{
  using tensor_core_type = boost::numeric::ublas::tensor_core<T>;
  return boost::numeric::ublas::detail::make_unary_tensor_expression<tensor_core_type> (lhs(), [rhs] (auto const& l) { return l - rhs; } );
}
template<class T, class L>
inline
  constexpr auto operator*(boost::numeric::ublas::detail::tensor_expression<boost::numeric::ublas::tensor_core<T>,L> const& lhs,
    typename boost::numeric::ublas::tensor_core<T>::const_reference rhs) noexcept
{
  using tensor_core_type = boost::numeric::ublas::tensor_core<T>;
  return boost::numeric::ublas::detail::make_unary_tensor_expression<tensor_core_type> (lhs(), [rhs] (auto const& l) { return l * rhs; } );
}
template<class T, class L>
inline
  constexpr auto operator/(boost::numeric::ublas::detail::tensor_expression<boost::numeric::ublas::tensor_core<T>,L> const& lhs,
    typename boost::numeric::ublas::tensor_core<T>::const_reference rhs) noexcept
{
  using tensor_core_type = boost::numeric::ublas::tensor_core<T>;
  return boost::numeric::ublas::detail::make_unary_tensor_expression<tensor_core_type> (lhs(), [rhs] (auto const& l) { return l / rhs; } );
}



template<class T, class D>
inline
  constexpr auto& operator += (boost::numeric::ublas::tensor_core<T>& lhs,
    const boost::numeric::ublas::detail::tensor_expression<boost::numeric::ublas::tensor_core<T>,D> &expr)
{
  boost::numeric::ublas::detail::eval(lhs, expr(), [](auto& l, auto const& r) { l+=r; } );
  return lhs;
}

template<class T, class D>
inline
  constexpr auto& operator -= (boost::numeric::ublas::tensor_core<T>& lhs,
    const boost::numeric::ublas::detail::tensor_expression<boost::numeric::ublas::tensor_core<T>,D> &expr)
{
  boost::numeric::ublas::detail::eval(lhs, expr(), [](auto& l, auto const& r) { l-=r; } );
  return lhs;
}

template<class T, class D>
inline
  constexpr auto& operator *= (boost::numeric::ublas::tensor_core<T>& lhs,
    const boost::numeric::ublas::detail::tensor_expression<boost::numeric::ublas::tensor_core<T>,D> &expr)
{
  boost::numeric::ublas::detail::eval(lhs, expr(), [](auto& l, auto const& r) { l*=r; } );
  return lhs;
}

template<class T, class D>
inline
  constexpr auto& operator /= (boost::numeric::ublas::tensor_core<T>& lhs,
    const boost::numeric::ublas::detail::tensor_expression<boost::numeric::ublas::tensor_core<T>,D> &expr)
{
  boost::numeric::ublas::detail::eval(lhs, expr(), [](auto& l, auto const& r) { l/=r; } );
  return lhs;
}




template<class TensorEngine>
inline
  constexpr auto& operator += (boost::numeric::ublas::tensor_core<TensorEngine>& lhs,
    typename boost::numeric::ublas::tensor_core<TensorEngine>::const_reference r)
{
  boost::numeric::ublas::detail::eval(lhs, [r](auto& l) { l+=r; } );
  return lhs;
}

template<typename TensorEngine>
inline
  constexpr auto& operator -= (boost::numeric::ublas::tensor_core<TensorEngine>& lhs,
    typename boost::numeric::ublas::tensor_core<TensorEngine>::const_reference r)
{
  boost::numeric::ublas::detail::eval(lhs, [r](auto& l) { l-=r; } );
  return lhs;
}

template<typename TensorEngine>
inline
  constexpr auto& operator *= (boost::numeric::ublas::tensor_core<TensorEngine>& lhs,
    typename boost::numeric::ublas::tensor_core<TensorEngine>::const_reference r)
{
  boost::numeric::ublas::detail::eval(lhs, [r](auto& l) { l*=r; } );
  return lhs;
}

template<typename TensorEngine>
constexpr auto& operator /= (boost::numeric::ublas::tensor_core<TensorEngine>& lhs, 
    typename boost::numeric::ublas::tensor_core<TensorEngine>::const_reference r)
{
  boost::numeric::ublas::detail::eval(lhs, [r](auto& l) { l/=r; } );
    return lhs;
}






template<class T, class D>
inline constexpr
  auto const& operator +(const boost::numeric::ublas::detail::tensor_expression<T,D>& lhs) noexcept{
  return lhs;
}

template<class T, class D>
inline constexpr
  auto operator -(boost::numeric::ublas::detail::tensor_expression<T,D> const& lhs) {
  return boost::numeric::ublas::detail::make_unary_tensor_expression<T> (lhs(), std::negate<>{} );
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
