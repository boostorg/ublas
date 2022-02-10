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

#ifndef BOOST_UBLAS_TENSOR_OPERATORS_COMPARISON_HPP
#define BOOST_UBLAS_TENSOR_OPERATORS_COMPARISON_HPP

#include <type_traits>
#include <functional>
#include <vector>
#include <utility>

#include "extents.hpp"
#include "expression.hpp"
#include "type_traits.hpp"
#include "expression_evaluation.hpp"

namespace boost::numeric::ublas {
template<class T>
class tensor_core;
} // namespace boost::numeric::ublas

namespace boost::numeric::ublas::detail
{

template<class T1, class T2, class BinaryPred>
[[nodiscard]] inline 
constexpr bool compare(tensor_core<T1> const& lhs, tensor_core<T2> const& rhs, BinaryPred&& pred) noexcept
    requires ( same_exp< BinaryPred, std::equal_to<> > || same_exp< BinaryPred, std::not_equal_to<> > )
{
    static_assert( std::is_same_v<typename tensor_core<T1>::value_type, typename tensor_core<T2>::value_type>,
        "boost::numeric::ublas::detail::compare(tensor_core<T1> const&, tensor_core<T2> const&, BinaryPred) : "
        "LHS and RHS both should have the same value type"
    );

    using lex_t = typename tensor_core<T1>::extents_type;
    using rex_t = typename tensor_core<T2>::extents_type;

    if constexpr(is_static_v<lex_t> && is_static_v<rex_t>){
        if constexpr(!same_exp<lex_t,rex_t>) 
            return false;
    }else{
        if(::operator!=(lhs.extents(),rhs.extents())){
            return false;
        }
    }

    return std::equal(lhs.begin(), lhs.end(), rhs.begin(), std::forward<BinaryPred>(pred));
}

template<class T1, class T2, class BinaryPred>
[[nodiscard]] inline 
constexpr bool compare(tensor_core<T1> const& lhs, tensor_core<T2> const& rhs, BinaryPred&& pred)
    noexcept( 
        is_static_v<typename tensor_core<T1>::extents_type> && 
        is_static_v<typename tensor_core<T2>::extents_type> 
    )
{
    static_assert( std::is_same_v<typename tensor_core<T1>::value_type, typename tensor_core<T2>::value_type>,
        "boost::numeric::ublas::detail::compare(tensor_core<T1> const&, tensor_core<T2> const&, BinaryPred) : "
        "LHS and RHS both should have the same value type"
    );

    using lex_t = typename tensor_core<T1>::extents_type;
    using rex_t = typename tensor_core<T2>::extents_type;

    if constexpr(is_static_v<lex_t> && is_static_v<rex_t>){
        static_assert(same_exp<lex_t,rex_t>, 
            "boost::numeric::ublas::detail::compare(tensor_core<T1> const&, tensor_core<T2> const&, BinaryPred) : "
            "cannot compare tensors with different shapes."
        );
    }else{
        if(::operator!=(lhs.extents(),rhs.extents())){
            throw std::runtime_error(
                "boost::numeric::ublas::detail::compare(tensor_core<T1> const&, tensor_core<T2> const&, BinaryPred) : "
                "cannot compare tensors with different shapes."
            );
        }
    }

    return std::equal(lhs.begin(), lhs.end(), rhs.begin(), std::forward<BinaryPred>(pred));
}

template<class T, class UnaryPred>
[[nodiscard]] inline 
constexpr bool compare(tensor_core<T> const& rhs, UnaryPred&& pred) noexcept
{
    return std::all_of(rhs.begin(), rhs.end(), std::forward<UnaryPred>(pred));
}


template<class T1, class T2, class L, class R, class BinaryPred>
[[nodiscard]]
constexpr bool compare(tensor_expression<T1,L> const& lhs, tensor_expression<T2,R> const& rhs, BinaryPred&& pred) noexcept
    requires (same_exp<T1,L> && same_exp<T2,R>)
{
    return compare(lhs(), rhs(), std::forward<BinaryPred>(pred));
}

template<class T1, class T2, class L, class R, class BinaryPred>
[[nodiscard]]
constexpr bool compare(tensor_expression<T1,L> const& lhs, tensor_expression<T2,R> const& rhs, BinaryPred&& pred)
    requires (same_exp<T1,L> && !same_exp<T2,R>)
{
    auto const r = T2(rhs); // FIXME: why are we constructing a whole new tensor?
    return compare(lhs(), r, std::forward<BinaryPred>(pred));
}

template<class T1, class T2, class L, class R, class BinaryPred>
[[nodiscard]]
constexpr bool compare(tensor_expression<T1,L> const& lhs, tensor_expression<T2,R> const& rhs, BinaryPred&& pred)
    requires (!same_exp<T1,L> && same_exp<T2,R>)
{
    auto const l = T1(lhs); // FIXME: why are we constructing a whole new tensor?
    return compare(l, rhs(), std::forward<BinaryPred>(pred));
}

template<class T1, class T2, class L, class R, class BinaryPred>
[[nodiscard]]
constexpr bool compare(tensor_expression<T1,L> const& lhs, tensor_expression<T2,R> const& rhs, BinaryPred&& pred)
{
    auto const l = T1(lhs); // FIXME: why are we constructing a whole new tensor?
    auto const r = T2(rhs); // FIXME: why are we constructing a whole new tensor?
    return compare(l, r, std::forward<BinaryPred>(pred));
}

template<class T, class D, class UnaryPred>
[[nodiscard]]
constexpr bool compare(tensor_expression<T,D> const& expr, UnaryPred&& pred) noexcept
    requires same_exp<T,D>
{
    return compare(expr(), std::forward<UnaryPred>(pred));
}

template<class T, class D, class UnaryPred>
[[nodiscard]]
constexpr bool compare(tensor_expression<T,D> const& expr, UnaryPred&& pred)
{
    auto const e = T(expr); // FIXME: why are we constructing a whole new tensor?
    return compare(e, std::forward<UnaryPred>(pred));
}

} // namespace boost::numeric::ublas::detail


template<class T1, class T2, class L, class R>
[[nodiscard]] inline
constexpr bool operator==(
    boost::numeric::ublas::detail::tensor_expression<T1,L> const& lhs,
    boost::numeric::ublas::detail::tensor_expression<T2,R> const& rhs) noexcept{
    return boost::numeric::ublas::detail::compare( lhs, rhs, std::equal_to<>{} );
}

template<class T1, class T2, class L, class R>
[[nodiscard]] inline 
constexpr auto operator!=(
    boost::numeric::ublas::detail::tensor_expression<T1,L> const& lhs,
    boost::numeric::ublas::detail::tensor_expression<T2,R> const& rhs) noexcept{
    return boost::numeric::ublas::detail::compare( lhs, rhs, std::not_equal_to<>{}  );
}

template<class T1, class T2, class L, class R>
[[nodiscard]] inline 
constexpr auto operator< (
    boost::numeric::ublas::detail::tensor_expression<T1,L> const& lhs,
    boost::numeric::ublas::detail::tensor_expression<T2,R> const& rhs) {
    return boost::numeric::ublas::detail::compare( lhs, rhs, std::less<>{} );
}

template<class T1, class T2, class L, class R>
[[nodiscard]] inline 
constexpr auto operator<=(
    boost::numeric::ublas::detail::tensor_expression<T1,L> const& lhs,
    boost::numeric::ublas::detail::tensor_expression<T2,R> const& rhs) {
    return boost::numeric::ublas::detail::compare( lhs, rhs, std::less_equal<>{} );
}

template<class T1, class T2, class L, class R>
[[nodiscard]] inline 
constexpr auto operator> (
    boost::numeric::ublas::detail::tensor_expression<T1,L> const& lhs,
    boost::numeric::ublas::detail::tensor_expression<T2,R> const& rhs) {
    return boost::numeric::ublas::detail::compare( lhs, rhs, std::greater<>{} );
}

template<class T1, class T2, class L, class R>
[[nodiscard]] inline 
constexpr auto operator>=(
    boost::numeric::ublas::detail::tensor_expression<T1,L> const& lhs,
    boost::numeric::ublas::detail::tensor_expression<T2,R> const& rhs) {
    return boost::numeric::ublas::detail::compare( lhs, rhs, std::greater_equal<>{} );
}





template<class T, class D>
[[nodiscard]] inline 
constexpr bool operator==( typename T::value_type lhs, boost::numeric::ublas::detail::tensor_expression<T,D> const& rhs) noexcept{
    return boost::numeric::ublas::detail::compare( rhs, [lhs](auto const& r){ return lhs == r; } );
}
template<class T, class D>
[[nodiscard]] inline 
constexpr auto operator!=( typename T::value_type lhs, boost::numeric::ublas::detail::tensor_expression<T,D> const& rhs) noexcept{
    return boost::numeric::ublas::detail::compare( rhs, [lhs](auto const& r){ return lhs != r; } );
}
template<class T, class D>
[[nodiscard]] inline 
constexpr auto operator< ( typename T::value_type lhs, boost::numeric::ublas::detail::tensor_expression<T,D> const& rhs) {
    return boost::numeric::ublas::detail::compare( rhs, [lhs](auto const& r){ return lhs <  r; } );
}
template<class T, class D>
[[nodiscard]] inline 
constexpr auto operator<=( typename T::value_type lhs, boost::numeric::ublas::detail::tensor_expression<T,D> const& rhs) {
    return boost::numeric::ublas::detail::compare( rhs, [lhs](auto const& r){ return lhs <= r; } );
}
template<class T, class D>
[[nodiscard]] inline 
constexpr auto operator> ( typename T::value_type lhs, boost::numeric::ublas::detail::tensor_expression<T,D> const& rhs) {
    return boost::numeric::ublas::detail::compare( rhs, [lhs](auto const& r){ return lhs >  r; } );
}
template<class T, class D>
[[nodiscard]] inline 
constexpr auto operator>=( typename T::value_type lhs, boost::numeric::ublas::detail::tensor_expression<T,D> const& rhs) {
    return boost::numeric::ublas::detail::compare( rhs, [lhs](auto const& r){ return lhs >= r; } );
}



template<class T, class D>
constexpr bool operator==( boost::numeric::ublas::detail::tensor_expression<T,D> const& lhs, typename T::value_type rhs) noexcept{
    return boost::numeric::ublas::detail::compare( lhs, [rhs](auto const& l){ return l == rhs; } );
}
template<class T, class D>
[[nodiscard]] inline 
constexpr auto operator!=( boost::numeric::ublas::detail::tensor_expression<T,D> const& lhs, typename T::value_type rhs) noexcept{
    return boost::numeric::ublas::detail::compare( lhs, [rhs](auto const& l){ return l != rhs; } );
}
template<class T, class D>
[[nodiscard]] inline 
constexpr auto operator< ( boost::numeric::ublas::detail::tensor_expression<T,D> const& lhs, typename T::value_type rhs) {
    return boost::numeric::ublas::detail::compare( lhs, [rhs](auto const& l){ return l <  rhs; } );
}
template<class T, class D>
[[nodiscard]] inline 
constexpr auto operator<=( boost::numeric::ublas::detail::tensor_expression<T,D> const& lhs, typename T::value_type rhs) {
    return boost::numeric::ublas::detail::compare( lhs, [rhs](auto const& l){ return l <= rhs; } );
}
template<class T, class D>
[[nodiscard]] inline 
constexpr auto operator> ( boost::numeric::ublas::detail::tensor_expression<T,D> const& lhs, typename T::value_type rhs) {
    return boost::numeric::ublas::detail::compare( lhs, [rhs](auto const& l){ return l >  rhs; } );
}
template<class T, class D>
[[nodiscard]] inline 
constexpr auto operator>=( boost::numeric::ublas::detail::tensor_expression<T,D> const& lhs, typename T::value_type rhs) {
    return boost::numeric::ublas::detail::compare( lhs, [rhs](auto const& l){ return l >= rhs; } );
}


#endif
