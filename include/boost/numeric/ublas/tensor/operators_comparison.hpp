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

template<typename T>
struct is_equality_functional_object
    : std::false_type
{};

template<typename T>
static constexpr bool is_equality_functional_object_v = is_equality_functional_object< std::decay_t<T> >::value;

template<typename T>
struct is_equality_functional_object< std::equal_to<T> >
    : std::true_type 
{};

template<typename T>
struct is_equality_functional_object< std::not_equal_to<T> >
    : std::true_type 
{};

template<>
struct is_equality_functional_object< std::equal_to<> >
    : std::true_type 
{};

template<>
struct is_equality_functional_object< std::not_equal_to<> >
    : std::true_type 
{};

template<class T1, class T2, class L, class R, class BinaryPred>
[[nodiscard]] inline 
constexpr bool compare(tensor_expression<T1,L> const& lhs, tensor_expression<T2,R> const& rhs, BinaryPred&& pred) noexcept
    requires is_equality_functional_object_v<BinaryPred>
{
    
    auto const& lexpr = cast_tensor_expression(lhs);
    auto const& rexpr = cast_tensor_expression(rhs);

    using lvalue_type = decltype(lexpr(0));
    using rvalue_type = decltype(rexpr(0));
    
    static_assert( same_exp< lvalue_type, rvalue_type >,
        "boost::numeric::ublas::detail::compare : "
        "both LHS and RHS should have the same value type"
    );

    static_assert(
        std::is_invocable_r_v<bool, BinaryPred, lvalue_type, rvalue_type>,
        "boost::numeric::ublas::detail::compare(lhs,rhs,pred) :"
        "predicate must be a binary predicate, and it must return a bool"
    );

    auto const& le = retrieve_extents(lexpr);
    auto const& re = retrieve_extents(rexpr);

    using size_type = typename T1::size_type;

    // returns the pair containing false if extents are not equal 
    // else true, and the size of the container.
    constexpr auto cal_size = [](auto const& le, auto const& re) 
        -> std::pair<bool, size_type>
    {
        using lex_t = std::decay_t< decltype(le) >;
        using rex_t = std::decay_t< decltype(re) >;

        if constexpr(is_static_v< lex_t > && is_static_v< rex_t >){
            if constexpr(!same_exp< lex_t, rex_t >) 
                return { false, size_type{} };

            return { true, product_v< lex_t > };
        } else {
            if(::operator!=(le,re))
                return { false, size_type{} };

            return { true, product( le ) };
        }
    };

    auto const [status, size] = cal_size(le, re);

    for(auto i = size_type{}; i < size; ++i){
        if(!std::invoke(pred, lexpr(i), rexpr(i)))
            return false;
    }

    // return false if the status is false
    return ( true & status );
}

template<class T1, class T2, class L, class R, class BinaryPred>
[[nodiscard]] inline 
constexpr bool compare(tensor_expression<T1,L> const& lhs, tensor_expression<T2,R> const& rhs, BinaryPred&& pred) 
    noexcept( 
        is_static_v< std::decay_t< decltype(retrieve_extents(lhs)) > > && 
        is_static_v< std::decay_t< decltype(retrieve_extents(rhs)) > > 
    )
    requires ( not is_equality_functional_object_v<BinaryPred> )
{
    auto const& lexpr = cast_tensor_expression(lhs);
    auto const& rexpr = cast_tensor_expression(rhs);

    using lvalue_type = decltype(lexpr(0));
    using rvalue_type = decltype(rexpr(0));
    
    static_assert( same_exp< lvalue_type, rvalue_type >,
        "boost::numeric::ublas::detail::compare : "
        "both LHS and RHS should have the same value type"
    );

    static_assert(
        std::is_invocable_r_v<bool, BinaryPred, lvalue_type, rvalue_type>,
        "boost::numeric::ublas::detail::compare(lhs,rhs,pred) :"
        "predicate must be a binary predicate, and it must return a bool"
    );

    auto const& le = retrieve_extents(lexpr);
    auto const& re = retrieve_extents(rexpr);

    using size_type = typename T1::size_type;

    // returns the size of the container
    constexpr auto cal_size = [](auto const& le, auto const& re) 
        -> size_type
    {
        using lex_t = std::decay_t< decltype(le) >;
        using rex_t = std::decay_t< decltype(re) >;

        if constexpr(is_static_v< lex_t > && is_static_v< rex_t >){
            static_assert(same_exp< lex_t, rex_t >, 
                "boost::numeric::ublas::detail::compare : "
                "cannot compare tensors with different shapes."
            );

            return product_v< lex_t >;
        }else{
            if(::operator!=(le,re)){
                throw std::runtime_error(
                    "boost::numeric::ublas::detail::compare : "
                    "cannot compare tensors with different shapes."
                );
            }

            return product( le );
        }
    };

    size_type const size = cal_size(le, re);

    for(auto i = size_type{}; i < size; ++i){
        if(!std::invoke(pred, lexpr(i), rexpr(i)))
            return false;
    }

    return true;
}


template<class T, class D, class UnaryPred>
[[nodiscard]]
constexpr bool compare(tensor_expression<T,D> const& expr, UnaryPred&& pred) noexcept
{
    auto const& ue = cast_tensor_expression(expr);
    auto const& e = retrieve_extents(ue);

    using size_type = typename T::size_type;
    
    static_assert(
        std::is_invocable_r_v<bool, UnaryPred, decltype(ue(0))>,
        "boost::numeric::ublas::detail::compare(expr,pred) :"
        "predicate must be an unary predicate, and it must return a bool"
    );

    // returns the size of the container
    constexpr auto cal_size = [](auto const& e) 
        -> size_type
    {
        using extents_t = std::decay_t< decltype(e) >;
        
        if constexpr(is_static_v< extents_t >)
            return product_v< extents_t >;
        else
            return product( e );
    };

    size_type const size = cal_size(e);

    for(auto i = size_type{}; i < size; ++i){
        if(!std::invoke(pred, ue(i)))
            return false;
    }

    return true;
}


} // namespace boost::numeric::ublas::detail


template<class T1, class T2, class L, class R>
[[nodiscard]] inline
constexpr bool operator==(
    boost::numeric::ublas::detail::tensor_expression<T1,L> const& lhs,
    boost::numeric::ublas::detail::tensor_expression<T2,R> const& rhs
) noexcept{
    using value_type = typename T1::value_type;
    return boost::numeric::ublas::detail::compare( lhs, rhs, std::equal_to<value_type>{} );
}

template<class T1, class T2, class L, class R>
[[nodiscard]] inline 
constexpr auto operator!=(
    boost::numeric::ublas::detail::tensor_expression<T1,L> const& lhs,
    boost::numeric::ublas::detail::tensor_expression<T2,R> const& rhs
) noexcept{
    using value_type = typename T1::value_type;
    return boost::numeric::ublas::detail::compare( lhs, rhs, std::not_equal_to<value_type>{}  );
}

template<class T1, class T2, class L, class R>
[[nodiscard]] inline 
constexpr auto operator< (
    boost::numeric::ublas::detail::tensor_expression<T1,L> const& lhs,
    boost::numeric::ublas::detail::tensor_expression<T2,R> const& rhs
)noexcept( 
    boost::numeric::ublas::is_static_v< std::decay_t< decltype(boost::numeric::ublas::detail::retrieve_extents(lhs)) > > && 
    boost::numeric::ublas::is_static_v< std::decay_t< decltype(boost::numeric::ublas::detail::retrieve_extents(rhs)) > > 
) {
    using value_type = typename T1::value_type;
    return boost::numeric::ublas::detail::compare( lhs, rhs, std::less<value_type>{} );
}

template<class T1, class T2, class L, class R>
[[nodiscard]] inline 
constexpr auto operator<=(
    boost::numeric::ublas::detail::tensor_expression<T1,L> const& lhs,
    boost::numeric::ublas::detail::tensor_expression<T2,R> const& rhs
)noexcept( 
    boost::numeric::ublas::is_static_v< std::decay_t< decltype(boost::numeric::ublas::detail::retrieve_extents(lhs)) > > && 
    boost::numeric::ublas::is_static_v< std::decay_t< decltype(boost::numeric::ublas::detail::retrieve_extents(rhs)) > > 
) {
    using value_type = typename T1::value_type;
    return boost::numeric::ublas::detail::compare( lhs, rhs, std::less_equal<value_type>{} );
}

template<class T1, class T2, class L, class R>
[[nodiscard]] inline 
constexpr auto operator> (
    boost::numeric::ublas::detail::tensor_expression<T1,L> const& lhs,
    boost::numeric::ublas::detail::tensor_expression<T2,R> const& rhs
)noexcept( 
    boost::numeric::ublas::is_static_v< std::decay_t< decltype(boost::numeric::ublas::detail::retrieve_extents(lhs)) > > && 
    boost::numeric::ublas::is_static_v< std::decay_t< decltype(boost::numeric::ublas::detail::retrieve_extents(rhs)) > > 
) {
    using value_type = typename T1::value_type;
    return boost::numeric::ublas::detail::compare( lhs, rhs, std::greater<value_type>{} );
}

template<class T1, class T2, class L, class R>
[[nodiscard]] inline 
constexpr auto operator>=(
    boost::numeric::ublas::detail::tensor_expression<T1,L> const& lhs,
    boost::numeric::ublas::detail::tensor_expression<T2,R> const& rhs
)noexcept( 
    boost::numeric::ublas::is_static_v< std::decay_t< decltype(boost::numeric::ublas::detail::retrieve_extents(lhs)) > > && 
    boost::numeric::ublas::is_static_v< std::decay_t< decltype(boost::numeric::ublas::detail::retrieve_extents(rhs)) > > 
) {
    using value_type = typename T1::value_type;
    return boost::numeric::ublas::detail::compare( lhs, rhs, std::greater_equal<value_type>{} );
}





template<class T, class D>
[[nodiscard]] inline 
constexpr bool operator==( typename T::value_type lhs, boost::numeric::ublas::detail::tensor_expression<T,D> const& rhs) noexcept{
    using value_type = typename T::value_type;
    return boost::numeric::ublas::detail::compare( rhs, [lhs](value_type const& r){ return lhs == r; } );
}
template<class T, class D>
[[nodiscard]] inline 
constexpr auto operator!=( typename T::value_type lhs, boost::numeric::ublas::detail::tensor_expression<T,D> const& rhs) noexcept{
    using value_type = typename T::value_type;
    return boost::numeric::ublas::detail::compare( rhs, [lhs](value_type const& r){ return lhs != r; } );
}
template<class T, class D>
[[nodiscard]] inline 
constexpr auto operator< ( typename T::value_type lhs, boost::numeric::ublas::detail::tensor_expression<T,D> const& rhs) noexcept{
    using value_type = typename T::value_type;
    return boost::numeric::ublas::detail::compare( rhs, [lhs](value_type const& r){ return lhs <  r; } );
}
template<class T, class D>
[[nodiscard]] inline 
constexpr auto operator<=( typename T::value_type lhs, boost::numeric::ublas::detail::tensor_expression<T,D> const& rhs) noexcept{
    using value_type = typename T::value_type;
    return boost::numeric::ublas::detail::compare( rhs, [lhs](value_type const& r){ return lhs <= r; } );
}
template<class T, class D>
[[nodiscard]] inline 
constexpr auto operator> ( typename T::value_type lhs, boost::numeric::ublas::detail::tensor_expression<T,D> const& rhs) noexcept{
    using value_type = typename T::value_type;
    return boost::numeric::ublas::detail::compare( rhs, [lhs](value_type const& r){ return lhs >  r; } );
}
template<class T, class D>
[[nodiscard]] inline 
constexpr auto operator>=( typename T::value_type lhs, boost::numeric::ublas::detail::tensor_expression<T,D> const& rhs) noexcept{
    using value_type = typename T::value_type;
    return boost::numeric::ublas::detail::compare( rhs, [lhs](value_type const& r){ return lhs >= r; } );
}



template<class T, class D>
[[nodiscard]] inline
constexpr bool operator==( boost::numeric::ublas::detail::tensor_expression<T,D> const& lhs, typename T::value_type rhs) noexcept{
    using value_type = typename T::value_type;
    return boost::numeric::ublas::detail::compare( lhs, [rhs](value_type const& l){ return l == rhs; } );
}
template<class T, class D>
[[nodiscard]] inline 
constexpr auto operator!=( boost::numeric::ublas::detail::tensor_expression<T,D> const& lhs, typename T::value_type rhs) noexcept{
    using value_type = typename T::value_type;
    return boost::numeric::ublas::detail::compare( lhs, [rhs](value_type const& l){ return l != rhs; } );
}
template<class T, class D>
[[nodiscard]] inline 
constexpr auto operator< ( boost::numeric::ublas::detail::tensor_expression<T,D> const& lhs, typename T::value_type rhs) noexcept{
    using value_type = typename T::value_type;
    return boost::numeric::ublas::detail::compare( lhs, [rhs](value_type const& l){ return l <  rhs; } );
}
template<class T, class D>
[[nodiscard]] inline 
constexpr auto operator<=( boost::numeric::ublas::detail::tensor_expression<T,D> const& lhs, typename T::value_type rhs) noexcept{
    using value_type = typename T::value_type;
    return boost::numeric::ublas::detail::compare( lhs, [rhs](value_type const& l){ return l <= rhs; } );
}
template<class T, class D>
[[nodiscard]] inline 
constexpr auto operator> ( boost::numeric::ublas::detail::tensor_expression<T,D> const& lhs, typename T::value_type rhs) noexcept{
    using value_type = typename T::value_type;
    return boost::numeric::ublas::detail::compare( lhs, [rhs](value_type const& l){ return l >  rhs; } );
}
template<class T, class D>
[[nodiscard]] inline 
constexpr auto operator>=( boost::numeric::ublas::detail::tensor_expression<T,D> const& lhs, typename T::value_type rhs) noexcept{
    using value_type = typename T::value_type;
    return boost::numeric::ublas::detail::compare( lhs, [rhs](value_type const& l){ return l >= rhs; } );
}


#endif
