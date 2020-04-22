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

#ifndef BOOST_UBLAS_TENSOR_OPERATORS_COMPARISON_HPP
#define BOOST_UBLAS_TENSOR_OPERATORS_COMPARISON_HPP

#include <boost/numeric/ublas/tensor/expression.hpp>
#include <boost/numeric/ublas/tensor/detail/type_traits.hpp>
#include <boost/numeric/ublas/tensor/expression_evaluation.hpp>
#include <type_traits>
#include <functional>
#include <vector>
#include <utility>

namespace boost::numeric::ublas {
template<class T>
class basic_tensor;
}

namespace boost::numeric::ublas::detail {

template<class T1, class T2, class BinaryPred>
inline constexpr bool compare(basic_tensor<T1> const& lhs, basic_tensor<T2> const& rhs, BinaryPred pred)
{
    static_assert( is_valid_tensor_v<T1> && is_valid_tensor_v<T2>,
        "boost::numeric::ublas::detail::compare() : LHS and RHS both should the tensor"
    );

    static_assert( std::is_same_v<typename T1::value_type, typename T2::value_type>,
        "boost::numeric::ublas::detail::compare() : LHS and RHS both should have same value type"
    );

    if(lhs.extents() != rhs.extents()){
        if constexpr(!std::is_same<BinaryPred,std::equal_to<>>::value && !std::is_same<BinaryPred,std::not_equal_to<>>::value)
            throw std::runtime_error("Error in boost::numeric::ublas::detail::compare: cannot compare tensors with different shapes.");
        else
            return false;
    }

    if constexpr(std::is_same<BinaryPred,std::greater<>>::value || std::is_same<BinaryPred,std::less<>>::value)
        if(lhs.empty())
            return false;

    for(auto i = 0u; i < lhs.size(); ++i)
        if(!pred(lhs(i), rhs(i)))
            return false;
    return true;
}

template<class T, class UnaryPred>
inline constexpr bool compare(basic_tensor<T> const& rhs, UnaryPred pred)
{
    static_assert( is_valid_tensor_v<T>,
        "boost::numeric::ublas::detail::compare() : Template typename T should the tensor type"
    );

    for(auto i = 0u; i < rhs.size(); ++i)
        if(!pred(rhs(i)))
            return false;
    return true;
}


template<class T1, class T2, class L, class R, class BinaryPred>
constexpr bool compare(tensor_expression<T1,L> const& lhs, tensor_expression<T2,R> const& rhs, BinaryPred pred)
{
    constexpr bool lhs_is_tensor = std::is_same<T1,L>::value;
    constexpr bool rhs_is_tensor = std::is_same<T2,R>::value;
    
    if constexpr (lhs_is_tensor && rhs_is_tensor)
        return compare(static_cast<T1 const&>( lhs ), static_cast<T2 const&>( rhs ), pred);
    else if constexpr (lhs_is_tensor && !rhs_is_tensor)
        return compare(static_cast<T1 const&>( lhs ), T2( rhs ), pred);
    else if constexpr (!lhs_is_tensor && rhs_is_tensor)
        return compare(T1( lhs ), static_cast<T2 const&>( rhs ), pred);
    else
        return compare(T1( lhs ), T2( rhs ), pred);

}

template<class T, class D, class UnaryPred>
constexpr bool compare(tensor_expression<T,D> const& expr, UnaryPred pred)
{
    if constexpr (std::is_same<T,D>::value)
        return compare(static_cast<T const&>( expr ), pred);
    else
        return compare(T( expr ), pred);
}

}


template<class T1, class T2, class L, class R>
inline constexpr bool operator==( boost::numeric::ublas::detail::tensor_expression<T1,L> const& lhs,
                                 boost::numeric::ublas::detail::tensor_expression<T2,R> const& rhs) {
    return boost::numeric::ublas::detail::compare( lhs, rhs, std::equal_to<>{} );
}
template<class T1, class T2, class L, class R>
inline constexpr auto operator!=(boost::numeric::ublas::detail::tensor_expression<T1,L> const& lhs,
                                boost::numeric::ublas::detail::tensor_expression<T2,R> const& rhs) {
    return boost::numeric::ublas::detail::compare( lhs, rhs, std::not_equal_to<>{}  );
}
template<class T1, class T2, class L, class R>
inline constexpr auto operator< ( boost::numeric::ublas::detail::tensor_expression<T1,L> const& lhs,
                                 boost::numeric::ublas::detail::tensor_expression<T2,R> const& rhs) {
    return boost::numeric::ublas::detail::compare( lhs, rhs, std::less<>{} );
}
template<class T1, class T2, class L, class R>
inline constexpr auto operator<=( boost::numeric::ublas::detail::tensor_expression<T1,L> const& lhs,
                                 boost::numeric::ublas::detail::tensor_expression<T2,R> const& rhs) {
    return boost::numeric::ublas::detail::compare( lhs, rhs, std::less_equal<>{} );
}
template<class T1, class T2, class L, class R>
inline constexpr auto operator> ( boost::numeric::ublas::detail::tensor_expression<T1,L> const& lhs,
                                 boost::numeric::ublas::detail::tensor_expression<T2,R> const& rhs) {
    return boost::numeric::ublas::detail::compare( lhs, rhs, std::greater<>{} );
}
template<class T1, class T2, class L, class R>
inline constexpr auto operator>=( boost::numeric::ublas::detail::tensor_expression<T1,L> const& lhs,
                                 boost::numeric::ublas::detail::tensor_expression<T2,R> const& rhs) {
    return boost::numeric::ublas::detail::compare( lhs, rhs, std::greater_equal<>{} );
}





template<class T, class D>
inline constexpr bool operator==( typename T::const_reference lhs, boost::numeric::ublas::detail::tensor_expression<T,D> const& rhs) {
    return boost::numeric::ublas::detail::compare( rhs, [lhs](auto const& r){ return lhs == r; } );
}
template<class T, class D>
inline constexpr auto operator!=( typename T::const_reference lhs, boost::numeric::ublas::detail::tensor_expression<T,D> const& rhs) {
    return boost::numeric::ublas::detail::compare( rhs, [lhs](auto const& r){ return lhs != r; } );
}
template<class T, class D>
inline constexpr auto operator< ( typename T::const_reference lhs, boost::numeric::ublas::detail::tensor_expression<T,D> const& rhs) {
    return boost::numeric::ublas::detail::compare( rhs, [lhs](auto const& r){ return lhs <  r; } );
}
template<class T, class D>
inline constexpr auto operator<=( typename T::const_reference lhs, boost::numeric::ublas::detail::tensor_expression<T,D> const& rhs) {
    return boost::numeric::ublas::detail::compare( rhs, [lhs](auto const& r){ return lhs <= r; } );
}
template<class T, class D>
inline constexpr auto operator> ( typename T::const_reference lhs, boost::numeric::ublas::detail::tensor_expression<T,D> const& rhs) {
    return boost::numeric::ublas::detail::compare( rhs, [lhs](auto const& r){ return lhs >  r; } );
}
template<class T, class D>
inline constexpr auto operator>=( typename T::const_reference lhs, boost::numeric::ublas::detail::tensor_expression<T,D> const& rhs) {
    return boost::numeric::ublas::detail::compare( rhs, [lhs](auto const& r){ return lhs >= r; } );
}



template<class T, class D>
bool operator==( boost::numeric::ublas::detail::tensor_expression<T,D> const& lhs, typename T::const_reference rhs) {
    return boost::numeric::ublas::detail::compare( lhs, [rhs](auto const& l){ return l == rhs; } );
}
template<class T, class D>
inline constexpr auto operator!=( boost::numeric::ublas::detail::tensor_expression<T,D> const& lhs, typename T::const_reference rhs) {
    return boost::numeric::ublas::detail::compare( lhs, [rhs](auto const& l){ return l != rhs; } );
}
template<class T, class D>
inline constexpr auto operator< ( boost::numeric::ublas::detail::tensor_expression<T,D> const& lhs, typename T::const_reference rhs) {
    return boost::numeric::ublas::detail::compare( lhs, [rhs](auto const& l){ return l <  rhs; } );
}
template<class T, class D>
inline constexpr auto operator<=( boost::numeric::ublas::detail::tensor_expression<T,D> const& lhs, typename T::const_reference rhs) {
    return boost::numeric::ublas::detail::compare( lhs, [rhs](auto const& l){ return l <= rhs; } );
}
template<class T, class D>
inline constexpr auto operator> ( boost::numeric::ublas::detail::tensor_expression<T,D> const& lhs, typename T::const_reference rhs) {
    return boost::numeric::ublas::detail::compare( lhs, [rhs](auto const& l){ return l >  rhs; } );
}
template<class T, class D>
inline constexpr auto operator>=( boost::numeric::ublas::detail::tensor_expression<T,D> const& lhs, typename T::const_reference rhs) {
    return boost::numeric::ublas::detail::compare( lhs, [rhs](auto const& l){ return l >= rhs; } );
}


#endif
