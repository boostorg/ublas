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

#ifndef BOOST_UBLAS_TENSOR_OPERATORS_ARITHMETIC_HPP
#define BOOST_UBLAS_TENSOR_OPERATORS_ARITHMETIC_HPP

#include "expression.hpp"
#include "expression_evaluation.hpp"
#include "multi_index_utility.hpp"
#include "functions.hpp"
#include <boost/numeric/ublas/tensor/detail/type_traits.hpp>

#include <type_traits>
#include <functional>
#include <algorithm>

namespace boost{
namespace numeric{
namespace ublas {


template<class T>
class basic_tensor;

template<class E>
class matrix_expression;


template<class E>
class vector_expression;

}
}
}

#define FIRST_ORDER_OPERATOR_RIGHT(OP, EXPR_TYPE_L, EXPR_TYPE_R) \
template<class T, class L, class R> \
auto operator OP ( boost::numeric::ublas:: EXPR_TYPE_L <T,L> const& lhs, boost::numeric::ublas:: EXPR_TYPE_R <R> const& rhs) { \
    return boost::numeric::ublas::detail::make_binary_tensor_expression<T> (lhs(), rhs(), \
      [](auto const& l, auto const& r){ return l OP r; }); \
} \

FIRST_ORDER_OPERATOR_RIGHT (*, detail:: tensor_expression , vector_expression)
FIRST_ORDER_OPERATOR_RIGHT (+, detail:: tensor_expression , vector_expression)
FIRST_ORDER_OPERATOR_RIGHT (-, detail:: tensor_expression , vector_expression)
FIRST_ORDER_OPERATOR_RIGHT (/, detail:: tensor_expression , vector_expression)

FIRST_ORDER_OPERATOR_RIGHT (*, detail:: tensor_expression , matrix_expression)
FIRST_ORDER_OPERATOR_RIGHT (+, detail:: tensor_expression , matrix_expression)
FIRST_ORDER_OPERATOR_RIGHT (-, detail:: tensor_expression , matrix_expression)
FIRST_ORDER_OPERATOR_RIGHT (/, detail:: tensor_expression , matrix_expression)


#define FIRST_ORDER_OPERATOR_LEFT(OP, EXPR_TYPE_L, EXPR_TYPE_R) \
template<class T, class L, class R> \
auto operator OP ( boost::numeric::ublas:: EXPR_TYPE_L <L> const& lhs, boost::numeric::ublas:: EXPR_TYPE_R <T,R> const& rhs) { \
    return boost::numeric::ublas::detail::make_binary_tensor_expression<T> (lhs(), rhs(), \
      [](auto const& l, auto const& r){ return l OP r; }); \
} \

FIRST_ORDER_OPERATOR_LEFT (*, vector_expression, detail:: tensor_expression)
FIRST_ORDER_OPERATOR_LEFT (+, vector_expression, detail:: tensor_expression)
FIRST_ORDER_OPERATOR_LEFT (-, vector_expression, detail:: tensor_expression)
FIRST_ORDER_OPERATOR_LEFT (/, vector_expression, detail:: tensor_expression)

FIRST_ORDER_OPERATOR_LEFT (*, matrix_expression, detail:: tensor_expression)
FIRST_ORDER_OPERATOR_LEFT (+, matrix_expression, detail:: tensor_expression)
FIRST_ORDER_OPERATOR_LEFT (-, matrix_expression, detail:: tensor_expression)
FIRST_ORDER_OPERATOR_LEFT (/, matrix_expression, detail:: tensor_expression)




template<class T1, class T2, class L, class R>
constexpr auto operator+( boost::numeric::ublas::detail::tensor_expression<T1,L> const& lhs, 
    boost::numeric::ublas::detail::tensor_expression<T2,R> const& rhs) 
{
    
    static_assert( std::is_same_v< typename T1::value_type, typename T2::value_type>,
        "operator+() : LHS tensor and RHS tensor should have same value type"
    );

    if constexpr( !std::is_same_v<T1,T2> ){
        auto e = boost::numeric::ublas::detail::retrieve_extents(rhs);
        
        if( !boost::numeric::ublas::detail::all_extents_equal(lhs,e) ){
            throw std::runtime_error("operator+() : LHS tensor and RHS tensor should have equal extents");
        }
    }

    return boost::numeric::ublas::detail::make_binary_tensor_expression<T1> (lhs(), rhs(), [](auto const& l, auto const& r){ return l + r; });
}
template<class T1, class T2, class L, class R>
constexpr auto operator-( boost::numeric::ublas::detail::tensor_expression<T1,L> const& lhs, 
    boost::numeric::ublas::detail::tensor_expression<T2,R> const& rhs) 
{
    
    static_assert( std::is_same_v< typename T1::value_type, typename T2::value_type>,
        "operator-() : LHS tensor and RHS tensor should have same value type"
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
constexpr auto operator*( boost::numeric::ublas::detail::tensor_expression<T1,L> const& lhs, 
    boost::numeric::ublas::detail::tensor_expression<T2,R> const& rhs) 
{
        
    static_assert( std::is_same_v< typename T1::value_type, typename T2::value_type>,
        "operator*() : LHS tensor and RHS tensor should have same value type"
    );

    if constexpr( !std::is_same_v<T1,T2> ){
        auto e = boost::numeric::ublas::detail::retrieve_extents(rhs);

        if( !boost::numeric::ublas::detail::all_extents_equal(lhs,e) ){
            throw std::runtime_error("operator+() : LHS tensor and RHS tensor should have equal extents");
        }
    }

    return boost::numeric::ublas::detail::make_binary_tensor_expression<T1> (lhs(), rhs(), [](auto const& l, auto const& r){ return l * r; });
}
template<class T1, class T2, class L, class R>
constexpr auto operator/( boost::numeric::ublas::detail::tensor_expression<T1,L> const& lhs, 
    boost::numeric::ublas::detail::tensor_expression<T2,R> const& rhs) 
{
        
    static_assert( std::is_same_v< typename T1::value_type, typename T2::value_type>,
        "operator/() : LHS tensor and RHS tensor should have same value type"
    );

    if constexpr( !std::is_same_v<T1,T2> ){
        auto e = boost::numeric::ublas::detail::retrieve_extents(rhs);

        if( !boost::numeric::ublas::detail::all_extents_equal(lhs,e) ){
            throw std::runtime_error("operator+() : LHS tensor and RHS tensor should have equal extents");
        }
    }

    return boost::numeric::ublas::detail::make_binary_tensor_expression<T1> (lhs(), rhs(), [](auto const& l, auto const& r){ return l / r; });
}


// Overloaded Arithmetic Operators with Scalars
template<class T, class R>
constexpr auto operator+(typename T::const_reference lhs, 
    boost::numeric::ublas::detail::tensor_expression<boost::numeric::ublas::basic_tensor<T>,R> const& rhs) 
{
    using basic_tensor_type = boost::numeric::ublas::basic_tensor<T>;
    return boost::numeric::ublas::detail::make_unary_tensor_expression<basic_tensor_type> (rhs(), [lhs](auto const& r){ return lhs + r; });
    //return boost::numeric::ublas::detail::make_lambda<T>( [&lhs,&rhs](std::size_t i) {return lhs + rhs(i); } );
}
template<class T, class R>
constexpr auto operator-(typename T::const_reference lhs, 
    boost::numeric::ublas::detail::tensor_expression<boost::numeric::ublas::basic_tensor<T>,R> const& rhs) 
{
    using basic_tensor_type = boost::numeric::ublas::basic_tensor<T>;
    return boost::numeric::ublas::detail::make_unary_tensor_expression<basic_tensor_type> (rhs(), [lhs](auto const& r){ return lhs - r; });
}
template<class T, class R>
constexpr auto operator*(typename T::const_reference lhs, 
    boost::numeric::ublas::detail::tensor_expression<boost::numeric::ublas::basic_tensor<T>,R> const& rhs) 
{
    using basic_tensor_type = boost::numeric::ublas::basic_tensor<T>;
    return boost::numeric::ublas::detail::make_unary_tensor_expression<basic_tensor_type> (rhs(), [lhs](auto const& r){ return lhs * r; });
}
template<class T, class R>
constexpr auto operator/(typename T::const_reference lhs, 
    boost::numeric::ublas::detail::tensor_expression<boost::numeric::ublas::basic_tensor<T>,R> const& rhs) 
{
    using basic_tensor_type = boost::numeric::ublas::basic_tensor<T>;
    return boost::numeric::ublas::detail::make_unary_tensor_expression<basic_tensor_type> (rhs(), [lhs](auto const& r){ return lhs / r; });
}


template<class T, class L>
constexpr auto operator+(boost::numeric::ublas::detail::tensor_expression<boost::numeric::ublas::basic_tensor<T>,L> const& lhs, 
    typename T::const_reference rhs) 
{
    using basic_tensor_type = boost::numeric::ublas::basic_tensor<T>;
    return boost::numeric::ublas::detail::make_unary_tensor_expression<basic_tensor_type> (lhs(), [rhs] (auto const& l) { return l + rhs; } );
}
template<class T, class L>
constexpr auto operator-(boost::numeric::ublas::detail::tensor_expression<boost::numeric::ublas::basic_tensor<T>,L> const& lhs, 
    typename T::const_reference rhs) 
{
    using basic_tensor_type = boost::numeric::ublas::basic_tensor<T>;
    return boost::numeric::ublas::detail::make_unary_tensor_expression<basic_tensor_type> (lhs(), [rhs] (auto const& l) { return l - rhs; } );
}
template<class T, class L>
constexpr auto operator*(boost::numeric::ublas::detail::tensor_expression<boost::numeric::ublas::basic_tensor<T>,L> const& lhs, 
    typename T::const_reference rhs) 
{
    using basic_tensor_type = boost::numeric::ublas::basic_tensor<T>;
    return boost::numeric::ublas::detail::make_unary_tensor_expression<basic_tensor_type> (lhs(), [rhs] (auto const& l) { return l * rhs; } );
}
template<class T, class L>
constexpr auto operator/(boost::numeric::ublas::detail::tensor_expression<boost::numeric::ublas::basic_tensor<T>,L> const& lhs, 
    typename T::const_reference rhs) 
{
    using basic_tensor_type = boost::numeric::ublas::basic_tensor<T>;
    return boost::numeric::ublas::detail::make_unary_tensor_expression<basic_tensor_type> (lhs(), [rhs] (auto const& l) { return l / rhs; } );
}



template<class T, class D>
constexpr auto& operator += (boost::numeric::ublas::basic_tensor<T>& lhs, 
    const boost::numeric::ublas::detail::tensor_expression<boost::numeric::ublas::basic_tensor<T>,D> &expr) 
{
    boost::numeric::ublas::detail::eval(lhs, expr(), [](auto& l, auto const& r) { l+=r; } );
    return lhs;
}

template<class T, class D>
constexpr auto& operator -= (boost::numeric::ublas::basic_tensor<T>& lhs, 
    const boost::numeric::ublas::detail::tensor_expression<boost::numeric::ublas::basic_tensor<T>,D> &expr) 
{
    boost::numeric::ublas::detail::eval(lhs, expr(), [](auto& l, auto const& r) { l-=r; } );
    return lhs;
}

template<class T, class D>
constexpr auto& operator *= (boost::numeric::ublas::basic_tensor<T>& lhs, 
    const boost::numeric::ublas::detail::tensor_expression<boost::numeric::ublas::basic_tensor<T>,D> &expr) 
{
    boost::numeric::ublas::detail::eval(lhs, expr(), [](auto& l, auto const& r) { l*=r; } );
    return lhs;
}

template<class T, class D>
constexpr auto& operator /= (boost::numeric::ublas::basic_tensor<T>& lhs, 
    const boost::numeric::ublas::detail::tensor_expression<boost::numeric::ublas::basic_tensor<T>,D> &expr) 
{
    boost::numeric::ublas::detail::eval(lhs, expr(), [](auto& l, auto const& r) { l/=r; } );
    return lhs;
}




template<class TensorType>
constexpr auto& operator += (boost::numeric::ublas::basic_tensor<TensorType>& lhs, typename TensorType::const_reference r) {
    static_assert( boost::numeric::ublas::is_valid_tensor_v<TensorType>, 
        "boost::numeric::ublas::operator +=() : tensor type should be valid tensor"
    );
    boost::numeric::ublas::detail::eval(lhs, [r](auto& l) { l+=r; } );
    return lhs;
}

template<typename TensorType>
constexpr auto& operator -= (boost::numeric::ublas::basic_tensor<TensorType>& lhs, typename TensorType::const_reference r) {
    static_assert( boost::numeric::ublas::is_valid_tensor_v<TensorType>, 
        "boost::numeric::ublas::operator -=() : tensor type should be valid tensor"
    );
    boost::numeric::ublas::detail::eval(lhs, [r](auto& l) { l-=r; } );
    return lhs;
}

template<typename TensorType>
constexpr auto& operator *= (boost::numeric::ublas::basic_tensor<TensorType>& lhs, typename TensorType::const_reference r) {
    static_assert( boost::numeric::ublas::is_valid_tensor_v<TensorType>, 
        "boost::numeric::ublas::operator *=() : tensor type should be valid tensor"
    );
    boost::numeric::ublas::detail::eval(lhs, [r](auto& l) { l*=r; } );
    return lhs;
}

template<typename TensorType>
constexpr auto& operator /= (boost::numeric::ublas::basic_tensor<TensorType>& lhs, typename TensorType::const_reference r) {
    static_assert( boost::numeric::ublas::is_valid_tensor_v<TensorType>, 
        "boost::numeric::ublas::operator /=() : tensor type should be valid tensor"
    );
    boost::numeric::ublas::detail::eval(lhs, [r](auto& l) { l/=r; } );
    return lhs;
}






template<class T, class D>
constexpr auto const& operator +(const boost::numeric::ublas::detail::tensor_expression<T,D>& lhs) {
    return lhs;
}

template<class T, class D>
constexpr auto operator -(boost::numeric::ublas::detail::tensor_expression<T,D> const& lhs) {
    return boost::numeric::ublas::detail::make_unary_tensor_expression<T> (lhs(), [] (auto const& l) { return -l; } );
}





/** @brief Performs a tensor contraction, not an elementwise multiplication
    *
*/

template<class tensor_type_left, class tuple_type_left, class tensor_type_right, class tuple_type_right>
auto operator*(
        std::pair< tensor_type_left  const&, tuple_type_left  > lhs,
        std::pair< tensor_type_right const&, tuple_type_right > rhs)
{

    using namespace boost::numeric::ublas;

    auto const& tensor_left  = lhs.first;
    auto const& tensor_right = rhs.first;

    auto multi_index_left = lhs.second;
    auto multi_index_right = rhs.second;

    static constexpr auto num_equal_ind = number_equal_indexes<tuple_type_left, tuple_type_right>::value;

    if constexpr ( num_equal_ind == 0  ){
        return tensor_left * tensor_right;
    }
    else if constexpr ( num_equal_ind==std::tuple_size<tuple_type_left>::value && std::is_same<tuple_type_left, tuple_type_right>::value ){

        return boost::numeric::ublas::inner_prod( tensor_left, tensor_right );
    }
    else {
        auto array_index_pairs = index_position_pairs(multi_index_left,multi_index_right);
        auto index_pairs = array_to_vector(  array_index_pairs  );
        return boost::numeric::ublas::prod( tensor_left, tensor_right, index_pairs.first, index_pairs.second );
    }

}

#endif
