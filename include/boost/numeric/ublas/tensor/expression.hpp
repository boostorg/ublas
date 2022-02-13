//
//  Copyright (c) 2018-2019, Cem Bassoy, cem.bassoy@gmail.com
// 	Copyright (c) 2022, Amit Singh, amitsingh19975@gmail.com
//
//  Distributed under the Boost Software License, Version 1.0. (See
//  accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)
//
//  The authors gratefully acknowledge the support of
//  Fraunhofer IOSB, Ettlingen, Germany
//

#ifndef BOOST_UBLAS_TENSOR_EXPRESSIONS_HPP
#define BOOST_UBLAS_TENSOR_EXPRESSIONS_HPP

#include <cstddef>
#include <type_traits>
#include <boost/numeric/ublas/expression_types.hpp>

#include "tags.hpp"

namespace boost::numeric::ublas::detail
{

template<class T, class E>
struct tensor_expression;

template<class T, class EL, class ER, class OP>
struct binary_tensor_expression;

template<class T, class E, class OP>
struct unary_tensor_expression;

template<typename E1, typename E2>
concept same_exp = std::is_same_v< std::decay_t<E1>, std::decay_t<E2> >;

template<typename T>
struct does_exp_need_cast : std::false_type{};

template<typename T>
static constexpr bool does_exp_need_cast_v = does_exp_need_cast< std::decay_t<T> >::value;

template<typename E, typename T>
struct does_exp_need_cast< tensor_expression<T,E> > : std::true_type{};

/**
 * @brief It is a safer way of casting `tensor_expression` because it handles
 * recursive expressions. Otherwise, in most of the cases, we try to access
 * `operator()`, which requires a parameter argument, that is not supported
 * by the `tensor_expression` class and might give an error if it is not casted
 * properly.
 * 
 * @tparam T type of the tensor
 * @tparam E type of the child stored inside tensor_expression
 * @param  e tensor_expression that needs to be casted
 * @return   child of tensor_expression that is not tensor_expression
 */
template<typename T, typename E>
constexpr auto const& cast_tensor_exression(tensor_expression<T,E> const& e) noexcept{
    auto const& res = e();
    if constexpr(does_exp_need_cast_v<decltype(res)>)
        return cast_tensor_exression(res);
    else 
        return res;
}


// FIXME: remove it when template expression support for the old matrix and vector is removed
/// @brief No Op: Any expression other than `tensor_expression`.
template<typename E>
constexpr auto const& cast_tensor_exression(E const& e) noexcept{ return e; }

template<typename E, typename T>
constexpr auto is_tensor_expression_impl(tensor_expression<T,E> const*) -> std::true_type;

constexpr auto is_tensor_expression_impl(void const*) -> std::false_type;

template<typename E>
constexpr auto is_matrix_expression_impl(matrix_expression<E> const*) -> std::true_type;

constexpr auto is_matrix_expression_impl(void const*) -> std::false_type;

template<typename E>
constexpr auto is_vector_expression_impl(vector_expression<E> const*) -> std::true_type;

constexpr auto is_vector_expression_impl(void const*) -> std::false_type;

template<typename E>
concept TensorExpression = decltype(is_tensor_expression_impl(static_cast<std::decay_t<E> const*>(nullptr)))::value;

// TODO: Remove this concept in the future when we have our own implementation of matrices.
template<typename E>
concept MatrixExpression = decltype(is_matrix_expression_impl(static_cast<std::decay_t<E> const*>(nullptr)))::value;

// TODO: Remove this concept in the future when we have our own implementation of vectors.
template<typename E>
concept VectorExpression = decltype(is_vector_expression_impl(static_cast<std::decay_t<E> const*>(nullptr)))::value;

template<typename Exp>
using expression_operand_t = std::conditional_t<
    !std::is_lvalue_reference_v<Exp>,
    std::decay_t<Exp>,
    std::add_lvalue_reference_t< std::add_const_t< std::decay_t<Exp> > >
>;

/** @\brief base class for tensor expressions
 *
 * \note implements crtp - no use of virtual function calls
 *
 * \tparam T type of the tensor
 * \tparam E type of the derived expression (crtp)
 *
 **/
template<class T, class E>
struct tensor_expression
    : public ublas_expression<E> // DISCUSS: Do we really need to do derive from ublas_expression?
{
    //	static const unsigned complexity = 0;
    using expression_type = E;
    using type_category = tensor_tag;
    using tensor_type = T;

    inline
    constexpr auto const& operator()() const noexcept { return *static_cast<const expression_type*> (this); }
    
    constexpr tensor_expression(tensor_expression&&) noexcept = default;
    constexpr tensor_expression& operator=(tensor_expression&&) noexcept = default;
    constexpr ~tensor_expression() = default;

    tensor_expression(const tensor_expression&) = delete;
    tensor_expression& operator=(const tensor_expression&) = delete;


protected :
    explicit tensor_expression() = default;
};


template<class T, class EL, class ER, class OP>
struct binary_tensor_expression
    : public tensor_expression <T, binary_tensor_expression<T,EL,ER,OP>>
{
    using self_type = binary_tensor_expression<T,EL,ER,OP>;
    using tensor_type  = T;
    using binary_operation = std::decay_t<OP>;
    using expression_type_left  = expression_operand_t<EL>;
    using expression_type_right = expression_operand_t<ER>;
    using derived_type =  tensor_expression <tensor_type,self_type>;

    using size_type = typename tensor_type::size_type;

    binary_tensor_expression() = delete;
    
    template<same_exp<EL> LeftExp, same_exp<ER> RightExp, typename OPType>
    explicit constexpr binary_tensor_expression(LeftExp&& l, RightExp&& r, OPType&& o) 
        : el(std::forward<LeftExp>(l)) 
        , er(std::forward<RightExp>(r)) 
        , op(std::forward<OPType>(o)) 
    {}
    constexpr binary_tensor_expression(binary_tensor_expression&& l) noexcept = default;
    constexpr binary_tensor_expression& operator=(binary_tensor_expression&& l) noexcept = default;
    constexpr ~binary_tensor_expression() = default;

    binary_tensor_expression(const binary_tensor_expression& l) = delete;
    binary_tensor_expression& operator=(binary_tensor_expression const& l) noexcept = delete;

    constexpr auto const& left_expr() const noexcept{ return cast_tensor_exression(el); }
    constexpr auto const& right_expr() const noexcept{ return cast_tensor_exression(er); }

    [[nodiscard]] inline 
    constexpr decltype(auto) operator()(size_type i) const {
        return op(left_expr()(i), right_expr()(i));
    }

private:
    expression_type_left el;
    expression_type_right er;
    binary_operation op;
};

/// @brief helper function to simply instantiation of lambda proxy class
template<typename T, typename EL, typename ER, typename OP>
    requires (
        ( MatrixExpression<EL> || VectorExpression<EL> || TensorExpression<EL> ) &&
        ( MatrixExpression<ER> || VectorExpression<ER> || TensorExpression<ER> ) 
    )
[[nodiscard]] inline
constexpr auto make_binary_tensor_expression( EL&& el, ER&& er, OP&& op) noexcept
{
    return binary_tensor_expression<T,EL,ER,OP>( 
        std::forward<EL>(el), 
        std::forward<ER>(er), 
        std::forward<OP>(op)
    );
}


template<class T, class E, class OP>
struct unary_tensor_expression
    : public tensor_expression <T, unary_tensor_expression<T,E,OP>>
{

    using self_type = unary_tensor_expression<T,E,OP>;
    using tensor_type  = T;
    using expression_type = expression_operand_t<E>;
    using unary_operation = std::decay_t<OP>;
    using derived_type = tensor_expression <T, unary_tensor_expression<T,E,OP>>;

    using size_type = typename tensor_type::size_type;

    template<same_exp<E> Exp, typename OPType>
    explicit constexpr unary_tensor_expression(Exp&& ee, OPType&& o) 
        : e(std::forward<Exp>(ee))
        , op(std::forward<OPType>(o)) 
    {}
    constexpr unary_tensor_expression(unary_tensor_expression&& l) noexcept = default;
    constexpr unary_tensor_expression& operator=(unary_tensor_expression&& l) noexcept = default;
    constexpr ~unary_tensor_expression() = default;

    constexpr unary_tensor_expression() = delete;
    unary_tensor_expression(unary_tensor_expression const& l) = delete;
    unary_tensor_expression& operator=(unary_tensor_expression const& l) noexcept = delete;

    constexpr auto const& expr() const noexcept{ return cast_tensor_exression(e); }

    [[nodiscard]] inline constexpr
    decltype(auto) operator()(size_type i) const {
        return op(expr()(i));
    }

private:
    expression_type e;
    unary_operation op;
};

// \brief helper function to simply instantiation of lambda proxy class
template<typename T, typename E, typename OP>
    requires ( MatrixExpression<E> || VectorExpression<E> || TensorExpression<E> )
[[nodiscard]] inline
constexpr auto make_unary_tensor_expression( E&& e, OP&& op) noexcept
{
    return unary_tensor_expression<T, E, OP>( 
        std::forward<E>(e), 
        std::forward<OP>(op)
    );
}

} // namespace boost::numeric::ublas::detail

#endif // BOOST_UBLAS_TENSOR_EXPRESSIONS_HPP
