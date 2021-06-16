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

#ifndef BOOST_UBLAS_TENSOR_EXPRESSIONS_HPP
#define BOOST_UBLAS_TENSOR_EXPRESSIONS_HPP

#include <cstddef>
#include <boost/numeric/ublas/expression_types.hpp>

#include "tags.hpp"

namespace boost::numeric::ublas::detail
{

/** @brief base class for tensor expressions
 *
 * \note implements crtp - no use of virtual function calls
 *
 * \tparam T type of the tensor
 * \tparam E type of the derived expression (crtp)
 *
 **/
template<class T, class E>
struct tensor_expression
    : public ublas_expression<E>
{
    //	static const unsigned complexity = 0;
    using expression_type = E;
    using type_category = tensor_tag;
    using tensor_type = T;

    inline
    constexpr auto const& operator()() const noexcept { return *static_cast<const expression_type*> (this); }
    
    ~tensor_expression() = default;
    tensor_expression(const tensor_expression&) = delete;
    tensor_expression(tensor_expression&&) noexcept = delete;
    tensor_expression& operator=(const tensor_expression&) = delete;
    tensor_expression& operator=(tensor_expression&&) noexcept = delete;


protected :
    explicit tensor_expression() = default;
};


template<class T, class EL, class ER, class OP>
struct binary_tensor_expression
    : public tensor_expression <T, binary_tensor_expression<T,EL,ER,OP>>
{
    using self_type = binary_tensor_expression<T,EL,ER,OP>;
    using tensor_type  = T;
    using binary_operation = OP;
    using expression_type_left  = EL;
    using expression_type_right = ER;
    using derived_type =  tensor_expression <tensor_type,self_type>;

    using size_type = typename tensor_type::size_type;

    explicit constexpr binary_tensor_expression(expression_type_left  const& l, expression_type_right const& r, binary_operation o) : el(l) , er(r) , op(std::move(o)) {}
    constexpr binary_tensor_expression(binary_tensor_expression&& l) noexcept = delete;
    constexpr binary_tensor_expression& operator=(binary_tensor_expression&& l) noexcept = delete;
    ~binary_tensor_expression() = default;

    binary_tensor_expression() = delete;
    binary_tensor_expression(const binary_tensor_expression& l) = delete;
    binary_tensor_expression& operator=(binary_tensor_expression const& l) noexcept = delete;



    [[nodiscard]] inline 
    constexpr decltype(auto) operator()(size_type i) const { return op(el(i), er(i)); }

    expression_type_left const& el;
    expression_type_right const& er;
    binary_operation op;
};

/// @brief helper function to simply instantiation of lambda proxy class
template<class T1, class T2, class EL, class ER, class OP>
[[nodiscard]] inline
constexpr auto make_binary_tensor_expression( tensor_expression<T1,EL> const& el, tensor_expression<T2,ER> const& er, OP op) noexcept
{
    static_assert( std::is_same_v< typename T1::value_type, typename T2::value_type>,
        "boost::numeric::ublas::make_binary_tensor_expression(T1,T2) : LHS tensor and RHS tensor should have same value type"
    );
    return binary_tensor_expression<T1,EL,ER,OP>( el(), er(), op) ;
}

template<class T, class EL, class ER, class OP>
[[nodiscard]] inline
constexpr auto make_binary_tensor_expression( matrix_expression<EL> const& el, tensor_expression<T,ER> const& er, OP op) noexcept
{
    return binary_tensor_expression<T,EL,ER,OP>( el(), er(), op) ;
}

template<class T, class EL, class ER, class OP>
[[nodiscard]] inline
constexpr auto make_binary_tensor_expression( tensor_expression<T,EL> const& el, matrix_expression<ER> const& er, OP op) noexcept
{
    return binary_tensor_expression<T,EL,ER,OP>( el(), er(), op) ;
}

template<class T, class EL, class ER, class OP>
[[nodiscard]] inline
constexpr auto make_binary_tensor_expression( vector_expression<EL> const& el, tensor_expression<T,ER> const& er, OP op) noexcept
{
    return binary_tensor_expression<T,EL,ER,OP>( el(), er(), op) ;
}

template<class T, class EL, class ER, class OP>
[[nodiscard]] inline
constexpr auto make_binary_tensor_expression( tensor_expression<T,EL> const& el, vector_expression<ER> const& er, OP op) noexcept
{
    return binary_tensor_expression<T,EL,ER,OP>( el(), er(), op) ;
}



template<class T, class E, class OP>
struct unary_tensor_expression
    : public tensor_expression <T, unary_tensor_expression<T,E,OP>>
{

    using self_type = unary_tensor_expression<T,E,OP>;
    using tensor_type  = T;
    using expression_type = E;
    using unary_operation = OP;
    using derived_type = tensor_expression <T, unary_tensor_expression<T,E,OP>>;

    using size_type = typename tensor_type::size_type;

    explicit constexpr unary_tensor_expression(expression_type const& ee, unary_operation o) : e(ee) , op(std::move(o)) {}
    constexpr unary_tensor_expression(unary_tensor_expression&& l) noexcept = delete;
    constexpr unary_tensor_expression& operator=(unary_tensor_expression&& l) noexcept = delete;

    constexpr unary_tensor_expression() = delete;
    unary_tensor_expression(unary_tensor_expression const& l) = delete;
    unary_tensor_expression& operator=(unary_tensor_expression const& l) noexcept = delete;
    ~unary_tensor_expression() = default;
    
    [[nodiscard]] inline constexpr
      decltype(auto) operator()(size_type i) const { return op(e(i)); }

    E const& e;
    OP op;
};

// \brief helper function to simply instantiation of lambda proxy class
template<class T, class E, class OP>
[[nodiscard]] inline
constexpr auto make_unary_tensor_expression( tensor_expression<T,E> const& e, OP op) noexcept
{
    return unary_tensor_expression<T,E,OP>( e() , op);
}

template<class T, class E, class OP>
[[nodiscard]] inline
constexpr auto make_unary_tensor_expression( matrix_expression<E> const& e, OP op) noexcept
{
    return unary_tensor_expression<T,E,OP>( e() , op);
}

template<class T, class E, class OP>
[[nodiscard]] inline
constexpr auto make_unary_tensor_expression( vector_expression<E> const& e, OP op) noexcept
{
    return unary_tensor_expression<T,E,OP>( e() , op);
}


} // namespace boost::numeric::ublas::detail

#endif // BOOST_UBLAS_TENSOR_EXPRESSIONS_HPP
