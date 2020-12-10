//
// 	Copyright (c) 2018-2020, Cem Bassoy, cem.bassoy@gmail.com
// 	Copyright (c) 2019-2020, Amit Singh, amitsingh19975@gmail.com
//
//  Distributed under the Boost Software License, Version 1.0. (See
//  accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)
//
//  The authors gratefully acknowledge the support of
//  Google and Fraunhofer IOSB, Ettlingen, Germany
//

#ifndef BOOST_UBLAS_TENSOR_EXPRESSIONS_EVALUATION_HPP
#define BOOST_UBLAS_TENSOR_EXPRESSIONS_EVALUATION_HPP

#include <type_traits>
#include <stdexcept>
#include <boost/numeric/ublas/tensor/type_traits.hpp>


namespace boost::numeric::ublas {

template<typename T>
class tensor_core;

template<typename ExtentType>
class basic_extents;

}

namespace boost::numeric::ublas::detail {

template<class T, class D>
struct tensor_expression;

template<class T, class EL, class ER, class OP>
struct binary_tensor_expression;

template<class T, class E, class OP>
struct unary_tensor_expression;

}

namespace boost::numeric::ublas::detail {

template<class T, class E>
struct has_tensor_types
{ static constexpr bool value = false; };

template<class T>
struct has_tensor_types<T,T>
{ static constexpr bool value = true; };

template<class T, class D>
struct has_tensor_types<T, tensor_expression<T,D>>
{ static constexpr bool value = std::is_same<T,D>::value || has_tensor_types<T,D>::value; };

template<class T, class EL, class ER, class OP>
struct has_tensor_types<T, binary_tensor_expression<T,EL,ER,OP>>
{ static constexpr bool value = std::is_same<T,EL>::value || std::is_same<T,ER>::value || has_tensor_types<T,EL>::value || has_tensor_types<T,ER>::value;  };

template<class T, class E, class OP>
struct has_tensor_types<T, unary_tensor_expression<T,E,OP>>
{ static constexpr bool value = std::is_same<T,E>::value || has_tensor_types<T,E>::value; };

} // namespace boost::numeric::ublas::detail


namespace boost::numeric::ublas::detail {





/** @brief Retrieves extents of the tensor_core
 *
*/
template<class TensorEngine>
[[nodiscard]]
constexpr auto& retrieve_extents(tensor_core<TensorEngine> const& t)
{
	return t.extents();
}

/** @brief Retrieves extents of the tensor_core expression
 *
 * @note tensor_core expression must be a binary tree with at least one tensor_core type
 *
 * @returns extents of the child expression if it is a tensor_core or extents of one child of its child.
*/
template<class T, class D>
[[nodiscard]]
constexpr auto& retrieve_extents(tensor_expression<T,D> const& expr)
{
	static_assert(detail::has_tensor_types<T,tensor_expression<T,D>>::value,
	              "Error in boost::numeric::ublas::detail::retrieve_extents: Expression to evaluate should contain tensors.");

	auto const& cast_expr = static_cast<D const&>(expr);

	if constexpr ( std::is_same<T,D>::value )
	    return cast_expr.extents();
	else
	return retrieve_extents(cast_expr);
}

// Disable warning for unreachable code for MSVC compiler
#ifdef _MSC_VER
	#pragma warning( push )
	#pragma warning( disable : 4702 )
#endif
/** @brief Retrieves extents of the binary tensor_core expression
 *
 * @note tensor_core expression must be a binary tree with at least one tensor_core type
 *
 * @returns extents of the (left and if necessary then right) child expression if it is a tensor_core or extents of a child of its (left and if necessary then right) child.
*/
template<class T, class EL, class ER, class OP>
[[nodiscard]]
constexpr auto& retrieve_extents(binary_tensor_expression<T,EL,ER,OP> const& expr)
{
	static_assert(detail::has_tensor_types<T,binary_tensor_expression<T,EL,ER,OP>>::value,
	              "Error in boost::numeric::ublas::detail::retrieve_extents: Expression to evaluate should contain tensors.");

	if constexpr ( std::is_same<T,EL>::value )
	    return expr.el.extents();

	if constexpr ( std::is_same<T,ER>::value )
	    return expr.er.extents();

	else if constexpr ( detail::has_tensor_types<T,EL>::value )
	    return retrieve_extents(expr.el);

	else if constexpr ( detail::has_tensor_types<T,ER>::value  )
	    return retrieve_extents(expr.er);
}

#ifdef _MSC_VER
	#pragma warning( pop )
#endif

/** @brief Retrieves extents of the binary tensor_core expression
 *
 * @note tensor_core expression must be a binary tree with at least one tensor_core type
 *
 * @returns extents of the child expression if it is a tensor_core or extents of a child of its child.
*/
template<class T, class E, class OP>
[[nodiscard]]
constexpr auto& retrieve_extents(unary_tensor_expression<T,E,OP> const& expr)
{

	static_assert(detail::has_tensor_types<T,unary_tensor_expression<T,E,OP>>::value,
	              "Error in boost::numeric::ublas::detail::retrieve_extents: Expression to evaluate should contain tensors.");

	if constexpr ( std::is_same<T,E>::value )
	    return expr.e.extents();

	else if constexpr ( detail::has_tensor_types<T,E>::value  )
	    return retrieve_extents(expr.e);
}

} // namespace boost::numeric::ublas::detail


///////////////

namespace boost::numeric::ublas::detail {

template<class TensorEngine, class Extents>
[[nodiscard]] inline
constexpr auto all_extents_equal(tensor_core<TensorEngine> const& t, Extents const& extents)
{
	static_assert(is_extents_v<Extents>,
		"Error in boost::numeric::ublas::detail::all_extents_equal: extents passed should be of extents type."
	);
	
	return extents == t.extents();
}

template<class T, class D, class Extents>
[[nodiscard]]
constexpr auto all_extents_equal(tensor_expression<T,D> const& expr, Extents const& extents)
{
	static_assert(is_extents_v<Extents>,
		"Error in boost::numeric::ublas::detail::all_extents_equal: extents passed should be of extents type."
	);

	static_assert(detail::has_tensor_types<T,tensor_expression<T,D>>::value,
	              "Error in boost::numeric::ublas::detail::all_extents_equal: Expression to evaluate should contain tensors.");

	auto const& cast_expr = static_cast<D const&>(expr);


	if constexpr ( std::is_same<T,D>::value )
	    if( extents != cast_expr.extents() )
	    return false;

	if constexpr ( detail::has_tensor_types<T,D>::value )
	    if ( !all_extents_equal(cast_expr, extents))
	    return false;

	return true;

}

template<class T, class EL, class ER, class OP, class Extents>
[[nodiscard]]
constexpr auto all_extents_equal(binary_tensor_expression<T,EL,ER,OP> const& expr, Extents const& extents)
{
	static_assert(is_extents_v<Extents>,
		"Error in boost::numeric::ublas::detail::all_extents_equal: extents passed should be of extents type."
	);

	static_assert(detail::has_tensor_types<T,binary_tensor_expression<T,EL,ER,OP>>::value,
	              "Error in boost::numeric::ublas::detail::all_extents_equal: Expression to evaluate should contain tensors.");

	if constexpr ( std::is_same<T,EL>::value )
	    if(extents !=  expr.el.extents())
	    return false;

	if constexpr ( std::is_same<T,ER>::value )
	    if(extents != expr.er.extents())
	    return false;

	if constexpr ( detail::has_tensor_types<T,EL>::value )
	    if(!all_extents_equal(expr.el, extents))
	    return false;

	if constexpr ( detail::has_tensor_types<T,ER>::value )
	    if(!all_extents_equal(expr.er, extents))
	    return false;

	return true;
}


template<class T, class E, class OP, class Extents>
[[nodiscard]]
constexpr auto all_extents_equal(unary_tensor_expression<T,E,OP> const& expr, Extents const& extents)
{
	static_assert(is_extents_v<Extents>,
		"Error in boost::numeric::ublas::detail::all_extents_equal: extents passed should be of extents type."
	);

	static_assert(detail::has_tensor_types<T,unary_tensor_expression<T,E,OP>>::value,
	              "Error in boost::numeric::ublas::detail::all_extents_equal: Expression to evaluate should contain tensors.");

	if constexpr ( std::is_same<T,E>::value )
	    if(extents != expr.e.extents())
	    	return false;

	if constexpr ( detail::has_tensor_types<T,E>::value )
	    if(!all_extents_equal(expr.e, extents))
	    	return false;

	return true;
}

} // namespace boost::numeric::ublas::detail


namespace boost::numeric::ublas::detail {


/** @brief Evaluates expression for a tensor_core
 *
 * Assigns the results of the expression to the tensor_core.
 *
 * \note Checks if shape of the tensor_core matches those of all tensors within the expression.
*/
template<class tensor_type, class derived_type>
inline void eval(tensor_type& lhs, tensor_expression<tensor_type, derived_type> const& expr)
{
	if constexpr (detail::has_tensor_types<tensor_type, tensor_expression<tensor_type,derived_type> >::value )
	    if(!detail::all_extents_equal(expr, lhs.extents() ))
	    	throw std::runtime_error("Error in boost::numeric::ublas::tensor_core: expression contains tensors with different shapes.");

#pragma omp parallel for
	for(auto i = 0u; i < lhs.size(); ++i)
		lhs(i) = expr()(i);
}

/** @brief Evaluates expression for a tensor_core
 *
 * Assigns the results of the expression to the tensor_core.
 *
 * \note Checks if shape of the tensor_core matches those of all tensors within the expression.
*/
template<typename tensor_type, typename other_tensor_type, typename derived_type>
inline void eval(tensor_type& lhs, tensor_expression<other_tensor_type, derived_type> const& expr)
{

	static_assert(is_valid_tensor_v<tensor_type> && is_valid_tensor_v<other_tensor_type>,
		"boost::numeric::ublas::detail::eval(tensor_type&, tensor_expression<other_tensor_type, derived_type> const&) : "
		"tensor_type and tensor_expresssion should be a valid tensor type"
	);

	static_assert(std::is_same_v<typename tensor_type::value_type, typename other_tensor_type::value_type>,
		"boost::numeric::ublas::detail::eval(tensor_type&, tensor_expression<other_tensor_type, derived_type> const&) : "
		"tensor_type and tensor_expresssion should have same value type"
	);

	if ( !detail::all_extents_equal(expr, lhs.extents() ) ){
		throw std::runtime_error("Error in boost::numeric::ublas::tensor_core: expression contains tensors with different shapes.");
	}   	
	
	#pragma omp parallel for
	for(auto i = 0u; i < lhs.size(); ++i)
		lhs(i) = expr()(i);
}

/** @brief Evaluates expression for a tensor_core
 *
 * Applies a unary function to the results of the expressions before the assignment.
 * Usually applied needed for unary operators such as A += C;
 *
 * \note Checks if shape of the tensor_core matches those of all tensors within the expression.
*/
template<class tensor_type, class derived_type, class unary_fn>
inline void eval(tensor_type& lhs, tensor_expression<tensor_type, derived_type> const& expr, unary_fn const fn)
{

	if constexpr (detail::has_tensor_types< tensor_type, tensor_expression<tensor_type,derived_type> >::value )
	    if(!detail::all_extents_equal( expr, lhs.extents() ))
	    	throw std::runtime_error("Error in boost::numeric::ublas::tensor_core: expression contains tensors with different shapes.");

	#pragma omp parallel for
	for(auto i = 0u; i < lhs.size(); ++i)
		fn(lhs(i), expr()(i));
}



/** @brief Evaluates expression for a tensor_core
 *
 * Applies a unary function to the results of the expressions before the assignment.
 * Usually applied needed for unary operators such as A += C;
 *
 * \note Checks if shape of the tensor_core matches those of all tensors within the expression.
*/
template<class tensor_type, class unary_fn>
inline void eval(tensor_type& lhs, unary_fn const& fn)
{
#pragma omp parallel for
	for(auto i = 0u; i < lhs.size(); ++i)
		fn(lhs(i));
}


}
#endif
