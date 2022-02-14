//
// 	Copyright (c) 2018, Cem Bassoy, cem.bassoy@gmail.com
// 	Copyright (c) 2019-2022, Amit Singh, amitsingh19975@gmail.com
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

#include <stdexcept>
#include <type_traits>



#include "extents.hpp"
#include "extents/extents_functions.hpp"
#include "type_traits.hpp"


namespace boost::numeric::ublas {

template<class tensor_engine>
class tensor_core;

} // namespace boost::numeric::ublas

namespace boost::numeric::ublas::detail {

template<class T, class D>
struct tensor_expression;

template<class T, class EL, class ER, class OP>
struct binary_tensor_expression;

template<class T, class E, class OP>
struct unary_tensor_expression;

} // namespace boost::numeric::ublas::detail

namespace boost::numeric::ublas::detail {

template<class T, class E>
struct has_tensor_types 
	: std::integral_constant< bool, same_exp<T,E> >
{};

template<class T, class E>
static constexpr bool has_tensor_types_v = has_tensor_types< std::decay_t<T>, std::decay_t<E> >::value;

template<class T, class D>
struct has_tensor_types<T, tensor_expression<T,D>>
{ 
	static constexpr bool value = 
		same_exp<T,D> || 
		has_tensor_types<T, std::decay_t<D> >::value; 
};

template<class T, class EL, class ER, class OP>
struct has_tensor_types<T, binary_tensor_expression<T,EL,ER,OP>>
{ 
	static constexpr bool value = 
		same_exp<T,EL> || 
		same_exp<T,ER> || 
		has_tensor_types<T, std::decay_t<EL> >::value || 
		has_tensor_types<T, std::decay_t<ER> >::value;  
};

template<class T, class E, class OP>
struct has_tensor_types<T, unary_tensor_expression<T,E,OP>>
{ 
	static constexpr bool value = 
		same_exp<T,E> || 
		has_tensor_types<T, std::decay_t<E> >::value; 
};

} // namespace boost::numeric::ublas::detail


namespace boost::numeric::ublas::detail
{

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
	static_assert(has_tensor_types_v<T,tensor_expression<T,D>>,
	              "Error in boost::numeric::ublas::detail::retrieve_extents: Expression to evaluate should contain tensors.");

	auto const& cast_expr = expr();

	if constexpr ( same_exp<T,D> )
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
	static_assert(has_tensor_types_v<T,binary_tensor_expression<T,EL,ER,OP>>,
	              "Error in boost::numeric::ublas::detail::retrieve_extents: Expression to evaluate should contain tensors.");

	auto const& lexpr = expr.left_expr();
	auto const& rexpr = expr.right_expr();

	if constexpr ( same_exp<T,EL> )
	    return lexpr.extents();

	else if constexpr ( same_exp<T,ER> )
	    return rexpr.extents();

	else if constexpr ( has_tensor_types_v<T,EL> )
	    return retrieve_extents(lexpr);

	else if constexpr ( has_tensor_types_v<T,ER> )
	    return retrieve_extents(rexpr);
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

	static_assert(has_tensor_types_v<T,unary_tensor_expression<T,E,OP>>,
	              "Error in boost::numeric::ublas::detail::retrieve_extents: Expression to evaluate should contain tensors.");
	
	auto const& uexpr = expr.expr();

	if constexpr ( same_exp<T,E> )
	    return uexpr.extents();

	else if constexpr ( has_tensor_types_v<T,E>  )
	    return retrieve_extents(uexpr);
}

} // namespace boost::numeric::ublas::detail


///////////////

namespace boost::numeric::ublas::detail {

template<class EN, std::size_t ... es>
[[nodiscard]] inline
  constexpr auto all_extents_equal(tensor_core<EN> const& t, extents<es...> const& e)
{
  return ::operator==(e,t.extents());
}

template<class T, class D, std::size_t ... es>
[[nodiscard]]
constexpr auto all_extents_equal(tensor_expression<T,D> const& expr, extents<es...> const& e)
{

	static_assert(has_tensor_types_v<T,tensor_expression<T,D>>,
	              "Error in boost::numeric::ublas::all_extents_equal: Expression to evaluate should contain tensors.");

	auto const& cast_expr = expr();

	using ::operator==;
	using ::operator!=;

	if constexpr ( same_exp<T,D> )
      if( e != cast_expr.extents() )
	    return false;

	if constexpr ( has_tensor_types_v<T,D> )
      if ( !all_extents_equal(cast_expr, e))
	    return false;

	return true;

}

template<class T, class EL, class ER, class OP, std::size_t... es>
[[nodiscard]]
constexpr auto all_extents_equal(binary_tensor_expression<T,EL,ER,OP> const& expr, extents<es...> const& e)
{
	static_assert(has_tensor_types_v<T,binary_tensor_expression<T,EL,ER,OP>>,
	              "Error in boost::numeric::ublas::all_extents_equal: Expression to evaluate should contain tensors.");

	using ::operator==;
	using ::operator!=;

	auto const& lexpr = expr.left_expr();
	auto const& rexpr = expr.right_expr();

	if constexpr ( same_exp<T,EL> )
      if(e !=  lexpr.extents())
	    return false;

	if constexpr ( same_exp<T,ER> )
      if(e != rexpr.extents())
	    return false;

	if constexpr ( has_tensor_types_v<T,EL> )
      if(!all_extents_equal(lexpr, e))
	    return false;

	if constexpr ( has_tensor_types_v<T,ER> )
      if(!all_extents_equal(rexpr, e))
	    return false;

	return true;
}


template<class T, class E, class OP, std::size_t... es>
[[nodiscard]]
constexpr auto all_extents_equal(unary_tensor_expression<T,E,OP> const& expr, extents<es...> const& e)
{
	static_assert(has_tensor_types_v<T,unary_tensor_expression<T,E,OP>>,
	              "Error in boost::numeric::ublas::all_extents_equal: Expression to evaluate should contain tensors.");

  	using ::operator==;

	auto const& uexpr = expr.expr();

	if constexpr ( same_exp<T,E> )
      if(e != uexpr.extents())
	    	return false;

	if constexpr ( has_tensor_types_v<T,E> )
      if(!all_extents_equal(uexpr, e))
	    	return false;

	return true;
}

} // namespace boost::numeric::ublas::detail


namespace boost::numeric::ublas::detail
{

/** @brief Evaluates expression for a tensor_core
 *
 * Applies a binary function to the results of the expressions before the assignment.
 * Usually applied needed for binary operators such as A += C;
 *
 * \note Checks if shape of the tensor_core matches those of all tensors within the expression.
*/
template<class TensorEngine, class BinaryFn>
	requires 
		std::is_invocable_r_v<void, BinaryFn, 
			typename tensor_core<TensorEngine>::reference,
			typename tensor_core<TensorEngine>::const_reference
		>
inline void eval(tensor_core<TensorEngine>& lhs, TensorExpression auto const& expr, BinaryFn&& fn)
{
	using rtensor_t   = typename std::decay_t<decltype(expr)>::tensor_type;
	using ltensor_t   = tensor_core<TensorEngine>;
	using lvalue_type = typename ltensor_t::value_type;
	using rvalue_type = typename rtensor_t::value_type;
	using lextents_t  = typename ltensor_t::extents_type;
	using rextents_t  = typename rtensor_t::extents_type;

	static_assert(std::is_same_v<lvalue_type, rvalue_type>,
		"boost::numeric::ublas::detail::eval(tensor_core<TensorEngine>&, TensorExpression auto const&) : "
		"both LHS and RHS tensors should have same value type"
	);

	if constexpr(is_static_v<lextents_t> && is_static_v<rextents_t>){
		static_assert(std::is_same_v<lextents_t,rextents_t>,
			"boost::numeric::ublas::tensor_core: "
			"both LHS and RHS tensors should have same shape."
		);
	}else{
		if ( !all_extents_equal( expr, lhs.extents() ) ){
			throw std::runtime_error("Error in boost::numeric::ublas::tensor_core: expression contains tensors with different shapes.");
		}   	
	}

	auto const& rhs = cast_tensor_expression(expr);

	#pragma omp parallel for
	for(auto i = 0u; i < lhs.size(); ++i)
		std::invoke(fn, lhs(i), rhs(i));
}

/** @brief Evaluates expression for a tensor_core
 *
 * Assigns the results of the expression to the tensor_core.
 *
 * \note Checks if shape of the tensor_core matches those of all tensors within the expression.
*/
template<typename TensorEngine>
inline void eval(tensor_core<TensorEngine>& lhs, TensorExpression auto const& expr)
{
	eval(lhs, expr, [](auto& l, auto const& r){
		l = r;
	});
}



/** @brief Evaluates expression for a tensor_core
 *
 * Applies a unary function to the results of the expressions before the assignment.
 * Usually applied needed for unary operators such as A += C;
 *
 * \note Checks if shape of the tensor_core matches those of all tensors within the expression.
*/
template<class TensorEngine, class UnaryFn>
	requires std::is_invocable_r_v<void, UnaryFn, typename tensor_core<TensorEngine>::reference>
inline void eval(tensor_core<TensorEngine>& lhs, UnaryFn&& fn)
{
	#pragma omp parallel for
	for(auto i = 0u; i < lhs.size(); ++i)
		std::invoke(fn, lhs(i));
}


} // namespace boost::numeric::ublas::detail
#endif
