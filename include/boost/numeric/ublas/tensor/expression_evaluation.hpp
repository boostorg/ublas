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

template<typename T>
struct is_tensor_type 
	: std::false_type
{};

template<typename E>
struct is_tensor_type< tensor_core<E> > 
	: std::true_type
{};

template<class T>
static constexpr bool is_tensor_type_v = is_tensor_type< std::decay_t<T> >::value;

template<typename T>
struct has_tensor_types 
	: is_tensor_type<T>
{};

template<class T>
static constexpr bool has_tensor_types_v = has_tensor_types< std::decay_t<T> >::value;

template<class T, class D>
struct has_tensor_types< tensor_expression<T,D> > 
	: has_tensor_types< std::decay_t<D> >
{};

template<class T, class EL, class ER, class OP>
struct has_tensor_types< binary_tensor_expression<T,EL,ER,OP> > 
	: std::integral_constant< bool, has_tensor_types_v<EL> || has_tensor_types_v<ER> >
{};

template<class T, class E, class OP>
struct has_tensor_types< unary_tensor_expression<T,E,OP> > 
	: has_tensor_types< std::decay_t<E> >
{};

} // namespace boost::numeric::ublas::detail


namespace boost::numeric::ublas::detail
{


// TODO: remove this place holder for the old ublas expression after we remove the
// support for them.
template<class E>
[[nodiscard]]
constexpr auto& retrieve_extents([[maybe_unused]] ublas_expression<E> const& /*unused*/) noexcept;

/** @brief Retrieves extents of the tensor_core
 *
*/
template<class TensorEngine>
[[nodiscard]]
constexpr auto& retrieve_extents(tensor_core<TensorEngine> const& t) noexcept
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
constexpr auto& retrieve_extents(tensor_expression<T,D> const& expr) noexcept
{
	static_assert(has_tensor_types_v<tensor_expression<T,D>>,
	              "Error in boost::numeric::ublas::detail::retrieve_extents: Expression to evaluate should contain tensors.");

	auto const& cast_expr = expr();
	
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
constexpr auto& retrieve_extents(binary_tensor_expression<T,EL,ER,OP> const& expr) noexcept
{
	static_assert(has_tensor_types_v<binary_tensor_expression<T,EL,ER,OP>>,
	              "Error in boost::numeric::ublas::detail::retrieve_extents: Expression to evaluate should contain tensors.");

	auto const& lexpr = expr.left_expr();
	auto const& rexpr = expr.right_expr();

	if constexpr ( is_tensor_type_v<EL> )
	    return lexpr.extents();

	else if constexpr ( is_tensor_type_v<ER> )
	    return rexpr.extents();

	else if constexpr ( has_tensor_types_v<EL>)
	    return retrieve_extents(lexpr);
	
	else if constexpr ( has_tensor_types_v<ER>)
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
constexpr auto& retrieve_extents(unary_tensor_expression<T,E,OP> const& expr) noexcept
{

	static_assert(has_tensor_types_v<unary_tensor_expression<T,E,OP>>,
	              "Error in boost::numeric::ublas::detail::retrieve_extents: Expression to evaluate should contain tensors.");
	
	auto const& uexpr = expr.expr();

	return retrieve_extents(uexpr);
}

} // namespace boost::numeric::ublas::detail


///////////////

namespace boost::numeric::ublas::detail {

// TODO: remove this place holder for the old ublas expression after we remove the
// support for them.
template<class E, std::size_t ... es>
[[nodiscard]] inline
  constexpr auto all_extents_equal([[maybe_unused]] ublas_expression<E> const& /*unused*/, [[maybe_unused]] extents<es...> const& /*unused*/) noexcept
{
  return true;
}

template<class EN, std::size_t ... es>
[[nodiscard]] inline
  constexpr auto all_extents_equal(tensor_core<EN> const& t, extents<es...> const& e) noexcept
{
  return ::operator==(e,t.extents());
}

template<class T, class D, std::size_t ... es>
[[nodiscard]]
constexpr auto all_extents_equal(tensor_expression<T,D> const& expr, extents<es...> const& e) noexcept
{

	static_assert(has_tensor_types_v<tensor_expression<T,D>>,
	              "Error in boost::numeric::ublas::all_extents_equal: Expression to evaluate should contain tensors.");

	auto const& cast_expr = expr();

    return all_extents_equal(cast_expr, e);
}

template<class T, class EL, class ER, class OP, std::size_t... es>
[[nodiscard]]
constexpr auto all_extents_equal(binary_tensor_expression<T,EL,ER,OP> const& expr, extents<es...> const& e) noexcept
{
	static_assert(has_tensor_types_v<binary_tensor_expression<T,EL,ER,OP>>,
	              "Error in boost::numeric::ublas::all_extents_equal: Expression to evaluate should contain tensors.");

	auto const& lexpr = expr.left_expr();
	auto const& rexpr = expr.right_expr();

	return 	all_extents_equal(lexpr, e) && 
			all_extents_equal(rexpr, e) ;
}


template<class T, class E, class OP, std::size_t... es>
[[nodiscard]]
constexpr auto all_extents_equal(unary_tensor_expression<T,E,OP> const& expr, extents<es...> const& e) noexcept
{
	static_assert(has_tensor_types_v<unary_tensor_expression<T,E,OP>>,
	              "Error in boost::numeric::ublas::all_extents_equal: Expression to evaluate should contain tensors.");

	auto const& uexpr = expr.expr();

	return all_extents_equal(uexpr, e);
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
	noexcept( 
		is_static_v< std::decay_t< decltype(retrieve_extents(lhs)) > > && 
		is_static_v< std::decay_t< decltype(retrieve_extents(expr)) > > 
	)
{
	auto const& rhs = cast_tensor_expression(expr);

	using ltensor_t   = tensor_core<TensorEngine>;
	using lvalue_type = typename ltensor_t::value_type;
	using lextents_t  = typename ltensor_t::extents_type;
	using rvalue_type = std::decay_t< decltype(rhs(0)) >;
	using rextents_t  = std::decay_t< decltype(retrieve_extents(expr)) >;

	static_assert(std::is_same_v<lvalue_type, rvalue_type>,
		"boost::numeric::ublas::detail::eval(tensor_core& lhs, tensor_expresion const& rhs, BinaryFn&& fn) : "
		"both LHS and RHS tensors should have same value type"
	);

	if constexpr(is_static_v<lextents_t> && is_static_v<rextents_t>){
		static_assert(std::is_same_v<lextents_t,rextents_t>,
			"boost::numeric::ublas::detail::eval(tensor_core& lhs, tensor_expresion const& rhs, BinaryFn&& fn) : "
			"both LHS and RHS tensors should have same shape."
		);
	}else{
		if ( !all_extents_equal( expr, lhs.extents() ) ){
			throw std::runtime_error(
				"boost::numeric::ublas::detail::eval(tensor_core& lhs, tensor_expresion const& rhs, BinaryFn&& fn) : "
				"both LHS and RHS tensors should have same shape."
			);
		}   	
	}

	auto const size = lhs.size();

	/// FIXME: add 'simd' clause and 'if' clause that will be used as a starting point
	/// for threads to start, otherwise, it's very expansive to use threads for small
	/// sized containers.
	/// @code #pragma omp parallel for simd if(size > SOME_SIZE) @endcode
	#pragma omp parallel for
	for(auto i = 0u; i < size; ++i)
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
	noexcept( 
		is_static_v< std::decay_t< decltype(retrieve_extents(lhs)) > > && 
		is_static_v< std::decay_t< decltype(retrieve_extents(expr)) > > 
	)
{
	using value_type = typename tensor_core<TensorEngine>::value_type;

	eval(lhs, expr, [](value_type& l, value_type const& r){ l = r; });
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
	noexcept( is_static_v< std::decay_t< decltype(retrieve_extents(lhs)) > > )
{
	auto const size = lhs.size();

	/// FIXME: add 'simd' clause and 'if' clause that will be used as a starting point
	/// for threads to start, otherwise, it's very expansive to use threads for small
	/// sized containers.
	/// @code #pragma omp parallel for simd if(size > SOME_SIZE) @endcode
	#pragma omp parallel for
	for(auto i = 0u; i < size; ++i)
		std::invoke(fn, lhs(i));
}


} // namespace boost::numeric::ublas::detail
#endif
