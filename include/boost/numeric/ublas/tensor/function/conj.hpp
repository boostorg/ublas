//
//  Copyright (c) 2018, Cem Bassoy, cem.bassoy@gmail.com
//  Copyright (c) 2020, Amit Singh, amitsingh19975@gmail.com
//
//  Distributed under the Boost Software License, Version 1.0. (See
//  accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)
//
//

#ifndef BOOST_NUMERIC_UBLAS_TENSOR_CONJ_HPP
#define BOOST_NUMERIC_UBLAS_TENSOR_CONJ_HPP

#include <cmath>
#include <stdexcept>
#include <type_traits>


#include "../extents/extents_functions.hpp"
#include "../traits/basic_type_traits.hpp"
#include "../expression.hpp"
#include "../expression_evaluation.hpp"

namespace boost::numeric::ublas
{
template<typename tensor_engine>
class tensor_core;

template<typename extents_type, typename layout_type, typename storage_type>
struct tensor_engine;

} // namespace boost::numeric::ublas

namespace boost::numeric::ublas
{
/** @brief Computes the complex conjugate component of tensor elements within a tensor expression
     *
     * @param[in] lhs tensor expression
     * @returns   unary tensor expression
    */
template<typename TensorEngine, class D,
          std::enable_if_t< is_complex_v<typename tensor_core< TensorEngine >::value_type>, int > = 0
          >
auto conj(detail::tensor_expression< tensor_core<TensorEngine>, D > const& expr)
{
  return detail::make_unary_tensor_expression< tensor_core<TensorEngine> > (expr(), [] (auto const& l) { return std::conj( l ); } );
}

/** @brief Computes the complex conjugate component of tensor elements within a tensor expression
     *
     * @param[in] expr tensor expression
     * @returns   complex tensor
    */
template<class T, class D>
auto conj(detail::tensor_expression<T,D> const& expr)
{
  using tensor_type       = T;
  using value_type        = typename tensor_type::value_type;
  using complex_type      = std::complex<value_type>;
  using layout_type       = typename tensor_type::layout_type;
  using container_type        = typename tensor_type::container_type;
  using extents_type      = typename tensor_type::extents_type;
  using return_container_type = typename container_traits<container_type>::template rebind<complex_type>;
  using return_tensor_type = tensor_core<tensor_engine<extents_type, layout_type, return_container_type >>;

  if( ublas::empty( detail::retrieve_extents( expr  ) ) ){
    throw std::runtime_error("error in boost::numeric::ublas::conj: tensors should not be empty.");
  }

  auto a = tensor_type( expr );
  auto c = return_tensor_type( a.extents() );

  std::transform( a.begin(), a.end(),  c.begin(), [](auto const& l){ return std::conj(l) ; }  );

  return c;
}


} // namespace boost::numeric::ublas

#endif   // BOOST_NUMERIC_UBLAS_TENSOR_PROD_DYNAMIC_HPP
