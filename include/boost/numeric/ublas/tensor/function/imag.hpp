//
//  Copyright (c) 2018, Cem Bassoy, cem.bassoy@gmail.com
//  Copyright (c) 2020, Amit Singh, amitsingh19975@gmail.com
//
//  Distributed under the Boost Software License, Version 1.0. (See
//  accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)
//
//

#ifndef BOOST_NUMERIC_UBLAS_TENSOR_IMAG_HPP
#define BOOST_NUMERIC_UBLAS_TENSOR_IMAG_HPP

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

/** @brief Extract the imaginary component of tensor elements within a tensor expression
     *
     * @param[in] lhs tensor expression
     * @returns   unary tensor expression
    */
template<class T, class D>
auto imag(detail::tensor_expression<T,D> const& lhs) {
  return detail::make_unary_tensor_expression<T> (lhs(), [] (auto const& l) { return std::imag( l ); } );
}


/** @brief Extract the imag component of tensor elements within a tensor expression
     *
     * @param[in] lhs tensor expression
     * @returns   unary tensor expression
    */
template<typename TE, class D,
          std::enable_if_t< is_complex_v<typename tensor_core< TE >::value_type>, int > = 0
          >
auto imag(detail::tensor_expression< tensor_core< TE > ,D> const& expr)
{
  using tensor_type       = tensor_core< TE >;
  using complex_type      = typename tensor_type::value_type;
  using value_type        = typename complex_type::value_type;
  using layout_type       = typename tensor_type::layout_type;
  using container_type        = typename tensor_type::container_type;
  using extents_type      = typename tensor_type::extents_type;
  using return_container_type = typename container_traits<container_type>::template rebind<value_type>;

  using return_tensor_type = tensor_core<tensor_engine<extents_type, layout_type, return_container_type >>;

  if( ublas::empty( detail::retrieve_extents( expr  ) ) ){
    throw std::runtime_error("error in boost::numeric::ublas::real: tensors should not be empty.");
  }

  auto a = tensor_type( expr );
  auto c = return_tensor_type( a.extents() );

  std::transform( a.begin(), a.end(),  c.begin(), [](auto const& l){ return std::imag(l) ; }  );

  return c;
}

} // namespace boost::numeric::ublas

#endif   // BOOST_NUMERIC_UBLAS_TENSOR_PROD_DYNAMIC_HPP
