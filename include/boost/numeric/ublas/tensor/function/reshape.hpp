//
//  Copyright (c) 2021, Cem Bassoy, cem.bassoy@gmail.com
//  Copyright (c) 2021, Amit Singh, amitsingh19975@gmail.com
//
//  Distributed under the Boost Software License, Version 1.0. (See
//  accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)
//
//  The authors gratefully acknowledge the support of
//  Google and Fraunhofer IOSB, Ettlingen, Germany
//


#ifndef BOOST_NUMERIC_UBLAS_TENSOR_RESHAPE_HPP
#define BOOST_NUMERIC_UBLAS_TENSOR_RESHAPE_HPP

#include "../extents.hpp"
#include "../tensor.hpp"

namespace boost::numeric::ublas{

/** Enables prod(ttt) if E1 or E1 is dynamic extents with static rank
 *
 * extents< > & extents<N>
 * extents<N> & extents< >
 * extents< > & extents< >
 *
 */
template<
  class TE,
  class  E = typename tensor_core< TE >::extents_type
  >
using enable_reshape_if_shape_is_dynamic = std::enable_if_t< is_dynamic_v<E> , bool >;

/** @brief Reshapes the basic_tensor
 *
 *
 * (1) @code auto b = a.reshape(extents{m,n,o});     @endcode or
 * (2) @code auto b = a.reshape(extents{m,n,o},4);   @endcode
 *
 * If the size of this smaller than the specified extents than
 * default constructed (1) or specified (2) value is appended.
 *
 * @note rank of the basic_tensor might also change.
 *
 * @param e extents with which the basic_tensor is reshaped.
 * @param v value which is appended if the basic_tensor is enlarged.
 */
template< class E, class D,
          enable_reshape_if_shape_is_dynamic<E> = true>
[[nodiscard]] constexpr auto reshape (tensor_core<E> const& t, extents_base<D> const& e, [[maybe_unused]] typename tensor_core<E>::value_type v = {})
{
  using from_engine_type     = E;
  using from_tensor_type     = tensor_core<from_engine_type>;
//  using from_extents_type    = typename from_tensor_type::extents_type;
  using from_container_type  = typename from_tensor_type::container_type;
  using from_layout_type     = typename from_tensor_type::layout_type;

  using to_extents_type    = D;
  using to_engine_type     = tensor_engine<to_extents_type,from_layout_type,from_container_type>;
  using to_tensor_type     = tensor_core<to_engine_type>;

  auto const& efrom  = t.extents();
  auto const& eto    = e();

  if( ::operator==(efrom,eto) )
    return t;

  auto const to_size   = product(eto);
  auto const from_size = product(efrom);


  auto r = to_tensor_type(eto);
  const auto m = std::min(to_size,from_size);
  std::copy(t.begin()  , t.begin()+m ,r.begin()  );

  if(m < to_size){
    const auto n = to_size - m;
    std::fill_n(r.begin()+m,n,v);
  }

  return r;
}

} // namespace boost::numeric::ublas

#endif   // BOOST_NUMERIC_UBLAS_TENSOR_PROD_COMMON_HPP
