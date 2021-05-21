//
//  Copyright (c) 2018, Cem Bassoy, cem.bassoy@gmail.com
//  Copyright (c) 2019, Amit Singh, amitsingh19975@gmail.com
//
//  Distributed under the Boost Software License, Version 1.0. (See
//  accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)
//
//  The authors gratefully acknowledge the support of
//  Google and Fraunhofer IOSB, Ettlingen, Germany
//

#ifndef BOOST_UBLAS_TENSOR_TENSOR_HPP
#define BOOST_UBLAS_TENSOR_TENSOR_HPP


#include "tensor/tensor_core.hpp"
#include "tensor/tensor_dynamic.hpp"
#include "tensor/tensor_engine.hpp"
#include "tensor/tensor_static_rank.hpp"
#include "tensor/tensor_static.hpp"
#include "concepts.hpp"



#if 0


#include "layout.hpp"
#include "extents/extents_base.hpp"

namespace boost::numeric::ublas{

template<integral T, T...>
class extents_core;

namespace detail{
template<class>                    struct product;
template<std::size_t... es> struct product<extents<es...>> { static constexpr auto value = sizeof...(es)==0 ? 0ul : ( 1 * ... * es ); };
} // namespace detail

template<class E>
constexpr inline auto product_vv = detail::product<E>::value;

template<typename E, typename L, typename ST>
struct tensor_engine;

template< class T >
class tensor_core;

template<
  class value_type,
  class layout_type = layout::first_order
  >
using tensor_dynamic = tensor_core<
  tensor_engine<
    extents<>,
    layout_type,
    std::vector<value_type>
    >
  >;

template<
  class value_type,
  class extents_type,
  class layout_type = layout::first_order>
using tensor_static = tensor_core<
  tensor_engine<
    extents_type,
    layout_type,
    std::array<value_type, product_vv<extents_type>>
    >
  >;

template<
  class value_type,
  std::size_t N,
  class layout_type = layout::first_order
  >
using fixed_rank_tensor = tensor_core<
  tensor_engine<
    extents<N>,
    layout_type,
    std::vector<value_type>
    >
  >;

} // namespace boost::numeric::ublas

#endif

#endif // BOOST_UBLAS_TENSOR_TENSOR_HPP
