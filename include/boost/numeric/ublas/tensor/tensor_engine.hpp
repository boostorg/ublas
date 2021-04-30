//
//  Copyright (c) 2020, Amit Singh, amitsingh19975@gmail.com
//  Copyright (c) 2021, Cem Bassoy, cem.bassoy@gmail.com
//
//  Distributed under the Boost Software License, Version 1.0. (See
//  accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)
//
//  The authors gratefully acknowledge the support of
//  Google and Fraunhofer IOSB, Ettlingen, Germany
//

#ifndef BOOST_UBLAS_TENSOR_ENGINE_HPP
#define BOOST_UBLAS_TENSOR_ENGINE_HPP

#include "tensor_core.hpp"

namespace boost::numeric::ublas{

template<typename...>
struct tensor_engine;

template<typename E, typename L, typename S, typename ST>
struct tensor_engine<E,L,S,ST>
{

  using extents_type 	        = E;
  using layout_type 	        = L;
  using strides_type 	        = typename S::template type<layout_type>;
  using storage_traits_type   = storage_traits<ST>;

  static_assert(is_extents_v<extents_type>,
                "boost::numeric::ublas::tensor_engine : please provide valid tensor extents type"
                );

  static_assert(is_strides_v<strides_type>,
                "boost::numeric::ublas::tensor_engine : please provide valid tensor layout type"
                );

};

template<typename E, typename L, typename ST>
struct tensor_engine<E,L,ST> : tensor_engine<E,L,strides<E>,ST> {};

} // namespace boost::numeric::ublas


#endif
