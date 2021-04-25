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

#ifndef BOOST_UBLAS_TENSOR_ENGINE_IMPL_HPP
#define BOOST_UBLAS_TENSOR_ENGINE_IMPL_HPP

#include "tensor_core.hpp"

namespace boost::numeric::ublas{

template<typename...>
struct tensor_engine;

template<typename extents_type__, typename layout_type__, typename strides_type__, typename storage_type__>
struct tensor_engine<extents_type__, layout_type__, strides_type__, storage_type__>
{

  using extents_type 	        = extents_type__;
  using layout_type 	        = layout_type__;
  using strides_type 	        = typename strides_type__::template type<layout_type>;
  using storage_traits_type   = storage_traits<storage_type__>;

  static_assert(is_extents_v<extents_type>,
                "boost::numeric::ublas::tensor_engine : please provide valid tensor extents type"
                );

  static_assert(is_strides_v<strides_type>,
                "boost::numeric::ublas::tensor_engine : please provide valid tensor layout type"
                );

};

template<typename ExtentsType, typename LayoutType, typename StorageType>
struct tensor_engine<ExtentsType, LayoutType, StorageType>
  : tensor_engine< ExtentsType, LayoutType, strides<ExtentsType>, StorageType >
{};

} // namespace boost::numeric::ublas


#endif
