//
//  Copyright (c) 2018-2020, Cem Bassoy, cem.bassoy@gmail.com
//  Copyright (c) 2019-2020, Amit Singh, amitsingh19975@gmail.com
//
//  Distributed under the Boost Software License, Version 1.0. (See
//  accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)
//
//  The authors gratefully acknowledge the support of
//  Google and Fraunhofer IOSB, Ettlingen, Germany
//

#ifndef BOOST_UBLAS_TENSOR_TYPE_TRAITS_STTAIC_STRIDES_HPP
#define BOOST_UBLAS_TENSOR_TYPE_TRAITS_STTAIC_STRIDES_HPP

#include <boost/numeric/ublas/tensor/traits/basic_type_traits.hpp>

namespace boost::numeric::ublas{
    
template <class ExtentsType, ExtentsType... E> class basic_static_extents;

template <class E, class L> class basic_static_strides;

} // namespace boost::numeric::ublas

namespace boost::numeric::ublas{
      
  template <class L, class T, T... E>
  struct is_strides< basic_static_strides< basic_static_extents<T, E...>, L > > : std::true_type {};

  template <class T, T... E, class L>
  struct is_static< basic_static_strides< basic_static_extents<T, E...>, L > > : std::true_type {};

  template <class T, T... E, class L>
  struct is_static_rank< basic_static_strides< basic_static_extents<T, E...>, L > > : std::true_type {};

  template <class T, T... Extents>
  struct strides<basic_static_extents<T, Extents...>>
  {
      template<typename Layout>
      using type = basic_static_strides<basic_static_extents<T, Extents...>, Layout>;
  };

} // namespace boost::numeric::ublas

#endif
