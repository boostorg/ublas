//
//  Copyright (c) 2018-2019, Cem Bassoy, cem.bassoy@gmail.com
//
//  Distributed under the Boost Software License, Version 1.0. (See
//  accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)
//
//  The authors gratefully acknowledge the support of
//  Fraunhofer IOSB, Ettlingen, Germany
//
/// \file strides.hpp Definition for the basic_strides template class

#ifndef BOOST_UBLAS_TENSOR_STRIDES_HPP
#define BOOST_UBLAS_TENSOR_STRIDES_HPP

#include <boost/numeric/ublas/tensor/dynamic_strides.hpp>
#include <boost/numeric/ublas/tensor/static_strides.hpp>

namespace boost::numeric::ublas {

template <class __int_type, class E, class __layout>
bool operator==(basic_strides<__int_type, __layout> const &lhs,
                static_strides<E, __layout> const &rhs) {
  if (lhs.size() != rhs.size()) {
    return false;
  }
  for (auto i = 0u; i < lhs.size(); i++) {
    if (lhs.at(i) != rhs.at(i)) {
      return false;
    }
  }
  return true;
}

template <class __int_type, class E, class __layout>
bool operator!=(basic_strides<__int_type, __layout> const &lhs,
                static_strides<E, __layout> const &rhs) {
  return !(lhs == rhs);
}

template <class __int_type, class E, class __layout>
bool operator==(static_strides<E, __layout> const &lhs,
                basic_strides<__int_type, __layout> const &rhs) {
  return rhs == lhs;
}

template <class __int_type, class E, class __layout>
bool operator!=(static_strides<E, __layout> const &lhs,
                basic_strides<__int_type, __layout> const &rhs) {
  return !(rhs == lhs);
}
} // namespace boost::numeric::ublas

#endif
