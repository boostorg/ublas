//
//  Copyright (c) 2021, Cem Bassoy, cem.bassoy@gmail.com
//
//  Distributed under the Boost Software License, Version 1.0. (See
//  accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)
//

#ifndef BOOST_UBLAS_TENSOR_EXTENTS_HPP
#define BOOST_UBLAS_TENSOR_EXTENTS_HPP

#include "extents/extents_base.hpp"
#include "extents/extents_dynamic_size.hpp"
#include "extents/extents_static_size.hpp"
#include "extents/extents_static.hpp"
#include "extents/extents_functions.hpp"
#include "extents/extents_static_functions.hpp"


template<std::size_t  m, std::size_t  n>
bool operator==(
  boost::numeric::ublas::extents<m> const& lhs,
  boost::numeric::ublas::extents<n> const& rhs )
{
  return size(lhs) == size(rhs) && std::equal( begin(lhs), end  (lhs), begin(rhs) );
}

template<std::size_t  n>
bool operator==(
  boost::numeric::ublas::extents<n> const& lhs,
  boost::numeric::ublas::extents<n> const& rhs )
{
  return size(lhs) == size(rhs) && std::equal( begin(lhs), end  (lhs), begin(rhs) );
}

template<std::size_t  m, std::size_t  n>
bool operator!=(
  boost::numeric::ublas::extents<m> const& lhs,
  boost::numeric::ublas::extents<n> const& rhs )
{
  return !( lhs == rhs) ;
}

template<std::size_t n>
bool operator!=(
  boost::numeric::ublas::extents<n> const& lhs,
  boost::numeric::ublas::extents<n> const& rhs )
{
  return !( lhs == rhs) ;
}


#endif // BOOST_UBLAS_TENSOR_EXTENTS_HPP
