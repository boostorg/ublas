//
//  Copyright (c) 2021, Cem Bassoy, cem.bassoy@gmail.com
//
//  Distributed under the Boost Software License, Version 1.0. (See
//  accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)
//
//


/// \file strides_base.hpp Definition for the basic_strides template class


#ifndef _BOOST_NUMERIC_UBLAS_TENSOR_STRIDES_BASE_HPP_
#define _BOOST_NUMERIC_UBLAS_TENSOR_STRIDES_BASE_HPP_

namespace boost::numeric::ublas {


template<class derived_type>
struct strides_base
{
  inline constexpr decltype(auto) operator()() const { return static_cast<const derived_type&>(*this); }
  inline constexpr decltype(auto) operator()()       { return static_cast<      derived_type&>(*this); }
};

} // namespace boost::numeric::ublas

#endif // _BOOST_NUMERIC_UBLAS_TENSOR_EXTENTS_BASE_HPP_
