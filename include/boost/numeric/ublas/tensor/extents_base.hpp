//
//  Copyright (c) 2020, Cem Bassoy, cem.bassoy@gmail.com
//
//  Distributed under the Boost Software License, Version 1.0. (See
//  accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)
//


#ifndef BOOST_NUMERIC_UBLAS_TENSOR_EXTENTS_BASE_HPP
#define BOOST_NUMERIC_UBLAS_TENSOR_EXTENTS_BASE_HPP


namespace boost::numeric::ublas {


template<class derived_type>
struct extents_base
{
  inline constexpr decltype(auto) operator()() const { return static_cast<const derived_type&>(*this); }
  inline constexpr decltype(auto) operator()()       { return static_cast<      derived_type&>(*this); }
};

} // namespace boost::numeric::ublas

#endif // _BOOST_NUMERIC_UBLAS_TENSOR_EXTENTS_BASE_HPP_
