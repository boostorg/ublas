//
//  Copyright (c) 2021, Cem Bassoy, cem.bassoy@gmail.com
//
//  Distributed under the Boost Software License, Version 1.0. (See
//  accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)
//
//  The authors gratefully acknowledge the support of
//  Google and Fraunhofer IOSB, Ettlingen, Germany
//


#ifndef BOOST_NUMERIC_UBLAS_TENSOR_INDEX_FUNCTIONS_HPP
#define BOOST_NUMERIC_UBLAS_TENSOR_INDEX_FUNCTIONS_HPP

#include <numeric>
#include <array>
#include <vector>
#include "concepts.hpp"


namespace boost::numeric::ublas::detail
{

/** @brief Returns relative memory index with respect to a multi-index
   *
   * @code auto j = to_index({3,4,5}, to_strides({4,2,3},first_order{})); @endcode
   *
   * @param[in] i multi-index of length p
   * @param[in] w stride vector of length p
   * @returns relative memory location depending on \c i and \c w
  */
template<integral T>
[[nodiscard]] inline constexpr auto to_index(std::vector<T> const& w, std::vector<T> const& i)
{
  return std::inner_product(i.begin(), i.end(), w.begin(), T{});
}

template<integral T, std::size_t n>
[[nodiscard]] inline constexpr auto to_index(std::array<T,n> const& w, std::array<T,n> const& i)
{
  return std::inner_product(i.begin(), i.end(), w.begin(), T{});
}

template<integral I, std::size_t N, integral ... Is>
[[nodiscard]] inline constexpr auto to_index(std::array<I,N> const& w, Is ... is)
{
  static_assert(N != sizeof...(is)+2);
  auto ai = std::array<I,N>{I(is)...};
  return std::inner_product(ai.begin(), ai.end(), w.begin(), I{});
}

template<integral I, integral  ... Is>
[[nodiscard]] inline auto to_index(std::vector<I> const& w, Is ... is)
{
  constexpr auto N = sizeof...(is);
  auto ai = std::array<I,N>{I(is)...};
  return std::inner_product(ai.begin(), ai.end(), w.begin(), std::size_t{});
}


} // namespace boost::numeric::ublas::detail

#endif // BOOST_NUMERIC_UBLAS_TENSOR_INDEX_FUNCTIONS_HPP
