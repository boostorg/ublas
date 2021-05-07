//
//  Copyright (c) 2021, Cem Bassoy, cem.bassoy@gmail.com
//  Copyright (c) 2019, Amit Singh, amitsingh19975@gmail.com
//
//  Distributed under the Boost Software License, Version 1.0. (See
//  accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)
//
//  The authors gratefully acknowledge the support of
//  Google and Fraunhofer IOSB, Ettlingen, Germany
//


#ifndef BOOST_NUMERIC_UBLAS_TENSOR_DETAIL_TENSOR_CORE_HPP
#define BOOST_NUMERIC_UBLAS_TENSOR_DETAIL_TENSOR_CORE_HPP

#include "../type_traits.hpp"
#include <tuple>

namespace boost::numeric::ublas::detail {

template <typename ExtentType,
          std::enable_if_t<is_dynamic_rank_v<ExtentType>, void>* = nullptr>
[[nodiscard]] constexpr ExtentType make_extent_from_legacy_ublas(std::size_t e1,
                                                                 std::size_t e2)
{
  return {e1, e2};
}

template <
  typename ExtentType,
  std::enable_if_t<is_static_rank_v<ExtentType> && !is_dynamic_v<ExtentType>,
                   void>* = nullptr>
[[nodiscard]] constexpr ExtentType make_extent_from_legacy_ublas(std::size_t e1,
                                                                 std::size_t e2)
{
  static_assert(
    std::tuple_size_v<ExtentType> == 2ul,
    "boost::numeric::ublas::detail::make_extent_from_legacy_ublas(std::size_t, "
    "std::size_t)"
    " : the rank of extents is not correct, it should be of the rank 2");

  if (ExtentType::at(0) != e1 ||
      ExtentType::at(1) != e2) {
    throw std::runtime_error(
      "boost::numeric::ublas::detail::make_extent_from_legacy_ublas(std::size_"
      "t, std::size_t)"
      " : please set the extents properly, the extents should contain the row "
      "and col of the matrix");
  }
  return {};
}

template <
  typename ExtentType,
  std::enable_if_t<is_static_rank_v<ExtentType> && is_dynamic_v<ExtentType>,
                   void>* = nullptr>
[[nodiscard]] constexpr ExtentType make_extent_from_legacy_ublas(std::size_t e1,
                                                                 std::size_t e2)
{
  if (ublas::size(ExtentType{}) != 2ul) {
    throw std::runtime_error(
      "boost::numeric::ublas::detail::make_extent_from_legacy_ublas(std::size_"
      "t, std::size_t)"
      " : the rank of extents is not correct, it should be of the rank 2");
  }
  return {e1, e2};
}

}   // namespace boost::numeric::ublas::detail


#endif   // BOOST_NUMERIC_UBLAS_TENSOR_DETAIL_TENSOR_CORE_HPP
