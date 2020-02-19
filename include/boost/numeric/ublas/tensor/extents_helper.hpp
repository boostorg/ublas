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

#ifndef BOOST_NUMERIC_UBLAS_TENSOR_EXTENTS_HELPER_HPP
#define BOOST_NUMERIC_UBLAS_TENSOR_EXTENTS_HELPER_HPP

#include <boost/numeric/ublas/tensor/shape_helper.hpp>
#include <initializer_list>
#include "meta_functions.hpp"

namespace boost::numeric::ublas::detail
{

/** @brief basic_extents_impl specialization for basic_shape
 *
 * tparam R of type ptrdiff_t which stands for Rank
 *
 */
template <ptrdiff_t R>
struct basic_extents_impl<R, basic_shape<>>
{
  // aliases the basic_extents_impl
  using next = basic_extents_impl;

  // stores the rank
  static constexpr ptrdiff_t Rank = 0;
  // stores the dynamic rank
  static constexpr ptrdiff_t DynamicRank = 0;

  static constexpr bool IsDynamic = false;
  // stores the both static
  static constexpr ptrdiff_t N = 1;

  ptrdiff_t S{1};

  /**
   * @returns extent at a given index
   **/
  TENSOR_AUTO_CONSTEXPR_RETURN at(int) const noexcept { return ptrdiff_t{1}; }

  TENSOR_AUTO_CONSTEXPR_RETURN &step(int) noexcept { return this->S; }
  
  TENSOR_AUTO_CONSTEXPR_RETURN &step(int) const noexcept { return this->S; }
  /**
   * @returns extent at a given index
   **/
  TENSOR_AUTO_CONSTEXPR_RETURN operator[](int) const noexcept { return at(0); }
  /**
   * @returns product of extents
   **/
  TENSOR_AUTO_CONSTEXPR_RETURN product(int) const noexcept { return ptrdiff_t{1}; }
  /**
   * @returns product of extents
   **/
  TENSOR_AUTO_CONSTEXPR_RETURN product() const noexcept { return ptrdiff_t{1}; }

  //@returns true if empty otherwise false
  TENSOR_AUTO_CONSTEXPR_RETURN empty() const noexcept { return true; }

  /**
   * @returns Rank of the extents
   **/
  TENSOR_AUTO_CONSTEXPR_RETURN size() const noexcept { return 0u; }

  // default constructor
  constexpr basic_extents_impl() noexcept = default;
  // copy constructor
  constexpr basic_extents_impl(basic_extents_impl const &other) noexcept =
      default;
  // copy assignment operator
  constexpr basic_extents_impl &
  operator=(basic_extents_impl const &other) noexcept = default;

  /** @brief assigns the value of dynamic extent
   *
   * @tparam IndexType
   * @param li of type std::initializer_list
   *
   **/
  template <typename IndexType>
  explicit constexpr basic_extents_impl(std::initializer_list<IndexType> li) {}

  /** @brief assigns the value of dynamic extent
   *
   * @tparam Iterator of type input_iterator and its reference value should be
   *integral
   * @param begin begining of a iterator
   * @param end ending of a iterator
   *
   **/
  template <typename Iterator>
  constexpr basic_extents_impl(Iterator, Iterator,
                               iterator_tag) {}

  template <typename Iterator>
  basic_extents_impl(Iterator, Iterator, invalid_iterator_tag)
  {
    throw std::runtime_error("boost::numeric::ublas::detail::basic_extents_impl: invalid iterator type");
  }

  ~basic_extents_impl() = default;
};

template <ptrdiff_t R, ptrdiff_t... E>
struct basic_extents_impl<R, basic_shape<dynamic_extent, E...>>
    : basic_extents_impl<R + 1, basic_shape<E...>>
{
  using next = basic_extents_impl<R + 1, basic_shape<E...>>;

  // store rank
  static constexpr ptrdiff_t Rank = 1 + next::Rank;
  // stores dynamic rank
  static constexpr ptrdiff_t DynamicRank = 1 + next::DynamicRank;

  static constexpr bool IsDynamic = true;
  // stores dynamic extent
  ptrdiff_t N{0};

  ptrdiff_t S{1};

  /**
   * @param k index of extent
   * @returns extent at a given index
   **/
  TENSOR_AUTO_CONSTEXPR_RETURN at(int k) const noexcept { return k == R ? N : next::at(k); }

  TENSOR_AUTO_CONSTEXPR_RETURN &step(int k) noexcept
  {
    return ( k == R ? S : next::step(k) );
  }

  TENSOR_AUTO_CONSTEXPR_RETURN &step(int k) const noexcept
  {
    return ( k == R ? S : next::step(k) );
  }
  /**
   * @param k index of extent
   * @returns extent at a given index
   **/
  TENSOR_AUTO_CONSTEXPR_RETURN operator[](int k) const noexcept { return at(k); }

  /**
   * @param k index of extent
   * @returns product of extents
   **/
  TENSOR_AUTO_CONSTEXPR_RETURN product(int k) const noexcept
  {
    return k == R ? N * next::product() : next::product(k);
  }

  /**
   * @returns product of extents
   **/
  TENSOR_AUTO_CONSTEXPR_RETURN product() const noexcept { return N * next::product(); }

  /**
   * @returns Rank of the extents
   **/
  TENSOR_AUTO_CONSTEXPR_RETURN size() const noexcept { return size_t(Rank); }

  //@returns true if empty otherwise false
  TENSOR_AUTO_CONSTEXPR_RETURN empty() const noexcept { return false; }

  // default constructor
  constexpr basic_extents_impl() noexcept : next(), N(0) {}
  // copy constructor
  constexpr basic_extents_impl(basic_extents_impl const &other) noexcept =
      default;
  // copy assignment operator
  constexpr basic_extents_impl &
  operator=(basic_extents_impl const &other) noexcept = default;

  /** @brief assigns the value of dynamic extent
   *
   * @tparam IndexType parameter pack
   * @param DynamicExtents parameter pack of indices
   *
   **/
  template <typename... IndexType>
  explicit constexpr basic_extents_impl(ptrdiff_t extent,
                                        IndexType... DynamicExtents)
      :next(DynamicExtents...), N(extent){
    static_assert(sizeof...(DynamicExtents) + 1 == DynamicRank,
                  "boost::numeric::ublas::detail::basic_extents_impl: number of extents doesnot match the dynamic rank");
    if (extent <= 0)
    {
      throw std::runtime_error("boost::numeric::ublas::detail::basic_extents_impl: extent should be greater than 0");
    }
  }

  /** @brief assigns the value of dynamic extent
   *
   * @tparam IndexType
   * @param li of type std::initializer_list
   *
   **/
  template <typename IndexType>
  explicit constexpr basic_extents_impl(std::initializer_list<IndexType> li)
      : basic_extents_impl(li.begin(), li.end(), iterator_tag{}) {}

  /** @brief assigns the value of dynamic extent
   *
   * @tparam Iterator of type input_iterator and its reference value should be
   *integral
   * @param begin begining of a iterator
   * @param end ending of a iterator
   *
   **/
  template <typename Iterator>
  constexpr basic_extents_impl(Iterator begin, Iterator end,
                               iterator_tag)
      : next(std::next(begin), end, iterator_tag{}), N(*begin) {
    if (*begin <= 0) {
      throw std::runtime_error("boost::numeric::ublas::detail::basic_extents_impl: extent should be greater than 0");
    }
  }

  template <typename Iterator>
  basic_extents_impl(Iterator, Iterator, invalid_iterator_tag)
  {
    throw std::runtime_error("boost::numeric::ublas::detail::basic_extents_impl: invalid iterator type");
  }

  ~basic_extents_impl() = default;
};

template <ptrdiff_t R, ptrdiff_t SE, ptrdiff_t... E>
struct basic_extents_impl<R, basic_shape<SE, E...>>
    : basic_extents_impl<R + 1, basic_shape<E...>>
{

  using next = basic_extents_impl<R + 1, basic_shape<E...>>;
  // stores the rank
  static constexpr ptrdiff_t Rank = 1 + next::Rank;
  // stores dynamic rank
  static constexpr ptrdiff_t DynamicRank = 0 + next::DynamicRank;

  static constexpr bool IsDynamic = false;
  // stores the static extent
  static constexpr ptrdiff_t N = SE;

  ptrdiff_t S{1};
  /**
   * @param k index of extent
   * @returns extent at given index
   **/
  TENSOR_AUTO_CONSTEXPR_RETURN at(int k) const noexcept { return k == R ? N : next::at(k); }

  TENSOR_AUTO_CONSTEXPR_RETURN &step(int k) noexcept
  {
    return ( k == R ? S : next::step(k) );
  }

  TENSOR_AUTO_CONSTEXPR_RETURN &step(int k) const noexcept
  {
    return ( k == R ? S : next::step(k) );
  }

  /**
   * @param k index of extent
   * @returns extent at a given index
   **/
  TENSOR_AUTO_CONSTEXPR_RETURN operator[](int k) const noexcept { return at(k); }

  /**
   * @param k index of extent
   * @returns product of extents
   **/
  TENSOR_AUTO_CONSTEXPR_RETURN product(int k) const noexcept
  {
    return k == R ? N * next::product() : next::product(k);
  }

  /**
   * @returns product of extents
   **/
  TENSOR_AUTO_CONSTEXPR_RETURN product() const noexcept { return N * next::product(); }

  /**
   * @returns Rank of the extents
   **/
  TENSOR_AUTO_CONSTEXPR_RETURN size() const noexcept { return size_t(Rank); }

  //@returns true if empty otherwise false
  TENSOR_AUTO_CONSTEXPR_RETURN empty() const noexcept { return false; }

  // default constructor
  constexpr basic_extents_impl() noexcept : next() {}
  // copy constructor
  constexpr basic_extents_impl(basic_extents_impl const &other) noexcept =
      default;
  // copy assignment operator
  constexpr basic_extents_impl &
  operator=(basic_extents_impl const &other) noexcept = default;

  /** @brief assigns the value of dynamic extent
   *
   * @tparam IndexType parameter pack
   * @param DynamicExtents parameter pack of indices
   *
   **/
  template <typename... IndexType>
  explicit constexpr basic_extents_impl(IndexType... DynamicExtents)
      : next(DynamicExtents...)
  {
    static_assert(sizeof...(DynamicExtents) == DynamicRank,
                  "boost::numeric::ublas::detail::basic_extents_impl: number of extents doesnot match the dynamic rank");
  }

  /** @brief assigns the value of dynamic extent
   *
   * @tparam IndexType
   * @param li of type std::initializer_list
   *
   **/
  template <typename IndexType>
  explicit constexpr basic_extents_impl(std::initializer_list<IndexType> li)
      : basic_extents_impl(li.begin(), li.end(), iterator_tag{}) {}

  /** @brief assigns the value of dynamic extent
   *
   * @tparam Iterator of type input_iterator and its reference value should be
   *integral
   * @param begin begining of a iterator
   * @param end ending of a iterator
   *
   **/
  template <typename Iterator>
  constexpr basic_extents_impl(Iterator begin, Iterator end,
                               iterator_tag) noexcept
      : next(begin, end, iterator_tag{}) {}

  template <typename Iterator>
  basic_extents_impl(Iterator, Iterator, invalid_iterator_tag)
  {
    throw std::runtime_error("boost::numeric::ublas::detail::basic_extents_impl: invalid iterator type");
  }

  ~basic_extents_impl() = default;
};

/**
   * @tparam IndexType type of index
   * @tparam Args parameter pack of indices with different types
   * @param idx index of extent
   * @param args parameter pack of indices
   * @returns true if in bound or false if not
   **/
template <size_t depth, class E, class IndexType, class... Args>
constexpr bool in_bounds(E const &e, IndexType const &idx, Args... args)
{
  if constexpr (sizeof...(args) == 0)
  {
    return 0 <= idx && idx < e.at(depth);
  }
  else
  {
    return 0 <= idx && idx < e.at(depth) && in_bounds<depth + 1>(e, args...);
  }
}

/**@returns true if nothing is passed*/
template <class E>
constexpr bool in_bounds(E const &e)
{
  return true;
}

} // namespace boost::numeric::ublas::detail

#endif