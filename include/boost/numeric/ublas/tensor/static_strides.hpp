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

#ifndef BOOST_UBLAS_TENSOR_STATIC_STRIDES_HPP
#define BOOST_UBLAS_TENSOR_STATIC_STRIDES_HPP

#include <algorithm>
#include <cassert>
#include <initializer_list>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <vector>

#include <boost/numeric/ublas/functional.hpp>
#include <boost/numeric/ublas/tensor/extents.hpp>
#include "fwd.hpp"

namespace boost::numeric::ublas
{

using first_order = column_major;
using last_order = row_major;

/** @brief Partial Specialization for first_order or column_major
 *
 * @code static_strides<basic_static_extents<4,1,2,3,4>, first_order> s @endcode
 *
 * @tparam R rank of basic_static_extents
 * @tparam Extents paramerter pack of extents
 *
 */
template <class T, ptrdiff_t R, ptrdiff_t... Extents>
struct static_strides<basic_static_extents<T, R, Extents...>, first_order>
    : basic_static_extents<T, R, Extents...>
{

  using extents_type = basic_static_extents<T, R, Extents...>;
  using base_type = std::vector<T>;
  using layout_type = first_order;
  using value_type = typename base_type::value_type;
  using reference = typename base_type::reference;
  using const_reference = typename base_type::const_reference;
  using size_type = typename base_type::size_type;
  using const_pointer = typename base_type::const_pointer;
  using const_iterator = typename base_type::const_iterator;

  /**
   * @param k pos of extent
   * @returns the element at given pos
   */
  TENSOR_AUTO_CONSTEXPR_RETURN at(int k) const
  {
    if ((k < 0 || k >= rank()) && !rank())
    {
      throw std::out_of_range(
          "boost::numeric::ublas::static_strides: Out Of Bound");
    }
    return stride(k);
  }

  TENSOR_AUTO_CONSTEXPR_RETURN operator[](int k) const { return at(k); }

  TENSOR_AUTO_CONSTEXPR_RETURN extent(int k) const noexcept { return impl::at(k); }

  //@returns the rank of basic_static_extents
  TENSOR_AUTO_CONSTEXPR_RETURN rank() const noexcept { return impl::Rank; }

  //@returns the rank of basic_static_extents
  TENSOR_AUTO_CONSTEXPR_RETURN size() const noexcept { return impl::Rank; }

  /**
   * @param k pos of extent
   * @returns Returns the number of elements a tensor holds with this from k
   * position ownwards
   */
  TENSOR_AUTO_CONSTEXPR_RETURN product(int k) const noexcept { return impl::product(k); }

  // default constructor
  constexpr static_strides() noexcept = default;

  // constructor for extents
  constexpr static_strides(extents_type const &e) noexcept : extents_type(e){};

  // default copy constructor
  constexpr static_strides(static_strides const &other) noexcept = default;
  // default assign constructor
  constexpr static_strides &
  operator=(static_strides const &other) noexcept = default;

  /** @brief assigns the extents to dynamic extents using parameter pack
   *
   * @code static_stride<basic_static_extents<2>,first_order> e( 2,3 ); @endcode
   *
   * @tparam IndexType
   *
   * @param DynamicExtents parameter pack of extents
   *
   * @note number of extents should be equal to dynamic rank
   */
  template <class... IndexType>
  explicit constexpr static_strides(value_type extent,
                                    IndexType... DynamicExtents) noexcept
      : extents_type(extent, DynamicExtents...) {}

  /** @brief assigns the extents to dynamic extents using initializer_list
   *
   * @code static_stride<basic_static_extents<2>,first_order> e = { 2, 3}; @endcode
   *
   * @tparam IndexType
   *
   * @param li of type initializer_list which constains the extents
   *
   * @note number of extents should be equal to dynamic rank
   */
  template <class Iterator, class = class std::enable_if<std::is_same<
                                detail::iterator_tag_t<Iterator>,
                                detail::iterator_tag>::value>::type>
  constexpr static_strides(Iterator begin, Iterator end,
                           detail::iterator_tag) noexcept
      : extents_type(begin, end, detail::iterator_tag{})
  {
  }

  private :
      /** @brief Base condition for calulating offset or relative memory for
       * indices when there is one index
       *
       * @tparam E type of basic_static_extents
       *
       * @param e basic_static_extents object
       *
       * @param k index
       *
       * @returns the offset when there is only one index left
       */
      template <class E>
      static constexpr value_type
      stride(E const &e, value_type k) noexcept
  {
    return k ? (e.N * stride((typename E::next const &)e, k - 1))
             : value_type{1};
  }

  /** @brief Base condition for calulating offset or relative memory for indices
   * when there is no index
   *
   * @tparam E type of basic_static_extents
   *
   * @param e basic_static_extents object
   *
   * @returns the offset when there is only no index left
   */
  template <class E>
  TENSOR_STATIC_AUTO_CONSTEXPR_RETURN access(E const &) noexcept
  {
    return 0;
  }

  /** @brief Helper function for calulating offset or relative memory for
   * indices
   *
   * @tparam E type of basic_static_extents
   *
   * @tparam IndexType
   *
   * @param e basic_static_extents object
   *
   * @param i_el first index of parameter pack
   *
   * @param idxs remaining indices of paramter pack
   *
   * @returns the offset or relative memory from given indices
   */
  template <class E, class... IndexType>
  TENSOR_STATIC_AUTO_CONSTEXPR_RETURN access(E const &e, value_type i_el,
                                             IndexType const &... idxs) noexcept
  {
    return i_el +
           e.N * static_strides::access((typename E::next const &)e, idxs...);
  }

  using impl = typename extents_type::impl;

public:
  /** @brief Gets the stride for a given position
   *
   * @param k index of the extent
   *
   * @returns the stride at position
   */
  TENSOR_AUTO_CONSTEXPR_RETURN stride(int k) const
  {
    if (this->empty())
      return value_type{1};

    if (!valid(static_cast<extents_type const &>(*this)))
      throw std::runtime_error(
          "Error in boost::numeric::ublas::static_strides() : shape is not "
          "valid.");

    if (is_vector(static_cast<extents_type const &>(*this)) || is_scalar(static_cast<extents_type const &>(*this)))
      return value_type{1};

    if (this->size() < 2)
      throw std::runtime_error(
          "Error in boost::numeric::ublas::static_strides() : size of strides "
          "must be greater or equal 2.");

    return static_cast<value_type>( stride((impl const &)*this, k) * this->step(k));
  }

  // /** @returns 0 if no indices is passed */
  // constexpr auto operator()() const noexcept { return 0; }

  // /** @brief Gets the relative memory or offset for given indices
  //  *
  //  * @tparam IndexType
  //  *
  //  * @param idxs parameter pack of the indices
  //  *
  //  * @returns the offset
  //  */
  // template <class... IndexType>
  // constexpr auto operator()(IndexType const &... idxs) const noexcept {
  //   return static_strides::access((impl const &)*this, idxs...);
  // }

  /** @brief Returns the std::vector containing strides */
  auto base() const
  {
    std::vector<value_type> temp(rank());
    for (auto i = 0u; i < temp.size(); i++)
    {
      temp[i] = this->at(i);
    }
    return temp;
  }
};

/** @brief Forward declaration of static_strides for specialization
 *
 * @code static_strides<basic_static_extents<4,1,2,3,4>, last_order> s @endcode
 *
 * @tparam R rank of basic_static_extents
 * @tparam Extents paramerter pack of extents
 *
 */
template <class T, ptrdiff_t R, ptrdiff_t... Extents>
struct static_strides<basic_static_extents<T, R, Extents...>, last_order>
    : basic_static_extents<T, R, Extents...>
{

  using parent_type = basic_static_extents<T, R, Extents...>;
  using extents_type = parent_type;
  using base_type = std::vector<T>;
  using layout_type = last_order;
  using value_type = typename base_type::value_type;
  using reference = typename base_type::reference;
  using const_reference = typename base_type::const_reference;
  using size_type = typename base_type::size_type;
  using const_pointer = typename base_type::const_pointer;
  using const_iterator = typename base_type::const_iterator;

  /**
   * @param k pos of extent
   * @returns the element at given pos
   */
  TENSOR_AUTO_CONSTEXPR_RETURN at(int k) const
  {
    if ((k < 0 || k >= rank()) && !rank())
    {
      throw std::out_of_range(
          "boost::numeric::ublas::static_strides: Out Of Bound");
    }
    return stride(k);
  }

  TENSOR_AUTO_CONSTEXPR_RETURN operator[](int k) const { return at(k); }

  //@returns the rank of basic_static_extents
  TENSOR_AUTO_CONSTEXPR_RETURN rank() const noexcept { return impl::Rank; }

  //@returns the rank of basic_static_extents
  TENSOR_AUTO_CONSTEXPR_RETURN size() const noexcept { return impl::Rank; }

  TENSOR_AUTO_CONSTEXPR_RETURN extent(int k) const noexcept { return impl::at(k); }

  // default constructor
  constexpr static_strides() noexcept = default;

  // constructor for extents
  constexpr static_strides(extents_type const &e) noexcept : extents_type(e){};

  // default copy constructor
  constexpr static_strides(static_strides const &other) noexcept = default;
  // default assign constructor
  constexpr static_strides &
  operator=(static_strides const &other) noexcept = default;

  /** @brief assigns the extents to dynamic extents using parameter pack
   *
   * @code static_stride<basic_static_extents<2>,first_order> e( 2,3 ); @endcode
   *
   * @tparam IndexType
   *
   * @param DynamicExtents parameter pack of extents
   *
   * @note number of extents should be equal to dynamic rank
   */
  template <class... IndexType>
  explicit constexpr static_strides(value_type extent,
                                    IndexType... DynamicExtents) noexcept
      : extents_type(extent, DynamicExtents...) {}

  /** @brief assigns the extents to dynamic extents using initializer_list
   *
   * @code static_stride<basic_static_extents<2>,first_order> e = { 2, 3}; @endcode
   *
   * @tparam IndexType
   *
   * @param li of type initializer_list which constains the extents
   *
   * @note number of extents should be equal to dynamic rank
   */
  template <class Iterator, class = class std::enable_if<std::is_same<
                                detail::iterator_tag_t<Iterator>,
                                detail::iterator_tag>::value>::type>
  constexpr static_strides(Iterator begin, Iterator end,
                           detail::iterator_tag) noexcept
      : extents_type(begin, end, detail::iterator_tag{})
  {
  }

  TENSOR_AUTO_CONSTEXPR_RETURN
  stride(int k) const
  {
    if (this->empty())
      return value_type{1};

    if (!valid(static_cast<extents_type const &>(*this)))
      throw std::runtime_error(
          "Error in boost::numeric::ublas::static_strides() : shape is not "
          "valid.");

    if (is_vector(static_cast<extents_type const &>(*this)) || is_scalar(static_cast<extents_type const &>(*this)))
      return value_type{1};

    if (this->size() < 2)
      throw std::runtime_error(
          "Error in boost::numeric::ublas::static_strides() : size of strides "
          "must be greater or equal 2.");

    return static_cast<value_type>(impl::next::product(k + 1) * this->step(k));
  }

  /** @brief Returns the std::vector containing strides */
  auto base() const
  {
    std::vector<value_type> temp(rank());
    for (auto i = 0u; i < temp.size(); i++)
    {
      temp[i] = this->at(i);
    }
    return temp;
  }

private:
  /** @brief Base condition for calulating offset or relative memory for
   * indices when there is one index
   *
   * @tparam E type of basic_static_extents
   *
   * @param e basic_static_extents object
   *
   * @param sum stores the offset
   *
   * @returns the offset when there is only one index left
   */
  template <class E>
  TENSOR_STATIC_AUTO_CONSTEXPR_RETURN access(E const &, value_type sum) noexcept
  {
    return sum;
  }

  /** @brief Helper function for calulating offset or relative memory for
   * indices
   *
   * @tparam E type of basic_static_extents
   *
   * @tparam IndexType
   *
   * @param e basic_static_extents object
   *
   * @param sum stores the offset
   *
   * @param i_el first index of parameter pack
   *
   * @param idxs remaining indices of paramter pack
   *
   * @returns the offset or relative memory from given indices
   */
  template <class E, class... IndexType>
  TENSOR_STATIC_AUTO_CONSTEXPR_RETURN access(E const &e, value_type sum, value_type i_el,
                                             IndexType const &... idxs) noexcept
  {
    return static_strides::access((typename E::next const &)e, sum * e.N + i_el,
                                  idxs...);
  }

  using impl = typename extents_type::impl;

  // public:
  //   /** @returns 0 if no indices is passed */
  //   constexpr auto operator()() const noexcept { return 0; }

  //   /** @brief Gets the relative memory or offset for given indices
  //    *
  //    * @tparam IndexType
  //    *
  //    * @param idxs parameter pack of the indices
  //    *
  //    * @returns the offset
  //    */
  //   template <class... IndexType>
  //   constexpr auto operator()(IndexType const &... idxs) const noexcept {
  //     return static_strides::access((typename impl::next const &)*this,
  //     idxs...);
  //   }
};

/** @brief Partial Specialization of stride_type for basic_static_extents
 *
 *
 * @tparam Layout either first_order or last_order
 *
 * @tparam R rank of extents
 *
 * @tparam Extents parameter pack of extents
 *
 */
template <class Layout, class T, ptrdiff_t R, ptrdiff_t... Extents>
struct stride_type<basic_static_extents<T, R, Extents...>, Layout>
{
  using type = static_strides<basic_static_extents<T, R, Extents...>, Layout>;
};

/** @brief Partial Specialization of stride_type for basic_extents
 *
 *
 * @tparam Layout either first_order or last_order
 *
 * @tparam T extents type
 *
 */
template <class Layout, class T>
struct stride_type<basic_extents<T>, Layout>
{
  using type = basic_strides<T, Layout>;
};

/** @brief type alias of result of stride_type::type
 *
 * @tparam E extents type either basic_extents or basic_static_extents
 *
 * @tparam Layout either first_order or last_order
 *
 */
template <class E, class Layout>
using strides_t = typename stride_type<E, Layout>::type;

template <class E, class __layout>
bool operator==(static_strides<E, __layout> const &lhs, static_strides<E, __layout> const &rhs)
{
  if (lhs.size() != rhs.size())
  {
    return false;
  }
  for (auto i = 0u; i < lhs.size(); i++)
  {
    if (lhs.at(i) != rhs.at(i))
    {
      return false;
    }
  }
  return true;
}

template <class E, class __layout>
bool operator!=(static_strides<E, __layout> const &lhs, static_strides<E, __layout> const &rhs)
{
  return !(lhs == rhs);
}

} // namespace boost::numeric::ublas

#endif