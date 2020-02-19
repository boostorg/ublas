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

#ifndef BOOST_NUMERIC_UBLAS_TENSOR_STATIC_EXTENTS_HPP
#define BOOST_NUMERIC_UBLAS_TENSOR_STATIC_EXTENTS_HPP

#include <boost/config.hpp>

#include <array>
#include <boost/numeric/ublas/tensor/dynamic_extents.hpp>
#include <boost/numeric/ublas/tensor/extents_helper.hpp>
#include <initializer_list>
#include <vector>
#include "fwd.hpp"

namespace boost::numeric::ublas {

/** @brief Template class for storing tensor extents for compile time.
 *
 * @code basic_static_extents<4,1,2,3,4> t @endcode
 * @tparam R rank of extents of type ptrdiff_t
 * @tparam E parameter pack of extents
 *
 */
template <class int_type, ptrdiff_t R, ptrdiff_t... E>
struct basic_static_extents<int_type,R,E...>
    : detail::basic_extents_impl<0, detail::make_basic_shape_t<R, E...>> {

  static_assert(sizeof...(E) == 0 || sizeof...(E) == R,
                "boost::numeric::ublas::basic_static_extents: number of extents should be equal to rank of extents");
  using parent_type = detail::basic_extents_impl<0, detail::make_basic_shape_t<R, E...>>;
  using base_type = std::vector<int_type>;
  using array_type = std::array<int_type,R>;
	using value_type = typename base_type::value_type;
	using const_reference = typename base_type::const_reference;
	using reference = typename base_type::reference;
	using const_pointer = typename base_type::const_pointer;
	using const_iterator = typename base_type::const_iterator;
	using size_type = std::size_t;

  //@returns the rank of basic_static_extents
  TENSOR_STATIC_AUTO_CONSTEXPR_RETURN size() noexcept { return impl::Rank; }
  
  //@returns the rank of basic_static_extents
  TENSOR_STATIC_AUTO_CONSTEXPR_RETURN rank() noexcept { return impl::Rank; }
  
  //@returns the dynamic rank of basic_static_extents
  TENSOR_STATIC_AUTO_CONSTEXPR_RETURN dynamic_rank() noexcept { return impl::DynamicRank; }

  /**
   * @param k pos of extent
   * @returns the element at given pos
   */
  TENSOR_AUTO_CONSTEXPR_RETURN at(size_type k) const { 
    if( k >= this->size() ) {
      throw std::out_of_range("boost::numeric::ublas::basic_static_extents: Out Of Bound");
    }else{
      return impl::at(k); 
    }
  }

  // default constructor
  constexpr basic_static_extents() = default;
 
  // default copy constructor
  constexpr basic_static_extents(basic_static_extents const &other) = default;
 
  // default assign constructor
  constexpr basic_static_extents &operator=(basic_static_extents const &other) = default;

  /** @brief assigns the extents to dynamic extents using parameter pack
   *
   * @code basic_static_extents<2> e( 2,3 ); @endcode
   *
   * @tparam IndexType
   *
   * @param DynamicExtents parameter pack of extents
   *
   * @note number of extents should be equal to dynamic rank
   */
  template <class... IndexType>
  constexpr basic_static_extents(IndexType... DynamicExtents)
      : impl(DynamicExtents...) {
    static_assert(sizeof...(DynamicExtents) == impl::DynamicRank,"boost::numeric::ublas::basic_static_extents: number of extents should be equal to rank of extents");
  }

  /** @brief assigns the extents to dynamic extents using initializer_list
   *
   * @code basic_static_extents<2> e = { 2, 3}; @endcode
   *
   * @tparam IndexType
   *
   * @param li of type initializer_list which constains the extents
   *
   * @note number of extents should be equal to dynamic rank
   */
  constexpr basic_static_extents(base_type const& b)
      : basic_static_extents(b.begin(), b.end()) {
  }

  /** @brief assigns the extents to dynamic extents using iterator
   *
   * @code basic_static_extents<2> e( a.begin(), a.end() ); @endcode
   *
   * @tparam I of type input iterator and valur_type should be integral
   *
   * @param begin start of iterator
   *
   * @param end end of iterator
   *
   * @note number of extents should be equal to dynamic rank
   *
   */
  template <class I, std::enable_if_t<detail::is_iterator<I>::value, int> = 0>
  constexpr basic_static_extents(I begin, I end)
      : impl(begin, end, detail::iterator_tag_t<I>{}) {
    if constexpr (std::is_same<detail::iterator_tag_t<I>,detail::iterator_tag>::value) {
      if ( std::distance(begin,end) != impl::DynamicRank) {
        throw std::runtime_error("boost::numeric::ublas::basic_static_extents: number of extents should be equal to rank of extents");
      }
    }
  }

  /** @brief Returns the std::vector containing extents */
  TENSOR_AUTO_RETURN to_vector() const{
    base_type temp(R);
    for (auto i = 0u; i < R; i++) {
      temp[i] = this->at(i);
    }
    return temp;
  }

  /** @brief Returns the std::vector containing extents */
  TENSOR_AUTO_RETURN base() const {
    return this->to_vector();
  }

  /** @brief Returns the std::array containing extents */
  TENSOR_AUTO_CONSTEXPR_RETURN to_array() const{
    array_type temp{};
    for (auto i = 0u; i < temp.size(); i++) {
      temp[i] = this->at(i);
    }
    return temp;
  }

  /** @brief Returns the basic_extents containing extents */
  TENSOR_AUTO_CONSTEXPR_RETURN to_dynamic_extents() const{
    return basic_extents<value_type>(this->to_vector());
  }

  /** @brief Checks if extents is empty or not
   *
   * @returns true if rank is 0 else false
   *
   */
  TENSOR_AUTO_CONSTEXPR_RETURN empty() const noexcept { return this->size() == size_type{0}; }

  /** @brief Returns true if both extents are equal else false */
  template <ptrdiff_t rhs_dims, ptrdiff_t... rhs>
  TENSOR_AUTO_CONSTEXPR_RETURN operator==(basic_static_extents<int_type,rhs_dims, rhs...> const &other) const{
    if (this->size() != other.size()) {
      return false;
    }
    for (auto i = 0u; i < this->size(); i++) {
      if (other.at(i) != this->at(i))
        return false;
    }
    return true;
  }

  /** @brief Returns false if both extents are equal else true */
  template <ptrdiff_t rhs_dims, ptrdiff_t... rhs>
  TENSOR_AUTO_CONSTEXPR_RETURN operator!=(basic_static_extents<int_type,rhs_dims, rhs...> const &other) const{
    return !(*this == other);
  }

  ~basic_static_extents() = default;

protected:
  using impl =
      detail::basic_extents_impl<0, detail::make_basic_shape_t<R, E...>>;
};

#if __cpp_deduction_guides
template<class T, class ...E>
basic_static_extents(T,E...) -> basic_static_extents<T,sizeof...(E)+1>;
#endif


/** @brief type alias of basic_extents or basic_static_extents depending on Rank
 *
 * @tparam R rank of extents
 *
 * @tparam E contains the extents as a parameter pack
 *
 */
template <class T, ptrdiff_t R, ptrdiff_t... E>
using shape_t =
    std::conditional_t<(R < 0), basic_extents<T>, basic_static_extents<T,R, E...>>;


template <ptrdiff_t R, ptrdiff_t... E>
using shape = shape_t<std::size_t,R,E...>;


template<ptrdiff_t... E>
using dynamic_extents = typename detail::dynamic_extents_impl<E...>::type;

template<ptrdiff_t... E>
using static_extents = basic_static_extents<std::size_t,sizeof...(E),E...>;

} // namespace boost::numeric::ublas
#endif