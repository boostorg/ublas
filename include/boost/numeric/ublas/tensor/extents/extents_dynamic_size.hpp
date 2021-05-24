//
//  Copyright (c) 2018, Cem Bassoy, cem.bassoy@gmail.com
//  Copyright (c) 2019, Amit Singh, amitsingh19975@gmail.com
//
//  Distributed under the Boost Software License, Version 1.0. (See
//  accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)
//
//  The authors gratefully acknowledge the support of
//  Google and Fraunhofer IOSB, Ettlingen, Germany
//


#ifndef BOOST_NUMERIC_UBLAS_TENSOR_EXTENTS_DYNAMIC_SIZE_HPP
#define BOOST_NUMERIC_UBLAS_TENSOR_EXTENTS_DYNAMIC_SIZE_HPP

#include <initializer_list>
#include <limits>
#include <stdexcept>
#include <vector>
#include <cassert>

#include "extents_base.hpp"
#include "extents_functions.hpp"

#include "../layout.hpp"
#include "../concepts.hpp"

namespace boost::numeric::ublas {



/** @brief Template class for storing tensor extents with runtime variable size.
 *
 * Proxy template class of std::vector<int_type>.
 *
 */
template<integral T>
class extents_core<T>
  : public extents_base<extents_core<T>>
{
  using super_type = extents_base<extents_core<T>>;

public:
  using base_type              = std::vector<T>;
  using value_type             = typename base_type::value_type;
  using const_reference        = typename base_type::const_reference;
  using reference              = typename base_type::reference;
  using size_type              = typename base_type::size_type;
  using const_pointer          = typename base_type::const_pointer;
  using const_iterator         = typename base_type::const_iterator;
  using const_reverse_iterator = typename base_type::const_reverse_iterator;


  extents_core() = default;

  explicit extents_core(base_type b)
    : _base(std::move(b))
  {
    if (!ublas::is_valid(*this)){
      throw std::invalid_argument("in boost::numeric::ublas::extents<> : "
        "could not intanstiate extents<> as provided extents are not valid.");
    }
  }

  /** @brief Constructs extents from an initializer list
     *
     * @code auto ex = extents<>{}; @endcode
     * @code auto ex = extents<>{3,2,4}; @endcode
     *
     * @note checks if size > 1 and all elements > 0
     *
     * @param l one-dimensional list of type std::initializer<int_type>
     */
  extents_core(std::initializer_list<value_type> l)
    : extents_core( base_type(l) )
  {
    if (!ublas::is_valid(*this)){
      throw std::invalid_argument("in boost::numeric::ublas::extents<> : "
        "could not intanstiate extents<> as provided extents are not valid.");
    }
  }

  /** @brief Constructs extents from a range specified by two iterators
     *
     * @code auto ex = extents<>(a.begin(), a.end());
     *
     * @note checks if size > 1 and all elements > 0
     *
     * @param first iterator pointing to the first element
     * @param last iterator pointing to the next position after the last element
     */

  template<class InputIt>
  constexpr extents_core(InputIt first, InputIt last)
    : extents_core ( base_type( first,last ) )
  {
    if (!ublas::is_valid(*this)){
      throw std::invalid_argument("in boost::numeric::ublas::extents<> : "
        "could not intanstiate extents<> as provided extents are not valid.");
    }
  }

  /** @brief Copy constructs extents */
  /*constexpr*/ extents_core(extents_core const& l )
    : _base(l._base)
  {
  }

  /** @brief Move constructs extents */
  /*constexpr*/ extents_core(extents_core && l ) noexcept
    : _base(std::move(l._base))
  {
  }

  ~extents_core() = default;

  // NOLINTNEXTLINE(cppcoreguidelines-special-member-functions,hicpp-special-member-functions)
  extents_core& operator=(extents_core other)
    noexcept(std::is_nothrow_swappable_v<base_type>)
  {
    swap (*this, other);
    return *this;
  }


  friend void swap(extents_core& lhs, extents_core& rhs)
        noexcept(std::is_nothrow_swappable_v<base_type>)
  {
    std::swap(lhs._base,rhs._base);
  }

  [[nodiscard]] inline /*constexpr*/ const_reference operator[] (size_type p) const { return this->_base[p];    }
  [[nodiscard]] inline /*constexpr*/ const_reference at         (size_type p) const { return this->_base.at(p); }

  [[nodiscard]] inline /*constexpr*/ auto          size() const noexcept { return this->_base.size(); }
  [[nodiscard]] inline /*constexpr*/ auto const&   base() const noexcept { return this->_base;        }
  [[nodiscard]] inline /*constexpr*/ const_pointer data() const noexcept { return this->_base.data(); }
private:
  base_type _base;
};

} // namespace boost::numeric::ublas


namespace boost::numeric::ublas{
template<integral T> struct is_extents      < extents_core<T> > : std::true_type {};
template<integral T> struct is_dynamic      < extents_core<T> > : std::true_type {};
template<integral T> struct is_dynamic_rank < extents_core<T> > : std::true_type {};

} // namespace boost::numeric::ublas


#endif
