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


#ifndef BOOST_NUMERIC_UBLAS_TENSOR_EXTENTS_STATIC_SIZE_HPP
#define BOOST_NUMERIC_UBLAS_TENSOR_EXTENTS_STATIC_SIZE_HPP

#include <array>
#include <initializer_list>
#include <limits>
#include <stdexcept>
#include <tuple>


#include "extents_functions.hpp"
#include "extents_base.hpp"

#include "../layout.hpp"
#include "../concepts.hpp"


namespace boost::numeric::ublas
{

/** @brief Class template for storing static-number of extents
 *
 * @code auto e = extents<3>{3,2,4}; @endcode
 *
 * @tparam N number of extents
 *
 */
template <integral T, T N>
class extents_core<T,N> : public extents_base<extents_core<T,N>>
{
public:
  using base_type       = std::array<std::size_t,N>;
  using value_type      = typename base_type::value_type;
  using size_type       = typename base_type::size_type;
  using reference       = typename base_type::reference;
  using const_reference = typename base_type::const_reference;
  using const_pointer   = typename base_type::const_pointer;
  using const_iterator  = typename base_type::const_iterator;
  using const_reverse_iterator = typename base_type::const_reverse_iterator;

  constexpr extents_core() = default;

  constexpr explicit extents_core(base_type data)
    : _base(std::move(data))
  {
    if ( !ublas::is_valid(*this) ){
      throw std::invalid_argument("in boost::numeric::ublas::extents<N> : "
        "could not intanstiate extents<N> as provided extents are not valid.");
    }
  }


  constexpr extents_core(std::initializer_list<value_type> const& li)
    : _base()
  {
    if( li.size() != ublas::size(*this) ){
      throw std::length_error("in boost::numeric::ublas::extents<N> : "
        "could not intanstiate extents<N> as number of indices exceed N.");
    }

    std::copy(li.begin(), li.end(), _base.begin());

    if ( !ublas::is_valid(*this) ){
      throw std::invalid_argument("in boost::numeric::ublas::extents<N> : "
        "could not intanstiate extents<N> as provided extents are not valid.");
    }
  }

  constexpr extents_core(const_iterator begin, const_iterator end)
  {
    if( std::distance(begin,end) < 0 || static_cast<std::size_t>(std::distance(begin,end)) > this->base().size()){
      throw std::out_of_range("in boost::numeric::ublas::extents<N> : "
        "initializer list size is greater than the rank");
    }

    std::copy(begin, end, _base.begin());

    if ( !ublas::is_valid(*this) ) {
      throw std::invalid_argument("in boost::numeric::ublas::extents<N>::ctor: "
        "could not intanstiate extents<N> as provided extents are not valid.");
    }
  }
  constexpr extents_core(extents_core const& other)
    : _base(other._base)
  {
    assert(ublas::is_valid(*this));
  }

  constexpr extents_core(extents_core && other) noexcept
    : _base( std::move(other._base) )
  {
  }

  // NOLINTNEXTLINE(cppcoreguidelines-special-member-functions,hicpp-special-member-functions)
  constexpr extents_core& operator=(extents_core other)
    noexcept(std::is_nothrow_swappable_v<base_type>)
  {
    swap(*this,other);
    return *this;
  }

  ~extents_core() = default;


  friend void swap(extents_core& lhs, extents_core& rhs)
    noexcept(std::is_nothrow_swappable_v<base_type>)
  {
    std::swap(lhs._base, rhs._base);
  }

  [[nodiscard]] inline constexpr const_reference at        (size_type k) const { return this->_base.at(k); }
  [[nodiscard]] inline constexpr const_reference operator[](size_type k) const { return this->_base[k]; }
  [[nodiscard]] inline constexpr auto const&     base ()        const noexcept { return this->_base; }
  [[nodiscard]] inline constexpr const_pointer   data ()        const noexcept { return this->_base.data(); }


private:
  base_type _base{};
};

} // namespace boost::numeric::ublas




namespace boost::numeric::ublas{
template <integral T, T n> struct is_extents     < extents_core<T,n> > : std::true_type {};
template <integral T, T n> struct is_dynamic     < extents_core<T,n> > : std::true_type {};
template <integral T, T n> struct is_static_rank < extents_core<T,n> > : std::true_type {};
} // namespace boost::numeric::ublas




#endif
