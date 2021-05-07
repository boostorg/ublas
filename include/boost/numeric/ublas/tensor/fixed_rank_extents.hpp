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


#ifndef BOOST_NUMERIC_UBLAS_TENSOR_FIXED_RANK_EXTENTS_HPP
#define BOOST_NUMERIC_UBLAS_TENSOR_FIXED_RANK_EXTENTS_HPP

#include <array>
#include <initializer_list>
#include <limits>
#include <stdexcept>


#include "detail/extents_functions.hpp"
#include "extents_base.hpp"
#include "type_traits.hpp"


namespace boost::numeric::ublas
{

/** @brief Class template for storing static-number of extents
 *
 * @code basic_fixed_rank_extents<unsigned,3> t; @endcode
 *
 * @tparam T element type of the extents
 * @tparam N number of extents
 *
 *
 */
template <class T, std::size_t N>
class basic_fixed_rank_extents : public extents_base<basic_fixed_rank_extents<T,N>>
{
  static_assert( std::is_unsigned_v<T> );
public:
  using base_type       = std::array<T,N>;
  using value_type      = typename base_type::value_type;
  using size_type       = typename base_type::size_type;
  using reference       = typename base_type::reference;
  using const_reference = typename base_type::const_reference;
  using const_pointer   = typename base_type::const_pointer;
  using const_iterator  = typename base_type::const_iterator;
  using const_reverse_iterator = typename base_type::const_reverse_iterator;

  constexpr basic_fixed_rank_extents() = default;

  constexpr explicit basic_fixed_rank_extents(base_type const& data)
    : _base(data)
  {
    if ( !is_valid(*this) ){
      throw std::invalid_argument("Error in basic_fixed_rank_extents : could not construct shape tuple"
        "shape tuple is not a valid permutation: has zero elements."
        );
    }
  }

  constexpr explicit basic_fixed_rank_extents(base_type&& data)
    : _base(std::move(data))
  {
    if ( !is_valid(*this) ){
      throw std::invalid_argument("Error in basic_fixed_rank_extents::basic_fixed_rank_extents(base_type &&) : "
        "shape tuple is not a valid permutation: has zero elements."
        );
    }
  }

  constexpr basic_fixed_rank_extents(basic_fixed_rank_extents const& other)
    : _base(other._base)
  {}

  constexpr basic_fixed_rank_extents(basic_fixed_rank_extents && other) noexcept
    : _base( std::move(other._base) )
  {}

  constexpr basic_fixed_rank_extents& operator=(basic_fixed_rank_extents const& other)
    noexcept(std::is_nothrow_swappable_v<base_type>)
  {
    basic_fixed_rank_extents temp(other);
    swap(*this,temp);
    return *this;
  }

  constexpr basic_fixed_rank_extents& operator=(basic_fixed_rank_extents && other)
    noexcept(std::is_nothrow_swappable_v<base_type>)
  {
    swap(*this,other);
    return *this;
  }

  ~basic_fixed_rank_extents() = default;

  constexpr basic_fixed_rank_extents(std::initializer_list<value_type> const& li)
    : _base()
  {
    if( li.size() != this->base().size() ){
      throw std::out_of_range("boost::numeric::ublas::basic_fixed_rank_extents: "
        "number of elements in std::initializer_list is greater than the size"
        );
    }

    std::copy(li.begin(), li.end(), _base.begin());

    if ( !is_valid(*this) ){
      throw std::invalid_argument("Error in boost::numeric::ublas::basic_fixed_rank_extents : "
        "shape tuple is not a valid permutation: has zero elements."
        );
    }
  }

  constexpr basic_fixed_rank_extents(const_iterator begin, const_iterator end)
  {
    if( std::distance(begin,end) < 0 || static_cast<std::size_t>(std::distance(begin,end)) > this->base().size()){
      throw std::out_of_range("Error in boost::numeric::ublas::basic_fixed_rank_extents(const_iterator, const_iterator): "
        "initializer list size is greater than the rank");
    }

    std::copy(begin, end, _base.begin());

    if ( !is_valid(*this) ) {
      throw std::length_error("Error in boost::numeric::ublas::basic_fixed_rank_extents(const_iterator, const_iterator) : "
        "shape tuple is not a valid permutation: has zero elements." );
    }
  }

  template<class OT>
  constexpr explicit basic_fixed_rank_extents(basic_fixed_rank_extents<OT,N> const& e)
  {
    std::copy(e.begin(), e.end(), _base.begin());
  }

  friend void swap(basic_fixed_rank_extents& lhs, basic_fixed_rank_extents& rhs)
    noexcept(std::is_nothrow_swappable_v<base_type>)
  {
    std::swap(lhs._base   , rhs._base   );
  }

  [[nodiscard]] inline constexpr const_reference at(size_type k) const { return this->_base.at(k); }
  [[nodiscard]] inline constexpr reference       at(size_type k)       { return this->_base.at(k); }

  [[nodiscard]] inline constexpr const_reference operator[](size_type k) const { return this->_base[k]; }
  [[nodiscard]] inline constexpr reference       operator[](size_type k)       { return this->_base[k]; }

  [[nodiscard]] inline constexpr const_iterator          begin() const noexcept { return this->_base.begin (); }
  [[nodiscard]] inline constexpr const_iterator          end  () const noexcept { return this->_base.end   (); }
  [[nodiscard]] inline constexpr const_reverse_iterator rbegin() const noexcept { return this->_base.rbegin(); }
  [[nodiscard]] inline constexpr const_reverse_iterator rend  () const noexcept { return this->_base.rend  (); }
  [[nodiscard]] inline constexpr base_type const&       base  () const noexcept { return this->_base;          }

private:
  base_type _base{};
};

} // namespace boost::numeric::ublas


namespace boost::numeric::ublas{
template <class T, std::size_t R> struct is_extents     < basic_fixed_rank_extents<T,R> > : std::true_type {};
template <class T, std::size_t R> struct is_dynamic     < basic_fixed_rank_extents<T,R> > : std::true_type {};
template <class T, std::size_t R> struct is_static_rank < basic_fixed_rank_extents<T,R> > : std::true_type {};
} // namespace boost::numeric::ublas

namespace std
{
template< class T, std::size_t N >
class tuple_size < boost::numeric::ublas::basic_fixed_rank_extents<T, N> > : public integral_constant<std::size_t, N> {};

template<size_t I, class T, size_t N>
[[nodiscard]] constexpr T get(boost::numeric::ublas::basic_fixed_rank_extents<T, N> const& e) noexcept{
  static_assert(I < N, 
    "std::get<I>(boost::numeric::ublas::basic_fixed_rank_extents<T, N>) : "
    "out of bound access"
  );
  return e[I];
}

} // namespace std



#endif
