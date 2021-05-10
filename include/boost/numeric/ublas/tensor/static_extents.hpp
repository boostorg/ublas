//
//  Copyright (c) 2018-2020, Cem Bassoy, cem.bassoy@gmail.com
//  Copyright (c) 2019-2020, Amit Singh, amitsingh19975@gmail.com
//
//  Distributed under the Boost Software License, Version 1.0. (See
//  accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)
//
//  The authors gratefully acknowledge the support of
//  Google and Fraunhofer IOSB, Ettlingen, Germany
//

#ifndef BOOST_NUMERIC_UBLAS_TENSOR_STATIC_EXTENTS_HPP
#define BOOST_NUMERIC_UBLAS_TENSOR_STATIC_EXTENTS_HPP

#include <array>
#include <initializer_list>
#include <limits>
#include <tuple>

#include "detail/extents_functions.hpp"
#include "extents_base.hpp"
#include "traits/basic_type_traits.hpp"


namespace boost::numeric::ublas {

template <class ExtentsType, ExtentsType... E>
class basic_static_extents;

/** @brief Template class for storing tensor extents for compile time.
 *
 * @code basic_static_extents<1,2,3,4> t @endcode
 * @tparam E parameter pack of extents
 *
 */
template <class ExtentsType, ExtentsType... E>
class basic_static_extents : public extents_base<basic_static_extents<ExtentsType, E...> >
{

public:

  static constexpr auto _size = sizeof...(E);
  
  using base_type       = std::array<ExtentsType,_size>;
  using value_type      = typename base_type::value_type;
  using size_type       = typename base_type::size_type;
  using reference       = typename base_type::reference;
  using const_reference = typename base_type::const_reference;
  using const_pointer   = typename base_type::const_pointer;
  using const_iterator  = typename base_type::const_iterator;
  using const_reverse_iterator = typename base_type::const_reverse_iterator;

  static_assert( std::numeric_limits<value_type>::is_integer, "Static error in basic_static_extents: type must be of type integer.");
  static_assert(!std::numeric_limits<value_type>::is_signed,  "Static error in basic_static_extents: type must be of type unsigned integer.");
  
  /**
   * @param k pos of extent
   * @returns the element at given pos
   */
  [[nodiscard]] inline
  static constexpr const_reference at(size_type k){
    return m_data.at(k); 
  }

  [[nodiscard]] inline
  constexpr const_reference operator[](size_type k) const{ 
    return m_data[k]; 
  }

  constexpr basic_static_extents() = default;

  constexpr basic_static_extents(basic_static_extents const&) noexcept = default;
  constexpr basic_static_extents(basic_static_extents &&) noexcept = default;
  
  constexpr basic_static_extents& operator=(basic_static_extents const&) noexcept = default;
  constexpr basic_static_extents& operator=(basic_static_extents &&) noexcept = default;

  ~basic_static_extents() = default;

  /** @brief Returns ref to the std::array containing extents */
  [[nodiscard]] inline
  constexpr base_type const& base() const noexcept{
    return m_data;
  }

  [[nodiscard]] inline
  constexpr const_reference back() const{
    return m_data.back();
  }

  [[nodiscard]] inline
  constexpr const_iterator begin() const noexcept{
    return m_data.begin();
  }

  [[nodiscard]] inline
  constexpr const_iterator end() const noexcept{
    return m_data.end();
  }

  [[nodiscard]] inline
  constexpr const_reverse_iterator
  rbegin() const noexcept
  {
      return m_data.rbegin();
  }

  [[nodiscard]] inline
  constexpr const_reverse_iterator
  rend() const noexcept
  {
      return m_data.rend();
  }

private:
  static constexpr base_type const m_data{E...};
  /// will be removed when we start using boost.mp11
  using tuple_type = std::tuple< std::integral_constant<ExtentsType,E>... >;
};


template<std::size_t... E>
using static_extents = basic_static_extents<std::size_t,E...>;

template <class T, T... E>           struct is_extents     < basic_static_extents<T, E...> > : std::true_type {};
template <class T, T... E>           struct is_static      < basic_static_extents<T, E...> > : std::true_type {};
template <class T, std::size_t... E> struct is_static_rank < basic_static_extents<T, E...> > : std::true_type {};

} // namespace boost::numeric::ublas

namespace std
{
template<std::size_t... E>
struct tuple_size< boost::numeric::ublas::static_extents<E...> >
  : std::integral_constant<std::size_t, sizeof...(E)>
{};

template<size_t I, class T, T... Es>
[[nodiscard]] constexpr
  auto get([[maybe_unused]] boost::numeric::ublas::basic_static_extents<T, Es...> /*e*/) noexcept
{
  static_assert(I < sizeof...(Es));
  return boost::numeric::ublas::basic_static_extents<T, Es...>::at(I);
}

} // namespace std

#endif
