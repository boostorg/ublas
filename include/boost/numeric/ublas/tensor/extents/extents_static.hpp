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

#ifndef BOOST_NUMERIC_UBLAS_TENSOR_EXTENTS_STATIC_HPP
#define BOOST_NUMERIC_UBLAS_TENSOR_EXTENTS_STATIC_HPP

#include <array>
#include <initializer_list>
#include <limits>
#include <tuple>

#include "extents_functions.hpp"
#include "extents_base.hpp"
#include "../concepts.hpp"


namespace boost::numeric::ublas {


/** @brief Template class for storing tensor extents for compile time.
 *
 * @code extents<1,2,3,4> t @endcode
 *
 * @tparam e parameter pack of extents
 *
 */
template <integral T, T e1, T e2, T ... e>
class extents_core<T,e1,e2,e...>
  : public extents_base<extents_core<T,e1,e2, e...>>
{
  static constexpr auto size = sizeof...(e)+2u;
public:
  
  using base_type       = std::array<T,size>;
  using value_type      = typename base_type::value_type;
  using size_type       = typename base_type::size_type;
  using reference       = typename base_type::reference;
  using const_reference = typename base_type::const_reference;
  using const_pointer   = typename base_type::const_pointer;
  using const_iterator  = typename base_type::const_iterator;
  using const_reverse_iterator = typename base_type::const_reverse_iterator;

  constexpr extents_core() = default;
  constexpr extents_core(extents_core const&) noexcept = default;
  constexpr extents_core(extents_core &&) noexcept = default;
  constexpr extents_core& operator=(extents_core const&) noexcept = default;
  constexpr extents_core& operator=(extents_core &&) noexcept = default;
  ~extents_core() = default;

  [[nodiscard]] inline constexpr const_reference  at        (size_type k) const{ return m_data.at(k);  }
  [[nodiscard]] inline constexpr const_reference  operator[](size_type k) const{ return m_data[k];     }
  [[nodiscard]] inline constexpr base_type const& base()         const noexcept{ return m_data; }
  [[nodiscard]] inline constexpr const_pointer    data ()        const noexcept{ return m_data.data(); }

private:
  static constexpr base_type const m_data{e1,e2,e...};

};

template <integral T, T e1, T e2, T... e> struct is_extents     < extents_core<T,e1, e2, e...> > : std::true_type {};
template <integral T, T e1, T e2, T... e> struct is_static      < extents_core<T,e1, e2, e...> > : std::true_type {};
template <integral T, T e1, T e2, T... e> struct is_static_rank < extents_core<T,e1, e2, e...> > : std::true_type {};

} // namespace boost::numeric::ublas




#endif
