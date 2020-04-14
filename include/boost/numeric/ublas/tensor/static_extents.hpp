//
// 	Copyright (c) 2018-2020, Cem Bassoy, cem.bassoy@gmail.com
// 	Copyright (c) 2019-2020, Amit Singh, amitsingh19975@gmail.com
//
//  Distributed under the Boost Software License, Version 1.0. (See
//  accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)
//
//  The authors gratefully acknowledge the support of
//  Google
//

#ifndef _BOOST_NUMERIC_UBLAS_TENSOR_STATIC_EXTENTS_HPP_
#define _BOOST_NUMERIC_UBLAS_TENSOR_STATIC_EXTENTS_HPP_

#include <array>
#include <initializer_list>
#include <vector>

namespace boost::numeric::ublas {

template <class ExtentsType, ExtentsType... E> struct basic_static_extents;
template <class ExtentsType> class basic_extents;

/** @brief Template class for storing tensor extents for compile time.
 *
 * @code basic_static_extents<1,2,3,4> t @endcode
 * @tparam E parameter pack of extents
 *
 */
template <class ExtentsType, ExtentsType... E>
struct basic_static_extents{

  static constexpr auto _size = sizeof...(E);
  
  using base_type       = std::array<ExtentsType,_size>;
	using value_type      = typename base_type::value_type;
	using const_reference = typename base_type::const_reference;
	using reference       = typename base_type::reference;
	using const_pointer   = typename base_type::const_pointer;
	using const_iterator  = typename base_type::const_iterator;
	using size_type       = typename base_type::size_type;

  static_assert( std::numeric_limits<value_type>::is_integer, "Static error in basic_static_extents: type must be of type integer.");
	static_assert(!std::numeric_limits<value_type>::is_signed,  "Static error in basic_static_extents: type must be of type unsigned integer.");

  //@returns the rank of basic_static_extents
  [[nodiscard]] inline 
  constexpr size_type size() const noexcept { return _size; }
  
  /**
   * @param k pos of extent
   * @returns the element at given pos
   */
  [[nodiscard]] inline
  static constexpr const_reference at(size_type k){
    return m_data.at(k); 
  }

  [[nodiscard]] inline
  constexpr const_reference operator[](size_type k) const noexcept{ 
    return m_data[k]; 
  }

  // default constructor
  constexpr basic_static_extents() = default;

  /** @brief Returns the std::vector containing extents */
  [[nodiscard]] inline
  auto to_vector() const {
    std::vector<value_type> temp(_size);
    std::copy(begin(), end(), temp.begin());
    return temp;
  }

  /** @brief Returns ref to the std::array containing extents */
  [[nodiscard]] inline
  constexpr base_type const& base() const noexcept{
    return m_data;
  }

  /** @brief Returns pointer to the std::array containing extents */
  [[nodiscard]] inline
  constexpr const_pointer data() const noexcept{
    return m_data.data();
  }

  /** @brief Returns the basic_extents containing extents */
  [[nodiscard]] inline
  auto to_dynamic_extents() const {
    return basic_extents<value_type>(this->to_vector());
  }

  /** @brief Checks if extents is empty or not
   *
   * @returns true if rank is 0 else false
   *
   */
  [[nodiscard]] inline
  constexpr bool empty() const noexcept { return m_data.empty(); }

  constexpr value_type back() const noexcept{
    return m_data.back();
  }

  /** @brief Returns true if both extents are equal else false */
  template <ExtentsType... RE>
  [[nodiscard]] inline
  constexpr bool operator==([[maybe_unused]] basic_static_extents<ExtentsType, RE...> const &rhs) const {
    if constexpr( _size != basic_static_extents<ExtentsType, RE...>::_size ){
      return false;
    }else{
      return std::equal(begin(), end(), rhs.begin());
    }
  }

  /** @brief Returns false if both extents are equal else true */
  template <ExtentsType... RE>
  [[nodiscard]] inline
  constexpr bool operator!=(basic_static_extents<ExtentsType, RE...> const &rhs) const {
    return !(*this == rhs);
  }

  [[nodiscard]] inline
  constexpr const_iterator begin() const noexcept{
    return m_data.begin();
  }

  [[nodiscard]] inline
  constexpr const_iterator end() const noexcept{
    return m_data.end();
  }

  ~basic_static_extents() = default;


private:
  static constexpr base_type const m_data{E...};
};

template<std::size_t... E>
using static_extents = basic_static_extents<std::size_t,E...>;

  
template<typename T> struct static_product;

template<typename T> 
inline static constexpr auto const static_product_v = static_product<T>::value;

template<typename ExtentsType, ExtentsType E0, ExtentsType... E>
struct static_product< basic_static_extents<ExtentsType, E0, E...> >{
  static constexpr auto const value = E0 * static_product_v< basic_static_extents<ExtentsType, E...> >;
};

template<typename ExtentsType, ExtentsType E0>
struct static_product< basic_static_extents<ExtentsType, E0> >{
  static constexpr auto const value = E0 ;
};

template<typename ExtentsType>
struct static_product< basic_static_extents<ExtentsType> >{
  static constexpr auto const value = ExtentsType(0) ;
};

} // namespace boost::numeric::ublas

#endif
