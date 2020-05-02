//
//  Copyright (c) 2018-2020, Cem Bassoy, cem.bassoy@gmail.com
//  Copyright (c) 2019-2020, Amit Singh, amitsingh19975@gmail.com
//
//  Distributed under the Boost Software License, Version 1.0. (See
//  accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)
//
//  The authors gratefully acknowledge the support of
//  Google
//

#ifndef _BOOST_NUMERIC_UBLAS_TENSOR_EXTENTS_FUNCTIONS_HPP_
#define _BOOST_NUMERIC_UBLAS_TENSOR_EXTENTS_FUNCTIONS_HPP_

#include <boost/numeric/ublas/tensor/type_traits.hpp>
#include <boost/numeric/ublas/tensor/detail/static_extents_traits.hpp>
#include <algorithm>
#include <sstream>
#include <iostream>
#include <numeric>
#include <iterator>

namespace boost::numeric::ublas::detail{

  template<typename T, T E, T...N>
  constexpr auto push_back(basic_static_extents<T, N...>) -> basic_static_extents<T, N..., E>;

  template<typename T, T E, T...N>
  constexpr auto push_front(basic_static_extents<T, N...>) -> basic_static_extents<T, E, N...>;

  template <typename T, T E0, T... E, T... N>
  constexpr auto squeeze_impl_remove_one( basic_static_extents<T, E0, E...>, basic_static_extents<T, N...> num = basic_static_extents<T>{} ){
    // executed when basic_static_extents is size of 1
    // @code basic_static_extents<T, E0> @endcode
    if constexpr( sizeof...(E) == 0ul ){
      // if element E0 is 1 we return number list but we do not append
      // it to the list
      if constexpr( E0 == T(1) ){
        return num;
      }else{
        // if element E0 is 1 we return number list but we append
        // it to the list
        return decltype(push_back<T, E0>(num)){};
      }
    }else{
      if constexpr( E0 == T(1) ){
        // if element E0 is 1 we return number list but we do not append
        // it to the list
        return squeeze_impl_remove_one(basic_static_extents<T, E...>{}, num);
      }else{
        // if element E0 is 1 we return number list but we append
        // it to the list
        auto n_num_list = decltype(push_back<T, E0>(num)){};
        return squeeze_impl_remove_one(basic_static_extents<T, E...>{}, n_num_list);
      }
    }
  }

  template <class T, T... E>
  constexpr auto squeeze_impl( basic_static_extents<T,E...> const& e ){
    
    using extents_type = basic_static_extents<T,E...>;

    if constexpr( extents_type::_size <= typename extents_type::size_type(2) ){
      return e;
    }

    using value_type = typename extents_type::value_type;
    using size_type = typename extents_type::size_type;

    auto one_free_static_extents = squeeze_impl_remove_one(e);

    // check after removing 1s from the list are they same
    // if same that means 1s does not exist and no need to
    // squeeze
    if constexpr( decltype(one_free_static_extents)::_size != extents_type::_size ){
      
      // after squeezing, all the extents are 1s we need to
      // return extents of (1, 1)
      if constexpr( decltype(one_free_static_extents)::_size == size_type(0) ){

        return basic_static_extents<T, value_type(1),value_type(1)>{};

      }else if constexpr( decltype(one_free_static_extents)::_size == (1) ){
        // to comply with GNU Octave this check is made
        // if position 2 contains 1 we push at back
        // else we push at front
        if constexpr( extents_type::at(1) == value_type(1) ){
          return decltype( push_back<T, value_type(1)>(one_free_static_extents) ){};
        }else{
          return decltype( push_front<T, value_type(1)>(one_free_static_extents) ){};
        }

      }else{
        return one_free_static_extents;
      }

    }else{
      return e;
    }
    
  }

  template <class T>
  inline
  constexpr auto squeeze_impl( basic_extents<T> const& e ){
    using extents_type  = basic_extents<T>;
    using base_type     = typename extents_type::base_type;
    using value_type    = typename extents_type::value_type;
    using size_type    = typename extents_type::size_type;
    
    if( e.size() <= size_type(2) ){
      return e;
    }

    auto not_one = [](auto const& el){
      return el != value_type(1);
    };

    // count non one values
    size_type size = std::count_if(e.begin(), e.end(), not_one);
    
    // reserve space
    base_type n_extents( std::max(size, size_type(2)), 1 );
    
    // copying non 1s to the new extents
    std::copy_if(e.begin(), e.end(), n_extents.begin(), not_one);

    // checking if extents size goes blow 2
    // if size of extents goes to 1
    // complying with GNU Octave
    // if position 2 contains 1 we 
    // swap the pos
    if( size < size_type(2) && e[1] != value_type(1) ){
      std::swap(n_extents[0], n_extents[1]);
    }
      
    return extents_type(n_extents); 
  }

  template <class T, std::size_t N>
  inline
  auto squeeze_impl( basic_fixed_rank_extents<T,N> const& e ){
    if constexpr( N <= 2 ){
      return e;
    }else{
      return squeeze_impl(basic_extents<T>(e));
    }
  }
    
    

} // namespace boost::numeric::ublas::detail

namespace boost::numeric::ublas {

/** @brief Returns true if size > 1 and all elements > 0 or size == 1 && e[0] == 1 */
template <class ExtentsType>
[[nodiscard]] inline 
constexpr bool valid(ExtentsType const &e) {
  
  static_assert(is_extents_v<ExtentsType>, "boost::numeric::ublas::valid() : invalid type, type should be an extents");

  auto greater_than_zero = [](auto const& a){ return a > 0u; };

  if( e.size() == 1u )  return e[0] == 1u;
  return !e.empty() && std::all_of(e.begin(), e.end(), greater_than_zero );
}

/**
 * @code static_extents<4,1,2,3,4> s;
 * std::cout<<to_string(extents); // {1,2,3,4}
 * @endcode
 * @returns the string of extents
 */

template <class T>
[[nodiscard]] inline
std::string to_string(T const &e) {
  
  using value_type = typename T::value_type;

  static_assert(is_extents_v<T> ||is_strides_v<T>, 
    "boost::numeric::ublas::to_string() : invalid type, type should be an extents or a strides");

  if ( e.empty() ) return "[]";

  std::stringstream ss;
  
  ss << "[ ";

  std::copy( e.begin(), e.end() - 1, std::ostream_iterator<value_type>(ss,", ") );
  
  ss << e.back() << " ]";

  return ss.str();
}

/** @brief Returns true if this has a scalar shape
 *
 * @returns true if (1,1,[1,...,1])
 */
template <class ExtentsType>
[[nodiscard]] inline 
constexpr bool is_scalar(ExtentsType const &e) {

  static_assert(is_extents_v<ExtentsType>, "boost::numeric::ublas::is_scalar() : invalid type, type should be an extents");
  
  auto equal_one = [](auto const &a) { return a == 1u; };

  return !e.empty() && std::all_of(e.begin(), e.end(), equal_one);
}

/** @brief Returns true if this has a vector shape
 *
 * @returns true if (1,n,[1,...,1]) or (n,1,[1,...,1]) with n > 1
 */
template <class ExtentsType>
[[nodiscard]] inline 
constexpr bool is_vector(ExtentsType const &e) {
  
  static_assert(is_extents_v<ExtentsType>, "boost::numeric::ublas::is_vector() : invalid type, type should be an extents");

  auto greater_one = [](auto const &a) { return a > 1u; };
  auto equal_one = [](auto const &a) { return a == 1u; };

  if      (e.size() == 0u) return false;
  else if (e.size() == 1u) return e[0] > 1u;
  else  return  std::any_of(e.begin(), e.begin() + 2, greater_one) &&
                std::any_of(e.begin(), e.begin() + 2, equal_one) &&
                std::all_of(e.begin() + 2, e.end(), equal_one);

}

/** @brief Returns true if this has a matrix shape
 *
 * @returns true if (m,n,[1,...,1]) with m > 1 and n > 1
 */
template <class ExtentsType>
[[nodiscard]] inline 
constexpr bool is_matrix(ExtentsType const &e) {
  
  static_assert(is_extents_v<ExtentsType>, "boost::numeric::ublas::is_matrix() : invalid type, type should be an extents");

  auto greater_one = [](auto const &a) { return a > 1u; };
  auto equal_one = [](auto const &a) { return a == 1u; };

  return  ( e.size() >= 2u ) && std::all_of(e.begin(), e.begin() + 2, greater_one) &&
          std::all_of(e.begin() + 2, e.end(), equal_one);
}

/** @brief Returns true if this is has a tensor shape
 *
 * @returns true if !empty() && !is_scalar() && !is_vector() && !is_matrix()
 */
template <class ExtentsType>
[[nodiscard]] inline 
constexpr bool is_tensor(ExtentsType const &e) {

  static_assert(is_extents_v<ExtentsType>, "boost::numeric::ublas::is_tensor() : invalid type, type should be an extents");
  
  auto greater_one = [](auto const &a) { return a > 1u;};
  
  return  ( e.size() >= 3u ) && std::any_of(e.begin() + 2, e.end(), greater_one);
}

/** @brief Eliminates singleton dimensions when size > 2
 *
 * squeeze {  1,1} -> {  1,1}
 * squeeze {  2,1} -> {  2,1}
 * squeeze {  1,2} -> {  1,2}
 *
 * squeeze {1,2,3} -> {  2,3}
 * squeeze {2,1,3} -> {  2,3}
 * squeeze {1,3,1} -> {  1,3}
 *
 * @returns basic_extents<int_type> with squeezed extents
 */
template <class ExtentsType>
[[nodiscard]] inline
auto squeeze(ExtentsType const &e) {
  
  static_assert(is_extents_v<ExtentsType>, "boost::numeric::ublas::squeeze() : invalid type, type should be an extents");

  return detail::squeeze_impl(e); 
}

/** @brief Returns the product of extents */
template <class ExtentsType>
[[nodiscard]] inline
constexpr auto product(ExtentsType const &e) {

  static_assert(is_extents_v<ExtentsType>, "boost::numeric::ublas::product() : invalid type, type should be an extents");
  
  if ( e.empty() ) return 0u;
  else return std::accumulate(e.begin(), e.end(), 1u, std::multiplies<>()) ;
}


template <class LExtents, class RExtents, 
  std::enable_if_t<
    is_extents_v<LExtents> && is_extents_v<RExtents>
  , int> = 0 
>
[[nodiscard]] inline
constexpr bool operator==(LExtents const& lhs, RExtents const& rhs) noexcept{
  
  static_assert( std::is_same_v<typename LExtents::value_type, typename RExtents::value_type>, 
    "boost::numeric::ublas::operator==(LExtents, RExtents) : LHS value type should be same as RHS value type");

  return ( lhs.size() == rhs.size() ) && std::equal(lhs.begin(), lhs.end(), rhs.begin());
}

template <class LExtents, class RExtents, 
  std::enable_if_t<
    is_extents_v<LExtents> && is_extents_v<RExtents>
  , int> = 0 
>
[[nodiscard]] inline
constexpr bool operator!=(LExtents const& lhs, RExtents const& rhs) noexcept{
  
  static_assert( std::is_same_v<typename LExtents::value_type, typename RExtents::value_type>, 
    "boost::numeric::ublas::operator!=(LExtents, RExtents) : LHS value type should be same as RHS value type");

  return !( lhs == rhs );
}

} // namespace boost::numeric::ublas

#endif
