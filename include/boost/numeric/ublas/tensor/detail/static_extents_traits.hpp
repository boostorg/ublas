//
//  Copyright (c) 2018-2020, Cem Bassoy, cem.bassoy@gmail.com
// 	Copyright (c) 2019-2020, Amit Singh, amitsingh19975@gmail.com
//
//  Distributed under the Boost Software License, Version 1.0. (See
//  accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)
//
//  The authors gratefully acknowledge the support of
//  Google
//

#ifndef _BOOST_NUMERIC_UBLAS_TENSOR_STATIC_TRAITS_HPP_
#define _BOOST_NUMERIC_UBLAS_TENSOR_STATIC_TRAITS_HPP_

#include <type_traits>

namespace boost::numeric::ublas{
  
  template <class ExtentsType, ExtentsType... E> class basic_static_extents;
  template <class ExtentsType, std::size_t N> class basic_fixed_rank_extents;
  template<class ExtentsType> class basic_extents;

} // namespace boost::numeric::ublas

namespace boost::numeric::ublas::static_traits{

  /** @brief Type-traits true if size > 1 and all elements > 0 or size == 1 && e[0] == 1 */
  template<typename T> 
  struct is_valid : std::false_type{};

  template<typename T> 
  inline static constexpr bool const is_valid_v = is_valid<T>::value;

  template<typename ExtentsType, ExtentsType... Es>
  struct is_valid< basic_static_extents<ExtentsType, Es...> >
    : std::integral_constant<bool,
        ( ( sizeof...(Es) == 1ul ) && ( ( Es == ExtentsType(1) ) && ... ) ) || 
        ( ( sizeof...(Es) > 1ul ) && ( ( Es > ExtentsType(0) ) && ... ) )
    >
  {};

  /** @brief Type-traits true if this has a scalar shape
   * true if (1,1,[1,...,1])
   */
  template<typename T> 
  struct is_scalar : std::false_type{};

  template<typename T> 
  inline static constexpr bool const is_scalar_v = is_scalar<T>::value;

  template<typename ExtentsType, ExtentsType... Es>
  struct is_scalar< basic_static_extents<ExtentsType, Es...> >
    : std::integral_constant<bool,
      ( sizeof...(Es) && ( ( Es == ExtentsType(1) ) && ... ) )
    >
  {};

  /** @brief Type-traits true if this has a vector shape
   * true if (1,n,[1,...,1]) or (n,1,[1,...,1]) with n > 1
   */
  template<typename T> 
  struct is_vector : std::false_type{};

  template<typename T> 
  inline static constexpr bool const is_vector_v = is_vector<T>::value;

  template<typename ExtentsType, ExtentsType E0, ExtentsType E1, ExtentsType... Es>
  struct is_vector< basic_static_extents<ExtentsType, E0, E1, Es...> >
    : std::integral_constant<bool,
        ( ( E0 > ExtentsType(1) ) ^ ( E1 > ExtentsType(1) ) ) &&
        ( ( sizeof...(Es) == 0 ) || is_scalar_v< basic_static_extents<ExtentsType, Es...> > )
    >
  {};

  template<typename ExtentsType, ExtentsType E0>
  struct is_vector< basic_static_extents<ExtentsType, E0> >
    : std::integral_constant<bool, ( E0 > ExtentsType(1) )>
  {};

  /** @brief Type-traits true if this has a matrix shape
   * true if (m,n,[1,...,1]) with m > 1 and n > 1
   */
  template<typename T> 
  struct is_matrix : std::false_type{};

  template<typename T> 
  inline static constexpr bool const is_matrix_v = is_matrix<T>::value;

  template<typename ExtentsType, ExtentsType E0, ExtentsType E1, ExtentsType... Es>
  struct is_matrix< basic_static_extents<ExtentsType, E0, E1, Es...> >
    : std::integral_constant<bool,
      ( ( E0 > ExtentsType(1) ) && ( E1 > ExtentsType(1) )  ) &&
      ( ( sizeof...(Es) == 0 ) || is_scalar_v< basic_static_extents<ExtentsType, Es...> > )
    >
  {};

  /** @brief Type-traits true if this has a any extent greater than 1 */
  template<typename T> 
  struct is_any_greater_than_one : std::false_type{};

  template<typename T> 
  inline static constexpr bool const is_any_greater_than_one_v = is_any_greater_than_one<T>::value;

  template<typename ExtentsType, ExtentsType... Es>
  struct is_any_greater_than_one< basic_static_extents<ExtentsType, Es... > > 
    : std::integral_constant<bool,
      sizeof...(Es) && ( ( Es > ExtentsType(1) ) || ... )
    >
    {};

  /** @brief Type-traits true if this is has a tensor shape
   * true if !empty() && !is_scalar() && !is_vector() && !is_matrix()
   */
  template<typename T> 
  struct is_tensor : std::false_type{};

  template<typename T> 
  inline static constexpr bool const is_tensor_v = is_tensor<T>::value;

  template<typename ExtentsType, ExtentsType E0, ExtentsType E1, ExtentsType... Es>
  struct is_tensor< basic_static_extents<ExtentsType, E0, E1, Es...> >
    : is_any_greater_than_one< basic_static_extents<ExtentsType, Es...> >
  {};

} // namespace boost::numeric::ublas::static_traits
#endif
