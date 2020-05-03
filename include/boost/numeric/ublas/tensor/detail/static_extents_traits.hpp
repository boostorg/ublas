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
  
  template <class ExtentsType, ExtentsType... E> struct basic_static_extents;
  template <class ExtentsType, std::size_t N> struct basic_fixed_rank_extents;
  template<class ExtentsType> class basic_extents;

} // namespace boost::numeric::ublas

namespace boost::numeric::ublas::static_traits{

  template<typename T> 
  struct is_valid : std::integral_constant<bool, false>{};

  template<typename T> 
  inline static constexpr bool const is_valid_v = is_valid<T>::value;

  template<typename ExtentsType, ExtentsType E0>
  struct is_valid< basic_static_extents<ExtentsType, E0> >{
    static constexpr bool const value = ( E0 == ExtentsType(1) );
  };

  template<typename ExtentsType, ExtentsType E0, ExtentsType E1>
  struct is_valid< basic_static_extents<ExtentsType, E0, E1 > >{
    static constexpr bool const value = ( E0 > ExtentsType(0) ) && ( E1 > ExtentsType(0) );
  };

  template<typename ExtentsType, ExtentsType E0, ExtentsType E1, ExtentsType E2, ExtentsType... E>
  struct is_valid< basic_static_extents<ExtentsType, E0, E1, E2, E...> >{
    static constexpr bool const value = ( E0 > ExtentsType(0) ) 
      && is_valid_v< basic_static_extents<ExtentsType, E1, E2, E...> >;
  };


  template<typename T> 
  struct is_scalar : std::integral_constant<bool, false>{};

  template<typename T> 
  inline static constexpr bool const is_scalar_v = is_scalar<T>::value;

  template<typename ExtentsType, ExtentsType E0, ExtentsType E1, ExtentsType... E>
  struct is_scalar< basic_static_extents<ExtentsType, E0, E1, E...> >{
    static constexpr bool const value = is_scalar_v< basic_static_extents<ExtentsType, E0> > && is_scalar_v< basic_static_extents<ExtentsType, E1, E...> >;
  };

  template<typename ExtentsType, ExtentsType E0>
  struct is_scalar< basic_static_extents<ExtentsType, E0> >{
    static constexpr bool const value = (E0 == 1);
  };

  template<typename ExtentsType>
  struct is_scalar< basic_static_extents<ExtentsType> > : std::false_type{};

  template<typename T> 
  struct is_vector : std::integral_constant<bool, false>{};

  template<typename T> 
  inline static constexpr bool const is_vector_v = is_vector<T>::value;

  template<typename ExtentsType, ExtentsType E0, ExtentsType E1, ExtentsType E2, ExtentsType... E>
  struct is_vector< basic_static_extents<ExtentsType, E0, E1, E2, E...> >{
    static constexpr bool const value = 
      is_vector_v< basic_static_extents<ExtentsType, E0, E1> > && 
      is_scalar_v< basic_static_extents<ExtentsType, E2, E...> >;
  };

  template<typename ExtentsType, ExtentsType E0, ExtentsType E1>
  struct is_vector< basic_static_extents<ExtentsType, E0, E1 > >{
    static constexpr bool const value = 
         ( E0 > ExtentsType(1) || E1 > ExtentsType(1) ) 
      && ( E0 == ExtentsType(1) || E1 == ExtentsType(1) );
  };

  template<typename ExtentsType, ExtentsType E0>
  struct is_vector< basic_static_extents<ExtentsType, E0> >{
    static constexpr bool const value = ( E0 > ExtentsType(1) );
  };

  template<typename T> 
  struct is_matrix : std::integral_constant<bool, false>{};

  template<typename T> 
  inline static constexpr bool const is_matrix_v = is_matrix<T>::value;

  template<typename ExtentsType, ExtentsType... E>
  struct is_matrix< basic_static_extents<ExtentsType, E...> > : std::false_type{};

  template<typename ExtentsType, ExtentsType E0, ExtentsType E1, ExtentsType E2, ExtentsType... E>
  struct is_matrix< basic_static_extents<ExtentsType, E0, E1, E2, E...> >{
    static constexpr bool const value = 
      is_matrix_v< basic_static_extents<ExtentsType, E0, E1> > && 
      is_scalar_v< basic_static_extents<ExtentsType, E2, E...> >;
  };

  template<typename ExtentsType, ExtentsType E0, ExtentsType E1>
  struct is_matrix< basic_static_extents<ExtentsType, E0, E1> >{
    static constexpr bool const value = ( E0 > ExtentsType(1) && E1 > ExtentsType(1) );
  };

  template<typename T> 
  struct is_any_greater_than_one : std::integral_constant<bool, false>{};

  template<typename T> 
  inline static constexpr bool const is_any_greater_than_one_v = is_any_greater_than_one<T>::value;

  template<typename ExtentsType, ExtentsType... E>
  struct is_any_greater_than_one< basic_static_extents<ExtentsType, E... > > : std::false_type{};

  template<typename ExtentsType, ExtentsType E0>
  struct is_any_greater_than_one< basic_static_extents<ExtentsType, E0> > {
    static constexpr bool const value = ( E0 > ExtentsType(1) );
  };

  template<typename ExtentsType, ExtentsType E0, ExtentsType... E>
  struct is_any_greater_than_one< basic_static_extents<ExtentsType, E0, E... > > {
    static constexpr bool const value = is_any_greater_than_one_v< basic_static_extents<ExtentsType, E0> > || is_any_greater_than_one_v< basic_static_extents<ExtentsType, E... > >;
  };

  template<typename T> 
  struct is_tensor : std::integral_constant<bool, false>{};

  template<typename T> 
  inline static constexpr bool const is_tensor_v = is_tensor<T>::value;

  template<typename ExtentsType, ExtentsType... E>
  struct is_tensor< basic_static_extents<ExtentsType, E... > > : std::false_type{};

  template<typename ExtentsType, ExtentsType E0, ExtentsType E1, ExtentsType E2, ExtentsType... E>
  struct is_tensor< basic_static_extents<ExtentsType, E0, E1, E2, E...> >{
    static constexpr bool const value = is_any_greater_than_one_v< basic_static_extents<ExtentsType, E2, E...> >;
  };

} // namespace boost::numeric::ublas::static_traits
#endif
