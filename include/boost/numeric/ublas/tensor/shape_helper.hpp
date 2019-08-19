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

#ifndef BOOST_NUMERIC_UBLAS_TENSOR_SHAPE_HELPER_HPP
#define BOOST_NUMERIC_UBLAS_TENSOR_SHAPE_HELPER_HPP

#include <iostream>
#include <type_traits>
#include "fwd.hpp"

namespace boost::numeric::ublas {

constexpr ptrdiff_t dynamic_extent{-1};
constexpr ptrdiff_t dynamic_rank{-2};

} // namespace boost::numeric::ublas

namespace boost::numeric::ublas::detail {

template <> struct shape_in_bounds<> : std::integral_constant<bool,true>{};

template <ptrdiff_t N, ptrdiff_t... E> struct shape_in_bounds<N, E...> {
  static constexpr bool value =
      (N >= 0 || N == dynamic_extent) && shape_in_bounds<E...>::value;
};
} // namespace boost::numeric::ublas::detail

namespace boost::numeric::ublas::detail {

/** @brief Stores the extents
 * 
 * @code using shape = typename basic_shape<1,2,3,4>::type
 * 
 * @tparam Extents is parameter pack of type ptrdiff_t which takes the extents
**/
template <ptrdiff_t... Extents> 
struct basic_shape 
{
  using type                        = basic_shape;
  static constexpr ptrdiff_t rank   = sizeof...(Extents);
  static constexpr bool empty    = false;
};

/** @brief basic_shape specialization with no template argument
 * 
 * @code using shape = typename basic_shape<>::type
 * 
**/
template <> 
struct basic_shape<> 
{
  using type                        = basic_shape;
  static constexpr ptrdiff_t rank   = 0;
  static constexpr bool empty    = true;
};

/** @brief type alias of basic_shape
 * 
 * @code using shape = make_static_basic_shape_t<1,2,3,4>;
 * 
 * @tparam Extents is parameter pack of type ptrdiff_t which takes the extents
**/
template<ptrdiff_t... Extents>
using make_static_basic_shape_t = typename basic_shape< Extents... >::type;

/** @brief type alias of make_dynamic_basic_shape_impl
 * 
 * @tparam start which is a start range
 * @tparam end which is a end of range
 * 
**/
template<ptrdiff_t start, ptrdiff_t end>
using make_dynamic_basic_shape_impl_t = typename make_dynamic_basic_shape_impl< start, end >::type;

/** @brief type alias of make_dynamic_basic_shape_impl_t but starts with 0
 * 
 * @tparam end which is a end of range
 * 
**/
template<ptrdiff_t end>
using make_dynamic_basic_shape_t = make_dynamic_basic_shape_impl_t< 0, end - 1 >;

/** @brief concat_basic_shape specialization which concat two basic_shapes
 * 
 * @tparam lhs parameter pack of extents of left basic_shape
 * @tparam rhs parameter pack of extents of right basic_shape
 * 
**/
template<ptrdiff_t... lhs, ptrdiff_t... rhs>
struct concat_basic_shape< basic_shape<lhs...>, basic_shape<rhs...> >
{
    //aliasing ther of concatenated basic_shape
    using type = basic_shape< lhs... , rhs... >;
};

/** @brief type alias of concat_basic_shape
 * 
 * @tparam S1 type of basic_shape
 * @tparam S2 type of basic_shape
 * 
**/
template<class S1, class S2>
using concat_basic_shape_t = typename concat_basic_shape< S1, S2 >::type;

/** @brief makes the range of dynamic_extents for a given range by concatenating two basic_shape 
 * 
 * @tparam start which is a start range
 * @tparam end which is a end of range
 * 
**/
template<ptrdiff_t start, ptrdiff_t end>
struct make_dynamic_basic_shape_impl
{
    //using divide and conquer to generate range of dynamic_extent
    //and aliasing the type after generating range  
    static_assert(start < end,"boost::numeric::ublas::detail::make_dynamic_basic_shape_impl invalid value of start and end");
    using type = concat_basic_shape_t< make_dynamic_basic_shape_impl_t< start, ( start + end ) / 2 > ,
        make_dynamic_basic_shape_impl_t< ( start + end ) / 2 + 1, end > >; 
};

/** @brief make_dynamic_basic_shape_impl specialization
 * 
 * @tparam I of type ptrdiff_t
 * 
**/
template<ptrdiff_t I>
struct make_dynamic_basic_shape_impl<I,I>
{
    using type = basic_shape< dynamic_extent >; 
};

/** @brief make_basic_shape specialization with no extents
 * 
 * @tparam rank which is a rank of tensor of type ptrdiff_t
 * 
**/
template<ptrdiff_t rank>
struct make_basic_shape<rank>
{
    static_assert(rank >= 0 , "boost::numeric::ublas::detail::make_basic_shape rank greater than or equal to 0 or -2");
    using type = make_dynamic_basic_shape_t<rank>;
};

/** @brief make_basic_shape specialization with a rank and extents
 * 
 * @tparam rank which is a rank of tensor of type ptrdiff_t
 * @tparam extent which is a extent of type ptrdiff_t
 * @tparam Extents which is a parameter pack of type ptrdiff_t containing extents
 * 
**/
template<ptrdiff_t rank, ptrdiff_t extent, ptrdiff_t... OtherExtents>
struct make_basic_shape< rank, extent, OtherExtents... >
{
    static_assert(rank >= 0 , "boost::numeric::ublas::detail::make_basic_shape rank greater than or equal to 0 or -2");
    static_assert(shape_in_bounds<extent,OtherExtents...>::value , "boost::numeric::ublas::detail::make_basic_shape extent greater than 0 or -1");
    static_assert(sizeof...(OtherExtents) + 1 == rank, "boost::numeric::ublas::detail::make_basic_shape rank should be equal to the extents");
    using type = make_static_basic_shape_t< extent, OtherExtents... >;
}; 

/** @brief make_basic_shape specialization with dynamic_rank
 * 
**/
template<>
struct make_basic_shape<dynamic_rank>
{
    using type = make_static_basic_shape_t<>;
};

/** @brief make_basic_shape specialization for rank 0
 * 
**/
template<>
struct make_basic_shape<0>
{
    using type = make_static_basic_shape_t<>;
};

/** @brief type alias of make_basic_shape
 * 
 * @tparam rank which is a rank of tensor of type ptrdiff_t
 * @tparam Extents which is a parameter pack of type ptrdiff_t containing extents
 * 
**/
template<ptrdiff_t rank, ptrdiff_t... Extents>
using make_basic_shape_t = typename make_basic_shape<rank, Extents...>::type; 

/** @brief checks if its a basic_shape of not
 * 
 * @tparam S of any type 
 * 
**/
template<class S>
struct is_basic_shape : std::integral_constant<bool,false>{};

/** @brief is_basic_shape specialization
 * 
 * @tparam Extents which is a parameter pack of type ptrdiff_t containing extents
 * 
**/
template<ptrdiff_t... Extents>
struct is_basic_shape< basic_shape< Extents... > > : std::integral_constant<bool,true>{};

/** @brief alias of is_basic_shape for the value
 * 
 * @tparam S any type
 * 
**/
template<class S>
inline constexpr bool is_basic_shape_v = is_basic_shape<S>::value;

/** @brief is_dynamic_basic_shape checks if its a dynamic_rank or not
 * 
 * @tparam S of type basic_shape
 * 
**/
template<class S>
struct is_dynamic_basic_shape : std::integral_constant<bool,false> 
{
    static_assert(is_basic_shape_v<S>,"boost::numeric::ublas::detail::is_dynamic_basic_shape template class type shoud be of basic_shape");
};

/** @brief is_dynamic_basic_shape specialization
 * 
**/
template<>
struct is_dynamic_basic_shape< basic_shape<> > : std::integral_constant<bool,true> {};

/** @brief alias of is_dynamic_basic_shape for the value
 * 
 * @tparam S any basic_shape
 * 
**/
template<class S>
inline constexpr bool is_dynamic_basic_shape_v = is_dynamic_basic_shape<S>::value;

} // namespace boost::numeric::ublas::detail

#endif