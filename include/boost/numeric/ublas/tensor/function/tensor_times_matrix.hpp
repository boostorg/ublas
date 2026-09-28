//
//  Copyright (c) 2021, Cem Bassoy, cem.bassoy@gmail.com
//  Copyright (c) 2021, Amit Singh, amitsingh19975@gmail.com
//
//  Distributed under the Boost Software License, Version 1.0. (See
//  accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)
//
//

#ifndef BOOST_NUMERIC_UBLAS_TENSOR_TTM_HPP
#define BOOST_NUMERIC_UBLAS_TENSOR_TTM_HPP

#include <cmath>
#include <stdexcept>
#include <type_traits>

#include "../extents.hpp"
#include "../type_traits.hpp"
#include "../tags.hpp"
#include "../tensor.hpp"


/** @brief Computes the m-mode tensor-times-matrix product
     *
     * Implements C[i1,...,im-1,j,im+1,...,ip] = A[i1,i2,...,ip] * B[j,im]
     *
     * @note calls ublas::ttm
     *
     * @param[in] a tensor object A with order p
     * @param[in] b vector object B
     * @param[in] m contraction dimension with 1 <= m <= p
     *
     * @returns tensor object C with order p, the same storage format and allocator type as A
    */

//namespace boost::numeric::ublas
//{

//template<class extents_type, class layout_type, class container_type>
//struct tensor_engine;

//template<typename tensor_engine>
//class tensor_core;

//template<class value_type, class layout, class allocator>
//class matrix;

//} // namespace boost::numeric::ublas

namespace boost::numeric::ublas
{

namespace detail {
template< class TE, class  E = typename tensor_core<TE>::extents_type >
using enable_ttm_if_extent_is_modifiable = std::enable_if_t<is_dynamic_v<E>, bool>;
} // namespace detail

template <typename TE,
          typename A,
          typename T = typename tensor_core<TE>::value_type,
          typename L = typename tensor_core<TE>::layout_type,
          detail::enable_ttm_if_extent_is_modifiable<TE> = true >
inline decltype(auto) prod( tensor_core< TE > const &a, matrix<T,L,A> const &b, const std::size_t m)
{
  using tensor_type    = tensor_core< TE >;
  using extents_type   = typename tensor_type::extents_type;
  using layout_type    = typename tensor_type::layout_type;
  using resizeable_tag = typename tensor_type::resizable_tag;

  static_assert(std::is_same_v<resizeable_tag, storage_resizable_container_tag> );
  static_assert(is_dynamic_v<extents_type>);

  auto const   p = a.rank();
  auto const& na = a.extents();
  auto        nb = extents<2>{std::size_t(b.size1()), std::size_t(b.size2())};

  assert( p != 0 );
  assert( p == ublas::size(na));

  if( m == 0 )       throw std::length_error("Error in boost::numeric::ublas::ttm: contraction mode must be greater than zero.");
  if( p <  m )       throw std::length_error("Error in boost::numeric::ublas::ttm: tensor order must be greater than or equal to the specified mode.");
  if(na[m-1]!=nb[1]) throw std::invalid_argument("Error in boost::numeric::ublas::ttm: 2nd extent of B and m-th extent of A must be equal.");


  auto nc_base = na.base();
  auto wb = ublas::to_strides(nb,layout_type{});
  nc_base[m-1] = nb[0];
  auto nc = extents_type(nc_base);
  auto c  = tensor_type(nc);

  assert( std::equal(begin(na)  , begin(na)+m-1, begin(nc)  ) );
  assert( std::equal(begin(na)+m, end  (na),     begin(nc)+m) );
  assert( nc[m-1] == nb[0] );

  auto const* bb = &(b(0, 0));
  ttm(m, p,
      c.data(), c.extents().data(), c.strides().data(),
      a.data(), a.extents().data(), a.strides().data(),
      bb,       nb.data(),          wb.data());

  return c;
}


/** @brief Computes the m-mode tensor-times-matrix product
     *
     * Implements C[i1,...,im-1,j,im+1,...,ip] = A[i1,i2,...,ip] * B[j,im]
     *
     * @note calls ublas::ttm
     *
     * @tparam    M contraction dimension with 1 <= M <= p
     * @tparam    N is a non contracting dimension
     * @tparam    TE TensorEngine is used for the tensor
     *
     * @param[in] a tensor object A with order p
     * @param[in] b vector object B
     *
     * @returns tensor object C with order p, the same storage format and allocator type as A
    */

namespace detail {
template< class TE, class  E = typename tensor_core< TE >::extents_type >
using enable_ttm_if_extent_is_not_resizable =
  std::enable_if_t<is_static_rank_v<E> && is_dynamic_v<E>,  bool>;
} // namespace detail

template <size_t m,
          typename TE,
          typename A,
          typename T = typename tensor_core< TE >::value_type,
          typename L = typename tensor_core< TE >::layout_type,
          detail::enable_ttm_if_extent_is_not_resizable<TE>>
inline decltype(auto) prod(tensor_core<TE> const &a, matrix<T,L,A> const &b)
{
  using tensor_type    = tensor_core<TE>;
  using extents_type   = typename tensor_type::extents_type;
  using layout_type    = typename tensor_type::layout_type;
  using resizeable_tag = typename tensor_type::resizable_tag;

  static_assert(std::is_same_v<resizeable_tag, storage_resizable_container_tag> );
  static_assert(is_dynamic_v<extents_type>);

  constexpr auto p = std::tuple_size_v<extents_type>;

  auto const& na = a.extents();
  auto        nb = extents<2>{std::size_t(b.size1()), std::size_t(b.size2())};

  static_assert( p != 0 );
  static_assert( p == a.rank());
  static_assert( m != 0);
  static_assert( p <  m);

  if(na[m-1]!=nb[1]) throw std::invalid_argument("Error in boost::numeric::ublas::ttm: 2nd extent of B and m-th extent of A must be equal.");

  auto nc_base = na.base();
  auto wb = ublas::to_strides(nb,layout_type{});

  std::get<m-1>(nc_base) = std::get<0>(nb.base());

  auto nc = extents_type(nc_base);
  auto c  = tensor_type(nc);

  assert(std::equal(begin(na)  , begin(na)+m-1, begin(nc)  ));
  assert(std::equal(begin(na)+m, end  (na),     begin(nc)+m));
  assert(nc[m-1] == nb[0]);

  auto bbdata = &(b(0, 0));

  auto const& wa = a.strides();
  auto const& wc = c.strides();

  ttm(m, p,
      c.data(), nc.data(), wc.data(),
      a.data(), na.data(), wa.data(),
      bbdata  , nb.data(), wb.data());

  return c;
}


//namespace detail {
//template<
//  class TEL,
//  class TER,
//  class  EL = typename TEL::extents_type,
//  class  ER = typename TER::extents_type
//  >
//using enable_ttm_if_extent_is_static =
//  std::enable_if_t<is_static_v<EL> && is_static_v<ER>,  bool>;
//} // namespace detail

//template <class TEL, class TER>
//inline decltype(auto) prod( tensor_core<TEL> const& a, tensor_core<TER> const &b)
//{
//  using tensorA    = tensor_core<TE>;
//  using extentsA   = typename tensorA::extents_type;
//  using layout    = typename tensorA::layout_type;
//  using resizeable_tag = typename tensorA::resizable_tag;

//  static_assert(std::is_same_v<resizeable_tag, storage_static_container_tag> );
//  static_assert(is_static_v<extentsA>);

//  constexpr auto p = size_v<extentsA>;


//  auto const& na = a.extents();
//  auto const& nb = b.extents();

//  static_assert( p != 0 );
//  static_assert( p == a.rank());
//  static_assert( m != 0);
//  static_assert( p <  m);

//  static_assert(get_v<extentsA,m-1> != get_v<extentsB,1>);

//  if(na[m-1]!=nb[1]) throw std::invalid_argument("Error in boost::numeric::ublas::ttm: 2nd extent of B and m-th extent of A must be equal.");

//  auto nc_base = na.base();
//  auto wb = ublas::to_strides(nb,layout{});

//  std::get<m-1>(nc_base) = std::get<0>(nb.base());

//  auto nc = extents_type(nc_base);
//  auto c  = tensor_type(nc);

//  assert(std::equal(na.begin()  , na.begin()+m-1, nc.begin()));
//  assert(std::equal(na.begin()+m, na.end,         nc.begin()));
//  assert(nc[m-1] == nb[0]);

//  auto bbdata = &(b(0, 0));

//  auto const& wa = a.strides();
//  auto const& wc = c.strides();

//  ttm(m, p,
//      c.data(), nc.data(), wc.data(),
//      a.data(), na.data(), wa.data(),
//      bbdata  , nb.data(), wb.data());

//  return c;
//}



//using value_type     = typename tensor_type::value_type;
//using container_type = typename tensor_type::container_type;
//using return_extents_type   = std::decay_t<decltype(nc)>;
//using return_container_type = rebind_storage_size_t<return_extents_type,container_type>;
//using return_tensor_type    = tensor_core<tensor_engine<return_extents_type, layout_type, return_container_type >>;

//auto c = return_tensor_type(value_type{});

} // namespace boost::numeric::ublas

#endif   // BOOST_NUMERIC_UBLAS_TENSOR_NORM_HPP
