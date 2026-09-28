//
//  Copyright (c) 2021, Cem Bassoy, cem.bassoy@gmail.com
//  Copyright (c) 2021, Amit Singh, amitsingh19975@gmail.com
//
//  Distributed under the Boost Software License, Version 1.0. (See
//  accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)
//
//

#ifndef BOOST_NUMERIC_UBLAS_TENSOR_TTV_HPP
#define BOOST_NUMERIC_UBLAS_TENSOR_TTV_HPP

#include <cmath>
#include <stdexcept>
#include <type_traits>

#include "../extents.hpp"
#include "../type_traits.hpp"
#include "../tags.hpp"

namespace boost::numeric::ublas
{


template<class extents, class layout, class container>
struct tensor_engine;

template<typename tensor_engine>
class tensor_core;

//template<class type, class allocator>
//class vector;

} // namespace boost::numeric::ublas

namespace boost::numeric::ublas
{

namespace detail {

/** Enables if extent E is dynamic with dynamic rank: extents< > */
template<
  class TE,
  class  E = typename tensor_core<TE>::extents_type
  >
using enable_ttv_if_extent_has_dynamic_rank = std::enable_if_t<is_dynamic_rank_v<E>,  bool>;

} // namespace detail


/** @brief Computes the m-mode tensor-times-vector product
     *
     * Implements C[i1,...,im-1,im+1,...,ip] = A[i1,i2,...,ip] * b[im]
     *
     * @note calls ublas::ttv
     *
     * @param[in] m contraction dimension with 1 <= m <= p
     * @param[in] a tensor object A with order p
     * @param[in] b vector object B
     *
     * @returns tensor object C with order p-1, the same storage format and allocator type as A
    */
template <class TE, class  A, class  T = typename tensor_core< TE >::value,
          detail::enable_ttv_if_extent_has_dynamic_rank<TE> = true >
inline decltype(auto) prod( tensor_core< TE > const &a, vector<T, A> const &b, const std::size_t m)
{

  using tensor            = tensor_core< TE >;
  using shape             = typename tensor::extents_type;
  using value             = typename tensor::value_type;
  using layout            = typename tensor::layout_type;
  using resize_tag        = typename tensor::resizable_tag;

  auto const p = a.rank();

  static_assert(std::is_same_v<resize_tag,storage_resizable_container_tag>);
  static_assert(is_dynamic_v<shape>);

  if (m == 0ul)  throw std::length_error("error in boost::numeric::ublas::prod(ttv): contraction mode must be greater than zero.");
  if (p < m)     throw std::length_error("error in boost::numeric::ublas::prod(ttv): rank of tensor must be greater than or equal to the contraction mode.");
  if (a.empty()) throw std::length_error("error in boost::numeric::ublas::prod(ttv): first argument tensor should not be empty.");
  if (b.empty()) throw std::length_error("error in boost::numeric::ublas::prod(ttv): second argument vector should not be empty.");

  auto const& na = a.extents();
  auto nb = extents<2>{std::size_t(b.size()),std::size_t(1ul)};
  auto wb = ublas::to_strides(nb,layout{} );

  auto const sz = std::max( std::size_t(ublas::size(na)-1u), std::size_t(2) );
  auto nc_base = typename shape::base_type(sz,1);

  for (auto i = 0ul, j = 0ul; i < p; ++i)
    if (i != m - 1)
      nc_base[j++] = na.at(i);

  auto nc = shape(nc_base);


  auto c = tensor( nc, value{} );
  auto const* bb = &(b(0));
  ttv(m, p,
      c.data(), c.extents().data(), c.strides().data(),
      a.data(), a.extents().data(), a.strides().data(),
      bb,       nb.data(),          wb.data());
  return c;
}


namespace detail {
/** Enables if extent E is dynamic with static rank: extents<N> */
template<
  class TE,
  class  E = typename tensor_core< TE >::extents_type
  >
using enable_ttv_if_extent_is_dynamic_with_static_rank =
  std::enable_if_t< is_static_rank_v< E > && is_dynamic_v< E >, bool>;

} // namespace detail


/** @brief Computes the m-mode tensor-times-vector product
     *
     * Implements C[i1,...,im-1,im+1,...,ip] = A[i1,i2,...,ip] * b[im]
     *
     * @note calls ublas::ttv
     *
     * @param[in] m contraction dimension with 1 <= m <= p
     * @param[in] a tensor object A with order p
     * @param[in] b vector object B
     *
     * @returns tensor object C with order p-1, the same storage format and allocator type as A
    */
template <class TE, class A, class T = typename tensor_core< TE >::value,
          detail::enable_ttv_if_extent_is_dynamic_with_static_rank<TE> = true
          >
inline auto prod( tensor_core< TE > const &a, vector<T, A> const &b, const std::size_t m)
{
  using tensor         = tensor_core< TE >;
  using shape          = typename tensor::extents_type;
  using container      = typename tensor::container_type;
  using layout         = typename tensor::layout_type;
  using resizeable_tag = typename tensor::resizable_tag;

  constexpr auto p  = std::tuple_size_v<shape>;
  constexpr auto sz = std::max(std::size_t(std::tuple_size_v<shape>-1U),std::size_t(2));

  using shape_b   = ublas::extents<2>;
  using shape_c   = ublas::extents<sz>;
  using tensor_c  = tensor_core<tensor_engine<shape_c,layout,container>>;

  static_assert(std::is_same_v<resizeable_tag,storage_resizable_container_tag >);

  if (m == 0ul)  throw std::length_error("error in boost::numeric::ublas::prod(ttv): contraction mode must be greater than zero.");
  if (p < m)     throw std::length_error("error in boost::numeric::ublas::prod(ttv): rank of tensor must be greater than or equal to the modus.");
  if (a.empty()) throw std::length_error("error in boost::numeric::ublas::prod(ttv): first argument tensor should not be empty.");
  if (b.empty()) throw std::length_error("error in boost::numeric::ublas::prod(ttv): second argument vector should not be empty.");

  auto const& na = a.extents();

  auto nc_base = typename shape_c::base_type{};
  std::fill(nc_base.begin(), nc_base.end(),std::size_t(1));
  for (auto i = 0ul, j = 0ul; i < p; ++i)
    if (i != m - 1)
      nc_base[j++] = na.at(i);

  auto nc = shape_c(std::move(nc_base));
  auto nb = shape_b{b.size(),1UL};
  auto wb = ublas::to_strides(nb,layout{});
  auto c  = tensor_c( std::move(nc) );
  auto const* bb = &(b(0));

  ttv(m, p,
      c.data(), c.extents().data(), c.strides().data(),
      a.data(), a.extents().data(), a.strides().data(),
      bb,       nb.data(),          wb.data() );
  return c;
}



/** @brief Computes the m-mode tensor-times-vector product
     *
     * Implements C[i1,...,im-1,im+1,...,ip] = A[i1,i2,...,ip] * b[im]
     *
     * @note calls ublas::ttv
     *
     * @tparam    M contraction dimension with 1 <= m <= p
     * @param[in] a tensor object A with order p
     * @param[in] b vector object B
     *
     * @returns tensor object C with order p-1, the same storage format and allocator type as A
    */

template <std::size_t m,
          typename TE,
          typename A,
          typename T = typename tensor_core< TE >::value>
inline auto prod( tensor_core< TE > const &a, vector<T, A> const &b)
{
  using tensor        = tensor_core< TE >;
  using container     = typename tensor::container;
  using shape         = typename tensor::extents;
  using layout        = typename tensor::layout;
  using shape_b       = extents<2>;
  using shape_c       = remove_element_t<m,shape>;
  using container_c   = rebind_storage_size_t<shape_c,container>;
  using tensor_c      = tensor_core<tensor_engine<shape_c,layout,container_c>>;

  static_assert( m != 0ul );
  static_assert(std::tuple_size_v<shape> != 0 );
  static_assert(std::tuple_size_v<shape> >= m );

  constexpr auto p = std::tuple_size_v<shape>;

  if (a.empty()) throw std::length_error("error in boost::numeric::ublas::prod(ttv): first argument tensor should not be empty.");
  if (b.empty()) throw std::length_error("error in boost::numeric::ublas::prod(ttv): second argument vector should not be empty.");

  auto const& na = a.extents();

  auto nc = shape_c{};
  auto nb = shape_b{std::size_t(b.size()),std::size_t(1)};

  auto c = tensor_c{};
  auto const* bb = &(b(0));

  auto const& wa = a.strides();
  auto const& wc = c.strides();
  auto wb        = ublas::to_strides(nb,layout{});

  ttv(m, p,
      c.data(), nc.data(), wc.data(),
      a.data(), na.data(), wa.data(),
      bb,       nb.data(), wb.data());

  return c;
}

} // namespace boost::numeric::ublas

#endif   // BOOST_NUMERIC_UBLAS_TENSOR_NORM_HPP
