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

#include "../multiplication.hpp"
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


namespace detail {
template <class TC, class TA, class  V, class EC>
inline auto scalar_scalar_prod(TA const &a, V const &b, EC const& nc_base)
{
  assert(ublas::is_scalar(a.extents()));
  using tensor = TC;
  using value = typename tensor::value_type;
  using shape = typename tensor::extents_type;
  return tensor(shape(nc_base),value(a[0]*b(0)));
}

template <class TC, class TA, class  V, class EC>
inline auto vector_vector_prod(TA const &a, V const &b, EC& nc_base, std::size_t m)
{
  auto const& na = a.extents();

  assert( ublas::is_vector(na));
  assert(!ublas::is_scalar(na));
  assert( ublas::size(na) > 1u);
  assert(m > 0);

  using tensor = TC;
  using value = typename tensor::value_type;
  using shape = typename tensor::extents_type;

  auto const n1 = na[0];
  auto const n2 = na[1];
  auto const s = b.size();

  // general
  // [n1 n2 1 ... 1] xj [s 1] for any 1 <= j <= p with n1==1 or n2==1


  // [n1 1 1 ... 1] x1 [n1 1] -> [1 1 1 ... 1]
  // [1 n2 1 ... 1] x2 [n2 1] -> [1 1 1 ... 1]


  assert(n1>1 || n2>1);

  if( (n1>1u && m==1u)  || (n2>1u && m==2u) ){
    if(m==1u) assert(n2==1u && n1==s);
    if(m==2u) assert(n1==1u && n2==s);
    auto cc = std::inner_product( a.begin(), a.end(), b.begin(), value(0) );
    return tensor(shape(nc_base),value(cc));
  }

  // [n1 1 1 ... 1] xj [1 1] -> [n1 1 1 ... 1] with j != 1
  // [1 n2 1 ... 1] xj [1 1] -> [1 n2 1 ... 1] with j != 2

//if( (n1>1u && m!=1u) && (n2>0u && m!=2u) ){

  if(n1>1u) assert(m!=1u);
  if(n2>1u) assert(m!=2u);
  assert(s==1u);

  if(n1>1u) assert(n2==1u);
  if(n2>1u) assert(n1==1u);

  if(n1>1u) nc_base[0] = n1;
  if(n2>1u) nc_base[1] = n2;

  auto bb = b(0);
  auto c = tensor(shape(nc_base));
  std::transform(a.begin(),a.end(),c.begin(),[bb](auto aa){ return aa*bb; });
  return c;
//}


}


/** Computes a matrix-vector product.
 *
 *
 *  @note assume stride 1 for specific dimensions and therefore requires refactoring for subtensor
 *
*/
template <class TC, class TA, class  V, class EC>
inline auto matrix_vector_prod(TA const &a, V const &b, EC& nc_base, std::size_t m)
{
  auto const& na = a.extents();

  assert( ublas::is_matrix(na));
  assert(!ublas::is_vector(na));
  assert(!ublas::is_scalar(na));
  assert( ublas::size(na) > 1u);
  assert(m > 0);

  using tensor = TC;
  using shape  = typename tensor::extents_type;
  using size_t = typename shape::value_type;

  auto const n1 = na[0];
  auto const n2 = na[1];
  auto const s = b.size();

  // general
  // [n1 n2 1 ... 1] xj [s 1] for any 1 <= j <= p with either n1>1 and n2>1


  // if [n1 n2 1 ... 1] xj [1 1] -> [n1 n2 1 ... 1] for j > 2
  if(m > 2){
    nc_base[0] = n1;
    nc_base[1] = n2;
    assert(s == 1);
    auto c  = tensor(shape(nc_base));
    auto const bb = b(0);
    std::transform(a.begin(),a.end(), c.begin(), [bb](auto aa){return aa*bb;});
    return c;
  }


  // [n1 n2 1 ... 1] x1 [n1 1] -> [n2 1 ... 1] -> vector-times-matrix
  // [n1 n2 1 ... 1] x2 [n2 1] -> [n1 1 ... 1] -> matrix-times-vector

  nc_base[0] = m==1 ? n2 : n1;

  auto c  = tensor(shape(nc_base));
  auto const& wa = a.strides();
  auto const* bdata = &(b(0));

  detail::recursive::mtv(m-1,n1,n2, c.data(), size_t(1), a.data(), wa[0], wa[1], bdata, size_t(1));

  return c;
}



template <class TC, class TA, class  V, class EC>
inline auto tensor_vector_prod(TA const &a, V const &b, EC& nc_base, std::size_t m)
{
  auto const& na = a.extents();

  assert( ublas::is_tensor(na));
  assert( ublas::size(na) > 1u);
  assert(m > 0);

  using tensor = TC;
  using shape  = typename tensor::extents_type;
  using layout = typename tensor::layout_type;

  auto const pa = a.rank();
  auto const nm = na[m-1];
  auto const s = b.size();

  auto nb = extents<2>{std::size_t(b.size()),std::size_t(1ul)};
  auto wb = ublas::to_strides(nb,layout{} );

  //TODO: Include an outer product when legacy vector becomes a new vector.

  for (auto i = 0ul, j = 0ul; i < pa; ++i)
    if (i != m - 1)
      nc_base[j++] = na.at(i);

  auto c  = tensor(shape(nc_base));

  // [n1 n2 ... nm ... np] xm [1 1] -> [n1 n2 ... nm-1 nm+1 ... np]

  if(s == 0){
    assert(nm == 1);
    auto const bb = b(0);
    std::transform(a.begin(),a.end(), c.begin(), [bb](auto aa){return aa*bb;});
    return c;
  }


  // if [n1 n2 n3 ... np] xm [nm 1] -> [n1 n2 ... nm-1 nm+1 ... np]

  auto const& nc = c.extents();
  auto const& wc = c.strides();
  auto const& wa = a.strides();
  auto const* bp = &(b(0));

  ttv(m, pa,
      c.data(), nc.data(), wc.data(),
      a.data(), na.data(), wa.data(),
      bp,       nb.data(), wb.data());

  return c;
}

}//namespace detail


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
inline auto prod( tensor_core< TE > const &a, vector<T, A> const &b, const std::size_t m)
{

  using tensor            = tensor_core< TE >;
  using shape             = typename tensor::extents_type;
  using resize_tag        = typename tensor::resizable_tag;

  auto const pa = a.rank();

  static_assert(std::is_same_v<resize_tag,storage_resizable_container_tag>);
  static_assert(is_dynamic_v<shape>);

  if (m == 0ul)  throw std::length_error("error in boost::numeric::ublas::prod(ttv): contraction mode must be greater than zero.");
  if (pa < m)    throw std::length_error("error in boost::numeric::ublas::prod(ttv): rank of tensor must be greater than or equal to the contraction mode.");
  if (a.empty()) throw std::length_error("error in boost::numeric::ublas::prod(ttv): first argument tensor should not be empty.");
  if (b.empty()) throw std::length_error("error in boost::numeric::ublas::prod(ttv): second argument vector should not be empty.");

  auto const& na = a.extents();

  if(b.size() != na[m-1]) throw std::length_error("error in boost::numeric::ublas::prod(ttv): dimension mismatch of tensor and vector.");

  auto const sz = std::max( std::size_t(ublas::size(na)-1u), std::size_t(2) );
  auto nc_base = typename shape::base_type(sz,1);

  // output scalar tensor
  if(ublas::is_scalar(na)){
    return detail::scalar_scalar_prod<tensor>(a,b,nc_base);
  }

  // output scalar tensor or vector tensor
  if (ublas::is_vector(na)){
    return detail::vector_vector_prod<tensor>(a,b,nc_base,m);
  }

  // output scalar tensor or vector tensor
  if (ublas::is_matrix(na)){
    return detail::matrix_vector_prod<tensor>(a,b,nc_base,m);
  }

  assert(ublas::is_tensor(na));
  return detail::tensor_vector_prod<tensor>(a,b,nc_base,m);


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


  // output scalar tensor
  if(ublas::is_scalar(na)){
    return detail::scalar_scalar_prod<tensor_c>(a,b,nc_base);
  }

  // output scalar tensor or vector tensor
  if (ublas::is_vector(na)){
    return detail::vector_vector_prod<tensor_c>(a,b,nc_base,m);
  }

  // output scalar tensor or vector tensor
  if (ublas::is_matrix(na)){
    return detail::matrix_vector_prod<tensor_c>(a,b,nc_base,m);
  }

  assert(ublas::is_tensor(na));
  return detail::tensor_vector_prod<tensor_c>(a,b,nc_base,m);
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
  using shape_c       = remove_element_t<m,shape>; // this is wrong
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
