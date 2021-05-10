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

//#include "../static_extents.hpp"
#include "../fixed_rank_extents.hpp"


#include "../detail/extents_functions.hpp"
#include "../traits/basic_type_traits.hpp"
#include "../traits/storage_traits.hpp"
#include "../tags.hpp"

namespace boost::numeric::ublas
{
template <class ExtentsType, ExtentsType... E>
class basic_static_extents;

template<typename ... >
struct tensor_engine;

template<typename tensor_engine>
class tensor_core;

template<class value_type, class layout, class allocator>
class matrix;

} // namespace boost::numeric::ublas

namespace boost::numeric::ublas
{

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
template <typename TE,
          typename A,
          typename T = typename tensor_core< TE >::value_type,
          typename L = typename tensor_core< TE >::layout_type>
inline decltype(auto) prod( tensor_core< TE > const &a, matrix<T,L,A> const &b, const std::size_t m)
{

  using tensor_type    = tensor_core< TE >;
  using extents_type   = typename tensor_type::extents_type;
  using value_type     = typename tensor_type::value_type;
  using layout_type    = typename tensor_type::layout_type;
  using resizeable_tag = typename tensor_type::resizable_tag;

  static_assert(std::is_same_v<resizeable_tag, storage_resizable_container_tag> );
  static_assert(is_dynamic_v<extents_type>);

  auto const p = a.rank();
  auto const size_b = b.size1()*b.size2();

  //  || m > ublas::size(a.extents())

  if (m == 0ul)    throw std::length_error("error in boost::numeric::ublas::prod(ttm): contraction mode must be greater than zero.");
  if (p < m)       throw std::length_error("error in boost::numeric::ublas::prod(ttm): rank of the tensor must be greater equal the modus.");
  if (a.empty())   throw std::length_error("error in boost::numeric::ublas::prod(ttm): first argument tensor should not be empty.");
  if (size_b==0ul) throw std::length_error("error in boost::numeric::ublas::prod(ttm): second argument matrix should not be empty.");

  const auto& na = a.extents();
  auto nc_base = na.base();
  auto nb = extents<2>{b.size1(), b.size2()};
  auto wb = basic_fixed_rank_strides<std::size_t,2,layout_type>(nb);

  nc_base[m-1] = nb[0];
  auto nc = extents_type(nc_base);

  auto c = tensor_type(nc, value_type{});

  auto bb = &(b(0, 0));
  ttm(m, p,
      c.data(), data(nc), c.strides().data(),
      a.data(), data(na), a.strides().data(),
      bb, data(nb), wb.data());

  return c;
}


namespace detail{
template<std::size_t I, std::size_t Value, typename ExtentsType, std::size_t... Is>
constexpr auto static_extents_set_at_helper( [[maybe_unused]] ExtentsType const& /*e*/, [[maybe_unused]] std::index_sequence<Is...> /*is*/){
  using extents_type = typename ExtentsType::value_type;
  constexpr auto res_arr = static_extents_set_at_impl<I,Value>(ExtentsType{});
  return basic_static_extents<extents_type, ( ..., res_arr[Is] ) >{};
}

template<std::size_t I, std::size_t Value, typename ExtentsType>
constexpr auto static_extents_set_at( [[maybe_unused]] ExtentsType const& /*e*/){
  static_assert(is_static_v<ExtentsType>);
  static_assert( I < size(ExtentsType{}), "boost::numeric::ublas::detail::static_extents_set_at(ExtentsType const&): out of bound");
  return static_extents_set_at_helper<I,Value>(ExtentsType{}, std::make_index_sequence<size(ExtentsType{})>{});
}
} // namespace detail



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
template <size_t M,
          size_t N,
          typename TE,
          typename A,
          typename T = typename tensor_core< TE >::value_type,
          typename L = typename tensor_core< TE >::layout_type>
inline decltype(auto) prod(tensor_core<TE> const &a, matrix<T,L,A> const &b)
{
  using tensor_type   = tensor_core<TE>;
  using extents_type  = typename tensor_type::extents_type;
  using layout_type   = typename tensor_type::layout_type;
  using value_type    = typename tensor_type::value_type;
  using array_type    = typename tensor_type::array_type;

  auto const p = a.rank();

  static_assert(M != 0ul);

  static_assert( std::tuple_size_v<extents_type> >= M);
  static_assert( std::tuple_size_v<extents_type> >  0);

  if (b.size1()*b.size2() == 0ul)
    throw std::length_error("error in boost::numeric::ublas::prod(ttm): second argument matrix should not be empty.");


  auto const& na = a.extents();
  auto nc = detail::static_extents_set_at<M-1, N>( na );
  auto nb = extents<2>{b.size1(), b.size2()};
  auto wb = basic_fixed_rank_strides<std::size_t,2,layout_type>(nb);

  using return_extents_type = std::decay_t<decltype(nc)>;
  using storage_type        = rebind_storage_size_t<return_extents_type,array_type>;
  using return_tensor_type  = tensor_core<tensor_engine<return_extents_type,layout_type, storage_type >>;

  auto c = return_tensor_type(value_type{});

  auto bbdata = &(b(0, 0));

  auto const& wa = a.strides();
  auto const& wc = c.strides();

  ttm(M, p,
      c.data(), data(nc), wc.data(),
      a.data(), data(na), wa.data(),
      bbdata  , data(nb), wb.data());

  return c;
}


} // namespace boost::numeric::ublas

#endif   // BOOST_NUMERIC_UBLAS_TENSOR_NORM_HPP
