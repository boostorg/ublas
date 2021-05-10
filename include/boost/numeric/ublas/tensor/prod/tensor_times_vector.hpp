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

#include "../static_extents.hpp"
#include "../fixed_rank_extents.hpp"

#include "../detail/extents_functions.hpp"
#include "../traits/basic_type_traits.hpp"
#include "../traits/storage_traits.hpp"
#include "../tags.hpp"

namespace boost::numeric::ublas
{

template<typename ... >
struct tensor_engine;

template<typename tensor_engine>
class tensor_core;

template<class type, class allocator>
class vector;

} // namespace boost::numeric::ublas

namespace boost::numeric::ublas
{

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
template <typename TE,
          typename A,
          typename T = typename tensor_core< TE >::value_type,
          typename E = typename tensor_core< TE >::extents_type,
          std::enable_if_t< is_dynamic_rank_v< E >, void >* = nullptr
          >
inline decltype(auto) prod( tensor_core< TE > const &a, vector<T, A> const &b, const std::size_t m)
{

  using tensor_type   = tensor_core< TE >;
  using extents_type  = typename tensor_type::extents_type;
  using value_type    = typename tensor_type::value_type;
  using resize_tag    = typename tensor_type::resizable_tag;
  using size_type     = typename extents_type::size_type;
  using extents_value_type = typename extents_type::value_type;
  using extents_base_type  = typename extents_type::base_type;

  auto const p = a.rank();

  static_assert(std::is_same_v<resize_tag,storage_resizable_container_tag>);
  static_assert(is_dynamic_v<extents_type>);

  if (m == 0ul)  throw std::length_error("error in boost::numeric::ublas::prod(ttv): contraction mode must be greater than zero.");
  if (p < m)     throw std::length_error("error in boost::numeric::ublas::prod(ttv): rank of tensor must be greater than or equal to the contraction mode.");
  if (a.empty()) throw std::length_error("error in boost::numeric::ublas::prod(ttv): first argument tensor should not be empty.");
  if (b.empty()) throw std::length_error("error in boost::numeric::ublas::prod(ttv): second argument vector should not be empty.");

  auto const& na = a.extents();
  auto nb = std::vector<extents_value_type>{b.size(), extents_value_type(1)};

  auto const sz = std::max( ublas::size(na) - 1, size_type(2) );
  auto nc_base = extents_base_type(sz,1);

  for (auto i = 0ul, j = 0ul; i < p; ++i)
    if (i != m - 1)
      nc_base[j++] = na.at(i);

  auto nc = extents_type(nc_base);


  auto c = tensor_type( nc, value_type{} );
  auto bb = &(b(0));
  ttv(m, p,
      c.data(), data(c.extents()), c.strides().data(),
      a.data(), data(a.extents()), a.strides().data(),
      bb, data(nb), data(nb));
  return c;
}



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
template <typename TE,
          typename A,
          typename E = typename tensor_core< TE >::extents_type,
          typename T = typename tensor_core< TE >::value_type,
          std::enable_if_t< is_static_rank_v< E > && is_dynamic_v< E >, void >* = nullptr
          >
inline decltype(auto) prod( tensor_core< TE > const &a, vector<T, A> const &b, const std::size_t m)
{

  using tensor_type    = tensor_core< TE >;
  using extents_type   = typename tensor_type::extents_type;
  using value_type     = typename tensor_type::value_type;
  using array_type     = typename tensor_type::array_type;
  using layout_type    = typename tensor_type::layout_type;
  using resizeable_tag = typename tensor_type::resizable_tag;
  using extents_value_type = typename extents_type::value_type;
  using size_type = typename extents_type::size_type;

  constexpr auto p = std::tuple_size_v<extents_type>;

  static_assert(std::is_same_v<resizeable_tag,storage_resizable_container_tag >);

  if (m == 0ul)  throw std::length_error("error in boost::numeric::ublas::prod(ttv): contraction mode must be greater than zero.");
  if (p < m)     throw std::length_error("error in boost::numeric::ublas::prod(ttv): rank of tensor must be greater than or equal to the modus.");
  if (a.empty()) throw std::length_error("error in boost::numeric::ublas::prod(ttv): first argument tensor should not be empty.");
  if (b.empty()) throw std::length_error("error in boost::numeric::ublas::prod(ttv): second argument vector should not be empty.");

  auto const& na = a.extents();

  constexpr size_type sz = std::max( std::tuple_size_v<extents_type> -1u , size_type(2) );
  using return_extents_type = ublas::extents<sz>;
  auto nc_base = typename return_extents_type::base_type{};
  std::fill(nc_base.begin(), nc_base.end(), size_type(1));
  for (auto i = 0ul, j = 0ul; i < p; ++i)
    if (i != m - 1)
      nc_base[j++] = na.at(i);

  auto nc = return_extents_type(nc_base);
  auto nb = std::vector<extents_value_type>{b.size(), extents_value_type(1)};

  using return_tensor_type = tensor_core<tensor_engine<return_extents_type,layout_type,array_type >>;

  auto c = return_tensor_type( nc, value_type{} );
  auto bb = &(b(0));
  ttv(m, p,
      c.data(), data(c.extents()), c.strides().data(),
      a.data(), data(a.extents()), a.strides().data(),
      bb, data(nb), data(nb));
  return c;
}





namespace detail{
template<typename T, std::size_t N>
constexpr auto array_of_ones() noexcept{
  std::array<T,N> ones{};
  std::fill(ones.begin(), ones.end(), T{1});
  return ones;
}

template<std::size_t M, typename ExtentsType>
constexpr auto extents_result_tensor_times_vector_impl(ExtentsType const& e) noexcept{
  static_assert(size(ExtentsType{}) > 0ul, "extents cannot be empty!");
  using extents_type = typename ExtentsType::value_type;
  constexpr auto sz = size(ExtentsType{}) - 1ul;
  auto res = array_of_ones<extents_type,sz>();

  auto j = 0ul;
  for(auto i = 0ul; i < sz; ++i){
    if(i != M - 1ul) res[j++] = e[i];
  }
  return res;
}

template<std::size_t M, typename ExtentsType, std::size_t... Is>
constexpr auto extents_result_tensor_times_vector_helper([[maybe_unused]] ExtentsType const& /*e*/, [[maybe_unused]] std::index_sequence<Is...> /*is*/) noexcept{
  using extents_type = typename ExtentsType::value_type;
  constexpr auto res_arr = extents_result_tensor_times_vector_impl<M>(ExtentsType{});
  return basic_static_extents<extents_type, ( ..., res_arr[Is] ) >{};
}

template<std::size_t M, typename ExtentsType>
constexpr auto extents_result_tensor_times_vector([[maybe_unused]] ExtentsType const& /*e*/) noexcept{
  static_assert(is_static_v<ExtentsType>);
  return extents_result_tensor_times_vector_helper<M>(ExtentsType{}, std::make_index_sequence<size(ExtentsType{})>{});
}

template<std::size_t I, std::size_t Value, typename ExtentsType>
constexpr auto static_extents_set_at_impl(ExtentsType const& e) noexcept{
  using extents_type = typename ExtentsType::value_type;

  auto res = e.base();

  res[I] = static_cast<extents_type>(Value);

  return res;
}

} // namespace detail

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

template <std::size_t M,
          typename TE,
          typename A,
          typename T = typename tensor_core< TE >::value_type>
inline decltype(auto) prod( tensor_core< TE > const &a, vector<T, A> const &b)
{
  using tensor_type   = tensor_core< TE >;
  using array_type    = typename tensor_type::array_type;
  using extents_type  = typename tensor_type::extents_type;
  using value_type    = typename tensor_type::value_type;
  using layout_type   = typename tensor_type::layout_type;
  using extents_value_type = typename extents_type::value_type;

  static_assert( M != 0ul );
  static_assert(std::tuple_size_v<extents_type> != 0 );
  static_assert(std::tuple_size_v<extents_type> >= M );

  constexpr auto p = std::tuple_size_v<extents_type>;

  if (a.empty()) throw std::length_error("error in boost::numeric::ublas::prod(ttv): first argument tensor should not be empty.");
  if (b.empty()) throw std::length_error("error in boost::numeric::ublas::prod(ttv): second argument vector should not be empty.");

  auto const& na = a.extents();
  auto nc = detail::extents_result_tensor_times_vector<M>(na);
  auto nb = std::vector<extents_value_type>{b.size(), extents_value_type(1)};

  using return_extents_type = std::decay_t<decltype(nc)>;
  using storage_type   = rebind_storage_size_t<return_extents_type,array_type>;
  using return_tensor_type = tensor_core<tensor_engine<return_extents_type,layout_type, storage_type > >;

  auto c = return_tensor_type(value_type{});
  auto bb = &(b(0));

  auto const& wa = a.strides();
  auto const& wc = c.strides();

//  auto const& na = a.extents().base();
//  auto const& nc = c.extents().base();

//  auto& a_static_strides = a.strides().base();
//  auto& c_static_strides = c.strides().base();

  ttv(M, p,
      c.data(), nc.data(), wc.data(),
      a.data(), na.data(), wa.data(),
      bb, data(nb), data(nb));

  return c;
}

} // namespace boost::numeric::ublas

#endif   // BOOST_NUMERIC_UBLAS_TENSOR_NORM_HPP
