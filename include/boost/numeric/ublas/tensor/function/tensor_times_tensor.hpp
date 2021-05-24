//
//  Copyright (c) 2021, Cem Bassoy, cem.bassoy@gmail.com
//  Copyright (c) 2021, Amit Singh, amitsingh19975@gmail.com
//
//  Distributed under the Boost Software License, Version 1.0. (See
//  accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)
//
//

#ifndef BOOST_NUMERIC_UBLAS_TENSOR_TTT_HPP
#define BOOST_NUMERIC_UBLAS_TENSOR_TTT_HPP

#include <cmath>
#include <stdexcept>
#include <type_traits>

#include "../extents.hpp"
#include "../tags.hpp"
#include "../tensor.hpp"
#include "../type_traits.hpp"


namespace boost::numeric::ublas
{

template<class extents_type, class layout_type, class container_type>
struct tensor_engine;

template<typename tensor_engine>
class tensor_core;

template<class value_type, class layout, class allocator>
class matrix;

} // namespace boost::numeric::ublas

namespace boost::numeric::ublas
{

namespace detail
{
/** Enables prod(ttt) if E1 or E1 is dynamic extents with static rank
 *
 * extents< > & extents<N>
 * extents<N> & extents< >
 * extents< > & extents< >
 *
 */
template<
  class TEA,
  class TEB,
  class  EA = typename tensor_core< TEA >::extents_type,
  class  EB = typename tensor_core< TEB >::extents_type
  >
using enable_ttt_if_one_extents_has_dynamic_rank = std::enable_if_t<
   ( is_dynamic_rank_v<EA> ||  is_dynamic_rank_v<EB>) &&
   (!is_static_v      <EA> || !is_static_v      <EB>) , bool >;
} // namespace detail
/** @brief Computes the q-mode tensor-times-tensor product
     *
     * Implements C[i1,...,ir,j1,...,js] = sum( A[i1,...,ir+q] * B[j1,...,js+q]  )
     *
     * @note calls ublas::ttt
     *
     * na[phia[x]] = nb[phib[x]] for 1 <= x <= q
     *
     * @param[in]  a  left-hand side tensor with order r+q
     * @param[in]  b  right-hand side tensor with order s+q
     * @param[in]  phia one-based permutation tuple of length q for the first input tensor a can be of type std::vector<std::size_t> or std::array<std::size_t,N>
     * @param[in]  phib one-based permutation tuple of length q for the second input tensor b can be of type std::vector<std::size_t> or std::array<std::size_t,N>
     * @result     tensor with order r+s
    */
template <class TEA, class TEB,
          detail::enable_ttt_if_one_extents_has_dynamic_rank<TEA,TEB> = true >
inline decltype(auto) prod(tensor_core< TEA > const &a,
                           tensor_core< TEB > const &b,
                           std::vector<std::size_t> const &phia,
                           std::vector<std::size_t> const &phib)
{
  using tensorA_type      = tensor_core< TEA >;
  using tensorB_type      = tensor_core< TEB >;
  using extentsA_type     = typename tensorA_type::extents_type;
  using extentsB_type     = typename tensorB_type::extents_type;
  using layoutA_type      = typename tensorA_type::layout_type;
  using container_type    = typename tensorA_type::container_type;
  using resizableA_tag    = typename tensorA_type::resizable_tag;
  using resizableB_tag    = typename tensorB_type::resizable_tag;
  using valueA_type       = typename tensorA_type::value_type;
  using valueB_type       = typename tensorB_type::value_type;

  static_assert(std::is_same_v<resizableA_tag, storage_resizable_container_tag>);
  static_assert(std::is_same_v<resizableB_tag, storage_resizable_container_tag>);
  static_assert(std::is_same_v<valueA_type, valueB_type>);

  static_assert(is_dynamic_rank_v<extentsA_type> || is_dynamic_rank_v<extentsB_type>);


  auto const pa = a.rank();
  auto const pb = b.rank();

  auto const q = std::size_t{phia.size()};

  if (pa == 0ul)        throw std::runtime_error("error in ublas::prod(ttt): order of left-hand side tensor must be greater than 0.");
  if (pb == 0ul)        throw std::runtime_error("error in ublas::prod(ttt): order of right-hand side tensor must be greater than 0.");
  if (pa < q)           throw std::runtime_error("error in ublas::prod(ttt): number of contraction dimensions cannot be greater than the order of the left-hand side tensor.");
  if (pb < q)           throw std::runtime_error("error in ublas::prod(ttt): number of contraction dimensions cannot be greater than the order of the right-hand side tensor.");
  if (q != phib.size()) throw std::runtime_error("error in ublas::prod(ttt): permutation tuples must have the same length.");
  if (pa < phia.size()) throw std::runtime_error("error in ublas::prod(ttt): permutation tuple for the left-hand side tensor cannot be greater than the corresponding order.");
  if (pb < phib.size()) throw std::runtime_error("error in ublas::prod(ttt): permutation tuple for the right-hand side tensor cannot be greater than the corresponding order.");

  auto const &na = a.extents();
  auto const &nb = b.extents();

  for (auto i = 0ul; i < q; ++i)
    if (na.at(phia.at(i) - 1) != nb.at(phib.at(i) - 1))
      throw std::runtime_error("error in ublas::prod: permutations of the extents are not correct.");

  auto const r = pa - q;
  auto const s = pb - q;

  auto phia1 = std::vector<std::size_t>(pa);
  auto phib1 = std::vector<std::size_t>(pb);
  std::iota(phia1.begin(), phia1.end(), std::size_t(1));
  std::iota(phib1.begin(), phib1.end(), std::size_t(1));

  using dynamic_extents = std::conditional_t<is_dynamic_rank_v<extentsA_type>, extentsA_type, extentsB_type>;
  using extents_base = typename dynamic_extents::base_type;
  auto const size = std::size_t(pa+pb-2*q);
  auto nc_base = extents_base (std::max(size,std::size_t{2}),std::size_t{1});

  //for (auto i = 0ul; i < phia.size(); ++i)
  for (auto p : phia)
    *std::remove(phia1.begin(), phia1.end(), p) = p;
  //phia1.erase( std::remove(phia1.begin(), phia1.end(), phia.at(i)),  phia1.end() )  ;

  for (auto i = 0ul; i < r; ++i)
    nc_base[i] = na[phia1[i] - 1];

  //for (auto i = 0ul; i < phib.size(); ++i)
  for (auto p : phib)
    *std::remove(phib1.begin(), phib1.end(), p) = p;
  //phib1.erase( std::remove(phib1.begin(), phib1.end(), phia.at(i)), phib1.end() )  ;

  for (auto i = 0ul; i < s; ++i)
    nc_base[r + i] = nb[phib1[i] - 1];

  assert(phia1.size() == pa);
  assert(phib1.size() == pb);

  auto nc = dynamic_extents(nc_base);

  using return_tensor_type = tensor_core<tensor_engine<dynamic_extents, layoutA_type, container_type>>;
  auto c = return_tensor_type( nc, valueA_type{} );

  ttt(pa, pb, q,
      phia1.data(), phib1.data(),
      c.data(), c.extents().data(), c.strides().data(),
      a.data(), a.extents().data(), a.strides().data(),
      b.data(), b.extents().data(), b.strides().data());

  return c;
}



/** @brief Computes the q-mode tensor-times-tensor product
     *
     * Implements C[i1,...,ir,j1,...,js] = sum( A[i1,...,ir+q] * B[j1,...,js+q]  )
     *
     * @note calls ublas::ttt
     *
     * na[phi[x]] = nb[phi[x]] for 1 <= x <= q
     *
     * @param[in]  a  left-hand side tensor with order r+q
     * @param[in]  b  right-hand side tensor with order s+q
     * @param[in]  phi one-based permutation tuple of length q for both input
     * tensors can be of type std::vector<std::size_t> or std::array<std::size_t,N>
     * @result     tensor with order r+s
    */
template <class TEA, class TEB, detail::enable_ttt_if_one_extents_has_dynamic_rank<TEA, TEB> = true >
inline decltype(auto) prod(tensor_core<TEA> const &a,
                           tensor_core<TEB> const &b,
                           std::vector<std::size_t> const &phi)
{
  return prod(a, b, phi, phi);
}



namespace detail
{

/** Enables if extents E1, E1 are dynamic extents with static rank
 *
 * extents<N> & extents<N>
 *
 */
template<
  class TE1,
  class TE2,
  class  E1 = typename tensor_core< TE1 >::extents_type,
  class  E2 = typename tensor_core< TE2 >::extents_type
  >
using enable_ttt_if_extents_have_static_rank = std::enable_if_t<
  (is_static_rank_v<E1> && is_dynamic_v<E1>) &&
  (is_static_rank_v<E2> && is_dynamic_v<E2>) ,  bool>;

} // namespace detail

/** @brief Computes the q-mode tensor-times-tensor product
     *
     * Implements C[i1,...,ir,j1,...,js] = sum( A[i1,...,ir+q] * B[j1,...,js+q]  )
     *
     * @note calls ublas::ttt
     *
     * na[phia[x]] = nb[phib[x]] for 1 <= x <= q
     *
     * @param[in]  a  left-hand side tensor with order r+q
     * @param[in]  b  right-hand side tensor with order s+q
     * @param[in]  phia one-based permutation tuple of length q for the first input tensor a
     * @param[in]  phib one-based permutation tuple of length q for the second input tensor b
     * @result     tensor with order r+s
    */
template <class TEA, class TEB, std::size_t Q,
          detail::enable_ttt_if_extents_have_static_rank<TEA, TEB> = true >
inline auto prod(tensor_core<TEA> const &a,
                 tensor_core<TEB> const &b,
                 std::array<std::size_t,Q> const &phia,
                 std::array<std::size_t,Q> const &phib)
{
  using tensorA_type      = tensor_core<TEA>;
  using tensorB_type      = tensor_core<TEB>;
  using extentsA_type     = typename tensorA_type::extents_type;
  using extentsB_type     = typename tensorB_type::extents_type;
  using valueA_type       = typename tensorA_type::value_type;
  using valueB_type       = typename tensorB_type::value_type;
  using layout_type       = typename tensorA_type::layout_type;
  using container_type    = typename tensorA_type::container_type;
  using resizeableA_tag   = typename tensorA_type::resizable_tag;
  using resizeableB_tag   = typename tensorB_type::resizable_tag;


  static_assert(std::is_same_v<resizeableA_tag, storage_resizable_container_tag>);
  static_assert(std::is_same_v<resizeableB_tag, storage_resizable_container_tag>);
  static_assert(std::is_same_v<valueA_type, valueB_type>);

  constexpr auto q = Q;
  constexpr auto pa = std::tuple_size_v<extentsA_type>;
  constexpr auto pb = std::tuple_size_v<extentsB_type>;

  static_assert(pa != 0);
  static_assert(pb != 0);
  static_assert(pa >= q);
  static_assert(pb >= q);

//  if (pa < phia.size()) throw std::runtime_error("error in ublas::prod: permutation tuple for the left-hand side tensor cannot be greater than the corresponding order.");
//  if (pb < phib.size()) throw std::runtime_error("error in ublas::prod: permutation tuple for the right-hand side tensor cannot be greater than the corresponding order.");

  auto const &na = a.extents();
  auto const &nb = b.extents();

  for (auto i = 0ul; i < q; ++i)
    if (na.at(phia.at(i) - 1) != nb.at(phib.at(i) - 1))
      throw std::runtime_error("error in ublas::prod: permutations of the extents are not correct.");

  constexpr auto r = pa - q;
  constexpr auto s = pb - q;

  auto phia1 = std::array<std::size_t,pa>{};
  auto phib1 = std::array<std::size_t,pb>{};
  std::iota(phia1.begin(), phia1.end(),std::size_t(1));
  std::iota(phib1.begin(), phib1.end(),std::size_t(1));

  constexpr auto const msz = std::max(std::size_t(r+s), std::size_t(2));
  using return_extents_type = extents<msz>;
  auto nc_base = std::array<std::size_t,msz>{};

  for (auto i = 0ul; i < phia.size(); ++i)
    *std::remove(phia1.begin(), phia1.end(), phia.at(i)) = phia.at(i);
  //phia1.erase( std::remove(phia1.begin(), phia1.end(), phia.at(i)),  phia1.end() )  ;

  for (auto i = 0ul; i < phib.size(); ++i)
    *std::remove(phib1.begin(), phib1.end(), phib.at(i)) = phib.at(i);
  //phib1.erase( std::remove(phib1.begin(), phib1.end(), phia.at(i)), phib1.end() )  ;

  for (auto i = 0ul; i < r; ++i)
    nc_base[i] = na[phia1[i] - 1];

  for (auto i = 0ul; i < s; ++i)
    nc_base[r+i] = nb[phib1[i] - 1];

  auto nc = return_extents_type(nc_base);

  using return_tensor_type = tensor_core<tensor_engine<return_extents_type,layout_type,container_type>>;

  auto c = return_tensor_type( nc );

  ttt(pa, pb, q,
      phia1.data(), phib1.data(),
      c.data(), c.extents().data(), c.strides().data(),
      a.data(), a.extents().data(), a.strides().data(),
      b.data(), b.extents().data(), b.strides().data());

  return c;
}


/** @brief Computes the q-mode tensor-times-tensor product
     *
     * Implements C[i1,...,ir,j1,...,js] = sum( A[i1,...,ir+q] * B[j1,...,js+q]  )
     *
     * @note calls ublas::ttt
     *
     * na[phi[x]] = nb[phi[x]] for 1 <= x <= q
     *
     * @param[in]  a  left-hand side tensor with order r+q
     * @param[in]  b  right-hand side tensor with order s+q
     * @param[in]  phi one-based permutation tuple of length q for both input
     * tensors can be of type std::vector<std::size_t> or std::array<std::size_t,N>
     * @result     tensor with order r+s
    */
template <class TEA, class TEB, std::size_t Q,
          detail::enable_ttt_if_extents_have_static_rank<TEA, TEB>* = nullptr >
inline decltype(auto) prod(tensor_core<TEA> const &a,
                           tensor_core<TEB> const &b,
                           std::array<std::size_t,Q> const &phi)
{
  return prod(a, b, phi, phi);
}




} // namespace boost::numeric::ublas

#endif   // BOOST_NUMERIC_UBLAS_TENSOR_NORM_HPP
