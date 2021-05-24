//
//  Copyright (c) 2018, Cem Bassoy, cem.bassoy@gmail.com
//  Copyright (c) 2020, Amit Singh, amitsingh19975@gmail.com
//
//  Distributed under the Boost Software License, Version 1.0. (See
//  accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)
//
//

#ifndef BOOST_NUMERIC_UBLAS_TENSOR_OUTER_HPP
#define BOOST_NUMERIC_UBLAS_TENSOR_OUTER_HPP

#include <stdexcept>
#include <type_traits>

#include "../extents.hpp"
#include "../multiplication.hpp"
#include "../type_traits.hpp"
#include "../tags.hpp"


namespace boost::numeric::ublas
{

template<class T, class L, class ST>
struct tensor_engine;

template<typename tensor_engine>
class tensor_core;

} // namespace boost::numeric::ublas

namespace boost::numeric::ublas
{


namespace detail{
/** Enables if E1 or E1 is dynamic extents with static rank
 *
 * extents< > & extents<N>
 * extents<N> & extents< >
 * extents< > & extents< >
 *
 */
template<
  class TEA,
  class TEB,
  class  EA = typename tensor_core<TEA>::extents_type,
  class  EB = typename tensor_core<TEB>::extents_type
  >
using enable_outer_if_one_extents_has_dynamic_rank = std::enable_if_t<
  ( is_dynamic_rank_v<EA> || is_dynamic_rank_v<EB>) &&
  (!is_static_v      <EA> || !is_static_v     <EB>) , bool >;

} // namespace detail

/** @brief Computes the outer product of two tensors
     *
     * Implements C[i1,...,ip,j1,...,jq] = A[i1,i2,...,ip] * B[j1,j2,...,jq]
     *
     * @note calls outer function
     *
     * @param[in] a tensor object A
     * @param[in] b tensor object B
     *
     * @returns tensor object C with the same storage format F and allocator type A1
    */
template <class TEA, class TEB, detail::enable_outer_if_one_extents_has_dynamic_rank<TEA,TEB> = true >
inline auto outer_prod( tensor_core< TEA > const &a, tensor_core< TEB > const &b)
{
  using tensorA   = tensor_core< TEA >;
  using tensorB   = tensor_core< TEB >;
  using valueA    = typename tensorA::value_type;
  using extentsA  = typename tensorA::extents_type;

  using valueB    = typename tensorB::value_type;
  using extentsB  = typename tensorB::extents_type;

  using tensorC   = std::conditional_t < is_dynamic_rank_v<extentsA>, tensorA, tensorB>;
//  using valueC    = typename tensorC::value_type;
  using extentsC  = typename tensorC::extents_type;

  static_assert( std::is_same_v<valueA, valueB> );
  static_assert( is_dynamic_rank_v<extentsA> || is_dynamic_rank_v<extentsB>);

  if (a.empty() || b.empty()){
    throw std::runtime_error("Error in boost::numeric::ublas::outer_prod: tensors should not be empty.");
  }

  auto const& na = a.extents();
  auto const& nb = b.extents();

  auto nc_base = typename extentsC::base_type(ublas::size(na)+ublas::size(nb));
  auto nci = std::copy(ublas::begin(na),ublas::end(na), std::begin(nc_base));
  std::copy(ublas::begin(nb),ublas::end(nb), nci);
  auto nc = extentsC(nc_base);

  auto c = tensorC( nc, valueA{} );

  outer(c.data(), c.rank(), nc.data(), c.strides().data(),
        a.data(), a.rank(), na.data(), a.strides().data(),
        b.data(), b.rank(), nb.data(), b.strides().data());

  return c;
}


namespace detail{
/** Enables if extents E1, E1
 *
 * both are dynamic extents with static rank
 *
 * extents<N> & extents<N>
 *
 */
template<
  class TEA,
  class TEB,
  class  E1 = typename tensor_core<TEA>::extents_type,
  class  E2 = typename tensor_core<TEB>::extents_type
  >
using enable_outer_if_both_extents_have_static_rank = std::enable_if_t<
  ( is_static_rank_v<E1> && is_dynamic_v<E1>) &&
  ( is_static_rank_v<E2> && is_dynamic_v<E2>) , bool >;
} // namespace detail

/** @brief Computes the outer product of two tensors
     *
     * Implements C[i1,...,ip,j1,...,jq] = A[i1,i2,...,ip] * B[j1,j2,...,jq]
     *
     * @note calls outer function
     *
     * @param[in] a tensor object A
     * @param[in] b tensor object B
     *
     * @returns tensor object C with the same storage format F and allocator type A1
    */
template <class TEA, class TEB, detail::enable_outer_if_both_extents_have_static_rank<TEA,TEB> = true >
inline auto outer_prod(tensor_core<TEA> const &a, tensor_core<TEB> const &b)
{
  using tensorA        = tensor_core<TEA>;
  using valueA         = typename tensorA::value_type;
  using layoutA        = typename tensorA::layout_type;
  using extentsA       = typename tensorA::extents_type;
  using containerA     = typename tensorA::container_type;
  using resizableA_tag = typename tensorA::resizable_tag;

  using tensorB    = tensor_core<TEB>;
  using valueB     = typename tensorB::value_type;
//  using layoutB    = typename tensorB::layout_type;
  using extentsB   = typename tensorB::extents_type;
  using resizableB_tag  = typename tensorB::resizable_tag;

  static_assert(std::is_same_v<valueA, valueB>);
  static_assert(is_static_rank_v<extentsA> || is_static_rank_v<extentsB>);

  static_assert(std::is_same_v<resizableA_tag, storage_resizable_container_tag>);
  static_assert(std::is_same_v<resizableB_tag, storage_resizable_container_tag>);

  if (a.empty() || b.empty())
    throw std::runtime_error("error in boost::numeric::ublas::outer_prod: tensors should not be empty.");

  auto const& na = a.extents();
  auto const& nb = b.extents();

  constexpr auto sizeA = std::tuple_size_v<extentsA>;
  constexpr auto sizeB = std::tuple_size_v<extentsB>;

  using extentsC = extents<sizeA+sizeB>;
  using tensorC = tensor_core<tensor_engine<extentsC,layoutA,containerA>>;

  auto nc_base = typename extentsC::base_type{};
  auto nci = std::copy(ublas::begin(na), ublas::end(na), std::begin(nc_base));
  std::copy(ublas::begin(nb),ublas::end(nb), nci);
  auto nc = extentsC( nc_base );

  auto c = tensorC( nc );

  outer(c.data(), c.rank(), nc.data(), c.strides().data(),
        a.data(), a.rank(), na.data(), a.strides().data(),
        b.data(), b.rank(), nb.data(), b.strides().data());

  return c;
}


namespace detail {

// concat two static_stride_list togather
// @code using type = typename concat< static_stride_list<int, 1,2,3>, static_stride_list<int, 4,5,6> >::type @endcode
template<typename L1, typename L2>
struct concat;

template<typename T, T... N1, T... N2>
struct concat< basic_static_extents<T, N1...>, basic_static_extents<T, N2...> > {
  using type = basic_static_extents<T, N1..., N2...>;
};

template<typename L1, typename L2>
using concat_t = typename concat<L1,L2>::type;

} // namespace detail

namespace detail {
/** Enables if extents E1, E1
 *
 * both are dynamic extents with static rank
 *
 * extents<N> & extents<N>
 *
 */
template<
  class TEA,
  class TEB,
  class  E1 = typename tensor_core<TEA>::extents_type,
  class  E2 = typename tensor_core<TEB>::extents_type
  >
using enable_outer_if_both_extents_are_static = std::enable_if_t<
  ( is_static_v<E1> && is_static_v<E2>) , bool>;

} // namespace detail
/** @brief Computes the outer product of two tensors
     *
     * Implements C[i1,...,ip,j1,...,jq] = A[i1,i2,...,ip] * B[j1,j2,...,jq]
     *
     * @note calls outer function
     *
     * @param[in] a tensor object A
     * @param[in] b tensor object B
     *
     * @returns tensor object C with the same storage format F and allocator type A1
    */
template <typename TEA, typename TEB, detail::enable_outer_if_both_extents_are_static<TEA,TEB> = true >
inline decltype(auto) outer_prod(tensor_core<TEA> const &a, tensor_core<TEB> const &b)
{
  using tensorA    = tensor_core<TEA>;
  using valueA     = typename tensorA::value_type;
  using layoutA    = typename tensorA::layout_type;
  using extentsA   = typename tensorA::extents_type;
  using arrayA     = typename tensorA::array_type;
//  using resizableA_tag  = typename tensorA::resizable_tag;

  using tensorB    = tensor_core<TEB>;
  using valueB     = typename tensorB::value_type;
//  using layoutB    = typename tensorB::layout_type;
  using extentsB   = typename tensorB::extents_type;
//  using resizableB_tag  = typename tensorB::resizable_tag;

  using extentsC   = ublas::cat_t<extentsA,extentsB>;//  detail::concat_t<extentsA, extentsB>;
  using layoutC    = layoutA;
  using valueC     = valueA;
  using storageC   = rebind_storage_size_t<extentsC,arrayA>;
  using tensorC    = tensor_core<tensor_engine<extentsC,layoutC,storageC>>;

  static_assert(std::is_same_v<valueA, valueB>);
  static_assert(is_static_v<extentsA> || is_static_v<extentsB>);

  constexpr auto extentsA_size = std::tuple_size_v<extentsA>;
  constexpr auto extentsB_size = std::tuple_size_v<extentsB>;


  if (a.empty() || b.empty())
    throw std::runtime_error("error in boost::numeric::ublas::outer_prod: tensors should not be empty.");

  auto nc = extentsC{};

  auto const& na = a.extents();
  auto const& nb = b.extents();

  auto c = tensorC(valueC{});

  outer(c.data(), c.rank(), data(nc), c.getStrides().data(),
        a.data(), a.rank(), data(na), a.getStrides().data(),
        b.data(), b.rank(), data(nb), b.getStrides().data());

  return c;
}


} // namespace boost::numeric::ublas

#endif   // BOOST_NUMERIC_UBLAS_TENSOR_PROD_DYNAMIC_HPP
