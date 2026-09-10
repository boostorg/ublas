//
//  Copyright (c) 2021, Cem Bassoy, cem.bassoy@gmail.com
//
//  Distributed under the Boost Software License, Version 1.0. (See
//  accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)
//
//

#ifndef BOOST_NUMERIC_UBLAS_TENSOR_FUNCTIONS_INIT_HPP
#define BOOST_NUMERIC_UBLAS_TENSOR_FUNCTIONS_INIT_HPP

#include "../extents.hpp"
#include "../tensor.hpp"
#include "../concepts.hpp"

//#include <concepts>


namespace boost::numeric::ublas
{

template<class V, class L, auto n>
struct init
{
  using container = std::vector<V>;
  using tensor = tensor_core<tensor_engine<extents<>, L, container>>;

  inline auto operator()(extents<> const& e) const
  {
    auto p = ublas::product(e);
    return tensor(e,container(p,V{n}));
  }


  template<integral ... Ns>
  inline auto operator()(Ns ... ns) const
  {
    auto p = ( std::size_t(1) * ... * std::size_t(ns) );
    return tensor(extents<>{std::size_t(ns)...},container(p,V{n}));
  }
};

template<class V = float, class L = layout::first_order>
using ones = init<V,L,1>;

template<class V = float, class L = layout::first_order>
using zeros = init<V,L,0>;



template<class V, class L, auto k>
struct init_static_rank
{
  using container = std::vector<V>;

  template<std::size_t n>
  inline auto operator()(extents<n> const& e) const
  {
    auto p = ublas::product(e);
    using tensor = tensor_core<tensor_engine<extents<n>, L, container>>;

    return tensor(e, container(p,V{k}));
  }


  template<integral ... Ns>
  inline auto operator()(Ns ... ns) const
  {
    constexpr auto n = sizeof...(ns);
    auto p = ( std::size_t(1) * ... * std::size_t(ns) );
    using tensor = tensor_core<tensor_engine<extents<n>, L, container>>;

    return tensor(extents<n>{std::size_t(ns)...}, container(p,V{k}));
  }
};

template<class V = float, class L = layout::first_order>
using ones_static_rank = init_static_rank<V,L,1>;

template<class V = float, class L = layout::first_order>
using zeros_static_rank = init_static_rank<V,L,0>;


template<class V , class L, auto k>
struct init_static
{
  template<class>
  struct inner;

  template<std::size_t ... is>
  struct inner<std::index_sequence<is...>>
  {
    static constexpr auto n  = sizeof...(is);
    // NOLINTNEXTLINE(bugprone-integer-division)
    static constexpr auto value = std::array<V,n>{  V(k*(is+1)/(is+1)) ... };
  };

  template<std::size_t ...ns>
  constexpr inline auto operator()(extents<ns...> const& /**/) const
  {
    using extents_type = extents<ns...>;
    constexpr auto p  = product_v<extents_type>;
    constexpr auto c = inner<std::make_index_sequence<p>>::value;
    using tensor = tensor_core<tensor_engine<extents_type, L, std::array<V,p>>>;
    return tensor(c);
  }
};


template<class V = float, class L = layout::first_order>
using ones_static = init_static<V,L,1>;

template<class V = float, class L = layout::first_order>
using zeros_static = init_static<V,L,0>;


} // namespace boost::numeric::ublas

#endif   // BOOST_NUMERIC_UBLAS_TENSOR_FUNCTIONS_ONES_HPP
