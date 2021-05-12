//
//  Copyright (c) 2018, Cem Bassoy, cem.bassoy@gmail.com
//  Copyright (c) 2019, Amit Singh, amitsingh19975@gmail.com
//
//  Distributed under the Boost Software License, Version 1.0. (See
//  accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)
//
//  The authors gratefully acknowledge the support of
//  Google and Fraunhofer IOSB, Ettlingen, Germany
//
//  And we acknowledge the support from all contributors.


#include <iostream>
#include <algorithm>
#include <boost/numeric/ublas/tensor.hpp>
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/vector.hpp>

#include <boost/test/unit_test.hpp>

#include "utility.hpp"

// BOOST_AUTO_TEST_SUITE ( test_tensor_functions, * boost::unit_test::depends_on("test_tensor_contraction") )
BOOST_AUTO_TEST_SUITE ( test_fixed_rank_tensor_functions)


using test_types = zip<int,float,std::complex<float>>::with_t<boost::numeric::ublas::layout::first_order, boost::numeric::ublas::layout::last_order>;

//using test_types = zip<int>::with_t<boost::numeric::ublas::layout::first_order>;


struct fixture
{
  template<size_t R>
  using fixed_rank_extents_type = boost::numeric::ublas::extents<R>;

  using dynamic_extents_type = boost::numeric::ublas::extents<>;
  fixture()
    : extents {
        dynamic_extents_type{1,1}, // 1
        dynamic_extents_type{2,3}, // 2
        dynamic_extents_type{2,3,1}, // 3
        dynamic_extents_type{4,2,3}, // 4
        dynamic_extents_type{4,2,3,5}} // 5
  {
  }

  std::tuple<
    fixed_rank_extents_type<2>,
    fixed_rank_extents_type<2>,
    fixed_rank_extents_type<3>,
    fixed_rank_extents_type<3>,
    fixed_rank_extents_type<4>
    > fixed_rank_extents{
      {1,1}, // 1
      {2,3}, // 2
      {2,3,1}, // 3
      {4,2,3}, // 4
      {4,2,3,5} // 5
    };


  std::vector<dynamic_extents_type> extents;

};



BOOST_FIXTURE_TEST_CASE_TEMPLATE( test_fixed_rank_tensor_prod_vector, value,  test_types, fixture )
{
  namespace ublas = boost::numeric::ublas;
  using value_type   = typename value::first_type;
  using layout_type  = typename value::second_type;

  for_each_in_tuple(fixed_rank_extents,[](auto const& /*unused*/, auto const& n){

    constexpr auto size = std::tuple_size_v<std::decay_t<decltype(n)>>;
    using tensor_type = ublas::fixed_rank_tensor<value_type, size, layout_type>;
    using vector_type = typename tensor_type::vector_type;
    auto a = tensor_type(n, value_type{2});

    for (auto m = 0u; m < ublas::size(n); ++m) {
      auto b = vector_type(n[m], value_type{1});

      auto c = ublas::prod(a, b, m + 1);

      for (auto i = 0u; i < c.size(); ++i)
        BOOST_CHECK_EQUAL(c[i], value_type( static_cast< inner_type_t<value_type> >(n[m]) ) * a[i]);
    }
    });

}

BOOST_FIXTURE_TEST_CASE_TEMPLATE( test_fixed_rank_tensor_prod_matrix, value,  test_types, fixture )
{
  namespace ublas = boost::numeric::ublas;
  using value_type   = typename value::first_type;
  using layout_type  = typename value::second_type;

  for_each_in_tuple(fixed_rank_extents,[](auto const& /*unused*/, auto const & n){
    constexpr auto size = std::tuple_size_v<std::decay_t<decltype(n)>>;
    using tensor_type = ublas::fixed_rank_tensor<value_type, size, layout_type>;
    using matrix_type = typename tensor_type::matrix_type;

    auto a = tensor_type(n, value_type{2});
    for (auto m = 0u; m < ublas::size(n); ++m) {

      auto b = matrix_type  ( n[m], n[m], value_type{1} );

      auto c = ublas::prod(a, b, m + 1);

      for (auto i = 0u; i < c.size(); ++i){
        BOOST_CHECK_EQUAL(c[i], value_type( static_cast< inner_type_t<value_type> >(n[m]) ) * a[i]);
      }
    }
  });

}


BOOST_FIXTURE_TEST_CASE_TEMPLATE( test_fixed_rank_tensor_prod_tensor_1, value,  test_types, fixture )
{
  namespace ublas = boost::numeric::ublas;
  using value_type   = typename value::first_type;
  using layout_type  = typename value::second_type;


  auto check = [&]<std::size_t ... qs>(auto const& a, auto const& b, std::index_sequence<qs...> /*unused*/)
  {
    namespace ublas = boost::numeric::ublas;

    constexpr auto q = sizeof...(qs);

    using tensorA = std::decay_t<decltype(a)>;
    using tensorB = std::decay_t<decltype(b)>;

    using extentsA = typename tensorA::extents_type;
    using extentsB = typename tensorB::extents_type;

    static_assert(!ublas::is_static_v<extentsA> && !ublas::is_static_v<extentsB> );

    constexpr auto one_of_extents_is_resizable = ublas::is_dynamic_rank_v<extentsA> ||
                                                 ublas::is_dynamic_rank_v<extentsB>;

    using phi_type = std::conditional_t<one_of_extents_is_resizable,
                                        std::vector<std::size_t>,
                                        std::array<std::size_t,q> >;

    auto phi = phi_type{};
    if constexpr(std::is_same_v<phi_type,std::vector<std::size_t>>){
      phi.resize(q);
    }
    std::iota(phi.begin(), phi.end(), 1ul);

    auto c = ublas::prod(a, b, phi);

    auto acc = value_type(1);
    for (auto i = 0ul; i < q; ++i){
      auto const& na = a.extents();
      acc *= static_cast<value_type>(na.at(phi.at(i)-1));
    }

    for (auto i = 0ul; i < c.size(); ++i){
      BOOST_CHECK_EQUAL(c[i], acc *a[0] * b[0]);
    }
  };


  for_each_in_tuple(fixed_rank_extents,[&](auto const& /*I*/, auto const& n){
    constexpr auto size = std::tuple_size_v<std::decay_t<decltype(n)>>;
    constexpr auto modes = std::make_index_sequence<size>{};
    using tensor_type = ublas::fixed_rank_tensor<value_type, size, layout_type>;
    auto a = tensor_type(n, value_type{2});
    auto b = tensor_type(n, value_type{3});
    for_each_in_index(modes, a,b, check );
  });

  for_each_in_tuple(fixed_rank_extents,[&](auto const& I, auto const& n){
    auto const& nA = n;
    auto const& nB = extents[I];
    constexpr auto sizeA = std::tuple_size_v<std::decay_t<decltype(n)>>;
    constexpr auto modes = std::make_index_sequence<sizeA>{};
    using tensorA_type = ublas::fixed_rank_tensor<value_type, sizeA , layout_type>;
    using tensorB_type = ublas::dynamic_tensor<value_type, layout_type>;
    auto a = tensorA_type(nA, value_type{2});
    auto b = tensorB_type(nB, value_type{3});
    for_each_in_index(modes, a,b, check );
  });

  for_each_in_tuple(fixed_rank_extents,[&](auto const& I, auto const& n){
    auto const& nA = extents[I];
    auto const& nB = n;
    constexpr auto sizeB = std::tuple_size_v<std::decay_t<decltype(n)>>;
    constexpr auto modes = std::make_index_sequence<sizeB>{};
    using tensor_type_1 = ublas::dynamic_tensor<value_type, layout_type>;
    using tensor_type_2 = ublas::fixed_rank_tensor<value_type, sizeB, layout_type>;
    auto a = tensor_type_1(nA, value_type{2});
    auto b = tensor_type_2(nB, value_type{3});
    for_each_in_index(modes, a,b, check );

  });
}

BOOST_FIXTURE_TEST_CASE_TEMPLATE( test_fixed_rank_tensor_prod_tensor_2, value,  test_types, fixture )
{
  namespace ublas = boost::numeric::ublas;
  using value_type   = typename value::first_type;
  using layout_type  = typename value::second_type;

  constexpr auto to_array = []<std::size_t ... is>(std::index_sequence<is...>/*unused*/) {
    return std::array<std::size_t,sizeof...(is)>{is...};
  };

  auto compute_factorial = []<std::size_t ... is>(std::index_sequence<is...>/*unused*/) {
    return ( 1 * ... * is );
  };
/*
  auto compute_factorial = [](auto const& p){
    auto f = 1ul;
    for(auto i = 1u; i <= p; ++i)
      f *= i;
    return f;
  };
*/
  auto permute_extents_dynamic_rank = [](auto const& pi, auto const& na){
    auto nb = ublas::extents<>(na.begin(),na.end());
    assert(std::size(pi) == ublas::size(na));
    for(auto j = 0u; j < std::size(pi); ++j)
      nb[pi[j]-1] = na[j];
    return nb;
  };

  auto permute_extents_static_rank = []<std::size_t size>(std::array<std::size_t,size> const& pi, auto const& na){
    //constexpr auto size = std::tuple_size_v<std::decay_t<decltype(na)>>;
    auto na_base = na.base();
    assert(std::size(pi) == size);
    for(auto j = 0u; j < std::size(pi); ++j)
      na_base[pi[j]-1] = na[j];
    return ublas::extents<size>(na_base.begin(),na_base.end());
  };

  for_each_in_tuple(fixed_rank_extents,[&](auto const& /*unused*/, auto const& n){
    auto const& na = n;
    constexpr auto size = std::tuple_size_v<std::decay_t<decltype(n)>>;
    using tensorA_type = ublas::fixed_rank_tensor<value_type, size, layout_type>;
    auto a = tensorA_type(na, value_type{2});
    assert(a.rank() == size);
//    auto const pa = a.rank();
    auto pi  = to_array(std::make_index_sequence<size>{});
    constexpr auto factorial = compute_factorial(std::make_index_sequence<size>{});
//    auto pi = std::vector<std::size_t>(pa);
//    auto fac = compute_factorial(pa);
//    std::iota(pi.begin(), pi.end(), 1);

    constexpr auto factorials = std::make_index_sequence<factorial>{};

//    for_each_in_tuple(factorials,[&](auto const& /*unused*/, auto const& /*unused*/){
//      using tensorB_type = ublas::dynamic_tensor<value_type, layout_type>;
//      const auto nb = permute_extents_dynamic_rank(pi, na);
//      const auto b = tensorB_type(nb, value_type{3});

//      constexpr auto modes  = std::make_index_sequence<size>{};

//      for_each_in_tuple(modes,[&](auto const& /*unused*/, auto const& /*unused*/){


//      const auto phia = to_array(std::make_index_sequence<Q>);
//      const auto phib = std::array<std::size_t>(q);

//    });

//    for (auto f = 0ul; f < fac; ++f) {
//      for (auto q = 0ul; q <= pa; ++q) {

//        auto phia = std::vector<std::size_t>(q);
//        auto phib = std::vector<std::size_t>(q);

//        std::iota(phia.begin(), phia.end(), 1ul);
//        std::transform(phia.begin(), phia.end(), phib.begin(),
//                       [&pi](std::size_t i) { return pi.at(i - 1); });

//        auto c = ublas::prod(a, b, phia, phib);

//        auto acc = value_type(1);
//        for (auto i = 0ul; i < q; ++i)
//          acc *= value_type( static_cast< inner_type_t<value_type> >( a.extents().at(phia.at(i) - 1) ) );

//        for (auto i = 0ul; i < c.size(); ++i)
//          BOOST_CHECK_EQUAL(c[i], acc *a[0] * b[0]);
//      }

//      std::next_permutation(pi.begin(), pi.end());
//    }
  });

  for_each_in_tuple(fixed_rank_extents,[&](auto const& /*unused*/, auto & n){
    auto const& na = n;
    constexpr auto size = std::tuple_size_v<std::decay_t<decltype(n)>>;
    using tensor_type_1 = ublas::fixed_rank_tensor<value_type, size, layout_type>;
    auto a = tensor_type_1(na, value_type{2});
    auto const pa = a.rank();

//    auto pi = std::vector<std::size_t>(pa);
//    auto fac = compute_factorial(pa);
//    std::iota(pi.begin(), pi.end(), 1);

//    for (auto f = 0ul; f < fac; ++f) {
//      auto nb = permute_extents_static_rank(pi, na);

//      using tensor_type_2 = ublas::fixed_rank_tensor<value_type, size, layout_type>;
//      auto b = tensor_type_2(nb, value_type{3});

//      for (auto q = 0ul; q <= pa; ++q) {

//        auto phia = std::vector<std::size_t>(q);
//        auto phib = std::vector<std::size_t>(q);

//        std::iota(phia.begin(), phia.end(), 1ul);
//        std::transform(phia.begin(), phia.end(), phib.begin(),
//                       [&pi](std::size_t i) { return pi.at(i - 1); });

//        auto c = ublas::prod(a, b, phia, phib);

//        auto acc = value_type(1);
//        for (auto i = 0ul; i < q; ++i){
//          acc *= value_type( static_cast< inner_type_t<value_type> >( a.extents().at(phia.at(i) - 1) ) );
//        }

//        for (auto i = 0ul; i < c.size(); ++i)
//          BOOST_CHECK_EQUAL(c[i], acc *a[0] * b[0]);
//      }

//      std::next_permutation(pi.begin(), pi.end());
//    }
  });

}


BOOST_FIXTURE_TEST_CASE_TEMPLATE( test_fixed_rank_tensor_inner_prod, value,  test_types, fixture )
{
  namespace ublas = boost::numeric::ublas;
  using value_type   = typename value::first_type;
  using layout_type  = typename value::second_type;

  auto const body = [&](auto const& a, auto const& b){
    auto c = ublas::inner_prod(a, b);
    auto r = std::inner_product(a.begin(),a.end(), b.begin(),value_type(0));

    BOOST_CHECK_EQUAL( c , r );
  };

  for_each_in_tuple(fixed_rank_extents,[&](auto const& /*unused*/, auto & n){
    constexpr auto size = std::tuple_size_v<std::decay_t<decltype(n)>>;
    using tensor_type = ublas::fixed_rank_tensor<value_type, size, layout_type>;
    auto a  = tensor_type(n, value_type(2));
    auto b  = tensor_type(n, value_type(1));
    body(a,b);

  });

  for_each_in_tuple(fixed_rank_extents,[&](auto const& I, auto & n){
    constexpr auto size = std::tuple_size_v<std::decay_t<decltype(n)>>;
    using tensor_type_1 = ublas::fixed_rank_tensor<value_type, size, layout_type>;
    using tensor_type_2 = ublas::dynamic_tensor<value_type, layout_type>;
    auto a  = tensor_type_1(n, value_type(2));
    auto b  = tensor_type_2(extents[I], value_type(1));
    body(a,b);

  });

  for_each_in_tuple(fixed_rank_extents,[&](auto const& I, auto & n){
    constexpr auto size = std::tuple_size_v<std::decay_t<decltype(n)>>;
    using tensor_type_1 = ublas::dynamic_tensor<value_type, layout_type>;
    using tensor_type_2 = ublas::fixed_rank_tensor<value_type, size, layout_type>;
    auto a  = tensor_type_1(extents[I], value_type(2));
    auto b  = tensor_type_2(n, value_type(1));
    body(a,b);

  });

}

BOOST_FIXTURE_TEST_CASE_TEMPLATE( test_fixed_rank_tensor_outer_prod, value,  test_types, fixture )
{
  namespace ublas = boost::numeric::ublas;
  using value_type   = typename value::first_type;
  using layout_type  = typename value::second_type;

  for_each_in_tuple(fixed_rank_extents,[&](auto const& /*unused*/, auto const& n1){
    constexpr auto size1 = std::tuple_size_v<std::decay_t<decltype(n1)>>;
    using tensor_type_1 = ublas::fixed_rank_tensor<value_type, size1, layout_type>;
    auto a  = tensor_type_1(n1, value_type(2));
    for_each_in_tuple(fixed_rank_extents,[&](auto const& /*J*/, auto const& n2){
      constexpr auto size2 = std::tuple_size_v<std::decay_t<decltype(n2)>>;
      using tensor_type_2 = ublas::fixed_rank_tensor<value_type, size2, layout_type>;
      auto b  = tensor_type_2(n2, value_type(1));
      auto c  = ublas::outer_prod(a, b);

      for(auto const& cc : c)
        BOOST_CHECK_EQUAL( cc , a[0]*b[0] );

    });

  });

  for_each_in_tuple(fixed_rank_extents,[&](auto const& I, auto const& /*n1*/){
    using tensor_type_1 = ublas::dynamic_tensor<value_type, layout_type>;
    auto a  = tensor_type_1(extents[I], value_type(2));
    for_each_in_tuple(fixed_rank_extents,[&](auto const& /*J*/, auto const& n2){
      constexpr auto size = std::tuple_size_v<std::decay_t<decltype(n2)>>;
      using tensor_type_2 = ublas::fixed_rank_tensor<value_type, size, layout_type>;
      auto b  = tensor_type_2(n2, value_type(1));
      auto c  = ublas::outer_prod(a, b);

      for(auto const& cc : c)
        BOOST_CHECK_EQUAL( cc , a[0]*b[0] );
    });

  });

  for_each_in_tuple(fixed_rank_extents,[&](auto const& /*unused*/, auto const& n1){
    constexpr auto size = std::tuple_size_v<std::decay_t<decltype(n1)>>;
    using tensor_type_1 = ublas::fixed_rank_tensor<value_type, size, layout_type>;
    auto a  = tensor_type_1(n1, value_type(2));
    for(auto const& n2 : extents){
      using tensor_type_2 = ublas::dynamic_tensor<value_type, layout_type>;
      auto b  = tensor_type_2(n2, value_type(1));
      auto c  = ublas::outer_prod(a, b);

      for(auto const& cc : c)
        BOOST_CHECK_EQUAL( cc , a[0]*b[0] );
    }

  });

}

BOOST_AUTO_TEST_SUITE_END()
