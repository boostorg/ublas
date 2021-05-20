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
BOOST_AUTO_TEST_SUITE ( test_tensor_extents_static_size_functions)


using test_types = zip<int,float,std::complex<float>>::with_t<boost::numeric::ublas::layout::first_order, boost::numeric::ublas::layout::last_order>;

//using test_types = zip<int>::with_t<boost::numeric::ublas::layout::first_order>;


struct fixture
{
  std::tuple<
    boost::numeric::ublas::extents<2>,
    boost::numeric::ublas::extents<2>,
    boost::numeric::ublas::extents<3>,
    boost::numeric::ublas::extents<3>,
    boost::numeric::ublas::extents<4>
    > extents_tuple{
      {1,1}, // 1
      {2,3}, // 2
      {2,3,1}, // 3
      {4,2,3}, // 4
      {4,2,3,5} // 5
    };


  std::vector<boost::numeric::ublas::extents<>> extents_vector =
    {
      {1,1}, // 1
      {2,3}, // 2
      {2,3,1}, // 3
      {4,2,3}, // 4
      {4,2,3,5} // 5
  };

};



BOOST_FIXTURE_TEST_CASE_TEMPLATE( test_tensor_extents_static_size_prod_vector, value,  test_types, fixture )
{
  namespace ublas    = boost::numeric::ublas;
  using value_t   = typename value::first_type;
  using layout_t  = typename value::second_type;

  for_each_in_tuple(extents_tuple,[](auto const& /*unused*/, auto const& n){

    constexpr auto size = std::tuple_size_v<std::decay_t<decltype(n)>>;
    using tensor_t = ublas::tensor_static_rank<value_t, size, layout_t>;
    using vector_t = typename tensor_t::vector_type;
    auto a = tensor_t(n);
    a = 2;

    for (auto m = 0u; m < ublas::size(n); ++m) {
      auto b = vector_t(n[m], value_t{1});

      auto c = ublas::prod(a, b, m + 1);

      for (auto i = 0u; i < c.size(); ++i)
        BOOST_CHECK_EQUAL(c[i], value_t( static_cast< inner_type_t<value_t> >(n[m]) ) * a[i]);
    }
  });

}

BOOST_FIXTURE_TEST_CASE_TEMPLATE( test_tensor_extents_static_size_prod_matrix, value,  test_types, fixture )
{
  namespace ublas = boost::numeric::ublas;
  using value_t   = typename value::first_type;
  using layout_t  = typename value::second_type;

  for_each_in_tuple(extents_tuple,[](auto const& /*unused*/, auto const & n){
    constexpr auto size = std::tuple_size_v<std::decay_t<decltype(n)>>;
    using tensor_t = ublas::tensor_static_rank<value_t, size, layout_t>;
    using matrix_t = typename tensor_t::matrix_type;

    auto a = tensor_t(n);
    a = 2;
    for (auto m = 0u; m < ublas::size(n); ++m) {

      auto b = matrix_t  ( n[m], n[m], value_t{1} );

      auto c = ublas::prod(a, b, m + 1);

      for (auto i = 0u; i < c.size(); ++i){
        BOOST_CHECK_EQUAL(c[i], value_t( static_cast< inner_type_t<value_t> >(n[m]) ) * a[i]);
      }
    }
  });

}


BOOST_FIXTURE_TEST_CASE_TEMPLATE( test_tensor_extents_static_size_prod_tensor_1, value,  test_types, fixture )
{
  namespace ublas = boost::numeric::ublas;
  using value_t   = typename value::first_type;
  using layout_t  = typename value::second_type;


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

    auto acc = value_t(1);
    for (auto i = 0ul; i < q; ++i){
      auto const& na = a.extents();
      acc *= static_cast<value_t>(na.at(phi.at(i)-1));
    }

    for (auto i = 0ul; i < c.size(); ++i){
      BOOST_CHECK_EQUAL(c[i], acc *a[0] * b[0]);
    }
  };


  for_each_in_tuple(extents_tuple,[&](auto const& /*I*/, auto const& n){
    constexpr auto size = std::tuple_size_v<std::decay_t<decltype(n)>>;
    constexpr auto modes = std::make_index_sequence<size>{};
    using tensor_t = ublas::tensor_static_rank<value_t, size, layout_t>;
    auto a = tensor_t(n);
    auto b = tensor_t(n);
    a = 2;
    b = 3;
    for_each_in_index(modes, a,b, check );
  });

  for_each_in_tuple(extents_tuple,[&](auto const& I, auto const& n){
    auto const& nA = n;
    auto const& nB = extents_vector[I];
    constexpr auto sizeA = std::tuple_size_v<std::decay_t<decltype(n)>>;
    constexpr auto modes = std::make_index_sequence<sizeA>{};
    using tensorA_type = ublas::tensor_static_rank<value_t, sizeA , layout_t>;
    using tensorB_type = ublas::tensor_dynamic<value_t, layout_t>;
    auto a = tensorA_type(nA);
    auto b = tensorB_type(nB);
    a = 2;
    b = 3;

    for_each_in_index(modes, a,b, check );
  });

  for_each_in_tuple(extents_tuple,[&](auto const& I, auto const& n){
    auto const& nA = extents_vector[I];
    auto const& nB = n;
    constexpr auto sizeB = std::tuple_size_v<std::decay_t<decltype(n)>>;
    constexpr auto modes = std::make_index_sequence<sizeB>{};
    using tensor_t_1 = ublas::tensor_dynamic<value_t, layout_t>;
    using tensor_t_2 = ublas::tensor_static_rank<value_t, sizeB, layout_t>;
    auto a = tensor_t_1(nA);
    auto b = tensor_t_2(nB);
    a = 2;
    b = 3;
    for_each_in_index(modes, a,b, check );

  });
}

// TODO:
#if 0

BOOST_FIXTURE_TEST_CASE_TEMPLATE( test_tensor_extents_static_size_prod_tensor_2, value,  test_types, fixture )
{
  namespace ublas = boost::numeric::ublas;
  using value_t   = typename value::first_type;
  using layout_t  = typename value::second_type;

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

  for_each_in_tuple(extents_tuple,[&](auto const& /*unused*/, auto const& n){
    auto const& na = n;
    constexpr auto size = std::tuple_size_v<std::decay_t<decltype(n)>>;
    using tensorA_type = ublas::tensor_static_rank<value_t, size, layout_t>;
    auto a = tensorA_type(na);
    a = 2;
    assert(a.rank() == size);
    //    auto const pa = a.rank();
    auto pi  = to_array(std::make_index_sequence<size>{});
    constexpr auto factorial = compute_factorial(std::make_index_sequence<size>{});
    //    auto pi = std::vector<std::size_t>(pa);
    //    auto fac = compute_factorial(pa);
    //    std::iota(pi.begin(), pi.end(), 1);

    constexpr auto factorials = std::make_index_sequence<factorial>{};

    //    for_each_in_tuple(factorials,[&](auto const& /*unused*/, auto const& /*unused*/){
    //      using tensorB_type = ublas::tensor_dynamic<value_t, layout_t>;
    //      const auto nb = permute_extents_dynamic_rank(pi, na);
    //      const auto b = tensorB_type(nb, value_t{3});

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

    //        auto acc = value_t(1);
    //        for (auto i = 0ul; i < q; ++i)
    //          acc *= value_t( static_cast< inner_type_t<value_t> >( a.extents().at(phia.at(i) - 1) ) );

    //        for (auto i = 0ul; i < c.size(); ++i)
    //          BOOST_CHECK_EQUAL(c[i], acc *a[0] * b[0]);
    //      }

    //      std::next_permutation(pi.begin(), pi.end());
    //    }
  });

  for_each_in_tuple(extents_tuple,[&](auto const& /*unused*/, auto & /*n*/){
    //    auto const& na = n;
    //    constexpr auto size = std::tuple_size_v<std::decay_t<decltype(n)>>;
    //    using tensor_t_1 = ublas::tensor_static_rank<value_t, size, layout_t>;
    //    auto a = tensor_t_1(na, value_t{2});
    //    auto const pa = a.rank();

    //    auto pi = std::vector<std::size_t>(pa);
    //    auto fac = compute_factorial(pa);
    //    std::iota(pi.begin(), pi.end(), 1);

    //    for (auto f = 0ul; f < fac; ++f) {
    //      auto nb = permute_extents_static_rank(pi, na);

    //      using tensor_t_2 = ublas::tensor_static_rank<value_t, size, layout_t>;
    //      auto b = tensor_t_2(nb, value_t{3});

    //      for (auto q = 0ul; q <= pa; ++q) {

    //        auto phia = std::vector<std::size_t>(q);
    //        auto phib = std::vector<std::size_t>(q);

    //        std::iota(phia.begin(), phia.end(), 1ul);
    //        std::transform(phia.begin(), phia.end(), phib.begin(),
    //                       [&pi](std::size_t i) { return pi.at(i - 1); });

    //        auto c = ublas::prod(a, b, phia, phib);

    //        auto acc = value_t(1);
    //        for (auto i = 0ul; i < q; ++i){
    //          acc *= value_t( static_cast< inner_type_t<value_t> >( a.extents().at(phia.at(i) - 1) ) );
    //        }

    //        for (auto i = 0ul; i < c.size(); ++i)
    //          BOOST_CHECK_EQUAL(c[i], acc *a[0] * b[0]);
    //      }

    //      std::next_permutation(pi.begin(), pi.end());
    //    }
  });

}
#endif

BOOST_FIXTURE_TEST_CASE_TEMPLATE( test_tensor_extents_static_size_inner_prod, value,  test_types, fixture )
{
  namespace ublas = boost::numeric::ublas;
  using value_t   = typename value::first_type;
  using layout_t  = typename value::second_type;


  using dtensor_t = ublas::tensor_dynamic<value_t, layout_t>;

  auto const body = [&](auto const& a, auto const& b){
    auto c = ublas::inner_prod(a, b);
    auto r = std::inner_product(a.begin(),a.end(), b.begin(),value_t(0));
    BOOST_CHECK_EQUAL( c , r );
  };

  for_each_in_tuple(extents_tuple,[&](auto const& /*unused*/, auto & n){
    constexpr auto size = std::tuple_size_v<std::decay_t<decltype(n)>>;
    using stensor_t = ublas::tensor_static_rank<value_t, size, layout_t>;
    auto a  = stensor_t(n);
    auto b  = stensor_t(n);
    a = 2;
    b = 3;
    body(a,b);

  });

  for_each_in_tuple(extents_tuple,[&](auto const& I, auto & n){
    constexpr auto size = std::tuple_size_v<std::decay_t<decltype(n)>>;
    using stensor_t = ublas::tensor_static_rank<value_t, size, layout_t>;
    auto a  = stensor_t(n);
    auto b  = dtensor_t(extents_vector[I]);
    a = 2;
    b = 1;

    body(a,b);

  });

  for_each_in_tuple(extents_tuple,[&](auto const& I, auto & n){
    constexpr auto size = std::tuple_size_v<std::decay_t<decltype(n)>>;
    using stensor_t = ublas::tensor_static_rank<value_t, size, layout_t>;
    auto a  = dtensor_t(extents_vector[I]);
    auto b  = stensor_t(n);
    a = 2;
    b = 1;
    body(a,b);

  });

}


BOOST_FIXTURE_TEST_CASE_TEMPLATE( test_tensor_extents_static_size_outer_prod, value,  test_types, fixture )
{
  namespace ublas = boost::numeric::ublas;
  using value_t   = typename value::first_type;
  using layout_t  = typename value::second_type;

  for_each_in_tuple(extents_tuple,[&](auto const& /*unused*/, auto const& n1){
    constexpr auto size1 = std::tuple_size_v<std::decay_t<decltype(n1)>>;
    using tensor_t_1 = ublas::tensor_static_rank<value_t, size1, layout_t>;
    auto a  = tensor_t_1(n1);
    a = 2;
    for_each_in_tuple(extents_tuple,[&](auto const& /*J*/, auto const& n2){
      constexpr auto size2 = std::tuple_size_v<std::decay_t<decltype(n2)>>;
      using tensor_t_2 = ublas::tensor_static_rank<value_t, size2, layout_t>;
      auto b  = tensor_t_2(n2);
      b = 1;
      auto c  = ublas::outer_prod(a, b);

      BOOST_CHECK ( std::all_of(c.begin(),c.end(), [&a,&b](auto cc){return cc == a[0]*b[0];}) );

    });

  });

  for_each_in_tuple(extents_tuple,[&](auto const& I, auto const& /*n1*/){
    using tensor_t_1 = ublas::tensor_dynamic<value_t, layout_t>;
    auto a  = tensor_t_1(extents_vector[I]);
    a = 2;
    for_each_in_tuple(extents_tuple,[&](auto const& /*J*/, auto const& n2){
      constexpr auto size = std::tuple_size_v<std::decay_t<decltype(n2)>>;
      using tensor_t_2 = ublas::tensor_static_rank<value_t, size, layout_t>;
      auto b  = tensor_t_2(n2);
      b = 1;
      auto c  = ublas::outer_prod(a, b);
      BOOST_CHECK ( std::all_of(c.begin(),c.end(), [&a,&b](auto cc){return cc == a[0]*b[0];}) );


//      for(auto const& cc : c)
//        BOOST_CHECK_EQUAL( cc , a[0]*b[0] );
    });

  });

  for_each_in_tuple(extents_tuple,[&](auto const& /*unused*/, auto const& n1){
    constexpr auto size = std::tuple_size_v<std::decay_t<decltype(n1)>>;
    using tensor_t_1 = ublas::tensor_static_rank<value_t, size, layout_t>;
    auto a  = tensor_t_1(n1);
    a = 2;
    for(auto const& n2 : extents_vector){
      using tensor_t_2 = ublas::tensor_dynamic<value_t, layout_t>;
      auto b  = tensor_t_2(n2);
      b = 1;
      auto c  = ublas::outer_prod(a, b);

      BOOST_CHECK ( std::all_of(c.begin(),c.end(), [&a,&b](auto cc){return cc == a[0]*b[0];}) );

//      for(auto const& cc : c)
//        BOOST_CHECK_EQUAL( cc , a[0]*b[0] );
    }

  });

}

BOOST_AUTO_TEST_SUITE_END()
