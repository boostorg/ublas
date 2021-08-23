//
// 	Copyright (c) 2020, Amit Singh, amitsingh19975@gmail.com
// 	Copyright (c) 2021, Cem Bassoy, cem.bassoy@gmail.com
//
//  Distributed under the Boost Software License, Version 1.0. (See
//  accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)
//
//  The authors gratefully acknowledge the support of
//  Google and Fraunhofer IOSB, Ettlingen, Germany
//




#include <boost/numeric/ublas/tensor/expression_evaluation.hpp>
#include <boost/numeric/ublas/tensor/expression.hpp>
#include <boost/numeric/ublas/tensor/extents.hpp>
#include <boost/numeric/ublas/tensor/tensor.hpp>
#include "utility.hpp"
#include <boost/test/unit_test.hpp>

#include <functional>

BOOST_AUTO_TEST_SUITE(test_tensor_expression)
using test_types = zip<int,float,std::complex<float>>::with_t<boost::numeric::ublas::layout::first_order, boost::numeric::ublas::layout::last_order>;

struct fixture
{
  using extents_t = boost::numeric::ublas::extents<>;

  const std::vector<extents_t> extents =
    {
//      extents_t{},            // 0

      extents_t{1,1},         // 1
      extents_t{1,2},         // 2
      extents_t{2,1},         // 3

      extents_t{2,3},         // 4
      extents_t{2,3,1},       // 5
      extents_t{1,2,3},       // 6
      extents_t{1,1,2,3},     // 7
      extents_t{1,2,3,1,1},   // 8

      extents_t{4,2,3},       // 9
      extents_t{4,2,1,3},     // 10
      extents_t{4,2,1,3,1},   // 11
      extents_t{1,4,2,1,3,1}  // 12
  };
};


BOOST_FIXTURE_TEST_CASE_TEMPLATE( test_subtensor_static_rank_expression_retrieve_extents, value,  test_types, fixture)
{
  namespace ublas  = boost::numeric::ublas;
  using value_t  = typename value::first_type;
  using layout_t = typename value::second_type;
  using tensor_t = ublas::tensor_dynamic<value_t, layout_t>;
  using subtensor = typename tensor_t::subtensor_type;

  auto uplus1 = [](auto const& a){return a + value_t(1); };
  auto uplus2 = [](auto const& a){return value_t(2) + a; };
  auto bplus  = std::plus <value_t>{};
  auto bminus = std::minus<value_t>{};

  for(auto const& e : extents) {

    auto t = tensor_t(e);
    auto v = value_t{};
    for(auto& tt: t){ tt = v; v+=value_t{1}; }
    auto s = subtensor(t);

    BOOST_CHECK( ublas::detail::retrieve_extents( s ) == e );

    // uexpr1 = s+1
    // uexpr2 = 2+s
    auto uexpr1 = ublas::detail::make_unary_tensor_expression<tensor_t>( s, uplus1 );
    auto uexpr2 = ublas::detail::make_unary_tensor_expression<tensor_t>( s, uplus2 );

    BOOST_CHECK( ublas::detail::retrieve_extents( uexpr1 ) == e );
    BOOST_CHECK( ublas::detail::retrieve_extents( uexpr2 ) == e );

    // bexpr_uexpr = (s+1) + (2+s)
    auto bexpr_uexpr = ublas::detail::make_binary_tensor_expression<tensor_t>( uexpr1, uexpr2, bplus );

    BOOST_CHECK( ublas::detail::retrieve_extents( bexpr_uexpr ) == e );


    // bexpr_bexpr_uexpr = ((s+1) + (2+s)) - s
    auto bexpr_bexpr_uexpr = ublas::detail::make_binary_tensor_expression<tensor_t>( bexpr_uexpr, s, bminus );

    BOOST_CHECK( ublas::detail::retrieve_extents( bexpr_bexpr_uexpr ) == e );

  }

  for(auto i = 0u; i < extents.size()-1; ++i)
  {

    auto v = value_t{};

    tensor_t t1(extents[i]);
    for(auto& tt: t1){ tt = v; v+=value_t{1}; }

    tensor_t t2(extents[i+1]);
    for(auto& tt: t2){ tt = v; v+=value_t{2}; }

    auto s1 = subtensor(t1);
    auto s2 = subtensor(t2);

    BOOST_CHECK( ublas::detail::retrieve_extents( s1 ) == ublas::detail::retrieve_extents( t1 ) );
    BOOST_CHECK( ublas::detail::retrieve_extents( s2 ) == ublas::detail::retrieve_extents( t2 ) );
    BOOST_CHECK( ublas::detail::retrieve_extents( s1 ) != ublas::detail::retrieve_extents( s2 ) );

    // uexpr1 = s1+1
    // uexpr2 = 2+s2
    auto uexpr1 = ublas::detail::make_unary_tensor_expression<tensor_t>( s1, uplus1 );
    auto uexpr2 = ublas::detail::make_unary_tensor_expression<tensor_t>( s2, uplus2 );

    BOOST_CHECK( ublas::detail::retrieve_extents( s1 )     == ublas::detail::retrieve_extents( uexpr1 ) );
    BOOST_CHECK( ublas::detail::retrieve_extents( s2 )     == ublas::detail::retrieve_extents( uexpr2 ) );
    BOOST_CHECK( ublas::detail::retrieve_extents( uexpr1 ) != ublas::detail::retrieve_extents( uexpr2 ) );


    // bexpr_uexpr = (s1+1) + (2+s2)
    auto bexpr_uexpr = ublas::detail::make_binary_tensor_expression<tensor_t>( uexpr1, uexpr2, bplus );

    BOOST_CHECK( ublas::detail::retrieve_extents( bexpr_uexpr ) == ublas::detail::retrieve_extents(s1) );


    // bexpr_bexpr_uexpr = ((s1+1) + (2+s2)) - s2
    auto bexpr_bexpr_uexpr1 = ublas::detail::make_binary_tensor_expression<tensor_t>( bexpr_uexpr, s2, bminus );

    BOOST_CHECK( ublas::detail::retrieve_extents( bexpr_bexpr_uexpr1 ) == ublas::detail::retrieve_extents(s2) );

    // bexpr_bexpr_uexpr = s2 - ((s1+1) + (2+s2))
    auto bexpr_bexpr_uexpr2 = ublas::detail::make_binary_tensor_expression<tensor_t>( s2, bexpr_uexpr, bminus );

    BOOST_CHECK( ublas::detail::retrieve_extents( bexpr_bexpr_uexpr2 ) == ublas::detail::retrieve_extents(s2) );

  }
}


BOOST_FIXTURE_TEST_CASE_TEMPLATE( test_subtensor_static_rank_expression_all_extents_equal, value,  test_types, fixture)
{
  namespace ublas  = boost::numeric::ublas;
  using value_t  = typename value::first_type;
  using layout_t = typename value::second_type;
  using tensor_t = ublas::tensor_dynamic<value_t, layout_t>;
  using subtensor = typename tensor_t::subtensor_type;

  auto uplus1 = [](auto const& a){return a + value_t(1); };
  auto uplus2 = [](auto const& a){return value_t(2) + a; };
  auto bplus  = std::plus <value_t>{};
  auto bminus = std::minus<value_t>{};

  for(auto const& e : extents) {

    auto t = tensor_t(e);
    auto v = value_t{};
    for(auto& tt: t){ tt = v; v+=value_t{1}; }

    auto s = subtensor(t);

    BOOST_CHECK( ublas::detail::all_extents_equal( s , e ) );


    // uexpr1 = s+1
    // uexpr2 = 2+s
    auto uexpr1 = ublas::detail::make_unary_tensor_expression<tensor_t>( s, uplus1 );
    auto uexpr2 = ublas::detail::make_unary_tensor_expression<tensor_t>( s, uplus2 );

    BOOST_CHECK( ublas::detail::all_extents_equal( uexpr1, e ) );
    BOOST_CHECK( ublas::detail::all_extents_equal( uexpr2, e ) );

    // bexpr_uexpr = (s+1) + (2+s)
    auto bexpr_uexpr = ublas::detail::make_binary_tensor_expression<tensor_t>( uexpr1, uexpr2, bplus );

    BOOST_CHECK( ublas::detail::all_extents_equal( bexpr_uexpr, e ) );


    // bexpr_bexpr_uexpr = ((s+1) + (2+s)) - s
    auto bexpr_bexpr_uexpr = ublas::detail::make_binary_tensor_expression<tensor_t>( bexpr_uexpr, s, bminus );

    BOOST_CHECK( ublas::detail::all_extents_equal( bexpr_bexpr_uexpr , e ) );

  };


  for(auto i = 0u; i < extents.size()-1; ++i)
  {

    auto v = value_t{};

    tensor_t t1(extents[i]);
    for(auto& tt: t1){ tt = v; v+=value_t{1}; }

    tensor_t t2(extents[i+1]);
    for(auto& tt: t2){ tt = v; v+=value_t{2}; }

    auto s1 = subtensor(t1);
    auto s2 = subtensor(t2);

    BOOST_CHECK( ublas::detail::all_extents_equal( t1, ublas::detail::retrieve_extents(t1) ) );
    BOOST_CHECK( ublas::detail::all_extents_equal( t2, ublas::detail::retrieve_extents(t2) ) );

    // uexpr1 = s1+1
    // uexpr2 = 2+s2
    auto uexpr1 = ublas::detail::make_unary_tensor_expression<tensor_t>( s1, uplus1 );
    auto uexpr2 = ublas::detail::make_unary_tensor_expression<tensor_t>( s2, uplus2 );

    BOOST_CHECK( ublas::detail::all_extents_equal( uexpr1, ublas::detail::retrieve_extents(uexpr1) ) );
    BOOST_CHECK( ublas::detail::all_extents_equal( uexpr2, ublas::detail::retrieve_extents(uexpr2) ) );

    // bexpr_uexpr = (s1+1) + (2+s2)
    auto bexpr_uexpr = ublas::detail::make_binary_tensor_expression<tensor_t>( uexpr1, uexpr2, bplus );

    BOOST_CHECK( ! ublas::detail::all_extents_equal( bexpr_uexpr, ublas::detail::retrieve_extents( bexpr_uexpr  ) ) );

    // bexpr_bexpr_uexpr = ((s1+1) + (2+s2)) - s2
    auto bexpr_bexpr_uexpr1 = ublas::detail::make_binary_tensor_expression<tensor_t>( bexpr_uexpr, s2, bminus );

    BOOST_CHECK( ! ublas::detail::all_extents_equal( bexpr_bexpr_uexpr1, ublas::detail::retrieve_extents( bexpr_bexpr_uexpr1  ) ) );

    // bexpr_bexpr_uexpr = s2 - ((s1+1) + (2+s2))
    auto bexpr_bexpr_uexpr2 = ublas::detail::make_binary_tensor_expression<tensor_t>( s2, bexpr_uexpr, bminus );

    BOOST_CHECK( ! ublas::detail::all_extents_equal( bexpr_bexpr_uexpr2, ublas::detail::retrieve_extents( bexpr_bexpr_uexpr2  ) ) );


    // bexpr_uexpr2 = (s1+1) + s2
    auto bexpr_uexpr2 = ublas::detail::make_binary_tensor_expression<tensor_t>( uexpr1, s2, bplus );
    BOOST_CHECK( ! ublas::detail::all_extents_equal( bexpr_uexpr2, ublas::detail::retrieve_extents( bexpr_uexpr2  ) ) );


    // bexpr_uexpr2 = ((s1+1) + s2) + s1
    auto bexpr_bexpr_uexpr3 = ublas::detail::make_binary_tensor_expression<tensor_t>( bexpr_uexpr2, s1, bplus );
    BOOST_CHECK( ! ublas::detail::all_extents_equal( bexpr_bexpr_uexpr3, ublas::detail::retrieve_extents( bexpr_bexpr_uexpr3  ) ) );

    // bexpr_uexpr2 = s1 + (((s1+1) + s2) + s1)
    auto bexpr_bexpr_uexpr4 = ublas::detail::make_binary_tensor_expression<tensor_t>( t1, bexpr_bexpr_uexpr3, bplus );
    BOOST_CHECK( ! ublas::detail::all_extents_equal( bexpr_bexpr_uexpr4, ublas::detail::retrieve_extents( bexpr_bexpr_uexpr4  ) ) );

  }

}

BOOST_AUTO_TEST_SUITE_END()
