//
//  Copyright (c) 2018-2020, Cem Bassoy, cem.bassoy@gmail.com
//  Copyright (c) 2019-2020, Amit Singh, amitsingh19975@gmail.com
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



BOOST_FIXTURE_TEST_CASE_TEMPLATE( test_tensor_expression_retrieve_extents, value,  test_types, fixture)
{
    namespace ublas = boost::numeric::ublas;
    using value_t  = typename value::first_type;
    using layout_t = typename value::second_type;
    using tensor_t = ublas::tensor_dynamic<value_t, layout_t>;

    auto uplus1 = [](auto const& a){ return a + value_t(1);  };
    auto uplus2 = [](auto const& a){ return value_t(2) + a;  };
    auto bplus  = std::plus <value_t>{};
    auto bminus = std::minus<value_t>{};

    for(auto const& e : extents) {

        auto t = tensor_t(e);
        auto v = value_t{};
        for(auto& tt: t){ tt = v; v+=value_t{1}; }


        BOOST_CHECK( ublas::detail::retrieve_extents( t ) == e );


        // uexpr1 = t+1
        // uexpr2 = 2+t
        auto uexpr1 = ublas::detail::make_unary_tensor_expression<tensor_t>( t, uplus1 );
        auto uexpr2 = ublas::detail::make_unary_tensor_expression<tensor_t>( t, uplus2 );

        BOOST_CHECK( ublas::detail::retrieve_extents( uexpr1 ) == e );
        BOOST_CHECK( ublas::detail::retrieve_extents( uexpr2 ) == e );

        // bexpr_uexpr = (t+1) + (2+t)
        auto bexpr_uexpr = ublas::detail::make_binary_tensor_expression<tensor_t>( uexpr1, uexpr2, bplus );

        BOOST_CHECK( ublas::detail::retrieve_extents( bexpr_uexpr ) == e );


        // bexpr_bexpr_uexpr = ((t+1) + (2+t)) - t
        auto bexpr_bexpr_uexpr = ublas::detail::make_binary_tensor_expression<tensor_t>( bexpr_uexpr, t, bminus );

        BOOST_CHECK( ublas::detail::retrieve_extents( bexpr_bexpr_uexpr ) == e );

    }


    for(auto i = 0u; i < extents.size()-1; ++i)
    {

        auto v = value_t{};

        auto t1 = tensor_t(extents[i]);
        for(auto& tt: t1){ tt = v; v+=value_t{1}; }

        auto t2 = tensor_t(extents[i+1]);
        for(auto& tt: t2){ tt = v; v+=value_t{2}; }

        BOOST_CHECK( ublas::detail::retrieve_extents( t1 ) != ublas::detail::retrieve_extents( t2 ) );

        // uexpr1 = t1+1
        // uexpr2 = 2+t2
        auto uexpr1 = ublas::detail::make_unary_tensor_expression<tensor_t>( t1, uplus1 );
        auto uexpr2 = ublas::detail::make_unary_tensor_expression<tensor_t>( t2, uplus2 );

        BOOST_CHECK( ublas::detail::retrieve_extents( t1 )     == ublas::detail::retrieve_extents( uexpr1 ) );
        BOOST_CHECK( ublas::detail::retrieve_extents( t2 )     == ublas::detail::retrieve_extents( uexpr2 ) );
        BOOST_CHECK( ublas::detail::retrieve_extents( uexpr1 ) != ublas::detail::retrieve_extents( uexpr2 ) );

        // bexpr_uexpr = (t1+1) + (2+t2)
        auto bexpr_uexpr = ublas::detail::make_binary_tensor_expression<tensor_t>( uexpr1, uexpr2, bplus );

        BOOST_CHECK( ublas::detail::retrieve_extents( bexpr_uexpr ) == ublas::detail::retrieve_extents(t1) );


        // bexpr_bexpr_uexpr = ((t1+1) + (2+t2)) - t2
        auto bexpr_bexpr_uexpr1 = ublas::detail::make_binary_tensor_expression<tensor_t>( bexpr_uexpr, t2, bminus );

        BOOST_CHECK( ublas::detail::retrieve_extents( bexpr_bexpr_uexpr1 ) == ublas::detail::retrieve_extents(t2) );


        // bexpr_bexpr_uexpr = t2 - ((t1+1) + (2+t2))
        auto bexpr_bexpr_uexpr2 = ublas::detail::make_binary_tensor_expression<tensor_t>( t2, bexpr_uexpr, bminus );

        BOOST_CHECK( ublas::detail::retrieve_extents( bexpr_bexpr_uexpr2 ) == ublas::detail::retrieve_extents(t2) );
    }
}







BOOST_FIXTURE_TEST_CASE_TEMPLATE( test_tensor_expression_all_extents_equal, value,  test_types, fixture)
{
    namespace ublas = boost::numeric::ublas;
    using value_t  = typename value::first_type;
    using layout_t = typename value::second_type;
    using tensor_t = ublas::tensor_dynamic<value_t, layout_t>;

    auto uplus1 = [](auto const& a){ return a + value_t(1);  };
    auto uplus2 = [](auto const& a){ return value_t(2) + a;  };
    auto bplus  = std::plus <value_t>{};
    auto bminus = std::minus<value_t>{};

    for(auto const& e : extents) {

        auto t = tensor_t(e);
        auto v = value_t{};
        for(auto& tt: t){ tt = v; v+=value_t{1}; }


        BOOST_CHECK( ublas::detail::all_extents_equal( t , e ) );


        // uexpr1 = t+1
        // uexpr2 = 2+t
        auto uexpr1 = ublas::detail::make_unary_tensor_expression<tensor_t>( t, uplus1 );
        auto uexpr2 = ublas::detail::make_unary_tensor_expression<tensor_t>( t, uplus2 );

        BOOST_CHECK( ublas::detail::all_extents_equal( uexpr1, e ) );
        BOOST_CHECK( ublas::detail::all_extents_equal( uexpr2, e ) );

        // bexpr_uexpr = (t+1) + (2+t)
        auto bexpr_uexpr = ublas::detail::make_binary_tensor_expression<tensor_t>( uexpr1, uexpr2, bplus );

        BOOST_CHECK( ublas::detail::all_extents_equal( bexpr_uexpr, e ) );


        // bexpr_bexpr_uexpr = ((t+1) + (2+t)) - t
        auto bexpr_bexpr_uexpr = ublas::detail::make_binary_tensor_expression<tensor_t>( bexpr_uexpr, t, bminus );

        BOOST_CHECK( ublas::detail::all_extents_equal( bexpr_bexpr_uexpr , e ) );

    }


    for(auto i = 0u; i < extents.size()-1; ++i)
    {

        auto v = value_t{};

        auto t1 = tensor_t(extents[i]);
        for(auto& tt: t1){ tt = v; v+=value_t{1}; }

        auto t2 = tensor_t(extents[i+1]);
        for(auto& tt: t2){ tt = v; v+=value_t{2}; }

        BOOST_CHECK( ublas::detail::all_extents_equal( t1, ublas::detail::retrieve_extents(t1) ) );
        BOOST_CHECK( ublas::detail::all_extents_equal( t2, ublas::detail::retrieve_extents(t2) ) );

        // uexpr1 = t1+1
        // uexpr2 = 2+t2
        auto uexpr1 = ublas::detail::make_unary_tensor_expression<tensor_t>( t1, uplus1 );
        auto uexpr2 = ublas::detail::make_unary_tensor_expression<tensor_t>( t2, uplus2 );

        BOOST_CHECK( ublas::detail::all_extents_equal( uexpr1, ublas::detail::retrieve_extents(uexpr1) ) );
        BOOST_CHECK( ublas::detail::all_extents_equal( uexpr2, ublas::detail::retrieve_extents(uexpr2) ) );

        // bexpr_uexpr = (t1+1) + (2+t2)
        auto bexpr_uexpr = ublas::detail::make_binary_tensor_expression<tensor_t>( uexpr1, uexpr2, bplus );

        BOOST_CHECK( ! ublas::detail::all_extents_equal( bexpr_uexpr, ublas::detail::retrieve_extents( bexpr_uexpr  ) ) );

        // bexpr_bexpr_uexpr = ((t1+1) + (2+t2)) - t2
        auto bexpr_bexpr_uexpr1 = ublas::detail::make_binary_tensor_expression<tensor_t>( bexpr_uexpr, t2, bminus );

        BOOST_CHECK( ! ublas::detail::all_extents_equal( bexpr_bexpr_uexpr1, ublas::detail::retrieve_extents( bexpr_bexpr_uexpr1  ) ) );

        // bexpr_bexpr_uexpr = t2 - ((t1+1) + (2+t2))
        auto bexpr_bexpr_uexpr2 = ublas::detail::make_binary_tensor_expression<tensor_t>( t2, bexpr_uexpr, bminus );

        BOOST_CHECK( ! ublas::detail::all_extents_equal( bexpr_bexpr_uexpr2, ublas::detail::retrieve_extents( bexpr_bexpr_uexpr2  ) ) );


        // bexpr_uexpr2 = (t1+1) + t2
        auto bexpr_uexpr2 = ublas::detail::make_binary_tensor_expression<tensor_t>( uexpr1, t2, bplus );
        BOOST_CHECK( ! ublas::detail::all_extents_equal( bexpr_uexpr2, ublas::detail::retrieve_extents( bexpr_uexpr2  ) ) );


        // bexpr_uexpr2 = ((t1+1) + t2) + t1
        auto bexpr_bexpr_uexpr3 = ublas::detail::make_binary_tensor_expression<tensor_t>( bexpr_uexpr2, t1, bplus );
        BOOST_CHECK( ! ublas::detail::all_extents_equal( bexpr_bexpr_uexpr3, ublas::detail::retrieve_extents( bexpr_bexpr_uexpr3  ) ) );

        // bexpr_uexpr2 = t1 + (((t1+1) + t2) + t1)
        auto bexpr_bexpr_uexpr4 = ublas::detail::make_binary_tensor_expression<tensor_t>( t1, bexpr_bexpr_uexpr3, bplus );
        BOOST_CHECK( ! ublas::detail::all_extents_equal( bexpr_bexpr_uexpr4, ublas::detail::retrieve_extents( bexpr_bexpr_uexpr4  ) ) );

    }
}

BOOST_AUTO_TEST_SUITE_END()
