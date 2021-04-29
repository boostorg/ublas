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
#include <boost/numeric/ublas/tensor/tensor.hpp>
#include "utility.hpp"
#include <boost/test/unit_test.hpp>

#include <functional>

BOOST_AUTO_TEST_SUITE(test_tensor_expression)
using test_types = zip<int,float,std::complex<float>>::with_t<boost::numeric::ublas::layout::first_order, boost::numeric::ublas::layout::last_order>;


struct fixture
{
    using extents_type = boost::numeric::ublas::extents<>;
    fixture()
      : extents{
          extents_type{},            // 0

          extents_type{1,1},         // 1
          extents_type{1,2},         // 2
          extents_type{2,1},         // 3

          extents_type{2,3},         // 4
          extents_type{2,3,1},       // 5
          extents_type{1,2,3},       // 6
          extents_type{1,1,2,3},     // 7
          extents_type{1,2,3,1,1},   // 8

          extents_type{4,2,3},       // 9
          extents_type{4,2,1,3},     // 10
          extents_type{4,2,1,3,1},   // 11
          extents_type{1,4,2,1,3,1}}  // 12
    {
    }
    std::vector<extents_type> extents;
};



BOOST_FIXTURE_TEST_CASE_TEMPLATE( test_tensor_expression_retrieve_extents, value,  test_types, fixture)
{
    namespace ublas = boost::numeric::ublas;
    using value_type  = typename value::first_type;
    using layout_type = typename value::second_type;
    using tensor_type = ublas::dynamic_tensor<value_type, layout_type>;

    auto uplus1 = [](auto const& a){ return a + value_type(1);  };
    auto uplus2 = [](auto const& a){ return value_type(2) + a;  };
    auto bplus  = std::plus <value_type>{};
    auto bminus = std::minus<value_type>{};

    for(auto const& e : extents) {

        auto t = tensor_type(e);
        auto v = value_type{};
        for(auto& tt: t){ tt = v; v+=value_type{1}; }


        BOOST_CHECK( ublas::detail::retrieve_extents( t ) == e );


        // uexpr1 = t+1
        // uexpr2 = 2+t
        auto uexpr1 = ublas::detail::make_unary_tensor_expression<tensor_type>( t, uplus1 );
        auto uexpr2 = ublas::detail::make_unary_tensor_expression<tensor_type>( t, uplus2 );

        BOOST_CHECK( ublas::detail::retrieve_extents( uexpr1 ) == e );
        BOOST_CHECK( ublas::detail::retrieve_extents( uexpr2 ) == e );

        // bexpr_uexpr = (t+1) + (2+t)
        auto bexpr_uexpr = ublas::detail::make_binary_tensor_expression<tensor_type>( uexpr1, uexpr2, bplus );

        BOOST_CHECK( ublas::detail::retrieve_extents( bexpr_uexpr ) == e );


        // bexpr_bexpr_uexpr = ((t+1) + (2+t)) - t
        auto bexpr_bexpr_uexpr = ublas::detail::make_binary_tensor_expression<tensor_type>( bexpr_uexpr, t, bminus );

        BOOST_CHECK( ublas::detail::retrieve_extents( bexpr_bexpr_uexpr ) == e );

    }


    for(auto i = 0u; i < extents.size()-1; ++i)
    {

        auto v = value_type{};

        auto t1 = tensor_type(extents[i]);
        for(auto& tt: t1){ tt = v; v+=value_type{1}; }

        auto t2 = tensor_type(extents[i+1]);
        for(auto& tt: t2){ tt = v; v+=value_type{2}; }

        BOOST_CHECK( ublas::detail::retrieve_extents( t1 ) != ublas::detail::retrieve_extents( t2 ) );

        // uexpr1 = t1+1
        // uexpr2 = 2+t2
        auto uexpr1 = ublas::detail::make_unary_tensor_expression<tensor_type>( t1, uplus1 );
        auto uexpr2 = ublas::detail::make_unary_tensor_expression<tensor_type>( t2, uplus2 );

        BOOST_CHECK( ublas::detail::retrieve_extents( t1 )     == ublas::detail::retrieve_extents( uexpr1 ) );
        BOOST_CHECK( ublas::detail::retrieve_extents( t2 )     == ublas::detail::retrieve_extents( uexpr2 ) );
        BOOST_CHECK( ublas::detail::retrieve_extents( uexpr1 ) != ublas::detail::retrieve_extents( uexpr2 ) );

        // bexpr_uexpr = (t1+1) + (2+t2)
        auto bexpr_uexpr = ublas::detail::make_binary_tensor_expression<tensor_type>( uexpr1, uexpr2, bplus );

        BOOST_CHECK( ublas::detail::retrieve_extents( bexpr_uexpr ) == ublas::detail::retrieve_extents(t1) );


        // bexpr_bexpr_uexpr = ((t1+1) + (2+t2)) - t2
        auto bexpr_bexpr_uexpr1 = ublas::detail::make_binary_tensor_expression<tensor_type>( bexpr_uexpr, t2, bminus );

        BOOST_CHECK( ublas::detail::retrieve_extents( bexpr_bexpr_uexpr1 ) == ublas::detail::retrieve_extents(t2) );


        // bexpr_bexpr_uexpr = t2 - ((t1+1) + (2+t2))
        auto bexpr_bexpr_uexpr2 = ublas::detail::make_binary_tensor_expression<tensor_type>( t2, bexpr_uexpr, bminus );

        BOOST_CHECK( ublas::detail::retrieve_extents( bexpr_bexpr_uexpr2 ) == ublas::detail::retrieve_extents(t2) );
    }
}







BOOST_FIXTURE_TEST_CASE_TEMPLATE( test_tensor_expression_all_extents_equal, value,  test_types, fixture)
{
    namespace ublas = boost::numeric::ublas;
    using value_type  = typename value::first_type;
    using layout_type = typename value::second_type;
    using tensor_type = ublas::dynamic_tensor<value_type, layout_type>;
    
    auto uplus1 = [](auto const& a){ return a + value_type(1);  };
    auto uplus2 = [](auto const& a){ return value_type(2) + a;  };
    auto bplus  = std::plus <value_type>{};
    auto bminus = std::minus<value_type>{};

    for(auto const& e : extents) {

        auto t = tensor_type(e);
        auto v = value_type{};
        for(auto& tt: t){ tt = v; v+=value_type{1}; }


        BOOST_CHECK( ublas::detail::all_extents_equal( t , e ) );


        // uexpr1 = t+1
        // uexpr2 = 2+t
        auto uexpr1 = ublas::detail::make_unary_tensor_expression<tensor_type>( t, uplus1 );
        auto uexpr2 = ublas::detail::make_unary_tensor_expression<tensor_type>( t, uplus2 );

        BOOST_CHECK( ublas::detail::all_extents_equal( uexpr1, e ) );
        BOOST_CHECK( ublas::detail::all_extents_equal( uexpr2, e ) );

        // bexpr_uexpr = (t+1) + (2+t)
        auto bexpr_uexpr = ublas::detail::make_binary_tensor_expression<tensor_type>( uexpr1, uexpr2, bplus );

        BOOST_CHECK( ublas::detail::all_extents_equal( bexpr_uexpr, e ) );


        // bexpr_bexpr_uexpr = ((t+1) + (2+t)) - t
        auto bexpr_bexpr_uexpr = ublas::detail::make_binary_tensor_expression<tensor_type>( bexpr_uexpr, t, bminus );

        BOOST_CHECK( ublas::detail::all_extents_equal( bexpr_bexpr_uexpr , e ) );

    }


    for(auto i = 0u; i < extents.size()-1; ++i)
    {

        auto v = value_type{};

        auto t1 = tensor_type(extents[i]);
        for(auto& tt: t1){ tt = v; v+=value_type{1}; }

        auto t2 = tensor_type(extents[i+1]);
        for(auto& tt: t2){ tt = v; v+=value_type{2}; }

        BOOST_CHECK( ublas::detail::all_extents_equal( t1, ublas::detail::retrieve_extents(t1) ) );
        BOOST_CHECK( ublas::detail::all_extents_equal( t2, ublas::detail::retrieve_extents(t2) ) );

        // uexpr1 = t1+1
        // uexpr2 = 2+t2
        auto uexpr1 = ublas::detail::make_unary_tensor_expression<tensor_type>( t1, uplus1 );
        auto uexpr2 = ublas::detail::make_unary_tensor_expression<tensor_type>( t2, uplus2 );

        BOOST_CHECK( ublas::detail::all_extents_equal( uexpr1, ublas::detail::retrieve_extents(uexpr1) ) );
        BOOST_CHECK( ublas::detail::all_extents_equal( uexpr2, ublas::detail::retrieve_extents(uexpr2) ) );

        // bexpr_uexpr = (t1+1) + (2+t2)
        auto bexpr_uexpr = ublas::detail::make_binary_tensor_expression<tensor_type>( uexpr1, uexpr2, bplus );

        BOOST_CHECK( ! ublas::detail::all_extents_equal( bexpr_uexpr, ublas::detail::retrieve_extents( bexpr_uexpr  ) ) );

        // bexpr_bexpr_uexpr = ((t1+1) + (2+t2)) - t2
        auto bexpr_bexpr_uexpr1 = ublas::detail::make_binary_tensor_expression<tensor_type>( bexpr_uexpr, t2, bminus );

        BOOST_CHECK( ! ublas::detail::all_extents_equal( bexpr_bexpr_uexpr1, ublas::detail::retrieve_extents( bexpr_bexpr_uexpr1  ) ) );

        // bexpr_bexpr_uexpr = t2 - ((t1+1) + (2+t2))
        auto bexpr_bexpr_uexpr2 = ublas::detail::make_binary_tensor_expression<tensor_type>( t2, bexpr_uexpr, bminus );

        BOOST_CHECK( ! ublas::detail::all_extents_equal( bexpr_bexpr_uexpr2, ublas::detail::retrieve_extents( bexpr_bexpr_uexpr2  ) ) );


        // bexpr_uexpr2 = (t1+1) + t2
        auto bexpr_uexpr2 = ublas::detail::make_binary_tensor_expression<tensor_type>( uexpr1, t2, bplus );
        BOOST_CHECK( ! ublas::detail::all_extents_equal( bexpr_uexpr2, ublas::detail::retrieve_extents( bexpr_uexpr2  ) ) );


        // bexpr_uexpr2 = ((t1+1) + t2) + t1
        auto bexpr_bexpr_uexpr3 = ublas::detail::make_binary_tensor_expression<tensor_type>( bexpr_uexpr2, t1, bplus );
        BOOST_CHECK( ! ublas::detail::all_extents_equal( bexpr_bexpr_uexpr3, ublas::detail::retrieve_extents( bexpr_bexpr_uexpr3  ) ) );

        // bexpr_uexpr2 = t1 + (((t1+1) + t2) + t1)
        auto bexpr_bexpr_uexpr4 = ublas::detail::make_binary_tensor_expression<tensor_type>( t1, bexpr_bexpr_uexpr3, bplus );
        BOOST_CHECK( ! ublas::detail::all_extents_equal( bexpr_bexpr_uexpr4, ublas::detail::retrieve_extents( bexpr_bexpr_uexpr4  ) ) );

    }
}

BOOST_AUTO_TEST_SUITE_END()
