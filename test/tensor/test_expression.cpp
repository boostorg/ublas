//  Copyright (c) 2018-2019 Cem Bassoy
//
//  Distributed under the Boost Software License, Version 1.0. (See
//  accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)
//
//  The authors gratefully acknowledge the support of
//  Fraunhofer and Google in producing this work
//  which started as a Google Summer of Code project.
//



#include <boost/numeric/ublas/tensor/extents.hpp>
#include <boost/numeric/ublas/tensor/expression.hpp>
#include <boost/numeric/ublas/tensor/tensor.hpp>
#include <boost/test/unit_test.hpp>
#include "utility.hpp"

#include <functional>
#include <complex>



using test_types = zip<int,float,std::complex<float>>::with_t<boost::numeric::ublas::layout::first_order, boost::numeric::ublas::layout::last_order>;


struct fixture
{
    using extents_type = boost::numeric::ublas::extents<>;

    const std::vector<extents_type> extents
    {
//      extents_type{ },            // 0

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
      extents_type{1,4,2,1,3,1}   // 12
      };
};


BOOST_FIXTURE_TEST_CASE_TEMPLATE( test_tensor_expression_access, value,  test_types, fixture)
{
    namespace ublas     = boost::numeric::ublas;
    using value_t       = typename value::first_type;
    using layout_t      = typename value::second_type;
    using tensor_t      = ublas::tensor_dynamic<value_t, layout_t>;
    using expression_t  = typename tensor_t::super_type;


    for(auto const& e : extents) {

      if(!ublas::is_valid(e)){
        continue;
      }

        auto v = value_t{};
        auto t = tensor_t(e);

        for(auto& tt: t){ tt = v; v+=value_t{1}; }
        const auto& tensor_expression_const = static_cast<expression_t const&>( t );

        for(auto i = 0ul; i < t.size(); ++i){
            BOOST_CHECK_EQUAL( tensor_expression_const()(i), t(i)  );
        }

    }
}



BOOST_FIXTURE_TEST_CASE_TEMPLATE( test_tensor_unary_expression, value,  test_types, fixture)
{
    namespace ublas = boost::numeric::ublas;
    using value_t  = typename value::first_type;
    using layout_t = typename value::second_type;
    using tensor_t = ublas::tensor_dynamic<value_t, layout_t>;

    auto uplus1 = [](auto const& a){ return a+value_t{1}; };
    //auto uplus1 = std::bind(  std::plus<value_t>{}, std::placeholders::_1, value_t(1) );

    for(auto const& e : extents) {

        auto t = tensor_t(e);
        auto v = value_t{};
        for(auto& tt: t) { tt = v; v+=value_t{1}; }

        const auto uexpr = ublas::detail::make_unary_tensor_expression<tensor_t>( t, uplus1 );

        for(auto i = 0ul; i < t.size(); ++i){
            BOOST_CHECK_EQUAL( uexpr(i), uplus1(t(i))  );
        }

        auto uexpr_uexpr = ublas::detail::make_unary_tensor_expression<tensor_t>( uexpr, uplus1 );

        for(auto i = 0ul; i < t.size(); ++i){
            BOOST_CHECK_EQUAL( uexpr_uexpr(i), uplus1(uplus1(t(i)))  );
        }

        const auto & uexpr_e = uexpr.e;

        BOOST_CHECK( ( std::is_same_v< std::decay_t< decltype(uexpr_e) >, tensor_t > )   );

        const auto & uexpr_uexpr_e_e = uexpr_uexpr.e.e;

        BOOST_CHECK( ( std::is_same_v< std::decay_t< decltype(uexpr_uexpr_e_e) >, tensor_t > )   );


    }
}


BOOST_FIXTURE_TEST_CASE_TEMPLATE( test_tensor_binary_expression, value,  test_types, fixture)
{
    namespace ublas = boost::numeric::ublas;
    using value_t  = typename value::first_type;
    using layout_t = typename value::second_type;
    using tensor_t = ublas::tensor_dynamic<value_t, layout_t>;

    auto uplus1 = [](auto const& a){ return a+value_t{1}; };
    auto uplus2 = [](auto const& a){ return a+value_t{2}; };
    //auto uplus1 = std::bind(  std::plus<value_t>{}, std::placeholders::_1, value_t(1) );
    //auto uplus2 = std::bind(  std::plus<value_t>{}, std::placeholders::_1, value_t(2) );
    auto bplus  = std::plus <value_t>{};
    auto bminus = std::minus<value_t>{};

    for(auto const& e : extents) {

        auto t = tensor_t(e);
        auto v = value_t{};
        for(auto& tt: t){ tt = v; v+=value_t{1}; }

        auto uexpr1 = ublas::detail::make_unary_tensor_expression<tensor_t>( t, uplus1 );
        auto uexpr2 = ublas::detail::make_unary_tensor_expression<tensor_t>( t, uplus2 );

        BOOST_CHECK( ( std::is_same_v< std::decay_t< decltype(uexpr1.e) >, tensor_t > )   );
        BOOST_CHECK( ( std::is_same_v< std::decay_t< decltype(uexpr2.e) >, tensor_t > )   );

        for(auto i = 0ul; i < t.size(); ++i){
            BOOST_CHECK_EQUAL( uexpr1(i), uplus1(t(i))  );
        }

        for(auto i = 0ul; i < t.size(); ++i){
            BOOST_CHECK_EQUAL( uexpr2(i), uplus2(t(i))  );
        }

        auto bexpr_uexpr = ublas::detail::make_binary_tensor_expression<tensor_t>( uexpr1, uexpr2, bplus );

        BOOST_CHECK( ( std::is_same_v< std::decay_t< decltype(bexpr_uexpr.el.e) >, tensor_t > )   );
        BOOST_CHECK( ( std::is_same_v< std::decay_t< decltype(bexpr_uexpr.er.e) >, tensor_t > )   );


        for(auto i = 0ul; i < t.size(); ++i){
            BOOST_CHECK_EQUAL( bexpr_uexpr(i), bplus(uexpr1(i),uexpr2(i))  );
        }

        auto bexpr_bexpr_uexpr = ublas::detail::make_binary_tensor_expression<tensor_t>( bexpr_uexpr, t, bminus );

        BOOST_CHECK( ( std::is_same_v< std::decay_t< decltype(bexpr_bexpr_uexpr.el.el.e) >, tensor_t > )   );
        BOOST_CHECK( ( std::is_same_v< std::decay_t< decltype(bexpr_bexpr_uexpr.el.er.e) >, tensor_t > )   );
        BOOST_CHECK( ( std::is_same_v< std::decay_t< decltype(bexpr_bexpr_uexpr.er) >, tensor_t > )   );
        BOOST_CHECK( ( std::is_same_v< std::decay_t< decltype(bexpr_bexpr_uexpr.er) >, tensor_t > )   );

        for(auto i = 0ul; i < t.size(); ++i){
            BOOST_CHECK_EQUAL( bexpr_bexpr_uexpr(i), bminus(bexpr_uexpr(i),t(i))  );
        }

    }


}
