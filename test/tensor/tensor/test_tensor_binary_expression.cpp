//
// 	Copyright (c) 2021  Cem Bassoy, cem.bassoy@gmail.com
// 	Copyright (c) 2021, Amit Singh, amitsingh19975@gmail.com
//
//  Distributed under the Boost Software License, Version 1.0. (See
//  accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)
//
//  The authors gratefully acknowledge the support of
//  Google and Fraunhofer IOSB, Ettlingen, Germany
//

#include <boost/test/unit_test.hpp>
#include "../fixture_utility.hpp"

BOOST_AUTO_TEST_SUITE(test_tensor_binary_expression, 
    * boost::unit_test::description("Validate Binary Expression")
)

BOOST_TEST_DECORATOR(
    *boost::unit_test::label("unary_expr")
    *boost::unit_test::description("Testing the binary expr for dynamic tensor")
)
BOOST_FIXTURE_TEST_CASE_TEMPLATE(test_tensor_dynamic,
    TestTupleType,
    boost::numeric::ublas::test_types,
    boost::numeric::ublas::tuple_fixture_tensor_dynamic<TestTupleType>
)
{
    namespace ublas = boost::numeric::ublas;
    using value_type = typename TestTupleType::first_type;
    using fixture_t = ublas::tuple_fixture_tensor_dynamic<TestTupleType>;

    constexpr auto uplus1 = [](auto const& a){ return a + value_type{1}; };
    constexpr auto uplus2 = [](auto const& a){ return a + value_type{2}; };
    constexpr auto bplus  = std::plus <value_type>{};
    constexpr auto bminus = std::minus<value_type>{};

    constexpr auto check = [uplus1,uplus2,bplus,bminus]<typename tensor_t>(auto /*id*/, tensor_t t){
        auto v = value_type{};

        BOOST_TEST_CONTEXT("[Dynamic Tensor Binary Expr] rank("<< t.rank() <<") dynamic tensor"){
            
            ublas::iota(t,v);
            
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
    };

    auto const& self = static_cast<fixture_t const&>(*this);
    ublas::for_each_fixture(self, check);
}

BOOST_TEST_DECORATOR(
    *boost::unit_test::label("unary_expr")
    *boost::unit_test::description("Testing the binary expr for static rank tensor")
)
BOOST_FIXTURE_TEST_CASE_TEMPLATE(test_tensor_static_rank,
    TestTupleType,
    boost::numeric::ublas::test_types,
    boost::numeric::ublas::tuple_fixture_tensor_static_rank<TestTupleType>
)
{
    namespace ublas = boost::numeric::ublas;
    using value_type = typename TestTupleType::first_type;
    using fixture_t = ublas::tuple_fixture_tensor_static_rank<TestTupleType>;

    constexpr auto uplus1 = [](auto const& a){ return a + value_type{1}; };
    constexpr auto uplus2 = [](auto const& a){ return a + value_type{2}; };
    constexpr auto bplus  = std::plus <value_type>{};
    constexpr auto bminus = std::minus<value_type>{};

    constexpr auto check = [uplus1,uplus2,bplus,bminus]<typename tensor_t>(auto /*id*/, tensor_t t){
        auto v = value_type{};

        BOOST_TEST_CONTEXT("[Static Rank Tensor Binary Expr] static rank("<< t.rank() <<") tensor"){
            
            ublas::iota(t,v);
            
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
    };

    auto const& self = static_cast<fixture_t const&>(*this);
    ublas::for_each_fixture(self, check);
}

BOOST_TEST_DECORATOR(
    *boost::unit_test::label("unary_expr")
    *boost::unit_test::description("Testing the binary expr for static tensor")
)
BOOST_FIXTURE_TEST_CASE_TEMPLATE(test_tensor_static,
    TestTupleType,
    boost::numeric::ublas::test_types,
    boost::numeric::ublas::tuple_fixture_tensor_static<TestTupleType>
)
{
    namespace ublas = boost::numeric::ublas;
    using value_type = typename TestTupleType::first_type;
    using fixture_t = ublas::tuple_fixture_tensor_static<TestTupleType>;
    
    constexpr auto uplus1 = [](auto const& a){ return a + value_type{1}; };
    constexpr auto uplus2 = [](auto const& a){ return a + value_type{2}; };
    constexpr auto bplus  = std::plus <value_type>{};
    constexpr auto bminus = std::minus<value_type>{};

    constexpr auto check = [uplus1,uplus2,bplus,bminus]<typename tensor_t>(auto /*id*/, tensor_t t){
        auto v = value_type{};

        BOOST_TEST_CONTEXT("[Static Tensor Binary Expr] rank("<< t.rank() <<") static tensor"){
            
            ublas::iota(t,v);
            
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
    };

    auto const& self = static_cast<fixture_t const&>(*this);
    ublas::for_each_fixture(self, check);
}


BOOST_AUTO_TEST_SUITE_END()
