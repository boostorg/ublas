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

BOOST_AUTO_TEST_SUITE(test_tensor_unary_expression, 
    * boost::unit_test::description("Validate Unary Expression")
)

BOOST_TEST_DECORATOR(
    *boost::unit_test::label("unary_expr")
    *boost::unit_test::description("Testing the unary expr for dynamic tensor")
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

    constexpr auto check = [uplus1]<typename tensor_t>(auto /*id*/, tensor_t t){
        auto v = value_type{};

        BOOST_TEST_CONTEXT("[Dynamic Tensor Unary Expr] rank("<< t.rank() <<") dynamic tensor"){
            
            ublas::iota(t,v);
            
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
    };

    auto const& self = static_cast<fixture_t const&>(*this);
    ublas::for_each_fixture(self, check);
}

BOOST_TEST_DECORATOR(
    *boost::unit_test::label("unary_expr")
    *boost::unit_test::description("Testing the unary expr for static rank tensor")
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

    constexpr auto check = [uplus1]<typename tensor_t>(auto /*id*/, tensor_t t){
        auto v = value_type{};

        BOOST_TEST_CONTEXT("[Static Rank Tensor Unary Expr] static rank("<< t.rank() <<") tensor"){
            
            ublas::iota(t,v);
            
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
    };

    auto const& self = static_cast<fixture_t const&>(*this);
    ublas::for_each_fixture(self, check);
}

BOOST_TEST_DECORATOR(
    *boost::unit_test::label("unary_expr")
    *boost::unit_test::description("Testing the unary expr for static tensor")
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

    constexpr auto check = [uplus1]<typename tensor_t>(auto /*id*/, tensor_t t){
        auto v = value_type{};

        BOOST_TEST_CONTEXT("[Static Tensor Unary Expr] rank("<< t.rank() <<") static tensor"){
            
            ublas::iota(t,v);
            
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
    };

    auto const& self = static_cast<fixture_t const&>(*this);
    ublas::for_each_fixture(self, check);
}


BOOST_AUTO_TEST_SUITE_END()
