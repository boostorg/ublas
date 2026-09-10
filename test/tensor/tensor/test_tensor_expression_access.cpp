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

BOOST_AUTO_TEST_SUITE(test_tensor_expression_access, 
    * boost::unit_test::description("Validate Expression access")
)

BOOST_TEST_DECORATOR(
    *boost::unit_test::label("access_expr")
    *boost::unit_test::description("Testing the access for dynamic tensor")
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

    constexpr auto check = []<typename tensor_t>(auto /*id*/, tensor_t t){
        auto v = value_type{};
        using expr_t = typename tensor_t::super_type;

        BOOST_TEST_CONTEXT("[Dynamic Tensor Expr Access] rank("<< t.rank() <<") dynamic tensor"){
            
            ublas::iota(t,v);
            auto const& expr = static_cast<expr_t const&>(t);
            
            for(auto i = 0ul; i < t.size(); ++i){
                BOOST_TEST_CHECKPOINT("[Dynamic Tensor Expr Access] testing tensor element(" << i << ") = " <<t[i]);
                BOOST_CHECK_EQUAL( expr()(i), t(i) );
            }
        }
    };

    auto const& self = static_cast<fixture_t const&>(*this);
    ublas::for_each_fixture(self, check);
}

BOOST_TEST_DECORATOR(
    *boost::unit_test::label("access_expr")
    *boost::unit_test::description("Testing the access for static rank tensor")
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

    constexpr auto check = []<typename tensor_t>(auto /*id*/, tensor_t t){
        auto v = value_type{};
        using expr_t = typename tensor_t::super_type;

        BOOST_TEST_CONTEXT("[Static Rank Tensor Expr Access] static rank("<< t.rank() <<") tensor"){
            
            ublas::iota(t,v);
            auto const& expr = static_cast<expr_t const&>(t);
            
            for(auto i = 0ul; i < t.size(); ++i){
                BOOST_TEST_CHECKPOINT("[Static Rank Tensor Expr Access] testing tensor element(" << i << ") = " <<t[i]);
                BOOST_CHECK_EQUAL( expr()(i), t(i) );
            }
        }
    };

    auto const& self = static_cast<fixture_t const&>(*this);
    ublas::for_each_fixture(self, check);
}

BOOST_TEST_DECORATOR(
    *boost::unit_test::label("access_expr")
    *boost::unit_test::description("Testing the access for static tensor")
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

    constexpr auto check = []<typename tensor_t>(auto /*id*/, tensor_t t){
        auto v = value_type{};
        using expr_t = typename tensor_t::super_type;

        BOOST_TEST_CONTEXT("[Static Tensor Expr Access] rank("<< t.rank() <<") static tensor"){
            
            ublas::iota(t,v);
            auto const& expr = static_cast<expr_t const&>(t);
            
            for(auto i = 0ul; i < t.size(); ++i){
                BOOST_TEST_CHECKPOINT("[Static Tensor Expr Access] testing tensor element(" << i << ") = " <<t[i]);
                BOOST_CHECK_EQUAL( expr()(i), t(i) );
            }
        }
    };

    auto const& self = static_cast<fixture_t const&>(*this);
    ublas::for_each_fixture(self, check);
}


BOOST_AUTO_TEST_SUITE_END()
