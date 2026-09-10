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

BOOST_AUTO_TEST_SUITE(test_tensor_comparision_with_tensor_expression, 
    * boost::unit_test::description("Validate Comparision Operators/Functions With Tensor Expression")
)

BOOST_TEST_DECORATOR(
    *boost::unit_test::label("compare_tensor")
    *boost::unit_test::description("Testing the dynamic tensor's comparision operators")
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

    constexpr auto check = [](auto /*id*/, auto t1){
        auto v = value_type{};
        BOOST_TEST_CONTEXT("[Comparision Operator] rank("<< t1.rank() <<") dynamic tensor"){
            auto t2 = t1;
            
            ublas::iota(t1, v);
            ublas::iota(t2, v + value_type{2});

            BOOST_CHECK(t1 == t1);
            BOOST_CHECK(t1 != t2);

            if(t1.empty()) return;

            auto const two = value_type(2);
            auto const three = value_type(3);

            // One cannot tell which complex number is greater or less.
            if constexpr(!ublas::is_complex_v<value_type>){
                BOOST_CHECK( !(t1             < t1)       );
                BOOST_CHECK( !(t1             > t1)       );
                BOOST_CHECK( t1               < (t2 + t1) );
                BOOST_CHECK( (t2 + t1)        > t1        );
                BOOST_CHECK( t1              <= (t1 + t1) );
                BOOST_CHECK( (t1 + t2)       >= t1        );
                BOOST_CHECK( (t2 + t2 + two) >= t1        );
                BOOST_CHECK( (two * t2)       > t1        );
                BOOST_CHECK( t1               < (two * t2));
                BOOST_CHECK( (two * t2)       > t1        );
                BOOST_CHECK( (two * t2)      >= t2        );
                BOOST_CHECK( t2              <= (two * t2));
                BOOST_CHECK( (three * t2)    >= t1        );
            }
        }
    };

    auto const& self = static_cast<fixture_t const&>(*this);
    ublas::for_each_fixture(self, check);
}

BOOST_TEST_DECORATOR(
    *boost::unit_test::label("compare_tensor")
    *boost::unit_test::description("Testing the static rank tensor's comparision operators")
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

    constexpr auto check = [](auto /*id*/, auto t1){
        auto v = value_type{};
        BOOST_TEST_CONTEXT("[Comparision Operator] static rank("<< t1.rank() <<") tensor"){
            auto t2 = t1;
            
            ublas::iota(t1, v);
            ublas::iota(t2, v + value_type{2});

            BOOST_CHECK(t1 == t1);
            BOOST_CHECK(t1 != t2);

            if(t1.empty()) return;

            auto const two = value_type(2);
            auto const three = value_type(3);

            // One cannot tell which complex number is greater or less.
            if constexpr(!ublas::is_complex_v<value_type>){
                BOOST_CHECK( !(t1             < t1)       );
                BOOST_CHECK( !(t1             > t1)       );
                BOOST_CHECK( t1               < (t2 + t1) );
                BOOST_CHECK( (t2 + t1)        > t1        );
                BOOST_CHECK( t1              <= (t1 + t1) );
                BOOST_CHECK( (t1 + t2)       >= t1        );
                BOOST_CHECK( (t2 + t2 + two) >= t1        );
                BOOST_CHECK( (two * t2)       > t1        );
                BOOST_CHECK( t1               < (two * t2));
                BOOST_CHECK( (two * t2)       > t1        );
                BOOST_CHECK( (two * t2)      >= t2        );
                BOOST_CHECK( t2              <= (two * t2));
                BOOST_CHECK( (three * t2)    >= t1        );
            }
        }
    };

    auto const& self = static_cast<fixture_t const&>(*this);
    ublas::for_each_fixture(self, check);
}

BOOST_TEST_DECORATOR(
    *boost::unit_test::label("compare_tensor")
    *boost::unit_test::description("Testing the static tensor's comparision operators")
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

    constexpr auto check = [](auto /*id*/, auto t1){
        auto v = value_type{};
        BOOST_TEST_CONTEXT("[Comparision Operator] rank("<< t1.rank() <<") static tensor"){
            auto t2 = t1;
            
            ublas::iota(t1, v);
            ublas::iota(t2, v + value_type{2});

            BOOST_CHECK(t1 == t1);
            BOOST_CHECK(t1 != t2);

            if(t1.empty()) return;

            auto const two = value_type(2);
            auto const three = value_type(3);

            // One cannot tell which complex number is greater or less.
            if constexpr(!ublas::is_complex_v<value_type>){
                BOOST_CHECK( !(t1             < t1)       );
                BOOST_CHECK( !(t1             > t1)       );
                BOOST_CHECK( t1               < (t2 + t1) );
                BOOST_CHECK( (t2 + t1)        > t1        );
                BOOST_CHECK( t1              <= (t1 + t1) );
                BOOST_CHECK( (t1 + t2)       >= t1        );
                BOOST_CHECK( (t2 + t2 + two) >= t1        );
                BOOST_CHECK( (two * t2)       > t1        );
                BOOST_CHECK( t1               < (two * t2));
                BOOST_CHECK( (two * t2)       > t1        );
                BOOST_CHECK( (two * t2)      >= t2        );
                BOOST_CHECK( t2              <= (two * t2));
                BOOST_CHECK( (three * t2)    >= t1        );
            }
        }
    };

    auto const& self = static_cast<fixture_t const&>(*this);
    ublas::for_each_fixture(self, check);
}


BOOST_AUTO_TEST_SUITE_END()
