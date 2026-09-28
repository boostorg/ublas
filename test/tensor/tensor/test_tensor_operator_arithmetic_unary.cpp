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

BOOST_AUTO_TEST_SUITE(test_tensor_unary_arithmetic, 
    * boost::unit_test::description("Validate Binary Arithmetic Operators/Functions")
)

BOOST_TEST_DECORATOR(
    *boost::unit_test::label("unary_op")
    *boost::unit_test::description("Testing the dynamic tensor's unary operators")
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

    constexpr auto check = []<typename tensor_t>(auto /*id*/, tensor_t t1){
        auto v = value_type{};
        auto one = value_type{1};
        auto two = value_type{2};
        auto three = value_type{3};
        auto four = value_type{4};
        if(t1.rank() == 0ul) return;
        
        BOOST_TEST_CONTEXT("[Unary Operator] rank("<< t1.rank() <<") dynamic tensor"){
            auto t2 = t1;

            ublas::iota(t1, v);
            ublas::iota(t2, v + two);

            tensor_t r1 = t1 + two + t1 + two;

            for(auto i = 0ul; i < t1.size(); ++i)
                BOOST_CHECK_EQUAL ( r1(i), two*t1(i) + four );

            tensor_t r2 = two + t1 + two + t1;

            for(auto i = 0ul; i < t1.size(); ++i)
                BOOST_CHECK_EQUAL ( r2(i), two*t1(i) + four );

            tensor_t r3 = (t1-two) + (t1-two);

            for(auto i = 0ul; i < t1.size(); ++i)
                BOOST_CHECK_EQUAL ( r3(i), two*t1(i) - four );

            tensor_t r4 = (t1*two) * (three*t1);

            for(auto i = 0ul; i < t1.size(); ++i)
                BOOST_CHECK_EQUAL ( r4(i), two*three*t1(i)*t1(i) );

            tensor_t r5 = (t2*two) / (two*t2) * t2;

            for(auto i = 0ul; i < t1.size(); ++i)
                BOOST_CHECK_EQUAL ( r5(i), (t2(i)*two) / (two*t2(i)) * t2(i) );

            tensor_t r6 = (t2/two+one) / (two/t2+one) / t2;

            for(auto i = 0ul; i < t1.size(); ++i)
                BOOST_CHECK_EQUAL ( r6(i), (t2(i)/two+one) / (two/t2(i)+one) / t2(i) );
        }
    };

    auto const& self = static_cast<fixture_t const&>(*this);
    ublas::for_each_fixture(self, check);
}

BOOST_TEST_DECORATOR(
    *boost::unit_test::label("unary_op")
    *boost::unit_test::description("Testing the static rank tensor's unary operators")
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

    constexpr auto check = []<typename tensor_t>(auto /*id*/, tensor_t t1){
        auto v = value_type{};
        auto one = value_type{1};
        auto two = value_type{2};
        auto three = value_type{3};
        auto four = value_type{4};
        if constexpr(t1.rank() > 0ul){

            BOOST_TEST_CONTEXT("[Unary Operator] static rank("<< t1.rank() <<") tensor"){
                auto t2 = t1;

                ublas::iota(t1, v);
                ublas::iota(t2, v + two);

                tensor_t r1 = t1 + two + t1 + two;

                for(auto i = 0ul; i < t1.size(); ++i)
                    BOOST_CHECK_EQUAL ( r1(i), two*t1(i) + four );

                tensor_t r2 = two + t1 + two + t1;

                for(auto i = 0ul; i < t1.size(); ++i)
                    BOOST_CHECK_EQUAL ( r2(i), two*t1(i) + four );

                tensor_t r3 = (t1-two) + (t1-two);

                for(auto i = 0ul; i < t1.size(); ++i)
                    BOOST_CHECK_EQUAL ( r3(i), two*t1(i) - four );

                tensor_t r4 = (t1*two) * (three*t1);

                for(auto i = 0ul; i < t1.size(); ++i)
                    BOOST_CHECK_EQUAL ( r4(i), two*three*t1(i)*t1(i) );

                tensor_t r5 = (t2*two) / (two*t2) * t2;

                for(auto i = 0ul; i < t1.size(); ++i)
                    BOOST_CHECK_EQUAL ( r5(i), (t2(i)*two) / (two*t2(i)) * t2(i) );

                tensor_t r6 = (t2/two+one) / (two/t2+one) / t2;

                for(auto i = 0ul; i < t1.size(); ++i)
                    BOOST_CHECK_EQUAL ( r6(i), (t2(i)/two+one) / (two/t2(i)+one) / t2(i) );
            }
        }
    };

    auto const& self = static_cast<fixture_t const&>(*this);
    ublas::for_each_fixture(self, check);
}

// FIXME: Enable after the strides computation is fixed [ issue #119 ]
// BOOST_TEST_DECORATOR(
//     *boost::unit_test::label("unary_op")
//     *boost::unit_test::description("Testing the static tensor's unary operators")
// )
// BOOST_FIXTURE_TEST_CASE_TEMPLATE(test_tensor_static,
//     TestTupleType,
//     boost::numeric::ublas::test_types,
//     boost::numeric::ublas::tuple_fixture_tensor_static<TestTupleType>
// )
// {
//     namespace ublas = boost::numeric::ublas;
//     using value_type = typename TestTupleType::first_type;
//     using fixture_t = ublas::tuple_fixture_tensor_static<TestTupleType>;

//     constexpr auto check = []<typename tensor_t>(auto /*id*/, tensor_t t1){
//         auto v = value_type{};
//         auto one = value_type{1};
//         auto two = value_type{2};
//         auto three = value_type{3};
//         auto four = value_type{4};
//         if constexpr(t1.rank() > 0ul){

//             BOOST_TEST_CONTEXT("[Unary Operator] rank("<< t1.rank() <<") static tensor"){
//                 tensor_t t2;

//                 ublas::iota(t1, v);
//                 ublas::iota(t2, v + two);

//                 tensor_t r1 = t1 + two + t1 + two;

//                 for(auto i = 0ul; i < t1.size(); ++i)
//                     BOOST_CHECK_EQUAL ( r1(i), two*t1(i) + four );

//                 tensor_t r2 = two + t1 + two + t1;

//                 for(auto i = 0ul; i < t1.size(); ++i)
//                     BOOST_CHECK_EQUAL ( r2(i), two*t1(i) + four );

//                 tensor_t r3 = (t1-two) + (t1-two);

//                 for(auto i = 0ul; i < t1.size(); ++i)
//                     BOOST_CHECK_EQUAL ( r3(i), two*t1(i) - four );

//                 tensor_t r4 = (t1*two) * (three*t1);

//                 for(auto i = 0ul; i < t1.size(); ++i)
//                     BOOST_CHECK_EQUAL ( r4(i), two*three*t1(i)*t1(i) );

//                 tensor_t r5 = (t2*two) / (two*t2) * t2;

//                 for(auto i = 0ul; i < t1.size(); ++i)
//                     BOOST_CHECK_EQUAL ( r5(i), (t2(i)*two) / (two*t2(i)) * t2(i) );

//                 tensor_t r6 = (t2/two+one) / (two/t2+one) / t2;

//                 for(auto i = 0ul; i < t1.size(); ++i)
//                     BOOST_CHECK_EQUAL ( r6(i), (t2(i)/two+one) / (two/t2(i)+one) / t2(i) );
//             }
//         }
//     };

//     auto const& self = static_cast<fixture_t const&>(*this);
//     ublas::for_each_fixture(self, check);
// }


BOOST_AUTO_TEST_SUITE_END()
