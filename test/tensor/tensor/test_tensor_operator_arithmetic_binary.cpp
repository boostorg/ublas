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

BOOST_AUTO_TEST_SUITE(test_tensor_binary_arithmetic, 
    * boost::unit_test::description("Validate Binary Arithmetic Operators/Functions")
)

BOOST_TEST_DECORATOR(
    *boost::unit_test::label("binary_op")
    *boost::unit_test::description("Testing the dynamic tensor's binary operators")
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
        auto one = value_type{1};
        auto three = value_type{3};
        auto four = value_type{4};
        
        if (t1.rank() <= 1ul) return;

        BOOST_TEST_CONTEXT("[Binary Operator] rank("<< t1.rank() <<") dynamic tensor"){
            auto t2 = t1;
            auto r = t1;

            ublas::iota(t1, v);
            ublas::iota(t2, v + value_type{2});

            r = t1 + t1 + t1 + t2;

            for(auto i = 0ul; i < t1.size(); ++i)
                BOOST_CHECK_EQUAL ( r(i), three*t1(i) + t2(i) );


            r = t2 / (t1+three) * (t1+one) - t2; // r = ( t2/ ((t+3)*(t+1)) ) - t2

            for(auto i = 0ul; i < t1.size(); ++i)
                BOOST_CHECK_EQUAL ( r(i), t2(i) / (t1(i)+three)*(t1(i)+one) - t2(i) );

            r = three+t2 / (t1+three) * (t1+one) * t1 - t2; // r = 3+( t2/ ((t+3)*(t+1)*t) ) - t2

            for(auto i = 0ul; i < t1.size(); ++i)
                BOOST_CHECK_EQUAL ( r(i), three+t2(i) / (t1(i)+three)*(t1(i)+one)*t1(i) - t2(i) );

            r = t2 - t1 + t2 - t1;

            for(auto i = 0ul; i < r.size(); ++i)
                BOOST_CHECK_EQUAL ( r(i), four );


            r = t1 * t1 * t1 * t2;

            for(auto i = 0ul; i < t1.size(); ++i)
                BOOST_CHECK_EQUAL ( r(i), t1(i)*t1(i)*t1(i)*t2(i) );

            r = (t2/t2) * (t2/t2);

            for(auto i = 0ul; i < t1.size(); ++i)
                BOOST_CHECK_EQUAL ( r(i), one );
        }
    };

    auto const& self = static_cast<fixture_t const&>(*this);
    ublas::for_each_fixture(self, check);
}

BOOST_TEST_DECORATOR(
    *boost::unit_test::label("binary_op")
    *boost::unit_test::description("Testing the static rank tensor's binary operators")
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
        if constexpr(t1.rank() > 1ul){
            auto v = value_type{};
            auto one = value_type{1};
            auto three = value_type{3};
            auto four = value_type{4};

            BOOST_TEST_CONTEXT("[Binary Operator] static rank("<< t1.rank() <<") tensor"){
                auto t2 = t1;
                auto r = t1;

                ublas::iota(t1, v);
                ublas::iota(t2, v + value_type{2});

                r = t1 + t1 + t1 + t2;

                for(auto i = 0ul; i < t1.size(); ++i)
                    BOOST_CHECK_EQUAL ( r(i), three*t1(i) + t2(i) );


                r = t2 / (t1+three) * (t1+one) - t2; // r = ( t2/ ((t+3)*(t+1)) ) - t2

                for(auto i = 0ul; i < t1.size(); ++i)
                    BOOST_CHECK_EQUAL ( r(i), t2(i) / (t1(i)+three)*(t1(i)+one) - t2(i) );

                r = three+t2 / (t1+three) * (t1+one) * t1 - t2; // r = 3+( t2/ ((t+3)*(t+1)*t) ) - t2

                for(auto i = 0ul; i < t1.size(); ++i)
                    BOOST_CHECK_EQUAL ( r(i), three+t2(i) / (t1(i)+three)*(t1(i)+one)*t1(i) - t2(i) );

                r = t2 - t1 + t2 - t1;

                for(auto i = 0ul; i < r.size(); ++i)
                    BOOST_CHECK_EQUAL ( r(i), four );


                r = t1 * t1 * t1 * t2;

                for(auto i = 0ul; i < t1.size(); ++i)
                    BOOST_CHECK_EQUAL ( r(i), t1(i)*t1(i)*t1(i)*t2(i) );

                r = (t2/t2) * (t2/t2);

                for(auto i = 0ul; i < t1.size(); ++i)
                    BOOST_CHECK_EQUAL ( r(i), one );
            }
        }
    };

    auto const& self = static_cast<fixture_t const&>(*this);
    ublas::for_each_fixture(self, check);
}

// FIXME: Enable after the strides computation is fixed [ issue #119 ]
// BOOST_TEST_DECORATOR(
//     *boost::unit_test::label("binary_op")
//     *boost::unit_test::description("Testing the static tensor's binary operators")
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

//     constexpr auto check = [](auto /*id*/, auto t1){
//         auto v = value_type{};
//         auto one = value_type{1};
//         auto three = value_type{3};
//         auto four = value_type{4};
//         if constexpr(t1.rank() > 0ul){

//             BOOST_TEST_CONTEXT("[Binary Operator] rank("<< t1.rank() <<") static tensor"){
//                 auto t2 = t1;
//                 auto r = t1;

//                 ublas::iota(t1, v);
//                 ublas::iota(t2, v + value_type{2});

//                 r = t1 + t1 + t1 + t2;

//                 for(auto i = 0ul; i < t1.size(); ++i)
//                     BOOST_CHECK_EQUAL ( r(i), three*t1(i) + t2(i) );


//                 r = t2 / (t1+three) * (t1+one) - t2; // r = ( t2/ ((t+3)*(t+1)) ) - t2

//                 for(auto i = 0ul; i < t1.size(); ++i)
//                     BOOST_CHECK_EQUAL ( r(i), t2(i) / (t1(i)+three)*(t1(i)+one) - t2(i) );

//                 r = three+t2 / (t1+three) * (t1+one) * t1 - t2; // r = 3+( t2/ ((t+3)*(t+1)*t) ) - t2

//                 for(auto i = 0ul; i < t1.size(); ++i)
//                     BOOST_CHECK_EQUAL ( r(i), three+t2(i) / (t1(i)+three)*(t1(i)+one)*t1(i) - t2(i) );

//                 r = t2 - t1 + t2 - t1;

//                 for(auto i = 0ul; i < r.size(); ++i)
//                     BOOST_CHECK_EQUAL ( r(i), four );


//                 r = t1 * t1 * t1 * t2;

//                 for(auto i = 0ul; i < t1.size(); ++i)
//                     BOOST_CHECK_EQUAL ( r(i), t1(i)*t1(i)*t1(i)*t2(i) );

//                 r = (t2/t2) * (t2/t2);

//                 for(auto i = 0ul; i < t1.size(); ++i)
//                     BOOST_CHECK_EQUAL ( r(i), one );
//             }
//         }
//     };

//     auto const& self = static_cast<fixture_t const&>(*this);
//     ublas::for_each_fixture(self, check);
// }


BOOST_AUTO_TEST_SUITE_END()
