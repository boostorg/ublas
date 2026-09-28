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

BOOST_AUTO_TEST_SUITE(test_tensor_assign_arithmetic, 
    * boost::unit_test::description("Validate Assignment of Arithmetic Operations")
)

BOOST_TEST_DECORATOR(
    *boost::unit_test::label("op_assignement")
    *boost::unit_test::description("Testing the dynamic tensor's assingment of arithmetic result")
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

        BOOST_TEST_CONTEXT("[Arithmatic Assignment] rank("<< t1.rank() <<") dynamic tensor"){
            auto t2 = t1;
            auto r = t1;

            ublas::iota(t1, v);
            ublas::iota(t2, v + two);

            r  = t1 + two;
            r += t1;
            r += two;

            for(auto i = 0ul; i < t1.size(); ++i)
                BOOST_CHECK_EQUAL ( r(i), two*t1(i) + four );

            r  = two + t1;
            r += t1;
            r += two;

            for(auto i = 0ul; i < t1.size(); ++i)
                BOOST_CHECK_EQUAL ( r(i), two*t1(i) + four );

            for(auto i = 0ul; i < t1.size(); ++i)
                BOOST_CHECK_EQUAL ( r(i), two*t1(i) + four );

            r = (t1-two);
            r += t1;
            r -= two;

            for(auto i = 0ul; i < t1.size(); ++i)
                BOOST_CHECK_EQUAL ( r(i), two*t1(i) - four );

            r  = (t1*two);
            r *= three;
            r *= t1;

            for(auto i = 0ul; i < t1.size(); ++i)
                BOOST_CHECK_EQUAL ( r(i), two*three*t1(i)*t1(i) );

            r  = (t2*two);
            r /= two;
            r /= t2;
            r *= t2;

            for(auto i = 0ul; i < t1.size(); ++i)
                BOOST_CHECK_EQUAL ( r(i), (t2(i)*two) / (two*t2(i)) * t2(i) );

            r  = (t2/two+one);
            r /= (two/t2+one);
            r /= t2;

            for(auto i = 0ul; i < t1.size(); ++i)
                BOOST_CHECK_EQUAL ( r(i), (t2(i)/two+one) / (two/t2(i)+one) / t2(i) );

            tensor_t q = -r;
            for(auto i = 0ul; i < t1.size(); ++i)
                BOOST_CHECK_EQUAL ( q(i), -r(i) );

            tensor_t p = +r;
            for(auto i = 0ul; i < t1.size(); ++i)
                BOOST_CHECK_EQUAL ( p(i), r(i) );
        }
    };

    auto const& self = static_cast<fixture_t const&>(*this);
    ublas::for_each_fixture(self, check);
}

BOOST_TEST_DECORATOR(
    *boost::unit_test::label("op_assignement")
    *boost::unit_test::description("Testing the static rank tensor's assingment of arithmetic result")
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

            BOOST_TEST_CONTEXT("[Arithmatic Assignment] static rank("<< t1.rank() <<") tensor"){
                auto t2 = t1;
                auto r = t1;

                ublas::iota(t1, v);
                ublas::iota(t2, v + two);

                r  = t1 + two;
                r += t1;
                r += two;

                for(auto i = 0ul; i < t1.size(); ++i)
                    BOOST_CHECK_EQUAL ( r(i), two*t1(i) + four );

                r  = two + t1;
                r += t1;
                r += two;

                for(auto i = 0ul; i < t1.size(); ++i)
                    BOOST_CHECK_EQUAL ( r(i), two*t1(i) + four );

                for(auto i = 0ul; i < t1.size(); ++i)
                    BOOST_CHECK_EQUAL ( r(i), two*t1(i) + four );

                r = (t1-two);
                r += t1;
                r -= two;

                for(auto i = 0ul; i < t1.size(); ++i)
                    BOOST_CHECK_EQUAL ( r(i), two*t1(i) - four );

                r  = (t1*two);
                r *= three;
                r *= t1;

                for(auto i = 0ul; i < t1.size(); ++i)
                    BOOST_CHECK_EQUAL ( r(i), two*three*t1(i)*t1(i) );

                r  = (t2*two);
                r /= two;
                r /= t2;
                r *= t2;

                for(auto i = 0ul; i < t1.size(); ++i)
                    BOOST_CHECK_EQUAL ( r(i), (t2(i)*two) / (two*t2(i)) * t2(i) );

                r  = (t2/two+one);
                r /= (two/t2+one);
                r /= t2;

                for(auto i = 0ul; i < t1.size(); ++i)
                    BOOST_CHECK_EQUAL ( r(i), (t2(i)/two+one) / (two/t2(i)+one) / t2(i) );

                tensor_t q = -r;
                for(auto i = 0ul; i < t1.size(); ++i)
                    BOOST_CHECK_EQUAL ( q(i), -r(i) );

                tensor_t p = +r;
                for(auto i = 0ul; i < t1.size(); ++i)
                    BOOST_CHECK_EQUAL ( p(i), r(i) );
            }
        }
    };

    auto const& self = static_cast<fixture_t const&>(*this);
    ublas::for_each_fixture(self, check);
}

// FIXME: Enable after the strides computation is fixed [ issue #119 ]
// BOOST_TEST_DECORATOR(
//     *boost::unit_test::label("op_assignement")
//     *boost::unit_test::description("Testing the static tensor's assingment of arithmetic result")
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
//         if constexpr (t1.rank() > 0ul){

//             BOOST_TEST_CONTEXT("[Arithmatic Assignment] rank("<< t1.rank() <<") static tensor"){
//                 auto t2 = t1;
//                 auto r = t1;

//                 ublas::iota(t1, v);
//                 ublas::iota(t2, v + two);

//                 r  = t1 + two;
//                 r += t1;
//                 r += two;

//                 for(auto i = 0ul; i < t1.size(); ++i)
//                     BOOST_CHECK_EQUAL ( r(i), two*t1(i) + four );

//                 r  = two + t1;
//                 r += t1;
//                 r += two;

//                 for(auto i = 0ul; i < t1.size(); ++i)
//                     BOOST_CHECK_EQUAL ( r(i), two*t1(i) + four );

//                 for(auto i = 0ul; i < t1.size(); ++i)
//                     BOOST_CHECK_EQUAL ( r(i), two*t1(i) + four );

//                 r = (t1-two);
//                 r += t1;
//                 r -= two;

//                 for(auto i = 0ul; i < t1.size(); ++i)
//                     BOOST_CHECK_EQUAL ( r(i), two*t1(i) - four );

//                 r  = (t1*two);
//                 r *= three;
//                 r *= t1;

//                 for(auto i = 0ul; i < t1.size(); ++i)
//                     BOOST_CHECK_EQUAL ( r(i), two*three*t1(i)*t1(i) );

//                 r  = (t2*two);
//                 r /= two;
//                 r /= t2;
//                 r *= t2;

//                 for(auto i = 0ul; i < t1.size(); ++i)
//                     BOOST_CHECK_EQUAL ( r(i), (t2(i)*two) / (two*t2(i)) * t2(i) );

//                 r  = (t2/two+one);
//                 r /= (two/t2+one);
//                 r /= t2;

//                 for(auto i = 0ul; i < t1.size(); ++i)
//                     BOOST_CHECK_EQUAL ( r(i), (t2(i)/two+one) / (two/t2(i)+one) / t2(i) );

//                 tensor_t q = -r;
//                 for(auto i = 0ul; i < t1.size(); ++i)
//                     BOOST_CHECK_EQUAL ( q(i), -r(i) );

//                 tensor_t p = +r;
//                 for(auto i = 0ul; i < t1.size(); ++i)
//                     BOOST_CHECK_EQUAL ( p(i), r(i) );
//             }
//         }
//     };

//     auto const& self = static_cast<fixture_t const&>(*this);
//     ublas::for_each_fixture(self, check);
// }


BOOST_AUTO_TEST_SUITE_END()
