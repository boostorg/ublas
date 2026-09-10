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

BOOST_AUTO_TEST_SUITE(test_tensor_comparision_with_scalar, 
    * boost::unit_test::description("Validate Comparision Operators/Functions With Scalar")
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
        BOOST_TEST_CONTEXT("[Comparision Operator] rank("<< t1.rank() <<") dynamic tensor"){
            auto const one = value_type{1};
            auto const two = value_type{2};
            auto const three = value_type{3};
            auto const five = value_type{5};
            auto const six = value_type{6};
            
            auto t2 = t1;
            auto t3 = t1;

            t1 = two;
            t2 = one;
            t3 = three;

            BOOST_CHECK(t1 == t1);
            BOOST_CHECK(t1 != t2);

            t2 = three;


            if(t1.empty()) return;

            // One cannot tell which complex number is greater or less.
            if constexpr(!ublas::is_complex_v<value_type>){
                BOOST_CHECK( !(t1  <  two)  );
                BOOST_CHECK( !(t1  >  two)  );
                BOOST_CHECK(  (t1 >= two)   );
                BOOST_CHECK(  (t1 <= two)   );
                BOOST_CHECK(  (t1 == two)   );
                BOOST_CHECK(  (t1 != three) );

                BOOST_CHECK( !(two  >  t1));
                BOOST_CHECK( !(two  <  t1));
                BOOST_CHECK(  (two <= t1) );
                BOOST_CHECK(  (two >= t1) );
                BOOST_CHECK(  (two == t1) );
                BOOST_CHECK(  (three != t1) );

                BOOST_CHECK( !( t1+three  <  five));
                BOOST_CHECK( !( t1+three  >  five));
                BOOST_CHECK(  ( t1+three >= five) );
                BOOST_CHECK(  ( t1+three <= five) );
                BOOST_CHECK(  ( t1+three == five) );
                BOOST_CHECK(  ( t1+three != six)  );


                BOOST_CHECK( !( five  >  t1+three));
                BOOST_CHECK( !( five  <  t1+three));
                BOOST_CHECK(  ( five >= t1+three) );
                BOOST_CHECK(  ( five <= t1+three) );
                BOOST_CHECK(  ( five == t1+three) );
                BOOST_CHECK(  ( six  != t1+three) );


                BOOST_CHECK( !( t1+t3  <  five));
                BOOST_CHECK( !( t1+t3  >  five));
                BOOST_CHECK(  ( t1+t3 >= five) );
                BOOST_CHECK(  ( t1+t3 <= five) );
                BOOST_CHECK(  ( t1+t3 == five) );
                BOOST_CHECK(  ( t1+t3 != six)  );


                BOOST_CHECK( !( five  >  t1+t3));
                BOOST_CHECK( !( five  <  t1+t3));
                BOOST_CHECK(  ( five >= t1+t3) );
                BOOST_CHECK(  ( five <= t1+t3) );
                BOOST_CHECK(  ( five == t1+t3) );
                BOOST_CHECK(  (  six != t1+t3) );
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
        BOOST_TEST_CONTEXT("[Comparision Operator] static rank("<< t1.rank() <<") tensor"){
            auto const one = value_type{1};
            auto const two = value_type{2};
            auto const three = value_type{3};
            auto const five = value_type{5};
            auto const six = value_type{6};
            
            auto t2 = t1;
            auto t3 = t1;

            t1 = two;
            t2 = one;
            t3 = three;

            BOOST_CHECK(t1 == t1);
            BOOST_CHECK(t1 != t2);

            t2 = three;

            if(t1.empty()) return;

            // One cannot tell which complex number is greater or less.
            if constexpr(!ublas::is_complex_v<value_type>){
                BOOST_CHECK( !(t1  <  two)  );
                BOOST_CHECK( !(t1  >  two)  );
                BOOST_CHECK(  (t1 >= two)   );
                BOOST_CHECK(  (t1 <= two)   );
                BOOST_CHECK(  (t1 == two)   );
                BOOST_CHECK(  (t1 != three) );

                BOOST_CHECK( !(two  >  t1));
                BOOST_CHECK( !(two  <  t1));
                BOOST_CHECK(  (two <= t1) );
                BOOST_CHECK(  (two >= t1) );
                BOOST_CHECK(  (two == t1) );
                BOOST_CHECK(  (three != t1) );

                BOOST_CHECK( !( t1+three  <  five));
                BOOST_CHECK( !( t1+three  >  five));
                BOOST_CHECK(  ( t1+three >= five) );
                BOOST_CHECK(  ( t1+three <= five) );
                BOOST_CHECK(  ( t1+three == five) );
                BOOST_CHECK(  ( t1+three != six)  );


                BOOST_CHECK( !( five  >  t1+three));
                BOOST_CHECK( !( five  <  t1+three));
                BOOST_CHECK(  ( five >= t1+three) );
                BOOST_CHECK(  ( five <= t1+three) );
                BOOST_CHECK(  ( five == t1+three) );
                BOOST_CHECK(  ( six  != t1+three) );


                BOOST_CHECK( !( t1+t3  <  five));
                BOOST_CHECK( !( t1+t3  >  five));
                BOOST_CHECK(  ( t1+t3 >= five) );
                BOOST_CHECK(  ( t1+t3 <= five) );
                BOOST_CHECK(  ( t1+t3 == five) );
                BOOST_CHECK(  ( t1+t3 != six)  );


                BOOST_CHECK( !( five  >  t1+t3));
                BOOST_CHECK( !( five  <  t1+t3));
                BOOST_CHECK(  ( five >= t1+t3) );
                BOOST_CHECK(  ( five <= t1+t3) );
                BOOST_CHECK(  ( five == t1+t3) );
                BOOST_CHECK(  (  six != t1+t3) );
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
        BOOST_TEST_CONTEXT("[Comparision Operator] rank("<< t1.rank() <<") static tensor"){
            auto const one = value_type{1};
            auto const two = value_type{2};
            auto const three = value_type{3};
            auto const five = value_type{5};
            auto const six = value_type{6};
            
            auto t2 = t1;
            auto t3 = t1;

            t1 = two;
            t2 = one;
            t3 = three;

            BOOST_CHECK(t1 == t1);
            BOOST_CHECK(t1 != t2);

            t2 = three;

            if(t1.empty()) return;

            // One cannot tell which complex number is greater or less.
            if constexpr(!ublas::is_complex_v<value_type>){
                BOOST_CHECK( !(t1  <  two)  );
                BOOST_CHECK( !(t1  >  two)  );
                BOOST_CHECK(  (t1 >= two)   );
                BOOST_CHECK(  (t1 <= two)   );
                BOOST_CHECK(  (t1 == two)   );
                BOOST_CHECK(  (t1 != three) );

                BOOST_CHECK( !(two  >  t1));
                BOOST_CHECK( !(two  <  t1));
                BOOST_CHECK(  (two <= t1) );
                BOOST_CHECK(  (two >= t1) );
                BOOST_CHECK(  (two == t1) );
                BOOST_CHECK(  (three != t1) );

                BOOST_CHECK( !( t1+three  <  five));
                BOOST_CHECK( !( t1+three  >  five));
                BOOST_CHECK(  ( t1+three >= five) );
                BOOST_CHECK(  ( t1+three <= five) );
                BOOST_CHECK(  ( t1+three == five) );
                BOOST_CHECK(  ( t1+three != six)  );


                BOOST_CHECK( !( five  >  t1+three));
                BOOST_CHECK( !( five  <  t1+three));
                BOOST_CHECK(  ( five >= t1+three) );
                BOOST_CHECK(  ( five <= t1+three) );
                BOOST_CHECK(  ( five == t1+three) );
                BOOST_CHECK(  ( six  != t1+three) );


                BOOST_CHECK( !( t1+t3  <  five));
                BOOST_CHECK( !( t1+t3  >  five));
                BOOST_CHECK(  ( t1+t3 >= five) );
                BOOST_CHECK(  ( t1+t3 <= five) );
                BOOST_CHECK(  ( t1+t3 == five) );
                BOOST_CHECK(  ( t1+t3 != six)  );


                BOOST_CHECK( !( five  >  t1+t3));
                BOOST_CHECK( !( five  <  t1+t3));
                BOOST_CHECK(  ( five >= t1+t3) );
                BOOST_CHECK(  ( five <= t1+t3) );
                BOOST_CHECK(  ( five == t1+t3) );
                BOOST_CHECK(  (  six != t1+t3) );
            }
        }
    };

    auto const& self = static_cast<fixture_t const&>(*this);
    ublas::for_each_fixture(self, check);
}


BOOST_AUTO_TEST_SUITE_END()
