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

BOOST_AUTO_TEST_SUITE(test_functions_inner,
    * boost::unit_test::description("Validate Inner Product")
)

BOOST_TEST_DECORATOR(
    *boost::unit_test::label("inner_prod")
    *boost::unit_test::description("Testing the inner product for dynamic tensor")
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

    auto const& self = static_cast<fixture_t const&>(*this);

    ublas::for_each_fixture(self, []<typename tensor_type>(auto /*id*/, tensor_type a){
        auto const rank = a.rank();
        
        if(rank <= 1ul) return;

        auto b = a;
        a = value_type{2};
        b = value_type{1};
        
        BOOST_TEST_CONTEXT("[Dynamic Inner Product] tensor with rank("<< rank <<")"){

            auto c = ublas::inner_prod(a, b);
            auto r = std::inner_product(a.begin(),a.end(), b.begin(),value_type(0));

            BOOST_CHECK_EQUAL( c , r );

        }
    });


}

BOOST_TEST_DECORATOR(
    *boost::unit_test::label("inner_prod")
    *boost::unit_test::description("Testing the inner product for static rank tensor")
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

    auto const& self = static_cast<fixture_t const&>(*this);

    ublas::for_each_fixture(self, []<typename tensor_type>(auto /*id*/, tensor_type a){
        
        constexpr auto rank = a.rank();

        if constexpr (rank > 1ul) {
        
            auto b = a;
            a = value_type{2};
            b = value_type{1};
            

            BOOST_TEST_CONTEXT("[Static Rank Inner Product] tensor with rank("<< rank <<")"){
                
                auto c = ublas::inner_prod(a, b);
                auto r = std::inner_product(a.begin(),a.end(), b.begin(),value_type(0));

                BOOST_CHECK_EQUAL( c , r );

            }
        }

    });


}

BOOST_AUTO_TEST_SUITE_END()
