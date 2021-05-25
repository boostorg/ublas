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

BOOST_AUTO_TEST_SUITE(test_functions_norm,
    * boost::unit_test::description("Validate Norm Function")
)

BOOST_TEST_DECORATOR(
    *boost::unit_test::label("norm_fn")
    *boost::unit_test::description("Testing the norm function for dynamic tensor")
)
BOOST_FIXTURE_TEST_CASE_TEMPLATE(test_tensor_dynamic,
    TestTupleType,
    boost::numeric::ublas::cpp_std_types,
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

        ublas::iota(a, value_type{1});
        
        BOOST_TEST_CONTEXT("[Dynamic Norm Function] tensor with rank("<< rank <<")"){
            auto c = ublas::inner_prod(a, a);
            auto r = std::inner_product(a.begin(),a.end(), a.begin(),value_type(0));

            tensor_type var = (a+a)/value_type{2};
            auto r2 = ublas::norm( var );

            BOOST_CHECK_EQUAL( c , r );
            BOOST_CHECK_EQUAL( std::sqrt( c ) , r2 );

        }
    });


}

BOOST_TEST_DECORATOR(
    *boost::unit_test::label("norm_fn")
    *boost::unit_test::description("Testing the norm function for static rank tensor")
)
BOOST_FIXTURE_TEST_CASE_TEMPLATE(test_tensor_static_rank,
    TestTupleType,
    boost::numeric::ublas::cpp_std_types,
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
        
            ublas::iota(a, value_type{1});

            BOOST_TEST_CONTEXT("[Static Rank Norm Function] tensor with rank("<< rank <<")"){
                
                auto c = ublas::inner_prod(a, a);
                auto r = std::inner_product(a.begin(),a.end(), a.begin(),value_type(0));

                tensor_type var = (a+a)/value_type{2};
                auto r2 = ublas::norm( var );

                BOOST_CHECK_EQUAL( c , r );
                BOOST_CHECK_EQUAL( std::sqrt( c ) , r2 );

            }
        }

    });


}

BOOST_AUTO_TEST_SUITE_END()
