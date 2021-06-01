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

BOOST_AUTO_TEST_SUITE(test_functions_outer,
    * boost::unit_test::description("Validate Outer Product")
)

BOOST_TEST_DECORATOR(
    *boost::unit_test::label("outer_prod")
    *boost::unit_test::description("Testing the outer product for dynamic tensor")
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

    ublas::for_each_fixture(self, [&self](auto /*id*/, auto a){
        auto const a_rank = a.rank();
        
        if(a_rank <= 1ul) return;
        
        a = value_type{2};

        ublas::for_each_fixture(self, [&a, a_rank](auto /*id*/, auto b){
            auto const b_rank = b.rank();
            if(b_rank <= 1ul) return;
            
            b = value_type{1};

            BOOST_TEST_CONTEXT("[Dynamic Outer Product] tensor with left rank("<< a_rank <<") : right rank (" << b_rank << ")"){

                auto c  = ublas::outer_prod(a, b);

                for(auto const& cc : c)
                    BOOST_CHECK_EQUAL( cc , a[0]*b[0] );

            }
        });
    });


}

BOOST_TEST_DECORATOR(
    *boost::unit_test::label("outer_prod")
    *boost::unit_test::description("Testing the outer product for static rank tensor")
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

    ublas::for_each_fixture(self, [&self](auto /*id*/, auto a){
        static constexpr auto a_rank = a.rank();
        
        if constexpr(a_rank > 1ul){
            a = value_type{2};

            ublas::for_each_fixture(self, [&a](auto /*id*/, auto b){
                constexpr auto b_rank = b.rank();
                
                if constexpr(b_rank > 1ul){
                    b = value_type{1};

                    BOOST_TEST_CONTEXT("[Static Rank Outer Product] tensor with left rank("<< a_rank <<") : right rank (" << b_rank << ")"){

                        auto c  = ublas::outer_prod(a, b);

                        for(auto const& cc : c)
                            BOOST_CHECK_EQUAL( cc , a[0]*b[0] );

                    }
                }
                
            });
        }
        
    });


}

BOOST_AUTO_TEST_SUITE_END()
