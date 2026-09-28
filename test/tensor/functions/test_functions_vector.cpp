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
#include <boost/numeric/ublas/vector.hpp>

BOOST_AUTO_TEST_SUITE(test_functions_vector, 
    * boost::unit_test::description("Validate Vector Product")
)

BOOST_TEST_DECORATOR(
    *boost::unit_test::label("vector_prod")
    *boost::unit_test::description("Testing the vector product for dynamic tensor")
)
BOOST_FIXTURE_TEST_CASE_TEMPLATE(test_tensor_dynamic,
    TestTupleType,
    boost::numeric::ublas::test_types,
    boost::numeric::ublas::tuple_fixture_tensor_dynamic<TestTupleType>
)
{
    namespace ublas = boost::numeric::ublas;
    using value_type = typename TestTupleType::first_type;
    using inner_t = inner_type_t<value_type>;
    using fixture_t = ublas::tuple_fixture_tensor_dynamic<TestTupleType>;

    auto const& self = static_cast<fixture_t const&>(*this);

    ublas::for_each_fixture(self, []<typename tensor_type>(auto /*id*/, tensor_type a){
        using vector_type  = typename tensor_type::vector_type;

        auto const rank = a.rank();
        
        if(rank == 0ul) return;

        a = value_type{2};

        BOOST_TEST_CONTEXT("[Dynamic Tensor Vector Product] tensor with rank("<< rank <<")"){
            
            for(auto m = 0u; m < rank; ++m){
                auto const em = a.size(m);

                auto b = vector_type  (em, value_type{1} );

                auto c = ublas::prod(a, b, m+1);

                for(auto i = 0u; i < c.size(); ++i)
                    BOOST_CHECK_EQUAL( c[i] , value_type( static_cast< inner_t >(em) ) * a[i] );

            }

        }
    });


}

BOOST_TEST_DECORATOR(
    *boost::unit_test::label("vector_prod")
    *boost::unit_test::description("Testing the vector product for static rank tensor")
)
BOOST_FIXTURE_TEST_CASE_TEMPLATE(test_tensor_static_rank,
    TestTupleType,
    boost::numeric::ublas::test_types,
    boost::numeric::ublas::tuple_fixture_tensor_static_rank<TestTupleType>
)
{
    namespace ublas = boost::numeric::ublas;
    using value_type = typename TestTupleType::first_type;
    using inner_t = inner_type_t<value_type>;
    using fixture_t = ublas::tuple_fixture_tensor_static_rank<TestTupleType>;

    auto const& self = static_cast<fixture_t const&>(*this);

    ublas::for_each_fixture(self, []<typename tensor_type>(auto /*id*/, tensor_type a){
        using vector_type  = typename tensor_type::vector_type;
        
        constexpr auto rank = a.rank();

        if constexpr (rank > 0ul) {
            
            a = value_type{2};

            BOOST_TEST_CONTEXT("[Static Rank Tensor Vector Product] tensor with rank("<< rank <<")"){
                
                for(auto m = 0u; m < rank; ++m){
                    auto const em = a.size(m);

                    auto b = vector_type  (em, value_type{1} );

                    auto c = ublas::prod(a, b, m+1);

                    for(auto i = 0u; i < c.size(); ++i)
                        BOOST_CHECK_EQUAL( c[i] , value_type( static_cast< inner_t >(em) ) * a[i] );

                }

            }
        }

    });


}

// TODO: Add product support for static tensor and enable this test
// BOOST_TEST_DECORATOR(
//     *boost::unit_test::label("vector_prod")
//     *boost::unit_test::description("Testing the vector product for static tensor")
// )
// BOOST_FIXTURE_TEST_CASE_TEMPLATE(test_tensor_static,
//     TestTupleType,
//     boost::numeric::ublas::test_types,
//     boost::numeric::ublas::tuple_fixture_tensor_static<TestTupleType>
// )
// {
//     namespace ublas = boost::numeric::ublas;
//     using value_type = typename TestTupleType::first_type;
//     using inner_t = inner_type_t<value_type>;
//     using fixture_t = ublas::tuple_fixture_tensor_static<TestTupleType>;

//     auto const& self = static_cast<fixture_t const&>(*this);

//     ublas::for_each_fixture(self, []<typename tensor_type>(auto /*id*/, tensor_type a){
//         using vector_type  = typename tensor_type::vector_type;
        
//         constexpr auto rank = a.rank();

//         if constexpr (rank > 0ul) {
            
//             a = value_type{2};

//             BOOST_TEST_CONTEXT("[Static Tensor Vector Product] tensor with rank("<< rank <<")"){
                
//                 for(auto m = 0u; m < rank; ++m){
//                     auto const em = a.size(m);

//                     auto b = vector_type  (em, value_type{1} );

//                     auto c = ublas::prod(a, b, m+1);

//                     for(auto i = 0u; i < c.size(); ++i)
//                         BOOST_CHECK_EQUAL( c[i] , value_type( static_cast< inner_t >(em) ) * a[i] );

//                 }

//             }
//         }

//     });
// }


BOOST_AUTO_TEST_SUITE_END()
