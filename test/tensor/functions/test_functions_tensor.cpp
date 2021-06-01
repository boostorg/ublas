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

BOOST_AUTO_TEST_SUITE(test_functions_tensor,
    * boost::unit_test::description("Validate Tensor Product")
)

BOOST_TEST_DECORATOR(
    *boost::unit_test::label("tensor_prod")
    *boost::unit_test::description("Testing the tensor product for dynamic tensor")
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
        auto const rank = a.rank();
        
        if(rank <= 1ul) return;
        
        auto b = tensor_type{a.extents(), value_type{3}};

        a = value_type{2};

        BOOST_TEST_CONTEXT("[Dynamic Tensor Product] tensor with rank("<< rank <<")"){

            // the number of contractions is changed.
            for( auto q = 0ul; q <= rank; ++q) { // rank

                auto phi = std::vector<std::size_t> ( q );

                std::iota(phi.begin(), phi.end(), 1ul);

                auto c = ublas::prod(a, b, phi);

                auto acc = value_type{1};
                for(auto i = 0ul; i < q; ++i)
                    acc *= value_type{ static_cast< inner_t >( a.size(phi.at(i)-1) ) };

                for(auto i = 0ul; i < c.size(); ++i)
                    BOOST_CHECK_EQUAL( c[i] , acc * a[0] * b[0] );

            }

        }
    });


}

BOOST_TEST_DECORATOR(
    *boost::unit_test::label("tensor_prod")
    *boost::unit_test::description("Testing the tensor product for static rank tensor")
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
        
        static constexpr auto rank = a.rank();

        if constexpr (rank > 1ul) {
        
            auto b = tensor_type(a.extents());
            a = value_type{2};
            b = value_type{3};

            BOOST_TEST_CONTEXT("[Static Rank Tensor Product] tensor with rank("<< rank <<")"){
                
                static_for_each<rank>([&]<typename IType>(IType /*id*/){
                    constexpr auto q = IType::value;

                    auto phi = std::array<std::size_t, q> ();

                    std::iota(phi.begin(), phi.end(), 1ul);

                    auto c = ublas::prod(a, b, phi);

                    auto acc = value_type{1};
                    for(auto i = 0ul; i < q; ++i)
                        acc *= value_type{ static_cast< inner_t >( a.size(phi.at(i)-1) ) };

                    for(auto i = 0ul; i < c.size(); ++i)
                        BOOST_CHECK_EQUAL( c[i] , acc * a[0] * b[0] );
                        
                });

            }
        }

    });


}

BOOST_AUTO_TEST_SUITE_END()
