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

BOOST_AUTO_TEST_SUITE(test_functions_tensor_permutation,
    * boost::unit_test::description("Validate Tensor Product with Permutation")
)


constexpr auto compute_factorial(std::size_t const& p) noexcept{
    std::size_t f = 1ul;
    for(std::size_t i = 1u; i <= p; ++i)
        f *= i;
    return f;
}

constexpr auto permute_extents(auto const& pi, auto const& na){
    namespace ublas = boost::numeric::ublas;
    using extents_type = std::decay_t< decltype(na) >;

    auto nb_base = na.base();
    assert(pi.size() == ublas::size(na));
    for(auto j = 0u; j < pi.size(); ++j)
        nb_base[pi[j]-1] = na[j];
    return extents_type(nb_base);
};

BOOST_TEST_DECORATOR(
    *boost::unit_test::label("tensor_prod_permutation")
    *boost::unit_test::description("Testing the tensor product with permutation for dynamic tensor")
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
        auto const pa = a.rank();
        
        if(pa <= 1ul) return;

        BOOST_TEST_CONTEXT("[Dynamic Tensor Product Permutation] tensor with rank("<< pa <<")"){
            
            auto const& na = a.extents();
            auto pi   = std::vector<std::size_t>(pa);
            auto fac = compute_factorial(pa);
            std::iota( pi.begin(), pi.end(), 1ul );

            for(auto f = 0ul; f < fac; ++f){

                auto nb = permute_extents( pi, na  );
                auto b  = tensor_type( nb, value_type{3} );

                // the number of contractions is changed.
                for( auto q = 0ul; q <= pa; ++q) { // pa

                    auto phia = std::vector<std::size_t> ( q );  // concatenation for a
                    auto phib = std::vector<std::size_t> ( q );  // concatenation for b

                    std::iota(phia.begin(), phia.end(), 1ul);
                    std::transform(  phia.begin(), phia.end(), phib.begin(),
                                    [&pi] ( std::size_t i ) { return pi.at(i-1); } );

                    auto c = ublas::prod(a, b, phia, phib);

                    auto acc = value_type{1};
                    for(auto i = 0ul; i < q; ++i)
                        acc *= value_type( static_cast< inner_t >( na.at(phia.at(i)-1) ) );

                    for(auto i = 0ul; i < c.size(); ++i)
                        BOOST_CHECK_EQUAL( c[i] , acc * a[0] * b[0] );

                }

                std::next_permutation(pi.begin(), pi.end());
            }

        }
    });


}

BOOST_TEST_DECORATOR(
    *boost::unit_test::label("tensor_prod_permutation")
    *boost::unit_test::description("Testing the tensor product with permutation for static rank tensor")
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
        
        constexpr auto rank = a.rank();

        if constexpr (rank > 1ul) {
        
            BOOST_TEST_CONTEXT("[Static Rank Tensor Product Permutation] tensor with rank("<< rank <<")"){
                auto const& na = a.extents();
                auto pi  = std::array<std::size_t,rank>();
                auto fac = compute_factorial(rank);
                std::iota( pi.begin(), pi.end(), 1ul );

                for(auto f = 0ul; f < fac; ++f){

                    auto nb = permute_extents( pi, na  );
                    auto b  = tensor_type(nb);
                    b = value_type{3};

                    // the number of contractions is changed.
                    static_for_each<rank>([&]<typename IType>(IType /*id*/){
                        constexpr auto q = IType::value;
                        
                        auto phia = std::array<std::size_t,q> ();  // concatenation for a
                        auto phib = std::array<std::size_t,q> ();  // concatenation for b

                        std::iota(phia.begin(), phia.end(), 1ul);
                        std::transform(  phia.begin(), phia.end(), phib.begin(),
                                        [&pi] ( std::size_t i ) { return pi.at(i-1); } );

                        auto c = ublas::prod(a, b, phia, phib);

                        auto acc = value_type{1};
                        for(auto i = 0ul; i < q; ++i)
                            acc *= value_type( static_cast< inner_t >( na.at(phia.at(i)-1) ) );

                        for(auto i = 0ul; i < c.size(); ++i)
                            BOOST_CHECK_EQUAL( c[i] , acc * a[0] * b[0] );
                            
                    });
                }

                std::next_permutation(pi.begin(), pi.end());
            }
        }

    });


}

BOOST_AUTO_TEST_SUITE_END()
