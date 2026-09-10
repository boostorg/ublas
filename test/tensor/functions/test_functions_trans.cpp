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

BOOST_AUTO_TEST_SUITE(test_functions_transpose,
    * boost::unit_test::description("Validate Tensor Transpose")
)


constexpr auto compute_factorial(std::size_t const& p) noexcept{
    std::size_t f = 1ul;
    for(std::size_t i = 1u; i <= p; ++i)
        f *= i;
    return f;
}

constexpr auto inverse(auto const& pi) noexcept{
    auto pi_inv = pi;
    for(auto j = 0u; j < pi.size(); ++j)
        pi_inv[pi[j]-1] = j+1;
    return pi_inv;
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
    *boost::unit_test::label("tensor_trans")
    *boost::unit_test::description("Testing the tensor transpose for dynamic tensor")
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

        BOOST_TEST_CONTEXT("[Dynamic Tensor Transpose] tensor with rank("<< rank <<")"){
            
            ublas::iota(a,value_type{0});
            auto const p = rank;
            auto aref = a;

            auto pi = std::vector<std::size_t>(p);
            std::iota(pi.begin(), pi.end(), 1);
            
            a = ublas::trans( a, pi );
            BOOST_CHECK_EQUAL_COLLECTIONS(a.begin(), a.end(), aref.begin(), aref.end());


            auto const factorial = compute_factorial(p);
            auto i = 0ul;
            
            for(; i < factorial - 1; ++i) {
                std::next_permutation(pi.begin(), pi.end());
                a = ublas::trans( a, pi );
            }
            std::next_permutation(pi.begin(), pi.end());
            
            for(; i > 0; --i) {
                std::prev_permutation(pi.begin(), pi.end());
                auto pi_inv = inverse(pi);
                a = ublas::trans( a, pi_inv );
            }

            BOOST_CHECK_EQUAL_COLLECTIONS(a.begin(), a.end(), aref.begin(), aref.end());

        }
    });


}

BOOST_TEST_DECORATOR(
    *boost::unit_test::label("tensor_trans")
    *boost::unit_test::description("Testing the tensor transpose for static rank tensor")
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

    ublas::for_each_fixture(self, [](auto /*id*/, auto a){
        constexpr auto rank = a.rank();
        
        if constexpr (rank > 1ul){

            BOOST_TEST_CONTEXT("[Static Rank Tensor Transpose] tensor with rank("<< rank <<")"){
                
                ublas::iota(a,value_type{0});
                constexpr auto p = rank;
                auto aref = a;

                auto pi = std::array<std::size_t,p>();
                std::iota(pi.begin(), pi.end(), 1);
                
                a = ublas::trans( a, pi );
                BOOST_CHECK_EQUAL_COLLECTIONS(a.begin(), a.end(), aref.begin(), aref.end());


                auto const factorial = compute_factorial(p);
                auto i = 0ul;
                
                for(; i < factorial - 1; ++i) {
                    std::next_permutation(pi.begin(), pi.end());
                    a = ublas::trans( a, pi );
                }

                std::next_permutation(pi.begin(), pi.end());
                
                for(; i > 0; --i) {
                    std::prev_permutation(pi.begin(), pi.end());
                    auto pi_inv = inverse(pi);
                    a = ublas::trans( a, pi_inv );
                }

                BOOST_CHECK_EQUAL_COLLECTIONS(a.begin(), a.end(), aref.begin(), aref.end());

            }
        }

    });


}

BOOST_AUTO_TEST_SUITE_END()
