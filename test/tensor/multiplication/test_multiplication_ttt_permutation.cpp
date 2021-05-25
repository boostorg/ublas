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
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/vector.hpp>

BOOST_AUTO_TEST_SUITE(test_tensor_ttt_permutation, 
    *boost::unit_test::description("Validate Tensor Times Tensor With Permuation")
)


constexpr auto compute_factorial(std::size_t const& p) noexcept{
    std::size_t f = 1ul;
    for(std::size_t i = 1u; i <= p; ++i)
        f *= i;
    return f;
}


constexpr auto compute_inverse_permutation(auto const& pi) noexcept{
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
        nb_base[j] = na[pi[j]-1];
    return extents_type(nb_base);
};

constexpr auto calculate_nc(auto& out, auto const& pia, auto const& pib_inv, auto const& na, auto const& nb, std::size_t r, std::size_t s){
    for(auto j = 0u; j < r; ++j)
        out[j] = na[pia[j]-1];

    for(auto j = 0u; j < s; ++j)
        out[r+j] = nb[ pib_inv[j]-1 ];
};

BOOST_TEST_DECORATOR(
    *boost::unit_test::label("ttt")
    *boost::unit_test::description("Testing ttt for dynamic tensor with permutation")
)
BOOST_FIXTURE_TEST_CASE_TEMPLATE(test_extents_dynamic,
    TestTupleType,
    boost::numeric::ublas::test_types,
    boost::numeric::ublas::fixture_extents_dynamic<std::size_t>
){
    namespace ublas = boost::numeric::ublas;
    using value_type = typename TestTupleType::first_type;
    using layout_type = typename TestTupleType::second_type;
    using fixture_type = ublas::fixture_extents_dynamic<std::size_t>;
    using vector_t  = std::vector<value_type>;
    using inner_t = inner_type_t<value_type>;

    auto const& self = static_cast<fixture_type const&>(*this);

    ublas::for_each_fixture(self, []<typename extents_type>(auto /*id*/, extents_type const& na){
        auto const rank = ublas::size(na);
        auto const p = ublas::product(na);
        using extents_base_t = typename extents_type::base_type;

        if(rank <= 1ul) return;

        constexpr auto one = std::size_t{1};
        constexpr auto two = std::size_t{2};

        BOOST_TEST_CONTEXT("[Permutated TTT Dynamic Tensor] testing for rank(" << rank << ")"){
            auto wa = ublas::to_strides(na,layout_type{});
            auto a  = vector_t(p, value_type{2});
            auto pa  = rank;
            auto pia = std::vector<std::size_t>(pa);
            std::iota( pia.begin(), pia.end(), one );

            auto pib     = pia;
            auto pib_inv = compute_inverse_permutation(pib);

            auto f = compute_factorial(pa);

            // for the number of possible permutations
            // only permutation tuple pib is changed.
            for(auto i = 0u; i < f; ++i) {

                auto nb = permute_extents( pib, na  );
                auto wb = ublas::to_strides(nb,layout_type{});
                auto b  = vector_t(ublas::product(nb), value_type{3});
                auto pb = ublas::size(nb);

                // the number of contractions is changed.
                for(auto q = std::size_t{0}; q <= pa; ++q) {

                    auto r  = pa - q;
                    auto s  = pb - q;

                    auto pc = r+s > 0 ? std::max(std::size_t{r+s},two) : two;

                    auto nc_base = extents_base_t(pc,one);
                    
                    calculate_nc(nc_base, pia, pib_inv, na, nb, r, s);

                    auto nc = extents_type ( nc_base );
                    auto wc = ublas::to_strides(nc,layout_type{});
                    auto c  = vector_t  ( ublas::product(nc), value_type(0) );

                    ublas::ttt(pa,pb,q,
                            pia.data(), pib_inv.data(),
                            c.data(), nc.data(), wc.data(),
                            a.data(), na.data(), wa.data(),
                            b.data(), nb.data(), wb.data());


                    auto acc = one;
                    for(auto j = r; j < pa; ++j)
                        acc *= na[pia[j]-1];

                    auto v = value_type{ static_cast<inner_t>(acc) } * a[0] * b[0];

                    BOOST_CHECK( std::all_of(c.begin(),c.end(), [v](auto cc){return cc == v; } ) );

                }

                std::next_permutation(pib.begin(), pib.end());
                pib_inv = compute_inverse_permutation(pib);
            }
        }
    });
}

BOOST_TEST_DECORATOR(
    *boost::unit_test::label("ttt")
    *boost::unit_test::description("Testing ttt for static rank tensor with permutation")
)
BOOST_FIXTURE_TEST_CASE_TEMPLATE(test_extents_static_rank,
    TestTupleType,
    boost::numeric::ublas::test_types,
    boost::numeric::ublas::fixture_extents_static_rank<std::size_t>
){
    namespace ublas = boost::numeric::ublas;
    using value_type = typename TestTupleType::first_type;
    using layout_type = typename TestTupleType::second_type;
    using fixture_type = ublas::fixture_extents_static_rank<std::size_t>;
    using vector_t  = std::vector<value_type>;
    using inner_t = inner_type_t<value_type>;

    auto const& self = static_cast<fixture_type const&>(*this);

    ublas::for_each_fixture(self, []<typename extents_type>(auto /*id*/, extents_type const& na){
        constexpr auto const rank = std::tuple_size_v<extents_type>;
        auto const p = ublas::product(na);
        using extents_value_type = typename extents_type::value_type;

        if constexpr(rank > 1ul){

            static constexpr auto one = std::size_t{1};
            static constexpr auto two = std::size_t{2};

            BOOST_TEST_CONTEXT("[Permutated TTT Static Rank Tensor] testing for rank(" << rank << ")"){
                auto wa = ublas::to_strides(na,layout_type{});
                auto a  = vector_t(p, value_type{2});
                static constexpr auto pa  = rank;
                auto pia = std::vector<std::size_t>(pa);
                std::iota( pia.begin(), pia.end(), one );

                auto pib     = pia;
                auto pib_inv = compute_inverse_permutation(pib);

                auto f = compute_factorial(pa);

                // for the number of possible permutations
                // only permutation tuple pib is changed.
                for(auto i = 0u; i < f; ++i) {

                    auto nb = permute_extents( pib, na  );
                    auto wb = ublas::to_strides(nb,layout_type{});
                    auto b  = vector_t(ublas::product(nb), value_type{3});
                    static constexpr auto pb = std::tuple_size_v<decltype(nb)>;

                    // the number of contractions is changed.
                    static_for_each<pa>([&a, &pia, &pib_inv, &nb, &wb, &b, &na, &wa]<typename IType>(IType /*id*/){
                        constexpr auto q = IType::value;
                        constexpr auto r  = pa - q;
                        constexpr auto s  = pb - q;

                        constexpr auto pc = r+s > 0ul ? std::max(std::size_t{r+s},two) : two;
                        using nc_type = ublas::extents_core< extents_value_type, pc >;
                        using nc_base_type = typename nc_type::base_type;
                        auto nc_base = nc_base_type();
                        std::fill(nc_base.begin(), nc_base.end(), one);
                        
                        calculate_nc(nc_base, pia, pib_inv, na, nb, r, s);

                        auto nc = nc_type ( nc_base );
                        auto wc = ublas::to_strides(nc,layout_type{});
                        auto c  = vector_t  ( ublas::product(nc), value_type{0} );

                        ublas::ttt(pa,pb,q,
                                pia.data(), pib_inv.data(),
                                c.data(), nc.data(), wc.data(),
                                a.data(), na.data(), wa.data(),
                                b.data(), nb.data(), wb.data());


                        auto acc = one;
                        for(auto j = r; j < pa; ++j)
                            acc *= na[pia[j]-1];

                        auto v = value_type{ static_cast<inner_t>(acc) } * a[0] * b[0];

                        BOOST_CHECK( std::all_of(c.begin(),c.end(), [v](auto cc){return cc == v; } ) );
                    });

                    std::next_permutation(pib.begin(), pib.end());
                    pib_inv = compute_inverse_permutation(pib);
                }
            }
        }

    });
}

BOOST_AUTO_TEST_SUITE_END()