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

BOOST_AUTO_TEST_SUITE(test_multiplication_ttm, 
    *boost::unit_test::description("Validate Tensor Times Matrix")
)


BOOST_TEST_DECORATOR(
    *boost::unit_test::label("ttm")
    *boost::unit_test::description("Testing ttm for dynamic tensor")
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

        if (rank <= 1ul) return;

        BOOST_TEST_CONTEXT("[TTM Dynamic Tensor] testing for rank(" << rank << ")"){
            auto a = vector_t(p, value_type{2});
            auto wa = ublas::to_strides(na,layout_type{});
            for(auto m = std::size_t{0}; m < rank; ++m){
                const auto nb = extents_type {na[m], na[m] };
                const auto b  = vector_t  (ublas::product(nb), value_type{1} );
                const auto wb = ublas::to_strides(nb,layout_type{});

                const auto& nc = na;
                const auto wc = ublas::to_strides(nc,layout_type{});
                auto c  = vector_t  (ublas::product(nc), value_type{0});

                ublas::ttm(m+1, rank,
                            c.data(), nc.data(), wc.data(),
                            a.data(), na.data(), wa.data(),
                            b.data(), nb.data(), wb.data());


                auto v = value_type{ static_cast<inner_t>(na[m]) };
                BOOST_CHECK(std::equal(c.begin(),c.end(),a.begin(), [v](auto cc, auto aa){return cc == v*aa;}));
            }
        }
    });
}

BOOST_TEST_DECORATOR(
    *boost::unit_test::label("ttm")
    *boost::unit_test::description("Testing ttm for static rank tensor")
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
        constexpr auto rank = std::tuple_size_v<extents_type>;
        auto const p = ublas::product(na);
        using extents_value_type = typename extents_type::value_type;

        if constexpr (rank > 1ul){

            BOOST_TEST_CONTEXT("[TTM Static Rank Tensor] testing for rank(" << rank << ")"){
                auto a = vector_t(p, value_type{2});
                auto wa = ublas::to_strides(na,layout_type{});
                for(auto m = std::size_t{0}; m < rank; ++m){
                    using nb_type = ublas::extents_core< extents_value_type, 2ul >;
                    const auto nb = nb_type{na[m], na[m]};
                    const auto b  = vector_t(ublas::product(nb), value_type{1} );
                    const auto wb = ublas::to_strides(nb,layout_type{});

                    const auto& nc = na;
                    const auto wc = ublas::to_strides(nc,layout_type{});
                    auto c  = vector_t(ublas::product(nc), value_type{0});

                    ublas::ttm(m+1, rank,
                                c.data(), nc.data(), wc.data(),
                                a.data(), na.data(), wa.data(),
                                b.data(), nb.data(), wb.data());


                    auto v = value_type{ static_cast<inner_t>(na[m]) };
                    BOOST_CHECK(std::equal(c.begin(),c.end(),a.begin(), [v](auto cc, auto aa){return cc == v*aa;}));
                }
            }
        }

    });
}

// FIXME: temp fix to the invalid computation of static strides,
// rempve this after the fix
template<typename L, typename T, std::size_t N>
constexpr auto get_strides(std::array<T,N> const& temp) noexcept{
    namespace ublas = boost::numeric::ublas;
    using extents_type = ublas::extents_core<T,N>;
    auto n = extents_type{temp};
    return ublas::to_strides(n, L{});
}

BOOST_TEST_DECORATOR(
    *boost::unit_test::label("ttm")
    *boost::unit_test::description("Testing ttm for static tensor")
)
BOOST_FIXTURE_TEST_CASE_TEMPLATE(test_extents_static,
    TestTupleType,
    boost::numeric::ublas::test_types,
    boost::numeric::ublas::fixture_extents_static<std::size_t>
){
    namespace ublas = boost::numeric::ublas;
    using value_type = typename TestTupleType::first_type;
    using layout_type = typename TestTupleType::second_type;
    using fixture_type = ublas::fixture_extents_static<std::size_t>;
    using inner_t = inner_type_t<value_type>;

    auto const& self = static_cast<fixture_type const&>(*this);

    ublas::for_each_fixture(self, []<typename extents_type>(auto /*id*/, extents_type const& /*na*/){
        using extents_value_type = typename extents_type::value_type;
        static constexpr auto rank = ublas::size_v<extents_type>;
        static constexpr auto p = ublas::product_v<extents_type>;

        if constexpr (rank > 1ul){
            
            BOOST_TEST_CONTEXT("[TTM Static Rank Tensor] testing for rank(" << rank << ")"){
                
                auto na = ublas::to_array_v<extents_type>;

                auto a  = std::array<value_type, p>();
                std::fill(a.begin(), a.end(), value_type{2});
                auto wa = get_strides<layout_type>(na);
                
                static_for_each<rank>([&a, &wa]<typename IType>(IType /*id*/){
                    constexpr auto na = ublas::to_array_v<extents_type>;
                    constexpr auto m = IType::value;

                    using nb_type = ublas::extents_core<extents_value_type, na[m], na[m] >;
                    auto nb = ublas::to_array_v<nb_type>;
                    auto wb = get_strides<layout_type>(nb);
                    auto b  = std::array<value_type, ublas::product_v<nb_type>>();
                    std::fill(b.begin(), b.end(), value_type{1});

                    auto nc = na;
                    auto wc = wa;;
                    auto c = std::array<value_type, ublas::product_v<extents_type>>();
                    
                    ublas::ttm(m+1, rank,
                                c.data(), nc.data(), wc.data(),
                                a.data(), na.data(), wa.data(),
                                b.data(), nb.data(), wb.data());

                    auto v = value_type{ static_cast<inner_t>(na[m]) };
                    BOOST_CHECK(std::equal(c.begin(),c.end(),a.begin(), [v](auto cc, auto aa){return cc == v*aa;}));
                });
            }

        }

    });
}

BOOST_AUTO_TEST_SUITE_END()