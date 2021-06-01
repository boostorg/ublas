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
#include <sstream>

BOOST_AUTO_TEST_SUITE(test_multiplication_outer, 
    *boost::unit_test::description("Validate Outer Product")
)

BOOST_TEST_DECORATOR(
    *boost::unit_test::label("inner_product")
    *boost::unit_test::description("Testing outer product for dynamic tensor")
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

    auto const& self = static_cast<fixture_type const&>(*this);

    ublas::for_each_fixture(self, [&self]<typename extents_type>(auto /*id*/, extents_type const& na){
        
        using base_t = typename extents_type::base_type;

        auto const a_rank = ublas::size(na);
        auto const ap = ublas::product(na);

        if(a_rank < 2ul) return;

        auto a = vector_t(ap, value_type{2});
        auto wa = ublas::to_strides(na,layout_type{});

        ublas::for_each_fixture(self, [&](auto /*id*/, extents_type const& nb) {
            auto const b_rank = ublas::size(nb);
            auto const bp = ublas::product(nb);

            if(b_rank < 2ul) return;

            auto b = vector_t(bp, value_type{3});
            auto wb = ublas::to_strides(nb,layout_type{});

            BOOST_TEST_CONTEXT("[Outer Product Dynamic Tensor] testing for outer rank(" << a_rank << ") : inner rank(" << b_rank << ")"){

                auto const c_rank = a_rank + b_rank;
                auto nc_base = base_t(c_rank);

                auto it = std::copy(ublas::begin(na), ublas::end(na), nc_base.begin());
                std::copy(ublas::begin(nb), ublas::end(nb), it);

                auto nc = extents_type(std::move(nc_base));
                auto const cp = ublas::product(nc);
                auto c = vector_t(cp);
                auto wc = ublas::to_strides(nc,layout_type{});
                
                ublas::outer(c.data(), c_rank, nc.data(), wc.data(),
                             a.data(), a_rank, na.data(), wa.data(),
                             b.data(), b_rank, nb.data(), wb.data());
                for(auto const& cc : c)
                    BOOST_REQUIRE_EQUAL( cc , a[0]*b[0] );
            }
        });

    });
}

BOOST_TEST_DECORATOR(
    *boost::unit_test::label("inner_product")
    *boost::unit_test::description("Testing outer product for static rank tensor")
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

    auto const& self = static_cast<fixture_type const&>(*this);

    ublas::for_each_fixture(self, [&self]<typename outer_extents_type>(auto /*id*/, outer_extents_type const& na){
        
        using extents_value_type = typename outer_extents_type::value_type;

        static constexpr auto a_rank = std::tuple_size_v<outer_extents_type>;
        auto const ap = ublas::product(na);

        if constexpr(a_rank >= 2ul){
            auto a = vector_t(ap, value_type{2});
            auto wa = ublas::to_strides(na,layout_type{});

            ublas::for_each_fixture(self, [&a, &wa, &na]<typename inner_extents_type>(auto /*id*/, inner_extents_type const& nb) {
                constexpr auto b_rank = std::tuple_size_v<inner_extents_type>;
                auto const bp = ublas::product(nb);

                if constexpr(b_rank >= 2ul){
                    auto b = vector_t(bp, value_type{3});
                    auto wb = ublas::to_strides(nb,layout_type{});

                    BOOST_TEST_CONTEXT("[Outer Product Statc Rank Tensor] testing for outer rank(" << a_rank << ") : inner rank(" << b_rank << ")"){

                        constexpr auto const c_rank = a_rank + b_rank;
                        using nc_type = ublas::extents_core<extents_value_type, c_rank>;
                        auto nc_base = typename nc_type::base_type();

                        auto it = std::copy(ublas::begin(na), ublas::end(na), nc_base.begin());
                        std::copy(ublas::begin(nb), ublas::end(nb), it);

                        auto nc = nc_type(nc_base);
                        auto const cp = ublas::product(nc);
                        auto c = vector_t(cp);
                        auto wc = ublas::to_strides(nc,layout_type{});
                        
                        ublas::outer(c.data(), c_rank, nc.data(), wc.data(),
                                    a.data(), a_rank, na.data(), wa.data(),
                                    b.data(), b_rank, nb.data(), wb.data());
                        for(auto const& cc : c)
                            BOOST_REQUIRE_EQUAL( cc , a[0]*b[0] );
                    }
                }

            });
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
    *boost::unit_test::label("inner_product")
    *boost::unit_test::description("Testing outer product for static tensor")
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

    auto const& self = static_cast<fixture_type const&>(*this);

    ublas::for_each_fixture(self, [&self]<typename outer_extents_type>(auto /*id*/, outer_extents_type const& na){

        static constexpr auto a_rank = ublas::size_v<outer_extents_type>;
        static constexpr auto ap = ublas::product_v<outer_extents_type>;

        if constexpr(a_rank >= 2ul){
            auto a = std::array<value_type, ap>();
            std::fill(a.begin(), a.end(), value_type{2});

            auto wa = get_strides<layout_type>(ublas::to_array_v<outer_extents_type>);

            ublas::for_each_fixture(self, [&a, &wa, &na]<typename inner_extents_type>(auto /*id*/, inner_extents_type const& nb) {
                constexpr auto b_rank = ublas::size_v<inner_extents_type>;
                constexpr auto bp = ublas::product_v<inner_extents_type>;

                if constexpr(b_rank >= 2ul){
                    auto b = std::array<value_type, bp>();
                    std::fill(b.begin(), b.end(), value_type{3});

                    auto wb = get_strides<layout_type>(ublas::to_array_v<inner_extents_type>);

                    BOOST_TEST_CONTEXT("[Outer Product Statc Rank Tensor] testing for outer rank(" << a_rank << ") : inner rank(" << b_rank << ")"){

                        constexpr auto const c_rank = a_rank + b_rank;
                        using nc_type = ublas::cat_t<outer_extents_type, inner_extents_type>;

                        constexpr auto cp = ublas::product_v<nc_type>;
                        auto c = std::array<value_type, cp>();
                        constexpr auto nc = ublas::to_array_v<nc_type>;
                        auto wc = get_strides<layout_type>(nc);
                        
                        ublas::outer(c.data(), c_rank, nc.data(), wc.data(),
                                     a.data(), a_rank, na.data(), wa.data(),
                                     b.data(), b_rank, nb.data(), wb.data());
                        for(auto const& cc : c)
                            BOOST_REQUIRE_EQUAL( cc , a[0]*b[0] );
                    }
                }

            });
        }


    });
}

BOOST_AUTO_TEST_SUITE_END()
