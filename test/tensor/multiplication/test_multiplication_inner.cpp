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

BOOST_AUTO_TEST_SUITE(test_multiplication_inner, 
    *boost::unit_test::description("Validate Inner Product")
)


BOOST_TEST_DECORATOR(
    *boost::unit_test::label("inner_product")
    *boost::unit_test::description("Testing inner product for dynamic tensor")
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

    ublas::for_each_fixture(self, []<typename extents_type>(auto /*id*/, extents_type const& n){
        auto const rank = ublas::size(n);
        auto const p = ublas::product(n);

        if(rank < 2ul) return;

        BOOST_TEST_CONTEXT("[Inner Product Dynamic Tensor] testing for rank(" << rank << ")"){
            auto a = vector_t(p, value_type{2});
            auto b = vector_t(p, value_type{3});
            auto w = ublas::to_strides(n,layout_type{});

            auto c = ublas::inner(rank, n.data(), a.data(), w.data(), b.data(), w.data(), value_type(0));
            auto cref = std::inner_product(a.begin(), a.end(), b.begin(), value_type(0));

            BOOST_CHECK_EQUAL( c , cref );
        }
    });
}

BOOST_TEST_DECORATOR(
    *boost::unit_test::label("inner_product")
    *boost::unit_test::description("Testing inner product for static rank tensor")
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

    ublas::for_each_fixture(self, []<typename extents_type>(auto /*id*/, extents_type const& n){
        constexpr auto rank = std::tuple_size_v<extents_type>;
        auto const p = ublas::product(n);

        if constexpr(rank >= 2ul){
            BOOST_TEST_CONTEXT("[Inner Product Static Rank Tensor] testing for rank(" << rank << ")"){
                auto a = vector_t(p, value_type{2});
                auto b = vector_t(p, value_type{3});
                auto w = ublas::to_strides(n,layout_type{});

                auto c = ublas::inner(rank, n.data(), a.data(), w.data(), b.data(), w.data(), value_type(0));
                auto cref = std::inner_product(a.begin(), a.end(), b.begin(), value_type(0));

                BOOST_CHECK_EQUAL( c , cref );
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
    *boost::unit_test::label("inner_product")
    *boost::unit_test::description("Testing inner product for static tensor")
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

    ublas::for_each_fixture(self, []<typename extents_type>(auto /*id*/, extents_type const& n){
        constexpr auto rank = ublas::size_v<extents_type>;
        constexpr auto p = ublas::product_v<extents_type>;

        if constexpr(rank >= 2ul){
            BOOST_TEST_CONTEXT("[Inner Product Static Tensor] testing for rank(" << rank << ")"){
                auto a = std::array<value_type, p>();
                std::fill(a.begin(), a.end(), value_type{2});
                auto b = std::array<value_type, p>();
                std::fill(a.begin(), a.end(), value_type{3});
                
                // FIXME: remove after the fix
                auto w = get_strides<layout_type>(ublas::to_array_v<extents_type>);

                auto c = ublas::inner(rank, n.data(), a.data(), w.data(), b.data(), w.data(), value_type(0));
                auto cref = std::inner_product(a.begin(), a.end(), b.begin(), value_type(0));

                BOOST_CHECK_EQUAL( c , cref );
            }
        }

    });
}

BOOST_AUTO_TEST_SUITE_END()
