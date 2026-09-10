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

BOOST_AUTO_TEST_SUITE(test_multiplication_mtm, 
    *boost::unit_test::description("Validate Matrix Times Matrix")
)


BOOST_TEST_DECORATOR(
    *boost::unit_test::label("mtm")
    *boost::unit_test::description("Testing mtm for dynamic tensor")
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

        if(rank != 2ul) return;

        BOOST_TEST_CONTEXT("[MTM Dynamic Tensor] testing for m(" << na[0] << "), n(" << na[1] <<")"){
            auto a  = vector_t  (p, value_type{2});
            auto wa = ublas::to_strides(na,layout_type{});

            auto nb = extents_type {na[1],na[0]};
            auto wb = ublas::to_strides(nb,layout_type{});
            auto b  = vector_t  (ublas::product(nb), value_type{1} );

            auto nc = extents_type {na[0],nb[1]};
            auto wc = ublas::to_strides(nc,layout_type{});
            auto c  = vector_t  (ublas::product(nc));


            ublas::detail::recursive::mtm(
                c.data(), nc.data(), wc.data(),
                a.data(), na.data(), wa.data(),
                b.data(), nb.data(), wb.data()
            );

            auto v = value_type{ static_cast<inner_t>(na[1]) }*a[0];
            BOOST_CHECK(std::all_of(c.begin(),c.end(), [v](auto cc){return cc == v;}));
        }
    });
}

BOOST_TEST_DECORATOR(
    *boost::unit_test::label("mtm")
    *boost::unit_test::description("Testing mtm for static rank tensor")
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

        if constexpr (rank == 2ul){

            BOOST_TEST_CONTEXT("[MTM Static Rank Tensor] testing for m(" << na[0] << "), n(" << na[1] <<")"){
                auto a  = vector_t  (p, value_type{2});
                auto wa = ublas::to_strides(na,layout_type{});

                auto nb = extents_type {na[1],na[0]};
                auto wb = ublas::to_strides(nb,layout_type{});
                auto b  = vector_t  (ublas::product(nb), value_type{1} );

                auto nc = extents_type {na[0],nb[1]};
                auto wc = ublas::to_strides(nc,layout_type{});
                auto c  = vector_t  (ublas::product(nc));


                ublas::detail::recursive::mtm(
                    c.data(), nc.data(), wc.data(),
                    a.data(), na.data(), wa.data(),
                    b.data(), nb.data(), wb.data()
                );

                auto v = value_type{ static_cast<inner_t>(na[1]) }*a[0];
                BOOST_CHECK(std::all_of(c.begin(),c.end(), [v](auto cc){return cc == v;}));
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
    *boost::unit_test::label("mtm")
    *boost::unit_test::description("Testing mtm for static tensor")
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
        constexpr auto rank = ublas::size_v<extents_type>;
        constexpr auto const p = ublas::product_v<extents_type>;

        if constexpr (rank == 2ul){
            
            constexpr auto na = ublas::to_array_v<extents_type>;
            
            BOOST_TEST_CONTEXT("[MTM Static Rank Tensor] testing for m(" << na[0] << "), n(" << na[1] <<"), k(" << na[1] <<")"){
                auto a  = std::array<value_type, p>();
                std::fill(a.begin(), a.end(), value_type{2});
                auto wa = get_strides<layout_type>(na);

                using nb_type = ublas::extents_core<extents_value_type, ublas::get_v<extents_type, 1ul>, ublas::get_v<extents_type, 0ul> >;
                auto nb = ublas::to_array_v<nb_type>;
                auto wb = get_strides<layout_type>(nb);
                auto b  = std::array<value_type, ublas::product_v<nb_type>>();
                std::fill(b.begin(), b.end(), value_type{1});

                using nc_type = ublas::extents_core<extents_value_type, ublas::get_v<extents_type, 0ul>, ublas::get_v<nb_type, 1ul> >;
                auto nc = ublas::to_array_v<nc_type>;
                auto wc = get_strides<layout_type>(nc);
                auto c  = std::array<value_type, ublas::product_v<nc_type>>();

                ublas::detail::recursive::mtm(
                    c.data(), nc.data(), wc.data(),
                    a.data(), na.data(), wa.data(),
                    b.data(), nb.data(), wb.data()
                );

                auto v = value_type{ static_cast<inner_t>(na[1]) }*a[0];
                BOOST_CHECK(std::all_of(c.begin(),c.end(), [v](auto cc){return cc == v;}));
            }

        }

    });
}

BOOST_AUTO_TEST_SUITE_END()