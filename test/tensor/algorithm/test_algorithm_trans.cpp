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

BOOST_AUTO_TEST_SUITE(test_algorithm_trans, 
    *boost::unit_test::description("Validate Transpose Algorithm")
    *boost::unit_test::depends_on("test_algorithm_copy")
)

BOOST_TEST_DECORATOR(
    *boost::unit_test::label("transpose_algorithm")
    *boost::unit_test::description("Testing transpose algorithm using dynamic extents")
)
BOOST_FIXTURE_TEST_CASE_TEMPLATE(test_extents_dynamic,
    TestTupleType,
    boost::numeric::ublas::test_types,
    boost::numeric::ublas::fixture_extents_dynamic<std::size_t>
){
    namespace ublas = boost::numeric::ublas;
    using fixture_t = ublas::fixture_extents_dynamic<std::size_t>;
    using value_type = typename TestTupleType::first_type;
    using layout_type = typename TestTupleType::second_type;
    using vector_t = std::vector<value_type>;
    using permutation_type = std::vector<std::size_t>;

    auto const& self = static_cast<fixture_t const&>(*this);
    ublas::for_each_fixture(self, []<typename extents_type>(auto /*id*/, extents_type const& n){
        using base_t = typename extents_type::base_type;

        auto const rank = ublas::size(n);
        auto const s = ublas::product(n);

        // FIXME: enbale tests for rank one after the enhancement #120
        if(rank < 2ul)
            return;

        BOOST_TEST_CONTEXT("[Transpose Algorithm] rank("<< rank <<") dynamic extents"){
            auto pi  = permutation_type(rank);
            auto a   = vector_t(s);
            auto b1  = vector_t(s);
            auto b2  = vector_t(s);
            auto c1  = vector_t(s);
            auto c2  = vector_t(s);

            auto wa = ublas::to_strides(n,layout_type{});

            ublas::iota(a, value_type{});

            // so wie last-order.
            std::iota(pi.rbegin(), pi.rend(), 1ul);

            auto nc_base = base_t(rank);
            for(auto i = 0u; i < rank; ++i)
                nc_base[pi[i]-1] = n[i];

            auto nc = extents_type(std::move(nc_base));

            auto wc    = ublas::to_strides(nc,layout_type{});
            auto wc_pi = base_t(rank);
            for(auto i = 0u; i < rank; ++i)
                wc_pi[pi[i]-1] = wc[i];

            ublas::copy ( rank, n.data(),            c1.data(), wc_pi.data(), a.data(), wa.data());
            ublas::trans( rank, n.data(), pi.data(), c2.data(), wc.data(),    a.data(), wa.data() );

            BOOST_REQUIRE_EQUAL_COLLECTIONS(std::begin(c1), std::end(c1), std::begin(c2), std::end(c2));

            auto nb_base = base_t(rank);
            for(auto i = 0u; i < rank; ++i)
            nb_base[pi[i]-1] = nc[i];

            auto nb = extents_type(std::move(nb_base));

            auto wb    = ublas::to_strides(nb,layout_type{});
            auto wb_pi = base_t(rank);
            for(auto i = 0u; i < rank; ++i)
            wb_pi[pi[i]-1] = wb[i];

            ublas::copy ( rank, nc.data(),            b1.data(), wb_pi.data(), c1.data(), wc.data());
            ublas::trans( rank, nc.data(), pi.data(), b2.data(), wb.data(),    c2.data(), wc.data() );

            BOOST_CHECK_EQUAL_COLLECTIONS(std::begin(b1), std::end(b1), std::begin(b2), std::end(b2));
            BOOST_CHECK_EQUAL_COLLECTIONS(std::begin(a), std::end(a), std::begin(b2), std::end(b2));
        }

        BOOST_TEST_CONTEXT("[Transpose Algorithm(Exception)] rank("<< rank <<") dynamic extents"){
            auto pi = permutation_type(rank);
            auto a  = vector_t(s);
            auto c  = vector_t(s);
            auto wa = ublas::to_strides(n,layout_type{});
            auto wc = ublas::to_strides(n,layout_type{});
            value_type* data = nullptr;
            std::size_t* size = nullptr;
            BOOST_REQUIRE_THROW( ublas::trans( rank, size     , pi.data(), c.data(), wc.data(),    a.data(), wa.data() ), std::runtime_error );
            BOOST_REQUIRE_THROW( ublas::trans( rank, n.data() , size     , c.data(), wc.data(),    a.data(), wa.data() ), std::runtime_error );
            BOOST_REQUIRE_THROW( ublas::trans( rank, n.data() , pi.data(), c.data(), size     ,    a.data(), size      ), std::runtime_error );
            BOOST_REQUIRE_THROW( ublas::trans( rank, n.data() , pi.data(), c.data(), wc.data(),    a.data(), size      ), std::runtime_error );
            BOOST_REQUIRE_THROW( ublas::trans( rank, n.data() , pi.data(), c.data(), size     ,    a.data(), wa.data() ), std::runtime_error );
            BOOST_REQUIRE_THROW( ublas::trans( rank, n.data() , pi.data(), data    , wc.data(),    data    , wa.data() ), std::runtime_error );
            BOOST_REQUIRE_THROW( ublas::trans( rank, n.data() , pi.data(), c.data(), wc.data(),    data    , wa.data() ), std::runtime_error );
            BOOST_REQUIRE_THROW( ublas::trans( rank, n.data() , pi.data(), data    , wc.data(),    a.data(), wa.data() ), std::runtime_error );
        }
    });

}

BOOST_TEST_DECORATOR(
    *boost::unit_test::label("transpose_algorithm")
    *boost::unit_test::description("Testing transpose algorithm using static rank extents")
)
BOOST_FIXTURE_TEST_CASE_TEMPLATE(test_extents_static_rank,
    TestTupleType,
    boost::numeric::ublas::test_types,
    boost::numeric::ublas::fixture_extents_static_rank<std::size_t>
){
    namespace ublas = boost::numeric::ublas;
    using fixture_t = ublas::fixture_extents_static_rank<std::size_t>;
    using value_type = typename TestTupleType::first_type;
    using layout_type = typename TestTupleType::second_type;
    using vector_t = std::vector<value_type>;
    using permutation_type = std::vector<std::size_t>;


    auto const& self = static_cast<fixture_t const&>(*this);
    ublas::for_each_fixture(self, []<typename extents_type>(auto /*id*/, extents_type const& n){
        using base_t = typename extents_type::base_type;
        auto const rank = ublas::size(n);
        auto const s = ublas::product(n);

        // FIXME: enbale tests for rank one after the enhancement #120
        if(rank < 2ul)
            return;

        BOOST_TEST_CONTEXT("[Transpose Algorithm] static rank("<< rank <<") extents"){
            auto pi  = permutation_type(rank);
            auto a   = vector_t(s);
            auto b1  = vector_t(s);
            auto b2  = vector_t(s);
            auto c1  = vector_t(s);
            auto c2  = vector_t(s);

            auto wa = ublas::to_strides(n,layout_type{});

            ublas::iota(a, value_type{});

            // so wie last-order.
            std::iota(pi.rbegin(), pi.rend(), 1ul);

            auto nc_base = base_t();
            for(auto i = 0u; i < rank; ++i)
                nc_base[pi[i]-1] = n[i];

            auto nc = extents_type(std::move(nc_base));

            auto wc    = ublas::to_strides(nc,layout_type{});
            auto wc_pi = base_t();
            for(auto i = 0u; i < rank; ++i)
                wc_pi[pi[i]-1] = wc[i];

            ublas::copy ( rank, n.data(),            c1.data(), wc_pi.data(), a.data(), wa.data());
            ublas::trans( rank, n.data(), pi.data(), c2.data(), wc.data(),    a.data(), wa.data() );

            BOOST_REQUIRE_EQUAL_COLLECTIONS(std::begin(c1), std::end(c1), std::begin(c2), std::end(c2));

            auto nb_base = base_t();
            for(auto i = 0u; i < rank; ++i)
                nb_base[pi[i]-1] = nc[i];

            auto nb = extents_type(std::move(nb_base));

            auto wb    = ublas::to_strides(nb,layout_type{});
            auto wb_pi = base_t();
            for(auto i = 0u; i < rank; ++i)
                wb_pi[pi[i]-1] = wb[i];

            ublas::copy ( rank, nc.data(),            b1.data(), wb_pi.data(), c1.data(), wc.data());
            ublas::trans( rank, nc.data(), pi.data(), b2.data(), wb.data(),    c2.data(), wc.data() );

            BOOST_CHECK_EQUAL_COLLECTIONS(std::begin(b1), std::end(b1), std::begin(b2), std::end(b2));
            BOOST_CHECK_EQUAL_COLLECTIONS(std::begin(a), std::end(a), std::begin(b2), std::end(b2));
        }

        BOOST_TEST_CONTEXT("[Transpose Algorithm(Exception)] static rank("<< rank <<") extents"){
            auto pi = permutation_type(rank);
            auto a  = vector_t(s);
            auto c  = vector_t(s);
            auto wa = ublas::to_strides(n,layout_type{});
            auto wc = ublas::to_strides(n,layout_type{});
            value_type* data = nullptr;
            std::size_t* size = nullptr;
            BOOST_REQUIRE_THROW( ublas::trans( rank, size     , pi.data(), c.data(), wc.data(),    a.data(), wa.data() ), std::runtime_error );
            BOOST_REQUIRE_THROW( ublas::trans( rank, n.data() , size     , c.data(), wc.data(),    a.data(), wa.data() ), std::runtime_error );
            BOOST_REQUIRE_THROW( ublas::trans( rank, n.data() , pi.data(), c.data(), size     ,    a.data(), size      ), std::runtime_error );
            BOOST_REQUIRE_THROW( ublas::trans( rank, n.data() , pi.data(), c.data(), wc.data(),    a.data(), size      ), std::runtime_error );
            BOOST_REQUIRE_THROW( ublas::trans( rank, n.data() , pi.data(), c.data(), size     ,    a.data(), wa.data() ), std::runtime_error );
            BOOST_REQUIRE_THROW( ublas::trans( rank, n.data() , pi.data(), data    , wc.data(),    data    , wa.data() ), std::runtime_error );
            BOOST_REQUIRE_THROW( ublas::trans( rank, n.data() , pi.data(), c.data(), wc.data(),    data    , wa.data() ), std::runtime_error );
            BOOST_REQUIRE_THROW( ublas::trans( rank, n.data() , pi.data(), data    , wc.data(),    a.data(), wa.data() ), std::runtime_error );
        }
    });

}

template<typename E>
constexpr auto generate_permuated_extents() noexcept{
    namespace ublas = boost::numeric::ublas;
    using value_type = typename E::value_type;
    constexpr auto sz = ublas::size_v<E>;

    
    constexpr auto helper = []<std::size_t... Is>(std::index_sequence<Is...> ids){
        constexpr auto helper1 = [](){
            std::array<std::size_t,sz> pi;
            (( pi[sz - Is - 1ul] = ublas::get_v<E,Is> ),...);
            return pi;
        };
        constexpr auto arr = helper1();
        return ublas::extents_core<value_type,arr[Is]...>{};
    };
    
    return helper(std::make_index_sequence<sz>{});
}

BOOST_TEST_DECORATOR(
    *boost::unit_test::label("transpose_algorithm")
    *boost::unit_test::description("Testing transpose algorithm using staticextents")
)
BOOST_FIXTURE_TEST_CASE_TEMPLATE(test_extents_static,
    TestTupleType,
    boost::numeric::ublas::test_types,
    boost::numeric::ublas::fixture_extents_static<std::size_t>
){
    namespace ublas = boost::numeric::ublas;
    using fixture_t = ublas::fixture_extents_static<std::size_t>;
    using value_type = typename TestTupleType::first_type;
    using layout_type = typename TestTupleType::second_type;
    using vector_t = std::vector<value_type>;
    using permutation_type = std::vector<std::size_t>;

    auto const& self = static_cast<fixture_t const&>(*this);
    ublas::for_each_fixture(self, []<typename extents_type>(auto /*id*/, extents_type const& n){
        using base_t = typename extents_type::base_type;
        constexpr auto rank = ublas::size_v<extents_type>;
        constexpr auto s = ublas::product_v<extents_type>;

        // FIXME: Enable test for the rank one and the rank two after the issue #119 has been fixed
        if constexpr(rank > 2ul){

            BOOST_TEST_CONTEXT("[Transpose Algorithm] rank("<< rank <<") static extents"){
                auto pi  = permutation_type(rank);
                auto a   = vector_t(s);
                auto b1  = vector_t(s);
                auto b2  = vector_t(s);
                auto c1  = vector_t(s);
                auto c2  = vector_t(s);

                constexpr auto wa = ublas::to_strides_v<extents_type, layout_type>;

                ublas::iota(a, value_type{});

                // so wie last-order.
                std::iota(pi.rbegin(), pi.rend(), 1ul);

                constexpr auto nc = generate_permuated_extents<extents_type>();
                constexpr auto wc    = ublas::to_strides_v<decltype(nc),layout_type>;

                auto wc_pi = base_t();
                for(auto i = 0u; i < rank; ++i)
                    wc_pi[pi[i]-1] = wc[i];

                ublas::copy ( rank, n.data(),            c1.data(), wc_pi.data(), a.data(), wa.data());
                ublas::trans( rank, n.data(), pi.data(), c2.data(), wc.data(),    a.data(), wa.data() );

                BOOST_REQUIRE_EQUAL_COLLECTIONS(std::begin(c1), std::end(c1), std::begin(c2), std::end(c2));

                constexpr auto nb = generate_permuated_extents<decltype(nc)>();
                constexpr auto wb    = ublas::to_strides_v<decltype(nb),layout_type>;

                auto wb_pi = base_t();
                for(auto i = 0u; i < rank; ++i)
                    wb_pi[pi[i]-1] = wb[i];

                ublas::copy ( rank, nc.data(),            b1.data(), wb_pi.data(), c1.data(), wc.data());
                ublas::trans( rank, nc.data(), pi.data(), b2.data(), wb.data(),    c2.data(), wc.data() );

                BOOST_CHECK_EQUAL_COLLECTIONS(std::begin(b1), std::end(b1), std::begin(b2), std::end(b2));
                BOOST_CHECK_EQUAL_COLLECTIONS(std::begin(a), std::end(a), std::begin(b2), std::end(b2));
            }

            BOOST_TEST_CONTEXT("[Transpose Algorithm(Exception)] rank("<< rank <<") static extents"){
                auto pi = permutation_type(rank);
                auto a  = vector_t(s);
                auto c  = vector_t(s);
                constexpr auto wa = ublas::to_strides_v<extents_type, layout_type>;
                constexpr auto wc = ublas::to_strides_v<extents_type, layout_type>;
                value_type* data = nullptr;
                std::size_t* size = nullptr;
                BOOST_REQUIRE_THROW( ublas::trans( rank, size     , pi.data(), c.data(), wc.data(),    a.data(), wa.data() ), std::runtime_error );
                BOOST_REQUIRE_THROW( ublas::trans( rank, n.data() , size     , c.data(), wc.data(),    a.data(), wa.data() ), std::runtime_error );
                BOOST_REQUIRE_THROW( ublas::trans( rank, n.data() , pi.data(), c.data(), size     ,    a.data(), size      ), std::runtime_error );
                BOOST_REQUIRE_THROW( ublas::trans( rank, n.data() , pi.data(), c.data(), wc.data(),    a.data(), size      ), std::runtime_error );
                BOOST_REQUIRE_THROW( ublas::trans( rank, n.data() , pi.data(), c.data(), size     ,    a.data(), wa.data() ), std::runtime_error );
                BOOST_REQUIRE_THROW( ublas::trans( rank, n.data() , pi.data(), data    , wc.data(),    data    , wa.data() ), std::runtime_error );
                BOOST_REQUIRE_THROW( ublas::trans( rank, n.data() , pi.data(), c.data(), wc.data(),    data    , wa.data() ), std::runtime_error );
                BOOST_REQUIRE_THROW( ublas::trans( rank, n.data() , pi.data(), data    , wc.data(),    a.data(), wa.data() ), std::runtime_error );
            }

        }

    });

}


BOOST_AUTO_TEST_SUITE_END()
