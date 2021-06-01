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

BOOST_AUTO_TEST_SUITE(test_algorithm_accumulate, 
    *boost::unit_test::description("Validate Accumulate Algorithm")
    *boost::unit_test::depends_on("test_extents_constructor")
    *boost::unit_test::depends_on("test_extents_product")
    *boost::unit_test::depends_on("test_strides")
)

BOOST_TEST_DECORATOR(
    *boost::unit_test::label("accumulate_algorithm")
    *boost::unit_test::description("Testing accumulate algorithm using dynamic extents")
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
    using inner_t = inner_type_t<value_type>;

    auto const& self = static_cast<fixture_t const&>(*this);
    ublas::for_each_fixture(self, [](auto /*id*/, auto const& n){
        if(ublas::empty(n))
            return;
        auto const rank = ublas::size(n);
        auto const s = ublas::product(n);

        auto const v = value_type{0};
        auto const one = std::size_t{1};
        auto const two = std::size_t{2};


        BOOST_TEST_CONTEXT("[Accumulate Algorithm] rank("<< rank <<") dynamic extents"){
            auto a = vector_t(s);

            auto const wa = ublas::to_strides(n, layout_type{});

            ublas::iota(a, one);
            
            auto const acc = ublas::accumulate( rank, n.data(), a.data(), wa.data(), v);

            auto const sum = value_type{ static_cast<inner_t>( (s * (s + one)) / two ) };

            BOOST_CHECK_EQUAL( acc, sum );

            auto zero = std::size_t{0};
            BOOST_CHECK_EQUAL( ublas::accumulate(zero, n.data(), a.data(), wa.data(),v), v );

            auto acc2 = ublas::accumulate( rank, n.data(), a.data(), wa.data(), v, std::plus<>{});

            BOOST_CHECK_EQUAL( acc2, sum );

            BOOST_CHECK_EQUAL( ublas::accumulate(zero, n.data(), a.data(), wa.data(), v, std::plus<>{}), v );
        }

        BOOST_TEST_CONTEXT("[Accumulate Algorithm(Exception)] rank("<< rank <<") dynamic extents"){
            {
                BOOST_TEST_CHECKPOINT("if input is null");
                value_type* a  = nullptr;
                auto const wa = ublas::to_strides(n,layout_type{});
                BOOST_REQUIRE_THROW( (void)ublas::accumulate( rank, n.data(), a, wa.data(), v ), std::runtime_error );
            }
            {
                BOOST_TEST_CHECKPOINT("if input is null and with predicate");
                value_type* a  = nullptr;
                auto const wa = ublas::to_strides(n,layout_type{});
                BOOST_REQUIRE_THROW( (void)ublas::accumulate( rank, n.data(), a, wa.data(), v, std::plus<>{} ), std::runtime_error );
            }
            {
                BOOST_TEST_CHECKPOINT("if stride is null");
                auto const a = vector_t(s);
                std::size_t const* wa = nullptr;
                BOOST_REQUIRE_THROW( (void)ublas::accumulate( rank, n.data(), a.data(), wa, v ), std::runtime_error );
            }
            {
                BOOST_TEST_CHECKPOINT("if stride is null and predicate");
                auto const a = vector_t(s);
                std::size_t const* wa = nullptr;
                BOOST_REQUIRE_THROW( (void)ublas::accumulate( rank, n.data(), a.data(), wa, v, std::plus<>{} ), std::runtime_error );
            }
        }
    });

}

BOOST_TEST_DECORATOR(
    *boost::unit_test::label("accumulate_algorithm")
    *boost::unit_test::description("Testing accumulate algorithm using static rank extents")
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
    using inner_t = inner_type_t<value_type>;

    auto const& self = static_cast<fixture_t const&>(*this);
    ublas::for_each_fixture(self, [](auto /*id*/, auto const& n){

        if(ublas::empty(n))
            return;

        auto const rank = ublas::size(n);
        auto const s = ublas::product(n);

        auto const v = value_type{0};
        auto const one = std::size_t{1};
        auto const two = std::size_t{2};

        BOOST_TEST_CONTEXT("[Accumulate Algorithm] static rank("<< rank <<") extents"){
            auto a = vector_t(s);

            auto wa = ublas::to_strides(n, layout_type{});

            ublas::iota(a, one);

            auto const acc = ublas::accumulate( rank, n.data(), a.data(), wa.data(), v);

            auto const sum = value_type{ static_cast<inner_t>( (s * (s + one)) / two ) };

            BOOST_CHECK_EQUAL( acc, sum );

            auto zero = std::size_t{0};
            BOOST_CHECK_EQUAL( ublas::accumulate(zero, n.data(), a.data(), wa.data(),v), v );

            auto acc2 = ublas::accumulate( rank, n.data(), a.data(), wa.data(), v, std::plus<>{});

            BOOST_CHECK_EQUAL( acc2, sum );

            BOOST_CHECK_EQUAL( ublas::accumulate(zero, n.data(), a.data(), wa.data(), v, std::plus<>{}), v );
        }

        BOOST_TEST_CONTEXT("[Accumulate Algorithm(Exception)] static rank("<< rank <<") extents"){
            {
                BOOST_TEST_CHECKPOINT("if input is null");
                value_type* a  = nullptr;
                auto const wa = ublas::to_strides(n,layout_type{});
                BOOST_REQUIRE_THROW( (void)ublas::accumulate( rank, n.data(), a, wa.data(), v ), std::runtime_error );
            }
            {
                BOOST_TEST_CHECKPOINT("if input is null and with predicate");
                value_type* a  = nullptr;
                auto const wa = ublas::to_strides(n,layout_type{});
                BOOST_REQUIRE_THROW( (void)ublas::accumulate( rank, n.data(), a, wa.data(), v, std::plus<>{} ), std::runtime_error );
            }
            {
                BOOST_TEST_CHECKPOINT("if stride is null");
                auto const a = vector_t(s);
                std::size_t const* wa = nullptr;
                BOOST_REQUIRE_THROW( (void)ublas::accumulate( rank, n.data(), a.data(), wa, v ), std::runtime_error );
            }
            {
                BOOST_TEST_CHECKPOINT("if stride is null and predicate");
                auto const a = vector_t(s);
                std::size_t const* wa = nullptr;
                BOOST_REQUIRE_THROW( (void)ublas::accumulate( rank, n.data(), a.data(), wa, v, std::plus<>{} ), std::runtime_error );
            }
        }
    });

}


BOOST_TEST_DECORATOR(
    *boost::unit_test::label("accumulate_algorithm")
    *boost::unit_test::description("Testing accumulate algorithm using static extents")
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
    using inner_t = inner_type_t<value_type>;

    auto const& self = static_cast<fixture_t const&>(*this);
    ublas::for_each_fixture(self, []<typename extents_type>(auto /*id*/, extents_type const& n){

        if constexpr(!ublas::empty_v<extents_type>){

            constexpr auto rank = ublas::size_v<extents_type>;
            constexpr auto s = ublas::product_v<extents_type>;

            auto const v = value_type{0};
            auto const one = std::size_t{1};
            auto const two = std::size_t{2};

            
            // TODO: Enable test for the rank one and the rank two after the issue #119 has been fixed
            if constexpr(rank > 2ul){

                BOOST_TEST_CONTEXT("[Accumulate Algorithm] rank("<< rank <<") static extents"){
                    auto a = vector_t(s);

                    auto wa = ublas::to_strides_v<extents_type, layout_type>;

                    ublas::iota(a, one);

                    auto const acc = ublas::accumulate( rank, n.data(), a.data(), wa.data(), v);

                    auto const sum = value_type{ static_cast<inner_t>( (s * (s + one)) / two ) };

                    BOOST_CHECK_EQUAL( acc, sum );

                    auto zero = std::size_t{0};
                    BOOST_CHECK_EQUAL( ublas::accumulate(zero, n.data(), a.data(), wa.data(),v), v );

                    auto acc2 = ublas::accumulate( rank, n.data(), a.data(), wa.data(), v, std::plus<>{});

                    BOOST_CHECK_EQUAL( acc2, sum );

                    BOOST_CHECK_EQUAL( ublas::accumulate(zero, n.data(), a.data(), wa.data(), v, std::plus<>{}), v );
                }

                BOOST_TEST_CONTEXT("[Accumulate Algorithm(Exception)] rank("<< rank <<") static extents"){
                    {
                        BOOST_TEST_CHECKPOINT("if input is null");
                        value_type* a  = nullptr;
                        auto const wa = ublas::to_strides_v<extents_type, layout_type>;
                        BOOST_REQUIRE_THROW( (void)ublas::accumulate( rank, n.data(), a, wa.data(), v ), std::runtime_error );
                    }
                    {
                        BOOST_TEST_CHECKPOINT("if input is null and with predicate");
                        value_type* a  = nullptr;
                        auto const wa = ublas::to_strides_v<extents_type, layout_type>;
                        BOOST_REQUIRE_THROW( (void)ublas::accumulate( rank, n.data(), a, wa.data(), v, std::plus<>{} ), std::runtime_error );
                    }
                    {
                        BOOST_TEST_CHECKPOINT("if stride is null");
                        auto const a = vector_t(s);
                        std::size_t const* wa = nullptr;
                        BOOST_REQUIRE_THROW( (void)ublas::accumulate( rank, n.data(), a.data(), wa, v ), std::runtime_error );
                    }
                    {
                        BOOST_TEST_CHECKPOINT("if stride is null and predicate");
                        auto const a = vector_t(s);
                        std::size_t const* wa = nullptr;
                        BOOST_REQUIRE_THROW( (void)ublas::accumulate( rank, n.data(), a.data(), wa, v, std::plus<>{} ), std::runtime_error );
                    }
                }
            }

        }

    });

}


BOOST_AUTO_TEST_SUITE_END()
