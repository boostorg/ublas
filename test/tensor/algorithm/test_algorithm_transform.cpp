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

BOOST_AUTO_TEST_SUITE(test_algorithm_transform, 
    *boost::unit_test::description("Validate Transfrom Algorithm")
    *boost::unit_test::depends_on("test_extents_constructor")
    *boost::unit_test::depends_on("test_extents_product")
    *boost::unit_test::depends_on("test_strides")
)

template<typename FnType>
constexpr auto check1(auto const& l, auto const& wl, auto const& r, auto const& wr, auto const& n, FnType&& pred) noexcept{
    for(auto i = 0ul; i < n[0]; ++i){
        auto const& le = l[ i * wl[0] ];
        auto const& re = r[ i * wr[0] ];
        BOOST_CHECK( std::invoke(std::forward<FnType>(pred), le, re) );
    }
};

template<typename FnType>
constexpr auto check2(auto const& l, auto const& wl, auto const& r, auto const& wr, auto const& n, FnType&& pred) noexcept{
    for(auto i = 0ul; i < n[0]; ++i){
        for(auto j = 0ul; j < n[1]; ++j){
            auto const& le = l[ i * wl[0] + j * wl[1] ];
            auto const& re = r[ i * wr[0] + j * wr[1] ];
            BOOST_CHECK( std::invoke(std::forward<FnType>(pred), le, re) );
        }
    }
};

template<typename FnType>
constexpr auto check3(auto const& l, auto const& wl, auto const& r, auto const& wr, auto const& n, FnType&& pred) noexcept{
    for(auto i = 0ul; i < n[0]; ++i){
        for(auto j = 0ul; j < n[1]; ++j){
            for(auto k = 0ul; k < n[2]; ++k){
                auto const& le = l[ i * wl[0] + j * wl[1] + k * wl[2] ];
                auto const& re = r[ i * wr[0] + j * wr[1] + k * wr[2] ];
                BOOST_CHECK( std::invoke(std::forward<FnType>(pred), le, re) );
            }
        }
    }
};

template<typename FnType>
constexpr auto check4(auto const& l, auto const& wl, auto const& r, auto const& wr, auto const& n, FnType&& pred) noexcept{
    for(auto i = 0ul; i < n[0]; ++i){
        for(auto j = 0ul; j < n[1]; ++j){
            for(auto k = 0ul; k < n[2]; ++k){
                for(auto m = 0ul; m < n[3]; ++m){
                    auto const& le = l[ i * wl[0] + j * wl[1] + k * wl[2] + m * wl[3] ];
                    auto const& re = r[ i * wr[0] + j * wr[1] + k * wr[2] + m * wr[3] ];
                    BOOST_CHECK( std::invoke(std::forward<FnType>(pred), le, re) );
                }
            }
        }
    }
};

BOOST_TEST_DECORATOR(
    *boost::unit_test::label("transform_algorithm")
    *boost::unit_test::description("Testing transform algorithm using dynamic extents")
)
BOOST_FIXTURE_TEST_CASE_TEMPLATE(test_extents_dynamic,
    ValueType,
    boost::numeric::ublas::test_types_with_no_layout,
    boost::numeric::ublas::fixture_extents_dynamic<std::size_t>
){
    namespace ublas = boost::numeric::ublas;
    using fixture_t = ublas::fixture_extents_dynamic<std::size_t>;
    using value_type = ValueType;
    using vector_t = std::vector<value_type>;

    constexpr auto check = []<typename FnType>(auto const& l, auto const& wl, auto const& r, auto const& wr, auto const& n, FnType&& pred){
        if(ublas::size(n) == 1ul)
            check1(l, wl, r, wr, n, std::forward<FnType>(pred));
        else if(ublas::size(n) == 2ul)
            check2(l, wl, r, wr, n, std::forward<FnType>(pred));
        else if(ublas::size(n) == 3ul)
            check3(l, wl, r, wr, n, std::forward<FnType>(pred));
        else if(ublas::size(n) == 4ul)
            check4(l, wl, r, wr, n, std::forward<FnType>(pred));
        else assert(false && "Not Implemented");
    };

    auto const& self = static_cast<fixture_t const&>(*this);
    ublas::for_each_fixture(self, [check](auto /*id*/, auto const& n){
        constexpr auto first_order = ublas::layout::first_order{};
        constexpr auto last_order = ublas::layout::last_order{};

        constexpr auto add_one = [](auto const& a){ return a + value_type{1}; };
        constexpr auto sub_one = [](auto const& a){ return a - value_type{1}; };

        if(ublas::empty(n))
            return;
        auto const rank = ublas::size(n);

        BOOST_TEST_CONTEXT("[Transform Algorithm] rank("<< rank <<") dynamic extents"){
            auto a = vector_t(ublas::product(n));
            auto b = vector_t(ublas::product(n));
            auto c = vector_t(ublas::product(n));

            auto wa = ublas::to_strides(n, first_order);
            auto wb = ublas::to_strides(n, last_order );
            auto wc = ublas::to_strides(n, first_order);

            ublas::iota(a, value_type{});

            ublas::transform( rank, n.data(), b.data(), wb.data(), a.data(), wa.data(), add_one );
            ublas::transform( rank, n.data(), c.data(), wc.data(), b.data(), wb.data(), sub_one );

            check(b, wb, a, wa, n, [](auto const& l, auto const& r){ return l == (r + value_type{1}); });
            check(c, wc, b, wb, n, [](auto const& l, auto const& r){ return l == (r - value_type{1}); });

            auto zero = std::size_t{0};
            ublas::transform(zero, n.data(), c.data(), wc.data(), b.data(), wb.data(), add_one );

            BOOST_CHECK_EQUAL_COLLECTIONS(std::begin(c), std::end(c), std::begin(a), std::end(a));
        }

        BOOST_TEST_CONTEXT("[Transform Algorithm(Exception)] rank("<< rank <<") dynamic extents"){
            {
                BOOST_TEST_CHECKPOINT("if input is null");
                value_type* a  = nullptr;
                auto c  = vector_t(ublas::product(n));

                auto wa = ublas::to_strides(n,first_order);
                auto wc = ublas::to_strides(n,first_order);

                BOOST_REQUIRE_THROW( ublas::transform( rank, n.data(), c.data(), wc.data(), a, wa.data(), add_one ), std::runtime_error );
            }
            {
                BOOST_TEST_CHECKPOINT("if input and output is null");
                value_type* a  = nullptr;
                value_type* c  = nullptr;

                auto wa = ublas::to_strides(n,first_order);
                auto wc = ublas::to_strides(n,first_order);

                BOOST_REQUIRE_THROW( ublas::transform( rank, n.data(), c, wc.data(), a, wa.data(), add_one), std::runtime_error );
            }
            {
                BOOST_TEST_CHECKPOINT("if output is null");
                auto a  = vector_t(ublas::product(n));
                value_type* c  = nullptr;

                auto wa = ublas::to_strides(n,first_order);
                auto wc = ublas::to_strides(n,first_order);

                BOOST_REQUIRE_THROW( ublas::transform( rank, n.data(), c, wc.data(), a.data(), wa.data(), add_one ), std::runtime_error );
            }
            {
                BOOST_TEST_CHECKPOINT("if input stride is null");
                auto a  = vector_t(ublas::product(n));
                auto c  = vector_t(ublas::product(n));

                std::size_t* wa = nullptr;
                auto wc = ublas::to_strides(n,first_order);

                BOOST_REQUIRE_THROW( ublas::transform( rank, n.data(), c.data(), wc.data(), a.data(), wa, add_one ), std::runtime_error );
            }
            {
                BOOST_TEST_CHECKPOINT("if input stride is null");
                auto a  = vector_t(ublas::product(n));
                auto c  = vector_t(ublas::product(n));

                std::size_t* wc = nullptr;
                auto wa = ublas::to_strides(n,first_order);

                BOOST_REQUIRE_THROW( ublas::transform( rank, n.data(), c.data(), wc, a.data(), wa.data(), add_one ), std::runtime_error );
            }
            {
                BOOST_TEST_CHECKPOINT("if extents is null");
                auto a  = vector_t(ublas::product(n));
                auto c  = vector_t(ublas::product(n));

                std::size_t* m = nullptr;
                auto wa = ublas::to_strides(n,first_order);
                auto wc = ublas::to_strides(n,first_order);

                BOOST_REQUIRE_THROW( ublas::transform( rank, m, c.data(), wc.data(), a.data(), wa.data(), add_one ), std::runtime_error );
            }
            
        }
    });

}

BOOST_TEST_DECORATOR(
    *boost::unit_test::label("transform_algorithm")
    *boost::unit_test::description("Testing transform algorithm using static rank extents")
)
BOOST_FIXTURE_TEST_CASE_TEMPLATE(test_extents_static_rank,
    ValueType,
    boost::numeric::ublas::test_types_with_no_layout,
    boost::numeric::ublas::fixture_extents_static_rank<std::size_t>
){
    namespace ublas = boost::numeric::ublas;
    using fixture_t = ublas::fixture_extents_static_rank<std::size_t>;
    using value_type = ValueType;
    using vector_t = std::vector<value_type>;

    constexpr auto check = []<typename FnType>(auto const& l, auto const& wl, auto const& r, auto const& wr, auto const& n, FnType&& pred){
        if(ublas::size(n) == 1ul)
            check1(l, wl, r, wr, n, std::forward<FnType>(pred));
        else if(ublas::size(n) == 2ul)
            check2(l, wl, r, wr, n, std::forward<FnType>(pred));
        else if(ublas::size(n) == 3ul)
            check3(l, wl, r, wr, n, std::forward<FnType>(pred));
        else if(ublas::size(n) == 4ul)
            check4(l, wl, r, wr, n, std::forward<FnType>(pred));
        else assert(false && "Not Implemented");
    };


    auto const& self = static_cast<fixture_t const&>(*this);
    ublas::for_each_fixture(self, [check](auto /*id*/, auto const& n){
        constexpr auto first_order = ublas::layout::first_order{};
        constexpr auto last_order = ublas::layout::last_order{};

        constexpr auto add_one = [](auto const& a){ return a + value_type{1}; };
        constexpr auto sub_one = [](auto const& a){ return a - value_type{1}; };

        if(ublas::empty(n))
            return;

        auto const rank = ublas::size(n);

        BOOST_TEST_CONTEXT("[Transform Algorithm] static rank("<< rank <<") extents"){
            auto a = vector_t(ublas::product(n));
            auto b = vector_t(ublas::product(n));
            auto c = vector_t(ublas::product(n));

            auto wa = ublas::to_strides(n, first_order);
            auto wb = ublas::to_strides(n, last_order );
            auto wc = ublas::to_strides(n, first_order);

            ublas::iota(a, value_type{});

            ublas::transform( rank, n.data(), b.data(), wb.data(), a.data(), wa.data(), add_one );
            ublas::transform( rank, n.data(), c.data(), wc.data(), b.data(), wb.data(), sub_one );

            check(b, wb, a, wa, n, [](auto const& l, auto const& r){ return l == (r + value_type{1}); });
            check(c, wc, b, wb, n, [](auto const& l, auto const& r){ return l == (r - value_type{1}); });

            auto zero = std::size_t{0};
            ublas::transform(zero, n.data(), c.data(), wc.data(), b.data(), wb.data(), add_one );

            BOOST_CHECK_EQUAL_COLLECTIONS(std::begin(c), std::end(c), std::begin(a), std::end(a));
        }

        BOOST_TEST_CONTEXT("[Transform Algorithm(Exception)] static rank("<< rank <<") extents"){
            {
                BOOST_TEST_CHECKPOINT("if input is null");
                value_type* a  = nullptr;
                auto c  = vector_t(ublas::product(n));

                auto wa = ublas::to_strides(n,first_order);
                auto wc = ublas::to_strides(n,first_order);

                BOOST_REQUIRE_THROW( ublas::transform( rank, n.data(), c.data(), wc.data(), a, wa.data(), add_one ), std::runtime_error );
            }
            {
                BOOST_TEST_CHECKPOINT("if input and output is null");
                value_type* a  = nullptr;
                value_type* c  = nullptr;

                auto wa = ublas::to_strides(n,first_order);
                auto wc = ublas::to_strides(n,first_order);

                BOOST_REQUIRE_THROW( ublas::transform( rank, n.data(), c, wc.data(), a, wa.data(), add_one), std::runtime_error );
            }
            {
                BOOST_TEST_CHECKPOINT("if output is null");
                auto a  = vector_t(ublas::product(n));
                value_type* c  = nullptr;

                auto wa = ublas::to_strides(n,first_order);
                auto wc = ublas::to_strides(n,first_order);

                BOOST_REQUIRE_THROW( ublas::transform( rank, n.data(), c, wc.data(), a.data(), wa.data(), add_one ), std::runtime_error );
            }
            {
                BOOST_TEST_CHECKPOINT("if input stride is null");
                auto a  = vector_t(ublas::product(n));
                auto c  = vector_t(ublas::product(n));

                std::size_t* wa = nullptr;
                auto wc = ublas::to_strides(n,first_order);

                BOOST_REQUIRE_THROW( ublas::transform( rank, n.data(), c.data(), wc.data(), a.data(), wa, add_one ), std::runtime_error );
            }
            {
                BOOST_TEST_CHECKPOINT("if input stride is null");
                auto a  = vector_t(ublas::product(n));
                auto c  = vector_t(ublas::product(n));

                std::size_t* wc = nullptr;
                auto wa = ublas::to_strides(n,first_order);

                BOOST_REQUIRE_THROW( ublas::transform( rank, n.data(), c.data(), wc, a.data(), wa.data(), add_one ), std::runtime_error );
            }
            {
                BOOST_TEST_CHECKPOINT("if extents is null");
                auto a  = vector_t(ublas::product(n));
                auto c  = vector_t(ublas::product(n));

                std::size_t* m = nullptr;
                auto wa = ublas::to_strides(n,first_order);
                auto wc = ublas::to_strides(n,first_order);

                BOOST_REQUIRE_THROW( ublas::transform( rank, m, c.data(), wc.data(), a.data(), wa.data(), add_one ), std::runtime_error );
            }
        }
    });

}


BOOST_TEST_DECORATOR(
    *boost::unit_test::label("transform_algorithm")
    *boost::unit_test::description("Testing transform algorithm using static extents")
)
BOOST_FIXTURE_TEST_CASE_TEMPLATE(test_extents_static,
    ValueType,
    boost::numeric::ublas::test_types_with_no_layout,
    boost::numeric::ublas::fixture_extents_static<std::size_t>
){
    namespace ublas = boost::numeric::ublas;
    using fixture_t = ublas::fixture_extents_static<std::size_t>;
    using value_type = ValueType;
    using vector_t = std::vector<value_type>;

    constexpr auto check = []<typename extents_type, typename FnType>(auto const& l, auto const& wl, auto const& r, auto const& wr, extents_type const& n, FnType&& pred){
        if constexpr(ublas::size_v<extents_type> == 1ul){
            for(auto i = 0ul; i < ublas::get_v<extents_type,0>; ++i){
                auto const& le = l[ i * wl[0] ];
                auto const& re = r[ i * wr[0] ];
                BOOST_CHECK( std::invoke(std::forward<FnType>(pred), le, re) );
            }
        }else if constexpr(ublas::size_v<extents_type> == 2ul)
            check2(l, wl, r, wr, n, std::forward<FnType>(pred));
        else if constexpr(ublas::size_v<extents_type> == 3ul)
            check3(l, wl, r, wr, n, std::forward<FnType>(pred));
        else if constexpr(ublas::size_v<extents_type> == 4ul)
            check4(l, wl, r, wr, n, std::forward<FnType>(pred));
        else assert(false && "Not Implemented");
    };


    auto const& self = static_cast<fixture_t const&>(*this);
    ublas::for_each_fixture(self, [check]<typename extents_type>(auto /*id*/, extents_type const& n){
        using first_order_t = ublas::layout::first_order;
        using last_order_t = ublas::layout::last_order;

        constexpr auto add_one = [](auto const& a){ return a + value_type{1}; };
        constexpr auto sub_one = [](auto const& a){ return a - value_type{1}; };


        if constexpr(!ublas::empty_v<extents_type>){

            constexpr auto rank = ublas::size_v<extents_type>;
            
            // TODO: Enable test for the rank one and the rank two after the issue #119 has been fixed
            if constexpr(rank > 2ul){

                BOOST_TEST_CONTEXT("[Transform Algorithm] rank("<< rank <<") static extents"){
                    auto a = vector_t(ublas::product_v<extents_type>);
                    auto b = vector_t(ublas::product_v<extents_type>);
                    auto c = vector_t(ublas::product_v<extents_type>);

                    auto wa = ublas::to_strides_v<extents_type, first_order_t>;
                    auto wb = ublas::to_strides_v<extents_type, last_order_t>;
                    auto wc = ublas::to_strides_v<extents_type, first_order_t>;

                    ublas::iota(a, value_type{});

                    ublas::transform( ublas::size(n), n.data(), b.data(), wb.data(), a.data(), wa.data(), add_one );
                    ublas::transform( ublas::size(n), n.data(), c.data(), wc.data(), b.data(), wb.data(), sub_one );

                    check(b, wb, a, wa, n, [](auto const& l, auto const& r){ return l == (r + value_type{1}); });
                    check(c, wc, b, wb, n, [](auto const& l, auto const& r){ return l == (r - value_type{1}); });

                    auto zero = std::size_t{0};
                    ublas::transform(zero, n.data(), c.data(), wc.data(), b.data(), wb.data(), add_one );

                    BOOST_CHECK_EQUAL_COLLECTIONS(std::begin(c), std::end(c), std::begin(a), std::end(a));
                }

                BOOST_TEST_CONTEXT("[Transform Algorithm(Exception)] rank("<< rank <<") static extents"){
                    {
                        BOOST_TEST_CHECKPOINT("if input is null");
                        value_type* a  = nullptr;
                        auto c  = vector_t(ublas::product_v<extents_type>);

                        auto wa = ublas::to_strides_v<extents_type, first_order_t>;
                        auto wc = ublas::to_strides_v<extents_type, first_order_t>;

                        BOOST_REQUIRE_THROW( ublas::transform( rank, n.data(), c.data(), wc.data(), a, wa.data(), add_one ), std::runtime_error );
                    }
                    {
                        BOOST_TEST_CHECKPOINT("if input and output is null");
                        value_type* a  = nullptr;
                        value_type* c  = nullptr;

                        auto wa = ublas::to_strides_v<extents_type, first_order_t>;
                        auto wc = ublas::to_strides_v<extents_type, first_order_t>;

                        BOOST_REQUIRE_THROW( ublas::transform( rank, n.data(), c, wc.data(), a, wa.data(), add_one), std::runtime_error );
                    }
                    {
                        BOOST_TEST_CHECKPOINT("if output is null");
                        auto a  = vector_t(ublas::product_v<extents_type>);
                        value_type* c  = nullptr;

                        auto wa = ublas::to_strides_v<extents_type, first_order_t>;
                        auto wc = ublas::to_strides_v<extents_type, first_order_t>;

                        BOOST_REQUIRE_THROW( ublas::transform( rank, n.data(), c, wc.data(), a.data(), wa.data(), add_one ), std::runtime_error );
                    }
                    {
                        BOOST_TEST_CHECKPOINT("if input stride is null");
                        auto a  = vector_t(ublas::product_v<extents_type>);
                        auto c  = vector_t(ublas::product_v<extents_type>);

                        std::size_t* wa = nullptr;
                        auto wc = ublas::to_strides_v<extents_type, first_order_t>;

                        BOOST_REQUIRE_THROW( ublas::transform( rank, n.data(), c.data(), wc.data(), a.data(), wa, add_one ), std::runtime_error );
                    }
                    {
                        BOOST_TEST_CHECKPOINT("if input stride is null");
                        auto a  = vector_t(ublas::product_v<extents_type>);
                        auto c  = vector_t(ublas::product_v<extents_type>);

                        std::size_t* wc = nullptr;
                        auto wa = ublas::to_strides_v<extents_type, first_order_t>;

                        BOOST_REQUIRE_THROW( ublas::transform( rank, n.data(), c.data(), wc, a.data(), wa.data(), add_one ), std::runtime_error );
                    }
                    {
                        BOOST_TEST_CHECKPOINT("if extents is null");
                        auto a  = vector_t(ublas::product_v<extents_type>);
                        auto c  = vector_t(ublas::product_v<extents_type>);

                        std::size_t* m = nullptr;
                        auto wa = ublas::to_strides_v<extents_type, first_order_t>;
                        auto wc = ublas::to_strides_v<extents_type, first_order_t>;

                        BOOST_REQUIRE_THROW( ublas::transform( rank, m, c.data(), wc.data(), a.data(), wa.data(), add_one ), std::runtime_error );
                    }
                }
            }

        }

    });

}


BOOST_AUTO_TEST_SUITE_END()
