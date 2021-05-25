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
#include "fixture_utility.hpp"

BOOST_AUTO_TEST_SUITE(test_strides, 
    *boost::unit_test::description("Validate Strides")
)


BOOST_TEST_DECORATOR(
    *boost::unit_test::label("boost::numeric::ublas::to_strides(extents<> const&, layout_type)")
    *boost::unit_test::description("Testing dynamic strides construction")
)
BOOST_FIXTURE_TEST_CASE_TEMPLATE(test_strides_dynamic_ctr,
    LayoutType,
    boost::numeric::ublas::layout_test_types,
    boost::numeric::ublas::fixture_extents_dynamic<std::size_t>
)
{
    namespace ublas = boost::numeric::ublas;

    constexpr auto check = [](auto const& e, std::size_t sz){
        auto const s = ublas::to_strides(e, LayoutType{});
        BOOST_CHECK      (!s.empty());
        BOOST_CHECK_EQUAL( s.size(), sz);
    };

    BOOST_TEST_CONTEXT("[Dynamic Strides Construction] rank(1) dynamic strides"){
        check(n1, 1ul);
        check(n2, 1ul);
    }

    BOOST_TEST_CONTEXT("[Dynamic Strides Construction] rank(2) dynamic strides"){
        check(n11, 2ul);
        check(n12, 2ul);
        check(n21, 2ul);
        check(n22, 2ul);
        check(n32, 2ul);
    }

    BOOST_TEST_CONTEXT("[Dynamic Strides Construction] rank(3) dynamic strides"){
        check(n111, 3ul);
        check(n112, 3ul);
        check(n121, 3ul);
        check(n123, 3ul);
        check(n211, 3ul);
        check(n213, 3ul);
        check(n321, 3ul);
        check(n432, 3ul);
    }

    BOOST_TEST_CONTEXT("[Dynamic Strides Construction] rank(4) dynamic strides"){
        check(n1111, 4ul);
        check(n4231, 4ul);
    }
}

BOOST_TEST_DECORATOR(
    *boost::unit_test::label("boost::numeric::ublas::to_strides(extents<N> const&, layout_type)")
    *boost::unit_test::description("Testing static rank strides construction")
)
BOOST_FIXTURE_TEST_CASE_TEMPLATE(test_strides_static_rank_ctr,
    LayoutType,
    boost::numeric::ublas::layout_test_types,
    boost::numeric::ublas::fixture_extents_static_rank<std::size_t>
)
{
    namespace ublas = boost::numeric::ublas;

    constexpr auto check = [](auto const& e, std::size_t sz){
        auto const s = ublas::to_strides(e, LayoutType{});
        BOOST_CHECK      (!s.empty());
        BOOST_CHECK_EQUAL( s.size(), sz);
    };

    BOOST_TEST_CONTEXT("[Static Rank Strides Construction] static rank(1) strides"){
        check(n1, 1ul);
        check(n2, 1ul);
    }

    BOOST_TEST_CONTEXT("[Static Rank Strides Construction] static rank(2) strides"){
        check(n11, 2ul);
        check(n12, 2ul);
        check(n21, 2ul);
        check(n22, 2ul);
        check(n32, 2ul);
    }

    BOOST_TEST_CONTEXT("[Static Rank Strides Construction] static rank(3) strides"){
        check(n111, 3ul);
        check(n112, 3ul);
        check(n121, 3ul);
        check(n123, 3ul);
        check(n211, 3ul);
        check(n213, 3ul);
        check(n321, 3ul);
        check(n432, 3ul);
    }

    BOOST_TEST_CONTEXT("[Static Rank Strides Construction] static rank(4) strides"){
        check(n1111, 4ul);
        check(n4231, 4ul);
    }
}

BOOST_TEST_DECORATOR(
    *boost::unit_test::label("boost::numeric::ublas::to_strides(extents<...> const&, layout_type)")
    *boost::unit_test::description("Testing static strides construction")
)
BOOST_FIXTURE_TEST_CASE_TEMPLATE(test_strides_static_ctr,
    LayoutType,
    boost::numeric::ublas::layout_test_types,
    boost::numeric::ublas::fixture_extents_static<std::size_t>
)
{
    namespace ublas = boost::numeric::ublas;

    constexpr auto check = []<typename E>(E const& /*e*/, std::size_t sz){
        constexpr auto s = ublas::to_strides_v<E, LayoutType>;
        BOOST_CHECK      (!s.empty());
        BOOST_CHECK_EQUAL( s.size(), sz);
    };

    BOOST_TEST_CONTEXT("[Static Strides Construction] rank(1) static strides"){
        check(n1, 1ul);
        check(n2, 1ul);
    }

    BOOST_TEST_CONTEXT("[Static Strides Construction] rank(2) static strides"){
        check(n11, 2ul);
        check(n12, 2ul);
        check(n21, 2ul);
        check(n22, 2ul);
        check(n32, 2ul);
    }

    BOOST_TEST_CONTEXT("[Static Strides Construction] rank(3) static strides"){
        check(n111, 3ul);
        check(n112, 3ul);
        check(n121, 3ul);
        check(n123, 3ul);
        check(n211, 3ul);
        check(n213, 3ul);
        check(n321, 3ul);
        check(n432, 3ul);
    }

    BOOST_TEST_CONTEXT("[Static Strides Construction] rank(4) static strides"){
        check(n1111, 4ul);
        check(n4231, 4ul);
    }
}

BOOST_FIXTURE_TEST_CASE(test_first_order_dynamic_strides, 
    boost::numeric::ublas::fixture_extents_dynamic<std::size_t>,
    *boost::unit_test::label("boost::numeric::ublas::to_strides(extents<> const&, first_order)")
    *boost::unit_test::description("Testing the first order dynamic strides"))
{
    
    namespace ublas = boost::numeric::ublas;
    constexpr auto check = [](auto const& e, std::initializer_list<std::size_t> l){
        auto const s = ublas::to_strides(e, ublas::layout::first_order{});
        BOOST_REQUIRE      (!s.empty());
        BOOST_REQUIRE_EQUAL( s.size(), l.size());
        BOOST_REQUIRE_EQUAL_COLLECTIONS(std::begin(s), std::end(s), std::begin(l), std::end(l));
    };

    BOOST_TEST_CONTEXT("[Dynamic Strides(First Order)] rank(1) dynamic strides"){
        check(n1, {1ul});
        check(n2, {1ul});
    }

    BOOST_TEST_CONTEXT("[Dynamic Strides(First Order)] rank(2) dynamic strides"){
        check(n11, {1ul, 1ul});
        check(n12, {1ul, 1ul});
        check(n21, {1ul, 1ul});
        check(n22, {1ul, 2ul});
        check(n32, {1ul, 3ul});
    }

    BOOST_TEST_CONTEXT("[Dynamic Strides(First Order)] rank(3) dynamic strides"){
        check(n111, {1ul, 1ul,  1ul});
        check(n112, {1ul, 1ul,  1ul});
        check(n121, {1ul, 1ul,  1ul});
        check(n123, {1ul, 1ul,  2ul});
        check(n211, {1ul, 1ul,  1ul});
        check(n213, {1ul, 2ul,  2ul});
        check(n321, {1ul, 3ul,  6ul});
        check(n432, {1ul, 4ul, 12ul});
    }

    BOOST_TEST_CONTEXT("[Dynamic Strides(First Order)] rank(4) dynamic strides"){
        check(n1111, {1ul, 1ul, 1ul, 1ul});
        check(n4231, {1ul, 4ul, 8ul, 24ul});
    }
}

BOOST_FIXTURE_TEST_CASE(test_first_order_static_rank_strides, 
    boost::numeric::ublas::fixture_extents_static_rank<std::size_t>,
    *boost::unit_test::label("boost::numeric::ublas::to_strides(extents<N> const&, first_order)")
    *boost::unit_test::description("Testing the first order static rank strides"))
{
    
    namespace ublas = boost::numeric::ublas;
    constexpr auto check = [](auto const& e, std::initializer_list<std::size_t> l){
        auto const s = ublas::to_strides(e, ublas::layout::first_order{});
        BOOST_REQUIRE      (!s.empty());
        BOOST_REQUIRE_EQUAL( s.size(), l.size());
        BOOST_REQUIRE_EQUAL_COLLECTIONS(std::begin(s), std::end(s), std::begin(l), std::end(l));
    };

    BOOST_TEST_CONTEXT("[Static Rank Strides(First Order)] static rank(1) dynamic strides"){
        check(n1, {1ul});
        check(n2, {1ul});
    }

    BOOST_TEST_CONTEXT("[Static Rank Strides(First Order)] static rank(2) dynamic strides"){
        check(n11, {1ul, 1ul});
        check(n12, {1ul, 1ul});
        check(n21, {1ul, 1ul});
        check(n22, {1ul, 2ul});
        check(n32, {1ul, 3ul});
    }

    BOOST_TEST_CONTEXT("[Static Rank Strides(First Order)] static rank(3) dynamic strides"){
        check(n111, {1ul, 1ul,  1ul});
        check(n112, {1ul, 1ul,  1ul});
        check(n121, {1ul, 1ul,  1ul});
        check(n123, {1ul, 1ul,  2ul});
        check(n211, {1ul, 1ul,  1ul});
        check(n213, {1ul, 2ul,  2ul});
        check(n321, {1ul, 3ul,  6ul});
        check(n432, {1ul, 4ul, 12ul});
    }

    BOOST_TEST_CONTEXT("[Static Rank Strides(First Order)] static rank(4) dynamic strides"){
        check(n1111, {1ul, 1ul, 1ul, 1ul});
        check(n4231, {1ul, 4ul, 8ul, 24ul});
    }
}

BOOST_FIXTURE_TEST_CASE(test_first_order_static_strides, 
    boost::numeric::ublas::fixture_extents_static<std::size_t>,
    *boost::unit_test::label("boost::numeric::ublas::to_strides(extents<...> const&, first_order)")
    *boost::unit_test::description("Testing the first order static strides"))
{
    
    namespace ublas = boost::numeric::ublas;
    constexpr auto check = []<typename E>(E const& /*e*/, std::initializer_list<std::size_t> l){
        constexpr auto s = ublas::to_strides_v<E, ublas::layout::first_order>;
        BOOST_REQUIRE      (!s.empty());
        BOOST_REQUIRE_EQUAL( s.size(), l.size());
        BOOST_REQUIRE_EQUAL_COLLECTIONS(std::begin(s), std::end(s), std::begin(l), std::end(l));
    };

    BOOST_TEST_CONTEXT("[Static Strides(First Order)] rank(1) static dynamic strides"){
        check(n1, {1ul});
        check(n2, {1ul});
    }

    // TODO: Enable after fixing the issue #119
    // BOOST_TEST_CONTEXT("[Static Strides(First Order)] rank(2) static dynamic strides"){
    //     check(n11, {1ul, 1ul});
    //     check(n12, {1ul, 1ul});
    //     check(n21, {1ul, 1ul});
    //     check(n22, {1ul, 2ul});
    //     check(n32, {1ul, 3ul});
    // }

    // BOOST_TEST_CONTEXT("[Static Strides(First Order)] rank(3) static dynamic strides"){
    //     check(n111, {1ul, 1ul,  1ul});
    //     check(n112, {1ul, 1ul,  1ul});
    //     check(n121, {1ul, 1ul,  1ul});
    //     check(n123, {1ul, 1ul,  2ul});
    //     check(n211, {1ul, 1ul,  1ul});
    //     check(n213, {1ul, 2ul,  2ul});
    //     check(n321, {1ul, 3ul,  6ul});
    //     check(n432, {1ul, 4ul, 12ul});
    // }

    // BOOST_TEST_CONTEXT("[Static Strides(First Order)] rank(4) static dynamic strides"){
    //     check(n1111, {1ul, 1ul, 1ul, 1ul});
    //     check(n4231, {1ul, 4ul, 8ul, 24ul});
    // }
}

BOOST_FIXTURE_TEST_CASE(test_last_order_dynamic_strides, 
    boost::numeric::ublas::fixture_extents_dynamic<std::size_t>,
    *boost::unit_test::label("boost::numeric::ublas::to_strides(extents<> const&, last_order)")
    *boost::unit_test::description("Testing the last order dynamic strides"))
{
    
    namespace ublas = boost::numeric::ublas;
    constexpr auto check = [](auto const& e, std::initializer_list<std::size_t> l){
        auto const s = ublas::to_strides(e, ublas::layout::last_order{});
        BOOST_REQUIRE      (!s.empty());
        BOOST_REQUIRE_EQUAL( s.size(), l.size());
        BOOST_REQUIRE_EQUAL_COLLECTIONS(std::begin(s), std::end(s), std::begin(l), std::end(l));
    };

    BOOST_TEST_CONTEXT("[Dynamic Strides(Last Order)] rank(1) dynamic strides"){
        check(n1, {1ul});
        check(n2, {1ul});
    }

    BOOST_TEST_CONTEXT("[Dynamic Strides(Last Order)] rank(2) dynamic strides"){
        check(n11, {1ul, 1ul});
        check(n12, {1ul, 1ul});
        check(n21, {1ul, 1ul});
        check(n22, {2ul, 1ul});
        check(n32, {2ul, 1ul});
    }

    BOOST_TEST_CONTEXT("[Dynamic Strides(Last Order)] rank(3) dynamic strides"){
        check(n111, {1ul, 1ul, 1ul});
        check(n112, {2ul, 2ul, 1ul});
        check(n121, {1ul, 1ul, 1ul});
        check(n123, {6ul, 3ul, 1ul});
        check(n211, {1ul, 1ul, 1ul});
        check(n213, {3ul, 3ul, 1ul});
        check(n321, {2ul, 1ul, 1ul});
        check(n432, {6ul, 2ul, 1ul});
    }

    BOOST_TEST_CONTEXT("[Dynamic Strides(Last Order)] rank(4) dynamic strides"){
        check(n1111, {1ul, 1ul, 1ul, 1ul});
        check(n4231, {6ul, 3ul, 1ul, 1ul});
    }
}

BOOST_FIXTURE_TEST_CASE(test_last_order_static_rank_strides, 
    boost::numeric::ublas::fixture_extents_static_rank<std::size_t>,
    *boost::unit_test::label("boost::numeric::ublas::to_strides(extents<N> const&, last_order)")
    *boost::unit_test::description("Testing the last order static rank strides"))
{
    
    namespace ublas = boost::numeric::ublas;
    constexpr auto check = [](auto const& e, std::initializer_list<std::size_t> l){
        auto const s = ublas::to_strides(e, ublas::layout::last_order{});
        BOOST_REQUIRE      (!s.empty());
        BOOST_REQUIRE_EQUAL( s.size(), l.size());
        BOOST_REQUIRE_EQUAL_COLLECTIONS(std::begin(s), std::end(s), std::begin(l), std::end(l));
    };

    BOOST_TEST_CONTEXT("[Static Rank Strides(Last Order)] static rank(1) dynamic strides"){
        check(n1, {1ul});
        check(n2, {1ul});
    }

    BOOST_TEST_CONTEXT("[Static Rank Strides(Last Order)] static rank(2) dynamic strides"){
        check(n11, {1ul, 1ul});
        check(n12, {1ul, 1ul});
        check(n21, {1ul, 1ul});
        check(n22, {2ul, 1ul});
        check(n32, {2ul, 1ul});
    }

    BOOST_TEST_CONTEXT("[Static Rank Strides(Last Order)] static rank(3) dynamic strides"){
        check(n111, {1ul, 1ul, 1ul});
        check(n112, {2ul, 2ul, 1ul});
        check(n121, {1ul, 1ul, 1ul});
        check(n123, {6ul, 3ul, 1ul});
        check(n211, {1ul, 1ul, 1ul});
        check(n213, {3ul, 3ul, 1ul});
        check(n321, {2ul, 1ul, 1ul});
        check(n432, {6ul, 2ul, 1ul});
    }

    BOOST_TEST_CONTEXT("[Static Rank Strides(Last Order)] static rank(4) dynamic strides"){
        check(n1111, {1ul, 1ul, 1ul, 1ul});
        check(n4231, {6ul, 3ul, 1ul, 1ul});
    }
}

BOOST_FIXTURE_TEST_CASE(test_last_order_static_strides, 
    boost::numeric::ublas::fixture_extents_static<std::size_t>,
    *boost::unit_test::label("boost::numeric::ublas::to_strides(extents<...> const&, last_order)")
    *boost::unit_test::description("Testing the last order static strides"))
{
    
    namespace ublas = boost::numeric::ublas;
    constexpr auto check = []<typename E>(E const& /*e*/, std::initializer_list<std::size_t> l){
        constexpr auto s = ublas::to_strides_v<E, ublas::layout::first_order>;
        BOOST_REQUIRE      (!s.empty());
        BOOST_REQUIRE_EQUAL( s.size(), l.size());
        BOOST_REQUIRE_EQUAL_COLLECTIONS(std::begin(s), std::end(s), std::begin(l), std::end(l));
    };

    BOOST_TEST_CONTEXT("[Static Strides(Last Order)] rank(1) static dynamic strides"){
        check(n1, {1ul});
        check(n2, {1ul});
    }

    // TODO: Enable after fixing the issue #119
    // BOOST_TEST_CONTEXT("[Static Strides(Last Order)] rank(2) static dynamic strides"){
    //     check(n11, {1ul, 1ul});
    //     check(n12, {1ul, 1ul});
    //     check(n21, {1ul, 1ul});
    //     check(n22, {2ul, 1ul});
    //     check(n32, {2ul, 1ul});
    // }

    // BOOST_TEST_CONTEXT("[Static Strides(Last Order)] rank(3) static dynamic strides"){
    //     check(n111, {1ul, 1ul, 1ul});
    //     check(n112, {2ul, 2ul, 1ul});
    //     check(n121, {1ul, 1ul, 1ul});
    //     check(n123, {6ul, 3ul, 1ul});
    //     check(n211, {1ul, 1ul, 1ul});
    //     check(n213, {3ul, 3ul, 1ul});
    //     check(n321, {2ul, 1ul, 1ul});
    //     check(n432, {6ul, 2ul, 1ul});
    // }

    // BOOST_TEST_CONTEXT("[Static Strides(Last Order)] rank(4) static dynamic strides"){
    //     check(n1111, {1ul, 1ul, 1ul, 1ul});
    //     check(n4231, {6ul, 3ul, 1ul, 1ul});
    // }
}

BOOST_AUTO_TEST_SUITE_END()
