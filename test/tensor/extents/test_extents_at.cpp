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

BOOST_AUTO_TEST_SUITE(test_extents_at, 
    *boost::unit_test::description("Validate At Method")
    *boost::unit_test::depends_on("test_extents_size")
)

constexpr auto check_at(auto const& e, std::initializer_list<std::size_t> l){
    auto i = 0ul;
    for(auto const& el : l){
        BOOST_REQUIRE_EQUAL(e.at(i++), el);
    }
};

BOOST_FIXTURE_TEST_CASE(test_extents_dynamic_function, 
    boost::numeric::ublas::fixture_extents_dynamic<std::size_t>,
    *boost::unit_test::label("[read]boost::numeric::ublas::extents<>::at")
    *boost::unit_test::description("Testing the dynamic extents' read [at] method"))
{
    namespace ublas = boost::numeric::ublas;


    BOOST_TEST_CONTEXT("[At Method] rank(1) dynamic extents"){
        check_at(n1, {1});
        check_at(n2, {2});
    }

    BOOST_TEST_CONTEXT("[At Method] rank(2) dynamic extents"){
        check_at(n11, {1,1});
        check_at(n12, {1,2});
        check_at(n21, {2,1});
        check_at(n22, {2,2});
        check_at(n32, {3,2});
    }
    
    BOOST_TEST_CONTEXT("[At Method] rank(3) dynamic extents"){
        check_at(n111, {1ul, 1ul, 1ul});
        check_at(n112, {1ul, 1ul, 2ul});
        check_at(n121, {1ul, 2ul, 1ul});
        check_at(n123, {1ul, 2ul, 3ul});
        check_at(n211, {2ul, 1ul, 1ul});
        check_at(n213, {2ul, 1ul, 3ul});
        check_at(n321, {3ul, 2ul, 1ul});
        check_at(n432, {4ul, 3ul, 2ul});
    }

    BOOST_TEST_CONTEXT("[At Method] rank(4) dynamic extents"){
        check_at(n1111, {1ul, 1ul, 1ul, 1ul});
        check_at(n4231, {4ul, 2ul, 3ul, 1ul});
    }

}

BOOST_FIXTURE_TEST_CASE(test_extents_static_rank_function,
    boost::numeric::ublas::fixture_extents_static_rank<std::size_t>,
    *boost::unit_test::label("[read]boost::numeric::ublas::extents<N>::at")
    *boost::unit_test::description("Testing the static rank extents' read [at] method"))
{
    namespace ublas = boost::numeric::ublas;

    BOOST_TEST_CONTEXT("[At Method] rank(1) static rank extents"){
        check_at(n1, {1});
        check_at(n2, {2});
    }

    BOOST_TEST_CONTEXT("[At Method] rank(2) static rank extents"){
        check_at(n11, {1,1});
        check_at(n12, {1,2});
        check_at(n21, {2,1});
        check_at(n22, {2,2});
        check_at(n32, {3,2});
    }
    
    BOOST_TEST_CONTEXT("[At Method] rank(3) static rank extents"){
        check_at(n111, {1ul, 1ul, 1ul});
        check_at(n112, {1ul, 1ul, 2ul});
        check_at(n121, {1ul, 2ul, 1ul});
        check_at(n123, {1ul, 2ul, 3ul});
        check_at(n211, {2ul, 1ul, 1ul});
        check_at(n213, {2ul, 1ul, 3ul});
        check_at(n321, {3ul, 2ul, 1ul});
        check_at(n432, {4ul, 3ul, 2ul});
    }

    BOOST_TEST_CONTEXT("[At Method] rank(4) static rank extents"){
        check_at(n1111, {1ul, 1ul, 1ul, 1ul});
        check_at(n4231, {4ul, 2ul, 3ul, 1ul});
    }
}

BOOST_FIXTURE_TEST_CASE(test_extents_static_function,
    boost::numeric::ublas::fixture_extents_static<std::size_t>,
    *boost::unit_test::label("[read]boost::numeric::ublas::extents<...>::at")
    *boost::unit_test::description("Testing the static extents' read [at] method"))
{
    namespace ublas = boost::numeric::ublas;

    BOOST_TEST_CONTEXT("[At Method] rank(2) static extents"){
        check_at(n11, {1,1});
        check_at(n12, {1,2});
        check_at(n21, {2,1});
        check_at(n22, {2,2});
        check_at(n32, {3,2});
    }
    
    BOOST_TEST_CONTEXT("[At Method] rank(3) static extents"){
        check_at(n111, {1ul, 1ul, 1ul});
        check_at(n112, {1ul, 1ul, 2ul});
        check_at(n121, {1ul, 2ul, 1ul});
        check_at(n123, {1ul, 2ul, 3ul});
        check_at(n211, {2ul, 1ul, 1ul});
        check_at(n213, {2ul, 1ul, 3ul});
        check_at(n321, {3ul, 2ul, 1ul});
        check_at(n432, {4ul, 3ul, 2ul});
    }

    BOOST_TEST_CONTEXT("[At Method] rank(4) static extents"){
        check_at(n1111, {1ul, 1ul, 1ul, 1ul});
        check_at(n4231, {4ul, 2ul, 3ul, 1ul});
    }
}

BOOST_AUTO_TEST_SUITE_END()
