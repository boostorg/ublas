//
// 	Copyright (c) 2021, Cem Bassoy, cem.bassoy@gmail.com
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


BOOST_AUTO_TEST_SUITE(test_extents_is_vector_functions, 
    *boost::unit_test::description("Validate extents' is_vector functions/traits")
)

BOOST_FIXTURE_TEST_CASE(test_extents_dynamic, 
    boost::numeric::ublas::fixture_extents_dynamic<std::size_t>,
    *boost::unit_test::label("is_vector(extents<>)")
    *boost::unit_test::description("Testing if dynamic extents is vector or not"))
{
    namespace ublas = boost::numeric::ublas;
    BOOST_TEST_CONTEXT("[Dynamic Extents] rank(0) dynamic extents"){
        BOOST_CHECK( !ublas::is_vector(n) );
    }

    BOOST_TEST_CONTEXT("[Dynamic Extents] rank(1) dynamic extents"){
        BOOST_CHECK(  ublas::is_vector(n1) );
        BOOST_CHECK(  ublas::is_vector(n2) );
    }

    BOOST_TEST_CONTEXT("[Dynamic Extents] rank(2) dynamic extents"){
        BOOST_CHECK(  ublas::is_vector(n11) );
        BOOST_CHECK(  ublas::is_vector(n12) );
        BOOST_CHECK(  ublas::is_vector(n21) );
        BOOST_CHECK( !ublas::is_vector(n22) );
        BOOST_CHECK( !ublas::is_vector(n32) );
    }

    BOOST_TEST_CONTEXT("[Dynamic Extents] rank(3) dynamic extents"){
        BOOST_CHECK(  ublas::is_vector(n111) );
        BOOST_CHECK( !ublas::is_vector(n112) );
        BOOST_CHECK(  ublas::is_vector(n121) );
        BOOST_CHECK( !ublas::is_vector(n123) );
        BOOST_CHECK(  ublas::is_vector(n211) );
        BOOST_CHECK( !ublas::is_vector(n213) );
        BOOST_CHECK( !ublas::is_vector(n321) );
        BOOST_CHECK( !ublas::is_vector(n432) );
    }

    BOOST_TEST_CONTEXT("[Dynamic Extents] rank(4) dynamic extents"){
        BOOST_CHECK(  ublas::is_vector(n1111) );
        BOOST_CHECK( !ublas::is_vector(n4231) );
    }
}

BOOST_FIXTURE_TEST_CASE(test_extents_static_rank, 
    boost::numeric::ublas::fixture_extents_static_rank<std::size_t>,
    *boost::unit_test::label("is_vector(extents<N>)")
    *boost::unit_test::description("Testing if static rank extents is vector or not"))
{
    namespace ublas = boost::numeric::ublas;
    BOOST_TEST_CONTEXT("[Static Rank Extents] static rank(0) extents"){
        BOOST_CHECK( !ublas::is_vector(n) );
    }

    BOOST_TEST_CONTEXT("[Static Rank Extents] static rank(1) extents"){
        BOOST_CHECK(  ublas::is_vector(n1) );
        BOOST_CHECK(  ublas::is_vector(n2) );
    }

    BOOST_TEST_CONTEXT("[Static Rank Extents] static rank(2) extents"){
        BOOST_CHECK(  ublas::is_vector(n11) );
        BOOST_CHECK(  ublas::is_vector(n12) );
        BOOST_CHECK(  ublas::is_vector(n21) );
        BOOST_CHECK( !ublas::is_vector(n22) );
        BOOST_CHECK( !ublas::is_vector(n32) );
    }

    BOOST_TEST_CONTEXT("[Static Rank Extents] static rank(3) extents"){
        BOOST_CHECK(  ublas::is_vector(n111) );
        BOOST_CHECK( !ublas::is_vector(n112) );
        BOOST_CHECK(  ublas::is_vector(n121) );
        BOOST_CHECK( !ublas::is_vector(n123) );
        BOOST_CHECK(  ublas::is_vector(n211) );
        BOOST_CHECK( !ublas::is_vector(n213) );
        BOOST_CHECK( !ublas::is_vector(n321) );
        BOOST_CHECK( !ublas::is_vector(n432) );
    }

    BOOST_TEST_CONTEXT("[Static Rank Extents] static rank(4) extents"){
        BOOST_CHECK(  ublas::is_vector(n1111) );
        BOOST_CHECK( !ublas::is_vector(n4231) );
    }
}

BOOST_FIXTURE_TEST_CASE(test_extents_static,
    boost::numeric::ublas::fixture_extents_static<std::size_t>,
    *boost::unit_test::label("is_vector_v<extents<...>>")
    *boost::unit_test::description("Testing if static extents is vector or not"))
{
    namespace ublas = boost::numeric::ublas;
    BOOST_TEST_CONTEXT("[Static Extents] rank(0) static extents"){
        BOOST_CHECK( !ublas::is_vector_v< n_type > );
    }

    BOOST_TEST_CONTEXT("[Static Extents] rank(1) static extents"){
        BOOST_CHECK(  ublas::is_vector_v< n1_type > );
        BOOST_CHECK(  ublas::is_vector_v< n2_type > );
    }

    BOOST_TEST_CONTEXT("[Static Extents] rank(2) static extents"){
        BOOST_CHECK(  ublas::is_vector_v< n11_type > );
        BOOST_CHECK(  ublas::is_vector_v< n12_type > );
        BOOST_CHECK(  ublas::is_vector_v< n21_type > );
        BOOST_CHECK( !ublas::is_vector_v< n22_type > );
        BOOST_CHECK( !ublas::is_vector_v< n32_type > );
    }

    BOOST_TEST_CONTEXT("[Static Extents] rank(3) static extents"){
        BOOST_CHECK(  ublas::is_vector_v< n111_type > );
        BOOST_CHECK( !ublas::is_vector_v< n112_type > );
        BOOST_CHECK(  ublas::is_vector_v< n121_type > );
        BOOST_CHECK( !ublas::is_vector_v< n123_type > );
        BOOST_CHECK(  ublas::is_vector_v< n211_type > );
        BOOST_CHECK( !ublas::is_vector_v< n213_type > );
        BOOST_CHECK( !ublas::is_vector_v< n321_type > );
        BOOST_CHECK( !ublas::is_vector_v< n432_type > );
    }

    BOOST_TEST_CONTEXT("[Static Extents] rank(4) static extents"){
        BOOST_CHECK(  ublas::is_vector_v< n1111_type > );
        BOOST_CHECK( !ublas::is_vector_v< n4231_type > );
    }
}

BOOST_AUTO_TEST_SUITE_END()
