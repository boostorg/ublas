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


BOOST_AUTO_TEST_SUITE(test_extents_product, 
    *boost::unit_test::description("Validate extents' product functions/traits")
)

BOOST_FIXTURE_TEST_CASE(test_extents_dynamic, 
    boost::numeric::ublas::fixture_extents_dynamic<std::size_t>,
    *boost::unit_test::label("product(extents<>)")
    *boost::unit_test::description("Testing the accumulated product of elements of the dynamic extents"))
{
    namespace ublas = boost::numeric::ublas;
    
    BOOST_TEST_CONTEXT("[Dynamic Extents] rank(0) dynamic extents"){
        BOOST_CHECK_EQUAL( ublas::product(n), 0ul );
    }

    BOOST_TEST_CONTEXT("[Dynamic Extents] rank(1) dynamic extents"){
        BOOST_CHECK_EQUAL( ublas::product(n1), 1ul );
        BOOST_CHECK_EQUAL( ublas::product(n2), 2ul );
    }

    BOOST_TEST_CONTEXT("[Dynamic Extents] rank(2) dynamic extents"){
        BOOST_CHECK_EQUAL( ublas::product(n11), 1ul );
        BOOST_CHECK_EQUAL( ublas::product(n12), 2ul );
        BOOST_CHECK_EQUAL( ublas::product(n21), 2ul );
        BOOST_CHECK_EQUAL( ublas::product(n22), 4ul );
        BOOST_CHECK_EQUAL( ublas::product(n32), 6ul );
    }

    BOOST_TEST_CONTEXT("[Dynamic Extents] rank(3) dynamic extents"){
        BOOST_CHECK_EQUAL( ublas::product(n111),  1ul );
        BOOST_CHECK_EQUAL( ublas::product(n112),  2ul );
        BOOST_CHECK_EQUAL( ublas::product(n121),  2ul );
        BOOST_CHECK_EQUAL( ublas::product(n123),  6ul );
        BOOST_CHECK_EQUAL( ublas::product(n211),  2ul );
        BOOST_CHECK_EQUAL( ublas::product(n213),  6ul );
        BOOST_CHECK_EQUAL( ublas::product(n321),  6ul );
        BOOST_CHECK_EQUAL( ublas::product(n432), 24ul );
    }

    BOOST_TEST_CONTEXT("[Dynamic Extents] rank(4) dynamic extents"){
        BOOST_CHECK_EQUAL( ublas::product(n1111),  1ul );
        BOOST_CHECK_EQUAL( ublas::product(n4231), 24ul );
    }
}

BOOST_FIXTURE_TEST_CASE(test_extents_static_rank, 
    boost::numeric::ublas::fixture_extents_static_rank<std::size_t>,
    *boost::unit_test::label("product(extents<N>)")
    *boost::unit_test::description("Testing the accumulated product of elements of the static rank extents"))
{
    namespace ublas = boost::numeric::ublas;

    BOOST_TEST_CONTEXT("[Static Rank Extents] static rank(0) extents"){
        BOOST_CHECK_EQUAL( ublas::product(n), 0ul );
    }

    BOOST_TEST_CONTEXT("[Static Rank Extents] static rank(1) extents"){
        BOOST_CHECK_EQUAL( ublas::product(n1), 1ul );
        BOOST_CHECK_EQUAL( ublas::product(n2), 2ul );
    }

    BOOST_TEST_CONTEXT("[Static Rank Extents] static rank(2) extents"){
        BOOST_CHECK_EQUAL( ublas::product(n11), 1ul );
        BOOST_CHECK_EQUAL( ublas::product(n12), 2ul );
        BOOST_CHECK_EQUAL( ublas::product(n21), 2ul );
        BOOST_CHECK_EQUAL( ublas::product(n22), 4ul );
        BOOST_CHECK_EQUAL( ublas::product(n32), 6ul );
    }

    BOOST_TEST_CONTEXT("[Static Rank Extents] static rank(3) extents"){
        BOOST_CHECK_EQUAL( ublas::product(n111),  1ul );
        BOOST_CHECK_EQUAL( ublas::product(n112),  2ul );
        BOOST_CHECK_EQUAL( ublas::product(n121),  2ul );
        BOOST_CHECK_EQUAL( ublas::product(n123),  6ul );
        BOOST_CHECK_EQUAL( ublas::product(n211),  2ul );
        BOOST_CHECK_EQUAL( ublas::product(n213),  6ul );
        BOOST_CHECK_EQUAL( ublas::product(n321),  6ul );
        BOOST_CHECK_EQUAL( ublas::product(n432), 24ul );
    }

    BOOST_TEST_CONTEXT("[Static Rank Extents] static rank(4) extents"){
        BOOST_CHECK_EQUAL( ublas::product(n1111),  1ul );
        BOOST_CHECK_EQUAL( ublas::product(n4231), 24ul );
    }
}

BOOST_FIXTURE_TEST_CASE(test_extents_static, 
    boost::numeric::ublas::fixture_extents_static<std::size_t>,
    *boost::unit_test::label("product_v<extents<...>>")
    *boost::unit_test::description("Testing the accumulated product of elements of the static extents"))
{
    namespace ublas = boost::numeric::ublas;
    
    BOOST_TEST_CONTEXT("[Static Extents] rank(0) static extents"){
        BOOST_CHECK_EQUAL( ublas::product_v< n_type >, 0ul );
    }

    BOOST_TEST_CONTEXT("[Static Extents] rank(1) static extents"){
        BOOST_CHECK_EQUAL( ublas::product_v< n1_type >, 1ul );
        BOOST_CHECK_EQUAL( ublas::product_v< n2_type >, 2ul );
    }

    BOOST_TEST_CONTEXT("[Static Extents] rank(2) static extents"){
        BOOST_CHECK_EQUAL( ublas::product_v< n11_type >, 1ul );
        BOOST_CHECK_EQUAL( ublas::product_v< n12_type >, 2ul );
        BOOST_CHECK_EQUAL( ublas::product_v< n21_type >, 2ul );
        BOOST_CHECK_EQUAL( ublas::product_v< n22_type >, 4ul );
        BOOST_CHECK_EQUAL( ublas::product_v< n32_type >, 6ul );
    }

    BOOST_TEST_CONTEXT("[Static Extents] rank(3) static extents"){
        BOOST_CHECK_EQUAL( ublas::product_v< n111_type >,  1ul );
        BOOST_CHECK_EQUAL( ublas::product_v< n112_type >,  2ul );
        BOOST_CHECK_EQUAL( ublas::product_v< n121_type >,  2ul );
        BOOST_CHECK_EQUAL( ublas::product_v< n123_type >,  6ul );
        BOOST_CHECK_EQUAL( ublas::product_v< n211_type >,  2ul );
        BOOST_CHECK_EQUAL( ublas::product_v< n213_type >,  6ul );
        BOOST_CHECK_EQUAL( ublas::product_v< n321_type >,  6ul );
        BOOST_CHECK_EQUAL( ublas::product_v< n432_type >, 24ul );
    }

    BOOST_TEST_CONTEXT("[Static Extents] rank(4) static extents"){
        BOOST_CHECK_EQUAL( ublas::product_v< n1111_type >,  1ul );
        BOOST_CHECK_EQUAL( ublas::product_v< n4231_type >, 24ul );
    }
}

BOOST_AUTO_TEST_SUITE_END()
