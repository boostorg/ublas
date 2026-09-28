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

BOOST_AUTO_TEST_SUITE(test_extents_assignement_operator, 
    *boost::unit_test::description("Validate extents operator=(...)")
    *boost::unit_test::depends_on("test_extents_constructor")
)


BOOST_FIXTURE_TEST_CASE(test_extents_dynamic, 
    boost::numeric::ublas::fixture_extents_dynamic<std::size_t>,
    *boost::unit_test::label("boost::numeric::ublas::extents<>::operator=(...)")
    *boost::unit_test::description("Testing dynamic extents assignment operator"))
{
    namespace ublas = boost::numeric::ublas;
    using extents_type = ublas::extents<>;
    
    BOOST_TEST_CONTEXT("[Dynamic Extents Assignment Operator] copy extents"){
        auto e = extents_type();
        BOOST_REQUIRE       ( ublas::empty(e));
        BOOST_REQUIRE_EQUAL ( ublas::size(e), 0ul);

        e = n4231;
        BOOST_REQUIRE       (!ublas::empty(e));
        BOOST_REQUIRE_EQUAL ( ublas::size(e), 4ul);
        BOOST_REQUIRE_EQUAL ( e[0], 4ul);
        BOOST_REQUIRE_EQUAL ( e[1], 2ul);
        BOOST_REQUIRE_EQUAL ( e[2], 3ul);
        BOOST_REQUIRE_EQUAL ( e[3], 1ul);
        BOOST_CHECK_THROW   ((void)e.at(5), std::out_of_range);
    }
    
    BOOST_TEST_CONTEXT("[Dynamic Extents Assignment Operator] copy extents"){
        auto temp = n4231;
        
        auto e = extents_type();
        BOOST_REQUIRE       ( ublas::empty(e));
        BOOST_REQUIRE_EQUAL ( ublas::size(e), 0ul);

        e = std::move(temp);
        BOOST_REQUIRE       ( ublas::empty(temp));
        BOOST_REQUIRE_EQUAL ( ublas::size(temp), 0ul);

        BOOST_REQUIRE       (!ublas::empty(e));
        BOOST_REQUIRE_EQUAL ( ublas::size(e), 4ul);
        BOOST_REQUIRE_EQUAL ( e[0], 4ul);
        BOOST_REQUIRE_EQUAL ( e[1], 2ul);
        BOOST_REQUIRE_EQUAL ( e[2], 3ul);
        BOOST_REQUIRE_EQUAL ( e[3], 1ul);
        BOOST_CHECK_THROW   ((void)e.at(5), std::out_of_range);
    }
    
}

BOOST_FIXTURE_TEST_CASE(test_extents_static_rank, 
    boost::numeric::ublas::fixture_extents_static_rank<std::size_t>,
    *boost::unit_test::label("boost::numeric::ublas::extents<N>::operator=(...)")
    *boost::unit_test::description("Testing static_rank extents assignment operator"))
{
    namespace ublas = boost::numeric::ublas;
    using extents_type = ublas::extents<4>;
    
    BOOST_TEST_CONTEXT("[Static Rank Extents Assignment Operator] copy extents"){
        auto e = extents_type();
        BOOST_REQUIRE       (!ublas::empty(e));
        BOOST_REQUIRE_EQUAL ( ublas::size(e), 4ul);

        e = n4231;
        BOOST_REQUIRE       (!ublas::empty(e));
        BOOST_REQUIRE_EQUAL ( ublas::size(e), 4ul);
        BOOST_REQUIRE_EQUAL ( e[0], 4ul);
        BOOST_REQUIRE_EQUAL ( e[1], 2ul);
        BOOST_REQUIRE_EQUAL ( e[2], 3ul);
        BOOST_REQUIRE_EQUAL ( e[3], 1ul);
        BOOST_CHECK_THROW   ((void)e.at(5), std::out_of_range);
    }
    
    BOOST_TEST_CONTEXT("[Static Rank Extents Assignment Operator] copy extents"){
        auto temp = n4231;
        
        auto e = extents_type();
        BOOST_REQUIRE       (!ublas::empty(e));
        BOOST_REQUIRE_EQUAL ( ublas::size(e), 4ul);

        e = std::move(temp);
        BOOST_REQUIRE       (!ublas::empty(temp));
        BOOST_REQUIRE_EQUAL ( ublas::size(temp), 4ul);

        BOOST_REQUIRE       (!ublas::empty(e));
        BOOST_REQUIRE_EQUAL ( ublas::size(e), 4ul);
        BOOST_REQUIRE_EQUAL ( e[0], 4ul);
        BOOST_REQUIRE_EQUAL ( e[1], 2ul);
        BOOST_REQUIRE_EQUAL ( e[2], 3ul);
        BOOST_REQUIRE_EQUAL ( e[3], 1ul);
        BOOST_CHECK_THROW   ((void)e.at(5), std::out_of_range);
    }
    
}

BOOST_AUTO_TEST_SUITE_END()
