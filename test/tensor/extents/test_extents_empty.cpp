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

BOOST_AUTO_TEST_SUITE(test_extents_empty, * boost::unit_test::description("Validate Empty Function and Trait"))


BOOST_FIXTURE_TEST_CASE(test_extents_dynamic_function, 
    boost::numeric::ublas::fixture_extents_dynamic<std::size_t>,
    *boost::unit_test::label("boost::numeric::ublas::empty(extents_base<D> const&)")
    *boost::unit_test::description("Testing free function [empty] for dynamic extents"))
{
    namespace ublas = boost::numeric::ublas;
    BOOST_TEST_CONTEXT("[Empty Free Function] rank(0) dynamic extents"){
        BOOST_REQUIRE( ublas::empty(n    ));
    }

    BOOST_TEST_CONTEXT("[Empty Free Function] rank(1) dynamic extents"){
        BOOST_REQUIRE(!ublas::empty(n1   ));
        BOOST_REQUIRE(!ublas::empty(n2   ));
    }

    BOOST_TEST_CONTEXT("[Empty Free Function] rank(2) dynamic extents"){
        BOOST_REQUIRE(!ublas::empty(n11  ));
        BOOST_REQUIRE(!ublas::empty(n12  ));
        BOOST_REQUIRE(!ublas::empty(n21  ));
        BOOST_REQUIRE(!ublas::empty(n22  ));
        BOOST_REQUIRE(!ublas::empty(n32  ));
    }

    BOOST_TEST_CONTEXT("[Empty Free Function] rank(3) dynamic extents"){
        BOOST_REQUIRE(!ublas::empty(n111 ));
        BOOST_REQUIRE(!ublas::empty(n112 ));
        BOOST_REQUIRE(!ublas::empty(n121 ));
        BOOST_REQUIRE(!ublas::empty(n123 ));
        BOOST_REQUIRE(!ublas::empty(n211 ));
        BOOST_REQUIRE(!ublas::empty(n213 ));
        BOOST_REQUIRE(!ublas::empty(n321 ));
        BOOST_REQUIRE(!ublas::empty(n432 ));
    }

    BOOST_TEST_CONTEXT("[Empty Free Function] rank(4) dynamic extents"){
        BOOST_REQUIRE(!ublas::empty(n1111));
        BOOST_REQUIRE(!ublas::empty(n4231));
    }
}

BOOST_FIXTURE_TEST_CASE(test_extents_static_rank_function,
    boost::numeric::ublas::fixture_extents_static_rank<std::size_t>,
    *boost::unit_test::label("boost::numeric::ublas::empty(extents_base<D> const&)")
    *boost::unit_test::description("Testing free function [empty] for static rank extents"))
{
    namespace ublas = boost::numeric::ublas;

    BOOST_TEST_CONTEXT("[Empty Free Function] rank(0) static rank extents"){
        BOOST_REQUIRE( ublas::empty(n    ));
    }

    BOOST_TEST_CONTEXT("[Empty Free Function] rank(1) static rank extents"){
        BOOST_REQUIRE(!ublas::empty(n1   ));
        BOOST_REQUIRE(!ublas::empty(n2   ));
    }

    BOOST_TEST_CONTEXT("[Empty Free Function] rank(2) static rank extents"){
        BOOST_REQUIRE(!ublas::empty(n11  ));
        BOOST_REQUIRE(!ublas::empty(n12  ));
        BOOST_REQUIRE(!ublas::empty(n21  ));
        BOOST_REQUIRE(!ublas::empty(n22  ));
        BOOST_REQUIRE(!ublas::empty(n32  ));
    }

    BOOST_TEST_CONTEXT("[Empty Free Function] rank(3) static rank extents"){
        BOOST_REQUIRE(!ublas::empty(n111 ));
        BOOST_REQUIRE(!ublas::empty(n112 ));
        BOOST_REQUIRE(!ublas::empty(n121 ));
        BOOST_REQUIRE(!ublas::empty(n123 ));
        BOOST_REQUIRE(!ublas::empty(n211 ));
        BOOST_REQUIRE(!ublas::empty(n213 ));
        BOOST_REQUIRE(!ublas::empty(n321 ));
        BOOST_REQUIRE(!ublas::empty(n432 ));
    }

    BOOST_TEST_CONTEXT("[Empty Free Function] rank(4) static rank extents"){
        BOOST_REQUIRE(!ublas::empty(n1111));
        BOOST_REQUIRE(!ublas::empty(n4231));
    }
}

BOOST_FIXTURE_TEST_CASE(test_extents_static_trait,
    boost::numeric::ublas::fixture_extents_static<std::size_t>,
    *boost::unit_test::label("boost::numeric::ublas::empty_v<extents_core<...>>")
    *boost::unit_test::description("Testing trait [empty_v] for static extents"))
{
    namespace ublas = boost::numeric::ublas;

    BOOST_TEST_CONTEXT("[Empty Trait] rank(0) static rank extents"){
        BOOST_REQUIRE( ublas::empty_v<n_type    >);
    }

    BOOST_TEST_CONTEXT("[Empty Trait] rank(1) static extents"){
        BOOST_REQUIRE(!ublas::empty_v<n1_type   >);
        BOOST_REQUIRE(!ublas::empty_v<n2_type   >);
    }

    BOOST_TEST_CONTEXT("[Empty Trait] rank(2) static extents"){
        BOOST_REQUIRE(!ublas::empty_v<n11_type  >);
        BOOST_REQUIRE(!ublas::empty_v<n12_type  >);
        BOOST_REQUIRE(!ublas::empty_v<n21_type  >);
        BOOST_REQUIRE(!ublas::empty_v<n22_type  >);
        BOOST_REQUIRE(!ublas::empty_v<n32_type  >);
    }

    BOOST_TEST_CONTEXT("[Empty Trait] rank(3) static extents"){
        BOOST_REQUIRE(!ublas::empty_v<n111_type >);
        BOOST_REQUIRE(!ublas::empty_v<n112_type >);
        BOOST_REQUIRE(!ublas::empty_v<n121_type >);
        BOOST_REQUIRE(!ublas::empty_v<n123_type >);
        BOOST_REQUIRE(!ublas::empty_v<n211_type >);
        BOOST_REQUIRE(!ublas::empty_v<n213_type >);
        BOOST_REQUIRE(!ublas::empty_v<n321_type >);
        BOOST_REQUIRE(!ublas::empty_v<n432_type >);
    }

    BOOST_TEST_CONTEXT("[Empty Trait] rank(4) static extents"){
        BOOST_REQUIRE(!ublas::empty_v<n1111_type>);
        BOOST_REQUIRE(!ublas::empty_v<n4231_type>);
    }
}

BOOST_AUTO_TEST_SUITE_END()
