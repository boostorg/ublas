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

BOOST_AUTO_TEST_SUITE(test_tensor_size, * boost::unit_test::description("Validate Size Function and Trait"))

BOOST_FIXTURE_TEST_CASE(test_tensor_dynamic,
    boost::numeric::ublas::fixture_tensor_dynamic<float>,
    *boost::unit_test::label("boost::numeric::ublas::tensor_dynamic::size")
    *boost::unit_test::description("Testing the dynamic tensor's [size] method"))
{
    namespace ublas = boost::numeric::ublas;

    BOOST_TEST_CONTEXT("[Size Method] rank(1) dynamic tensor"){
        BOOST_REQUIRE_EQUAL(t2.size()   , 2ul);
    }

    BOOST_TEST_CONTEXT("[Size Method] rank(2) dynamic tensor"){
        BOOST_REQUIRE_EQUAL(t11.size()  , 1ul);
        BOOST_REQUIRE_EQUAL(t12.size()  , 2ul);
        BOOST_REQUIRE_EQUAL(t21.size()  , 2ul);
        BOOST_REQUIRE_EQUAL(t22.size()  , 4ul);
        BOOST_REQUIRE_EQUAL(t32.size()  , 6ul);
    }

    BOOST_TEST_CONTEXT("[Size Method] rank(3) dynamic tensor"){
        BOOST_REQUIRE_EQUAL(t111.size() ,  1ul);
        BOOST_REQUIRE_EQUAL(t112.size() ,  2ul);
        BOOST_REQUIRE_EQUAL(t121.size() ,  2ul);
        BOOST_REQUIRE_EQUAL(t123.size() ,  6ul);
        BOOST_REQUIRE_EQUAL(t211.size() ,  2ul);
        BOOST_REQUIRE_EQUAL(t213.size() ,  6ul);
        BOOST_REQUIRE_EQUAL(t321.size() ,  6ul);
        BOOST_REQUIRE_EQUAL(t432.size() , 24ul);
    }

    BOOST_TEST_CONTEXT("[Size Method] rank(4) dynamic tensor"){
        BOOST_REQUIRE_EQUAL(t1111.size(),  1ul);
        BOOST_REQUIRE_EQUAL(t4231.size(), 24ul);
    }
    
}

BOOST_FIXTURE_TEST_CASE(test_tensor_static_rank,
    boost::numeric::ublas::fixture_tensor_static_rank<float>,
    *boost::unit_test::label("boost::numeric::ublas::tensor_static_rank::size")
    *boost::unit_test::description("Testing the static rank tensor's [size] method"))
{
    namespace ublas = boost::numeric::ublas;

    BOOST_TEST_CONTEXT("[Size Method] rank(1) static_rank tensor"){
        BOOST_REQUIRE_EQUAL(t2.size()   , 2ul);
    }

    BOOST_TEST_CONTEXT("[Size Method] rank(2) static_rank tensor"){
        BOOST_REQUIRE_EQUAL(t11.size()  , 1ul);
        BOOST_REQUIRE_EQUAL(t12.size()  , 2ul);
        BOOST_REQUIRE_EQUAL(t21.size()  , 2ul);
        BOOST_REQUIRE_EQUAL(t22.size()  , 4ul);
        BOOST_REQUIRE_EQUAL(t32.size()  , 6ul);
    }

    BOOST_TEST_CONTEXT("[Size Method] rank(3) static_rank tensor"){
        BOOST_REQUIRE_EQUAL(t111.size() ,  1ul);
        BOOST_REQUIRE_EQUAL(t112.size() ,  2ul);
        BOOST_REQUIRE_EQUAL(t121.size() ,  2ul);
        BOOST_REQUIRE_EQUAL(t123.size() ,  6ul);
        BOOST_REQUIRE_EQUAL(t211.size() ,  2ul);
        BOOST_REQUIRE_EQUAL(t213.size() ,  6ul);
        BOOST_REQUIRE_EQUAL(t321.size() ,  6ul);
        BOOST_REQUIRE_EQUAL(t432.size() , 24ul);
    }

    BOOST_TEST_CONTEXT("[Size Method] rank(4) static_rank tensor"){
        BOOST_REQUIRE_EQUAL(t1111.size(),  1ul);
        BOOST_REQUIRE_EQUAL(t4231.size(), 24ul);
    }
    
}

BOOST_FIXTURE_TEST_CASE(test_tensor_static,
    boost::numeric::ublas::fixture_tensor_static<float>,
    *boost::unit_test::label("boost::numeric::ublas::tensor_static::size")
    *boost::unit_test::description("Testing the static tensor's [size] method"))
{
    namespace ublas = boost::numeric::ublas;

    // BOOST_TEST_CONTEXT("[Size Method] rank(1) static tensor"){
    //     BOOST_REQUIRE_EQUAL(t2.size()   , 2ul);
    // }

    BOOST_TEST_CONTEXT("[Size Method] rank(2) static tensor"){
        BOOST_REQUIRE_EQUAL(t11.size()  , 1ul);
        BOOST_REQUIRE_EQUAL(t12.size()  , 2ul);
        BOOST_REQUIRE_EQUAL(t21.size()  , 2ul);
        BOOST_REQUIRE_EQUAL(t22.size()  , 4ul);
        BOOST_REQUIRE_EQUAL(t32.size()  , 6ul);
    }

    BOOST_TEST_CONTEXT("[Size Method] rank(3) static tensor"){
        BOOST_REQUIRE_EQUAL(t111.size() ,  1ul);
        BOOST_REQUIRE_EQUAL(t112.size() ,  2ul);
        BOOST_REQUIRE_EQUAL(t121.size() ,  2ul);
        BOOST_REQUIRE_EQUAL(t123.size() ,  6ul);
        BOOST_REQUIRE_EQUAL(t211.size() ,  2ul);
        BOOST_REQUIRE_EQUAL(t213.size() ,  6ul);
        BOOST_REQUIRE_EQUAL(t321.size() ,  6ul);
        BOOST_REQUIRE_EQUAL(t432.size() , 24ul);
    }

    BOOST_TEST_CONTEXT("[Size Method] rank(4) static tensor"){
        BOOST_REQUIRE_EQUAL(t1111.size(),  1ul);
        BOOST_REQUIRE_EQUAL(t4231.size(), 24ul);
    }
    
}


BOOST_AUTO_TEST_SUITE_END()
