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

BOOST_AUTO_TEST_SUITE(test_tensor_empty, * boost::unit_test::description("Validate Empty Method"))


BOOST_FIXTURE_TEST_CASE(test_tensor_dynamic,
    boost::numeric::ublas::fixture_tensor_dynamic<float>,
    *boost::unit_test::label("boost::numeric::ublas::tensor_dynamic::empty")
    *boost::unit_test::description("Testing the dynamic tensor's [empty] method"))
{
    namespace ublas = boost::numeric::ublas;

    BOOST_TEST_CONTEXT("[Empty Method] rank(1) dynamic tensor"){
        BOOST_REQUIRE(!t2.empty()   );
    }

    BOOST_TEST_CONTEXT("[Empty Method] rank(2) dynamic tensor"){
        BOOST_REQUIRE(!t11.empty()  );
        BOOST_REQUIRE(!t12.empty()  );
        BOOST_REQUIRE(!t21.empty()  );
        BOOST_REQUIRE(!t22.empty()  );
        BOOST_REQUIRE(!t32.empty()  );
    }

    BOOST_TEST_CONTEXT("[Empty Method] rank(3) dynamic tensor"){
        BOOST_REQUIRE(!t111.empty() );
        BOOST_REQUIRE(!t112.empty() );
        BOOST_REQUIRE(!t121.empty() );
        BOOST_REQUIRE(!t123.empty() );
        BOOST_REQUIRE(!t211.empty() );
        BOOST_REQUIRE(!t213.empty() );
        BOOST_REQUIRE(!t321.empty() );
        BOOST_REQUIRE(!t432.empty() );
    }

    BOOST_TEST_CONTEXT("[Empty Method] rank(4) dynamic tensor"){
        BOOST_REQUIRE(!t1111.empty());
        BOOST_REQUIRE(!t4231.empty());
    }
    
}

BOOST_FIXTURE_TEST_CASE(test_tensor_static_rank,
    boost::numeric::ublas::fixture_tensor_static_rank<float>,
    *boost::unit_test::label("boost::numeric::ublas::tensor_static_rank::empty")
    *boost::unit_test::description("Testing the static rank tensor's [empty] method"))
{
    namespace ublas = boost::numeric::ublas;

    BOOST_TEST_CONTEXT("[Empty Method] rank(1) static_rank tensor"){
        BOOST_REQUIRE(!t2.empty()   );
    }

    BOOST_TEST_CONTEXT("[Empty Method] rank(2) static_rank tensor"){
        BOOST_REQUIRE(!t11.empty()  );
        BOOST_REQUIRE(!t12.empty()  );
        BOOST_REQUIRE(!t21.empty()  );
        BOOST_REQUIRE(!t22.empty()  );
        BOOST_REQUIRE(!t32.empty()  );
    }

    BOOST_TEST_CONTEXT("[Empty Method] rank(3) static_rank tensor"){
        BOOST_REQUIRE(!t111.empty() );
        BOOST_REQUIRE(!t112.empty() );
        BOOST_REQUIRE(!t121.empty() );
        BOOST_REQUIRE(!t123.empty() );
        BOOST_REQUIRE(!t211.empty() );
        BOOST_REQUIRE(!t213.empty() );
        BOOST_REQUIRE(!t321.empty() );
        BOOST_REQUIRE(!t432.empty() );
    }

    BOOST_TEST_CONTEXT("[Empty Method] rank(4) static_rank tensor"){
        BOOST_REQUIRE(!t1111.empty());
        BOOST_REQUIRE(!t4231.empty());
    }
    
}

BOOST_FIXTURE_TEST_CASE(test_tensor_static,
    boost::numeric::ublas::fixture_tensor_static<float>,
    *boost::unit_test::label("boost::numeric::ublas::tensor_static::empty")
    *boost::unit_test::description("Testing the static tensor's [empty] method"))
{
    namespace ublas = boost::numeric::ublas;

    // BOOST_TEST_CONTEXT("[Empty Method] rank(1) static tensor"){
    //     BOOST_REQUIRE(!t2.empty()   );
    // }

    BOOST_TEST_CONTEXT("[Empty Method] rank(2) static tensor"){
        BOOST_REQUIRE(!t11.empty()  );
        BOOST_REQUIRE(!t12.empty()  );
        BOOST_REQUIRE(!t21.empty()  );
        BOOST_REQUIRE(!t22.empty()  );
        BOOST_REQUIRE(!t32.empty()  );
    }

    BOOST_TEST_CONTEXT("[Empty Method] rank(3) static tensor"){
        BOOST_REQUIRE(!t111.empty() );
        BOOST_REQUIRE(!t112.empty() );
        BOOST_REQUIRE(!t121.empty() );
        BOOST_REQUIRE(!t123.empty() );
        BOOST_REQUIRE(!t211.empty() );
        BOOST_REQUIRE(!t213.empty() );
        BOOST_REQUIRE(!t321.empty() );
        BOOST_REQUIRE(!t432.empty() );
    }

    BOOST_TEST_CONTEXT("[Empty Method] rank(4) static tensor"){
        BOOST_REQUIRE(!t1111.empty());
        BOOST_REQUIRE(!t4231.empty());
    }
    
}


BOOST_AUTO_TEST_SUITE_END()
