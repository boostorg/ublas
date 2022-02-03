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


BOOST_AUTO_TEST_SUITE(test_extents_compare, 
    *boost::unit_test::description("Validate extents' comparision operator/function")
)


BOOST_FIXTURE_TEST_CASE(test_extents_dynamic, 
    boost::numeric::ublas::fixture_extents_dynamic<std::size_t>,
    *boost::unit_test::label("cmp_equality")
    *boost::unit_test::description("Testing equality comparision for the dynamic extents"))
{
    namespace ublas = boost::numeric::ublas;
    
    BOOST_TEST_CONTEXT("[Dynamic Extents] rank(0) dynamic extents"){
        BOOST_CHECK( n == n );
    }

    BOOST_TEST_CONTEXT("[Dynamic Extents] rank(1) dynamic extents"){
        BOOST_CHECK( n1 == n1 );
        BOOST_CHECK( n2 == n2 );

        BOOST_CHECK( n1 != n2 );
    }

    BOOST_TEST_CONTEXT("[Dynamic Extents] rank(2) dynamic extents"){
        BOOST_CHECK( n11 == n11 );
        BOOST_CHECK( n12 == n12 );
        BOOST_CHECK( n21 == n21 );
        BOOST_CHECK( n22 == n22 );
        BOOST_CHECK( n32 == n32 );
        
        BOOST_CHECK( n11 != n12 );
        BOOST_CHECK( n12 != n21 );
        BOOST_CHECK( n21 != n22 );
        BOOST_CHECK( n22 != n32 );
    }

    BOOST_TEST_CONTEXT("[Dynamic Extents] rank(3) dynamic extents"){
        BOOST_CHECK( n111 == n111 );
        BOOST_CHECK( n112 == n112 );
        BOOST_CHECK( n121 == n121 );
        BOOST_CHECK( n123 == n123 );
        BOOST_CHECK( n211 == n211 );
        BOOST_CHECK( n213 == n213 );
        BOOST_CHECK( n321 == n321 );
        BOOST_CHECK( n432 == n432 );
        
        BOOST_CHECK( n111 != n112 );
        BOOST_CHECK( n112 != n121 );
        BOOST_CHECK( n121 != n123 );
        BOOST_CHECK( n123 != n211 );
        BOOST_CHECK( n211 != n213 );
        BOOST_CHECK( n213 != n321 );
        BOOST_CHECK( n321 != n432 );
    }

    BOOST_TEST_CONTEXT("[Dynamic Extents] rank(4) dynamic extents"){
        BOOST_CHECK( n1111 == n1111 );
        BOOST_CHECK( n4231 == n4231 );
        
        BOOST_CHECK( n1111 != n4231 );
    }

    BOOST_TEST_CONTEXT("[Dynamic Extents] different rank dynamic extents"){
        BOOST_CHECK( n    != n1     );
        BOOST_CHECK( n1   != n11    );
        BOOST_CHECK( n11  != n111   );
        BOOST_CHECK( n111 != n1111  );
    }
}

BOOST_FIXTURE_TEST_CASE(test_extents_static_rank, 
    boost::numeric::ublas::fixture_extents_static_rank<std::size_t>,
    *boost::unit_test::label("cmp_equality")
    *boost::unit_test::description("Testing equality comparision for the static rank extents"))
{
    namespace ublas = boost::numeric::ublas;
    
    BOOST_TEST_CONTEXT("[Static Rank Extents] static rank(0) extents"){
        BOOST_CHECK( n == n );
    }

    BOOST_TEST_CONTEXT("[Static Rank Extents] static rank(1) extents"){
        BOOST_CHECK( n1 == n1 );
        BOOST_CHECK( n2 == n2 );

        BOOST_CHECK( n1 != n2 );
    }

    BOOST_TEST_CONTEXT("[Static Rank Extents] static rank(2) extents"){
        BOOST_CHECK( n11 == n11 );
        BOOST_CHECK( n12 == n12 );
        BOOST_CHECK( n21 == n21 );
        BOOST_CHECK( n22 == n22 );
        BOOST_CHECK( n32 == n32 );
        
        BOOST_CHECK( n11 != n12 );
        BOOST_CHECK( n12 != n21 );
        BOOST_CHECK( n21 != n22 );
        BOOST_CHECK( n22 != n32 );
    }

    BOOST_TEST_CONTEXT("[Static Rank Extents] static rank(3) extents"){
        BOOST_CHECK( n111 == n111 );
        BOOST_CHECK( n112 == n112 );
        BOOST_CHECK( n121 == n121 );
        BOOST_CHECK( n123 == n123 );
        BOOST_CHECK( n211 == n211 );
        BOOST_CHECK( n213 == n213 );
        BOOST_CHECK( n321 == n321 );
        BOOST_CHECK( n432 == n432 );
        
        BOOST_CHECK( n111 != n112 );
        BOOST_CHECK( n112 != n121 );
        BOOST_CHECK( n121 != n123 );
        BOOST_CHECK( n123 != n211 );
        BOOST_CHECK( n211 != n213 );
        BOOST_CHECK( n213 != n321 );
        BOOST_CHECK( n321 != n432 );
    }

    BOOST_TEST_CONTEXT("[Static Rank Extents] static rank(4) extents"){
        BOOST_CHECK( n1111 == n1111 );
        BOOST_CHECK( n4231 == n4231 );
        
        BOOST_CHECK( n1111 != n4231 );
    }

    BOOST_TEST_CONTEXT("[Static Rank Extents] different static rank extents"){
        BOOST_CHECK( n    != n1     );
        BOOST_CHECK( n1   != n11    );
        BOOST_CHECK( n11  != n111   );
        BOOST_CHECK( n111 != n1111  );
    }
}

BOOST_FIXTURE_TEST_CASE(test_extents_static, 
    boost::numeric::ublas::fixture_extents_static<std::size_t>,
    *boost::unit_test::label("cmp_equality")
    *boost::unit_test::description("Testing equality comparision for the static extents"))
{
    namespace ublas = boost::numeric::ublas;
    
        
    BOOST_TEST_CONTEXT("[Static Extents] rank(0) static extents"){
        BOOST_CHECK( n_type{} == n_type{} );
    }

    BOOST_TEST_CONTEXT("[Static Extents] rank(1) static extents"){
        BOOST_CHECK( n1 == n1 );
        BOOST_CHECK( n2 == n2 );

        BOOST_CHECK( n1 != n2 );
    }

    BOOST_TEST_CONTEXT("[Static Extents] rank(2) static extents"){
        BOOST_CHECK( n11 == n11 );
        BOOST_CHECK( n12 == n12 );
        BOOST_CHECK( n21 == n21 );
        BOOST_CHECK( n22 == n22 );
        BOOST_CHECK( n32 == n32 );
        
        BOOST_CHECK( n11 != n12 );
        BOOST_CHECK( n12 != n21 );
        BOOST_CHECK( n21 != n22 );
        BOOST_CHECK( n22 != n32 );
    }

    BOOST_TEST_CONTEXT("[Static Extents] rank(3) static extents"){
        BOOST_CHECK( n111 == n111 );
        BOOST_CHECK( n112 == n112 );
        BOOST_CHECK( n121 == n121 );
        BOOST_CHECK( n123 == n123 );
        BOOST_CHECK( n211 == n211 );
        BOOST_CHECK( n213 == n213 );
        BOOST_CHECK( n321 == n321 );
        BOOST_CHECK( n432 == n432 );
        
        BOOST_CHECK( n111 != n112 );
        BOOST_CHECK( n112 != n121 );
        BOOST_CHECK( n121 != n123 );
        BOOST_CHECK( n123 != n211 );
        BOOST_CHECK( n211 != n213 );
        BOOST_CHECK( n213 != n321 );
        BOOST_CHECK( n321 != n432 );
    }

    BOOST_TEST_CONTEXT("[Static Extents] rank(4) static extents"){
        BOOST_CHECK( n1111 == n1111 );
        BOOST_CHECK( n4231 == n4231 );
        
        BOOST_CHECK( n1111 != n4231 );
    }

    BOOST_TEST_CONTEXT("[Static Extents] different rank static extents"){
        BOOST_CHECK( n_type{}   != n1     );
        BOOST_CHECK( n1         != n11    );
        BOOST_CHECK( n11        != n111   );
        BOOST_CHECK( n111       != n1111  );
    }
}

BOOST_AUTO_TEST_SUITE_END()
