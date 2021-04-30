//  Copyright (c) 2018, Cem Bassoy, cem.bassoy@gmail.com
//  Copyright (c) 2019, Amit Singh, amitsingh19975@gmail.com
//
//  Distributed under the Boost Software License, Version 1.0. (See
//  accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)
//
//  The authors gratefully acknowledge the support of
//  Google and Fraunhofer IOSB, Ettlingen, Germany
//



#include <boost/test/unit_test.hpp>
#include <boost/numeric/ublas/tensor/strides.hpp>
#include <boost/numeric/ublas/tensor/dynamic_extents.hpp>

BOOST_AUTO_TEST_SUITE(test_strides)

using test_types = std::tuple<boost::numeric::ublas::layout::first_order, boost::numeric::ublas::layout::last_order>;

BOOST_AUTO_TEST_CASE_TEMPLATE( test_strides_ctor, value, test_types)
{
    namespace ublas = boost::numeric::ublas;

    using extents_type  = ublas::basic_extents<unsigned>;
    using strides_type = ublas::strides_t<extents_type,ublas::layout::first_order>;

    strides_type         s0{};
    BOOST_CHECK        ( s0.empty());
    BOOST_CHECK_EQUAL  ( s0.size(), 0);

    strides_type        s1{extents_type{1,1}};
    BOOST_CHECK       (!s1.empty());
    BOOST_CHECK_EQUAL ( s1.size(), 2);

    strides_type        s2{extents_type{1,2}};
    BOOST_CHECK       (!s2.empty());
    BOOST_CHECK_EQUAL ( s2.size(), 2);

    strides_type        s3{extents_type{2,1}};
    BOOST_CHECK       (!s3.empty());
    BOOST_CHECK_EQUAL ( s3.size(), 2);

    strides_type        s4{extents_type{2,3}};
    BOOST_CHECK       (!s4.empty());
    BOOST_CHECK_EQUAL ( s4.size(), 2);

    strides_type        s5{extents_type{2,3,1}};
    BOOST_CHECK       (!s5.empty());
    BOOST_CHECK_EQUAL ( s5.size(), 3);

    strides_type        s6{extents_type{1,2,3}};
    BOOST_CHECK       (!s6.empty());
    BOOST_CHECK_EQUAL ( s6.size(), 3);

    strides_type        s7{extents_type{4,2,3}};
    BOOST_CHECK       (!s7.empty());
    BOOST_CHECK_EQUAL ( s7.size(), 3);
}

BOOST_AUTO_TEST_CASE( test_strides_ctor_access_first_order)
{
    namespace ublas = boost::numeric::ublas;

    using extents_type  = ublas::basic_extents<unsigned>;
    using strides_type = ublas::strides_t<extents_type,ublas::layout::first_order>;

    strides_type         s1{extents_type{1,1}};
    BOOST_REQUIRE_EQUAL( s1.size(),2);
    BOOST_CHECK_EQUAL  ( s1[0], 1);
    BOOST_CHECK_EQUAL  ( s1[1], 1);

    strides_type          s2{extents_type{1,2}};
    BOOST_REQUIRE_EQUAL ( s2.size(),2);
    BOOST_CHECK_EQUAL   ( s2[0], 1);
    BOOST_CHECK_EQUAL   ( s2[1], 1);

    strides_type          s3{extents_type{2,1}};
    BOOST_REQUIRE_EQUAL ( s3.size(),2);
    BOOST_CHECK_EQUAL   ( s3[0], 1);
    BOOST_CHECK_EQUAL   ( s3[1], 1);

    strides_type          s4{extents_type{2,3}};
    BOOST_REQUIRE_EQUAL ( s4.size(),2);
    BOOST_CHECK_EQUAL   ( s4[0], 1);
    BOOST_CHECK_EQUAL   ( s4[1], 2);

    strides_type          s5{extents_type{2,3,1}};
    BOOST_REQUIRE_EQUAL ( s5.size(),3);
    BOOST_CHECK_EQUAL   ( s5[0], 1);
    BOOST_CHECK_EQUAL   ( s5[1], 2);
    BOOST_CHECK_EQUAL   ( s5[2], 6);

    strides_type          s6{extents_type{1,2,3}};
    BOOST_REQUIRE_EQUAL ( s6.size(),3);
    BOOST_CHECK_EQUAL   ( s6[0], 1);
    BOOST_CHECK_EQUAL   ( s6[1], 1);
    BOOST_CHECK_EQUAL   ( s6[2], 2);

    strides_type          s7{extents_type{2,1,3}};
    BOOST_REQUIRE_EQUAL ( s7.size(),3);
    BOOST_CHECK_EQUAL   ( s7[0], 1);
    BOOST_CHECK_EQUAL   ( s7[1], 2);
    BOOST_CHECK_EQUAL   ( s7[2], 2);

    strides_type          s8{extents_type{4,2,3}};
    BOOST_REQUIRE_EQUAL ( s8.size(),3);
    BOOST_CHECK_EQUAL   ( s8[0], 1);
    BOOST_CHECK_EQUAL   ( s8[1], 4);
    BOOST_CHECK_EQUAL   ( s8[2], 8);
}

BOOST_AUTO_TEST_CASE( test_strides_ctor_access_last_order)
{
    namespace ublas = boost::numeric::ublas;

    using extents_type  = ublas::basic_extents<unsigned>;
    using strides_type = ublas::strides_t<extents_type,ublas::layout::last_order>;

    strides_type         s1{extents_type{1,1}};
    BOOST_REQUIRE_EQUAL( s1.size(),2);
    BOOST_CHECK_EQUAL  ( s1[0], 1);
    BOOST_CHECK_EQUAL  ( s1[1], 1);

    strides_type          s2{extents_type{1,2}};
    BOOST_REQUIRE_EQUAL ( s2.size(),2);
    BOOST_CHECK_EQUAL   ( s2[0], 1);
    BOOST_CHECK_EQUAL   ( s2[1], 1);

    strides_type          s3{extents_type{2,1}};
    BOOST_REQUIRE_EQUAL ( s3.size(),2);
    BOOST_CHECK_EQUAL   ( s3[0], 1);
    BOOST_CHECK_EQUAL   ( s3[1], 1);

    strides_type          s4{extents_type{2,3}};
    BOOST_REQUIRE_EQUAL ( s4.size(),2);
    BOOST_CHECK_EQUAL   ( s4[0], 3);
    BOOST_CHECK_EQUAL   ( s4[1], 1);

    strides_type          s5{extents_type{2,3,1}};
    BOOST_REQUIRE_EQUAL ( s5.size(),3);
    BOOST_CHECK_EQUAL   ( s5[0], 3);
    BOOST_CHECK_EQUAL   ( s5[1], 1);
    BOOST_CHECK_EQUAL   ( s5[2], 1);

    strides_type          s6{extents_type{1,2,3}};
    BOOST_REQUIRE_EQUAL ( s6.size(),3);
    BOOST_CHECK_EQUAL   ( s6[0], 6);
    BOOST_CHECK_EQUAL   ( s6[1], 3);
    BOOST_CHECK_EQUAL   ( s6[2], 1);

    strides_type          s7{extents_type{2,1,3}};
    BOOST_REQUIRE_EQUAL ( s7.size(),3);
    BOOST_CHECK_EQUAL   ( s7[0], 3);
    BOOST_CHECK_EQUAL   ( s7[1], 3);
    BOOST_CHECK_EQUAL   ( s7[2], 1);

    strides_type          s8{extents_type{4,2,3}};
    BOOST_REQUIRE_EQUAL ( s8.size(),3);
    BOOST_CHECK_EQUAL   ( s8[0], 6);
    BOOST_CHECK_EQUAL   ( s8[1], 3);
    BOOST_CHECK_EQUAL   ( s8[2], 1);
}

BOOST_AUTO_TEST_SUITE_END()
