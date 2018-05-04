//  Copyright (c) 2018 Cem Bassoy
//
//  Distributed under the Boost Software License, Version 1.0. (See
//  accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#include <boost/test/unit_test.hpp>
#include <boost/numeric/ublas/tensor/extents.hpp>
#include <vector>

BOOST_AUTO_TEST_SUITE ( test_extents );



BOOST_AUTO_TEST_CASE(test_extents_ctor,
										 *boost::unit_test::label("extents")
										 *boost::unit_test::label("constructor"))
{
	using namespace boost::numeric;


	ublas::extents      e0{};
	BOOST_CHECK       ( e0.empty());
	BOOST_CHECK_EQUAL ( e0.size(),0);

	ublas::extents      e1{1,1};
	BOOST_CHECK       (!e1.empty());
	BOOST_CHECK_EQUAL ( e1.size(),2);

	ublas::extents      e2{1,2};
	BOOST_CHECK       (!e2.empty());
	BOOST_CHECK_EQUAL ( e2.size(),2);

	ublas::extents       e3{2,1};
	BOOST_CHECK        (!e3.empty());
	BOOST_CHECK_EQUAL  ( e3.size(),2);

	ublas::extents      e4{2,3};
	BOOST_CHECK       (!e4.empty());
	BOOST_CHECK_EQUAL ( e4.size(),2);

	ublas::extents       e5{2,3,1};
	BOOST_CHECK        (!e5.empty());
	BOOST_CHECK_EQUAL  ( e5.size(),3);

	ublas::extents      e6{1,2,3}; // 6
	BOOST_CHECK       (!e6.empty());
	BOOST_CHECK_EQUAL ( e6.size(),3);

	ublas::extents      e7{4,2,3};  // 7
	BOOST_CHECK       (!e7.empty());
	BOOST_CHECK_EQUAL ( e7.size(),3);

	BOOST_CHECK_THROW( ublas::extents({1,0}), std::length_error );
	BOOST_CHECK_THROW( ublas::extents({0}  ), std::length_error );
	BOOST_CHECK_THROW( ublas::extents({3}  ), std::length_error );
	BOOST_CHECK_THROW( ublas::extents({0,1}), std::length_error );
}

struct fixture {
	using extents_type = boost::numeric::ublas::extents;
	fixture() : extents{
								extents_type{},    // 0
								extents_type{1,1}, // 1
								extents_type{1,2}, // 2
								extents_type{2,1}, // 3
								extents_type{2,3}, // 4
								extents_type{2,3,1}, // 5
								extents_type{1,2,3}, // 6
								extents_type{4,2,3}}  // 7
	{}
	std::vector<extents_type> extents;
};

BOOST_FIXTURE_TEST_CASE(test_extents_access, fixture,
										 *boost::unit_test::label("extents")
										 *boost::unit_test::label("access"))
{
	using namespace boost::numeric;
	BOOST_REQUIRE_EQUAL(extents.size(),8);

	BOOST_CHECK_EQUAL(extents[0].size(), 0);

	BOOST_REQUIRE_EQUAL(extents[1].size(), 2);
	BOOST_REQUIRE_EQUAL(extents[2].size(), 2);
	BOOST_REQUIRE_EQUAL(extents[3].size(), 2);
	BOOST_REQUIRE_EQUAL(extents[4].size(), 2);

	BOOST_CHECK_EQUAL(extents[1][0],1);
	BOOST_CHECK_EQUAL(extents[1][1],1);

	BOOST_CHECK_EQUAL(extents[2][0],1);
	BOOST_CHECK_EQUAL(extents[2][1],2);

	BOOST_CHECK_EQUAL(extents[3][0],2);
	BOOST_CHECK_EQUAL(extents[3][1],1);

	BOOST_CHECK_EQUAL(extents[4][0],2);
	BOOST_CHECK_EQUAL(extents[4][1],3);


	BOOST_REQUIRE_EQUAL(extents[5].size(), 3);
	BOOST_REQUIRE_EQUAL(extents[6].size(), 3);
	BOOST_REQUIRE_EQUAL(extents[7].size(), 3);

	BOOST_CHECK_EQUAL(extents[5][0],2);
	BOOST_CHECK_EQUAL(extents[5][1],3);
	BOOST_CHECK_EQUAL(extents[5][2],1);

	BOOST_CHECK_EQUAL(extents[6][0],1);
	BOOST_CHECK_EQUAL(extents[6][1],2);
	BOOST_CHECK_EQUAL(extents[6][2],3);

	BOOST_CHECK_EQUAL(extents[7][0],4);
	BOOST_CHECK_EQUAL(extents[7][1],2);
	BOOST_CHECK_EQUAL(extents[7][2],3);


}

BOOST_AUTO_TEST_SUITE_END();

