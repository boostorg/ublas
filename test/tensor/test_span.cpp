//  Copyright (c) 2018 Cem Bassoy
//
//  Distributed under the Boost Software License, Version 1.0. (See
//  accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)
//
//  The authors gratefully acknowledge the support of
//  Fraunhofer and Google in producing this work
//  which started as a Google Summer of Code project.
//

#include <boost/numeric/ublas/tensor/tags.hpp>
#include <boost/test/unit_test.hpp>
#include <boost/numeric/ublas/tensor/span.hpp>


BOOST_AUTO_TEST_SUITE( span_testsuite );

struct fixture {
	using span_type  = boost::numeric::ublas::strided_span;

	fixture() :
		spans {
				span_type{},      // 0
				span_type(0,0,0), // 1
				span_type(0,1,0), // 2
				span_type(0,1,2), // 3
				span_type(1,1,2), // 4
				span_type(0,2,4), // 5
				span_type(1,2,4), // 6
				span_type(1,3,5), // 7
				span_type(1,3,7)  // 8
				}
	{}
	std::vector<span_type> spans;
};



BOOST_FIXTURE_TEST_CASE( ctor_test, fixture )
{
	using span_type = boost::numeric::ublas::strided_span;

	BOOST_CHECK_EQUAL (spans[0].first(),0);
	BOOST_CHECK_EQUAL (spans[0].step (),0);
	BOOST_CHECK_EQUAL (spans[0].last (),0);
	BOOST_CHECK_EQUAL (spans[0].size (),0);

	BOOST_CHECK_EQUAL (spans[1].first(),0);
	BOOST_CHECK_EQUAL (spans[1].step (),0);
	BOOST_CHECK_EQUAL (spans[1].last (),0);
	BOOST_CHECK_EQUAL (spans[1].size (),1);

	BOOST_CHECK_EQUAL (spans[2].first(),0);
	BOOST_CHECK_EQUAL (spans[2].step (),1);
	BOOST_CHECK_EQUAL (spans[2].last (),0);
	BOOST_CHECK_EQUAL (spans[2].size (),1);

	BOOST_CHECK_EQUAL (spans[3].first(),0);
	BOOST_CHECK_EQUAL (spans[3].step (),1);
	BOOST_CHECK_EQUAL (spans[3].last (),2);
	BOOST_CHECK_EQUAL (spans[3].size (),3);

	BOOST_CHECK_EQUAL (spans[4].first(),1);
	BOOST_CHECK_EQUAL (spans[4].step (),1);
	BOOST_CHECK_EQUAL (spans[4].last (),2);
	BOOST_CHECK_EQUAL (spans[4].size (),2);

	BOOST_CHECK_EQUAL (spans[5].first(),0);
	BOOST_CHECK_EQUAL (spans[5].step (),2);
	BOOST_CHECK_EQUAL (spans[5].last (),4);
	BOOST_CHECK_EQUAL (spans[5].size (),3);

	BOOST_CHECK_EQUAL (spans[6].first(),1);
	BOOST_CHECK_EQUAL (spans[6].step (),2);
	BOOST_CHECK_EQUAL (spans[6].last (),3);
	BOOST_CHECK_EQUAL (spans[6].size (),2);

	BOOST_CHECK_EQUAL (spans[7].first(),1);
	BOOST_CHECK_EQUAL (spans[7].step (),3);
	BOOST_CHECK_EQUAL (spans[7].last (),4);
	BOOST_CHECK_EQUAL (spans[7].size (),2);

	BOOST_CHECK_EQUAL (spans[8].first(),1);
	BOOST_CHECK_EQUAL (spans[8].step (),3);
	BOOST_CHECK_EQUAL (spans[8].last (),7);
	BOOST_CHECK_EQUAL (spans[8].size (),3);


	BOOST_CHECK_THROW ( span_type( 1,0,3 ), std::runtime_error  );
	BOOST_CHECK_THROW ( span_type( 1,2,0 ), std::runtime_error  );

}



BOOST_FIXTURE_TEST_CASE( copy_ctor_test, fixture )
{
	using span_type = boost::numeric::ublas::strided_span;


	BOOST_CHECK_EQUAL (span_type(spans[0]).first(),0);
	BOOST_CHECK_EQUAL (span_type(spans[0]).step (),0);
	BOOST_CHECK_EQUAL (span_type(spans[0]).last (),0);
	BOOST_CHECK_EQUAL (span_type(spans[0]).size (),0);

	BOOST_CHECK_EQUAL (span_type(spans[1]).first(),0);
	BOOST_CHECK_EQUAL (span_type(spans[1]).step (),0);
	BOOST_CHECK_EQUAL (span_type(spans[1]).last (),0);
	BOOST_CHECK_EQUAL (span_type(spans[1]).size (),1);

	BOOST_CHECK_EQUAL (span_type(spans[2]).first(),0);
	BOOST_CHECK_EQUAL (span_type(spans[2]).step (),1);
	BOOST_CHECK_EQUAL (span_type(spans[2]).last (),0);
	BOOST_CHECK_EQUAL (span_type(spans[2]).size (),1);

	BOOST_CHECK_EQUAL (span_type(spans[3]).first(),0);
	BOOST_CHECK_EQUAL (span_type(spans[3]).step (),1);
	BOOST_CHECK_EQUAL (span_type(spans[3]).last (),2);
	BOOST_CHECK_EQUAL (span_type(spans[3]).size (),3);

	BOOST_CHECK_EQUAL (span_type(spans[4]).first(),1);
	BOOST_CHECK_EQUAL (span_type(spans[4]).step (),1);
	BOOST_CHECK_EQUAL (span_type(spans[4]).last (),2);
	BOOST_CHECK_EQUAL (span_type(spans[4]).size (),2);


	BOOST_CHECK_EQUAL (span_type(spans[5]).first(),0);
	BOOST_CHECK_EQUAL (span_type(spans[5]).step (),2);
	BOOST_CHECK_EQUAL (span_type(spans[5]).last (),4);
	BOOST_CHECK_EQUAL (span_type(spans[5]).size (),3);

	BOOST_CHECK_EQUAL (span_type(spans[6]).first(),1);
	BOOST_CHECK_EQUAL (span_type(spans[6]).step (),2);
	BOOST_CHECK_EQUAL (span_type(spans[6]).last (),3);
	BOOST_CHECK_EQUAL (span_type(spans[6]).size (),2);

	BOOST_CHECK_EQUAL (span_type(spans[7]).first(),1);
	BOOST_CHECK_EQUAL (span_type(spans[7]).step (),3);
	BOOST_CHECK_EQUAL (span_type(spans[7]).last (),4);
	BOOST_CHECK_EQUAL (span_type(spans[7]).size (),2);

	BOOST_CHECK_EQUAL (span_type(spans[8]).first(),1);
	BOOST_CHECK_EQUAL (span_type(spans[8]).step (),3);
	BOOST_CHECK_EQUAL (span_type(spans[8]).last (),7);
	BOOST_CHECK_EQUAL (span_type(spans[8]).size (),3);


}


BOOST_FIXTURE_TEST_CASE( assignment_operator_test, fixture )
{
	auto c0 = spans[1];
	BOOST_CHECK_EQUAL ((c0=spans[0]).first(),0);
	BOOST_CHECK_EQUAL ((c0=spans[0]).step (),0);
	BOOST_CHECK_EQUAL ((c0=spans[0]).last (),0);
	BOOST_CHECK_EQUAL ((c0=spans[0]).size (),0);

	auto c1 = spans[2];
	BOOST_CHECK_EQUAL ((c1=spans[1]).first(),0);
	BOOST_CHECK_EQUAL ((c1=spans[1]).step (),0);
	BOOST_CHECK_EQUAL ((c1=spans[1]).last (),0);
	BOOST_CHECK_EQUAL ((c1=spans[1]).size (),1);

	auto c2 = spans[3];
	BOOST_CHECK_EQUAL ((c2=spans[2]).first(),0);
	BOOST_CHECK_EQUAL ((c2=spans[2]).step (),1);
	BOOST_CHECK_EQUAL ((c2=spans[2]).last (),0);
	BOOST_CHECK_EQUAL ((c2=spans[2]).size (),1);

	auto c3 = spans[4];
	BOOST_CHECK_EQUAL ((c3=spans[3]).first(),0);
	BOOST_CHECK_EQUAL ((c3=spans[3]).step (),1);
	BOOST_CHECK_EQUAL ((c3=spans[3]).last (),2);
	BOOST_CHECK_EQUAL ((c3=spans[3]).size (),3);

	auto c4 = spans[5];
	BOOST_CHECK_EQUAL ((c4=spans[4]).first(),1);
	BOOST_CHECK_EQUAL ((c4=spans[4]).step (),1);
	BOOST_CHECK_EQUAL ((c4=spans[4]).last (),2);
	BOOST_CHECK_EQUAL ((c4=spans[4]).size (),2);

	auto c5 = spans[6];
	BOOST_CHECK_EQUAL ((c5=spans[5]).first(),0);
	BOOST_CHECK_EQUAL ((c5=spans[5]).step (),2);
	BOOST_CHECK_EQUAL ((c5=spans[5]).last (),4);
	BOOST_CHECK_EQUAL ((c5=spans[5]).size (),3);

	auto c6 = spans[7];
	BOOST_CHECK_EQUAL ((c6=spans[6]).first(),1);
	BOOST_CHECK_EQUAL ((c6=spans[6]).step (),2);
	BOOST_CHECK_EQUAL ((c6=spans[6]).last (),3);
	BOOST_CHECK_EQUAL ((c6=spans[6]).size (),2);

	auto c7 = spans[8];
	BOOST_CHECK_EQUAL ((c7=spans[7]).first(),1);
	BOOST_CHECK_EQUAL ((c7=spans[7]).step (),3);
	BOOST_CHECK_EQUAL ((c7=spans[7]).last (),4);
	BOOST_CHECK_EQUAL ((c7=spans[7]).size (),2);

}

BOOST_FIXTURE_TEST_CASE( access_operator_test, fixture )
{

	BOOST_CHECK_EQUAL(spans[0][0], 0);

	BOOST_CHECK_EQUAL(spans[1][0], 0);

	BOOST_CHECK_EQUAL(spans[2][0], 0);

	BOOST_CHECK_EQUAL(spans[3][0], 0);
	BOOST_CHECK_EQUAL(spans[3][1], 1);
	BOOST_CHECK_EQUAL(spans[3][2], 2);

	BOOST_CHECK_EQUAL(spans[4][0], 1);
	BOOST_CHECK_EQUAL(spans[4][1], 2);

	BOOST_CHECK_EQUAL(spans[5][0], 0);
	BOOST_CHECK_EQUAL(spans[5][1], 2);
	BOOST_CHECK_EQUAL(spans[5][2], 4);

	BOOST_CHECK_EQUAL(spans[6][0], 1);
	BOOST_CHECK_EQUAL(spans[6][1], 3);

	BOOST_CHECK_EQUAL(spans[7][0], 1);
	BOOST_CHECK_EQUAL(spans[7][1], 4);

	BOOST_CHECK_EQUAL(spans[8][0], 1);
	BOOST_CHECK_EQUAL(spans[8][1], 4);
	BOOST_CHECK_EQUAL(spans[8][2], 7);

}

BOOST_FIXTURE_TEST_CASE( ran_test, fixture )
{
	using namespace boost::numeric::ublas;

	BOOST_CHECK ( ( ran(0,0,0) == spans[0]) );

	BOOST_CHECK ( ( ran(0,1,0) == spans[2]) );
	BOOST_CHECK ( ( ran(0,  0) == spans[2]) );


	BOOST_CHECK ( ( ran(0,1,2) == spans[3]) );
	BOOST_CHECK ( ( ran(0,  2) == spans[3]) );

	BOOST_CHECK ( ( ran(1,1,2) == spans[4]) );
	BOOST_CHECK ( ( ran(1,  2) == spans[4]) );

	BOOST_CHECK ( ( ran(0,2,4) == spans[5]) );
	BOOST_CHECK ( ( ran(1,2,4) == spans[6]) );
	BOOST_CHECK ( ( ran(1,3,5) == spans[7]) );
	BOOST_CHECK ( ( ran(1,3,7) == spans[8]) );
}

BOOST_AUTO_TEST_SUITE_END();
