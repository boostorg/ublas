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
	using span_type  = boost::numeric::ublas::span<>;

	fixture() :
		spans {
				span_type{},      // 0
				span_type(0,4),   // 1
				span_type(2,6),   // 2
				span_type(0,0,0), // 3
				span_type(0,1,0), // 4
				span_type(0,1,2), // 5
				span_type(1,1,2), // 6
				span_type(0,2,4), // 7
				span_type(1,2,4), // 8
				span_type(1,3,5), // 9
				span_type(1,3,7)  // 10
				}
	{}
	std::vector<span_type> spans;
};



BOOST_FIXTURE_TEST_CASE( ctor_test, fixture )
{
	using span_type = boost::numeric::ublas::span<>;

	BOOST_CHECK_EQUAL (spans[0].first(),0);
	BOOST_CHECK_EQUAL (spans[0].step (),1);
	BOOST_CHECK_EQUAL (spans[0].last (),boost::numeric::ublas::max);

	BOOST_CHECK_EQUAL (spans[1].first(),0);
	BOOST_CHECK_EQUAL (spans[1].step (),1);
	BOOST_CHECK_EQUAL (spans[1].last (),4);
	BOOST_CHECK_EQUAL (spans[1].size (),5);

	BOOST_CHECK_EQUAL (spans[2].first(),2);
	BOOST_CHECK_EQUAL (spans[2].step (),1);
	BOOST_CHECK_EQUAL (spans[2].last (),6);
	BOOST_CHECK_EQUAL (spans[2].size (),5);

	BOOST_CHECK_EQUAL (spans[3].first(),0);
	BOOST_CHECK_EQUAL (spans[3].step (),1);
	BOOST_CHECK_EQUAL (spans[3].last (),0);
	BOOST_CHECK_EQUAL (spans[3].size (),1);

	BOOST_CHECK_EQUAL (spans[4].first(),0);
	BOOST_CHECK_EQUAL (spans[4].step (),1);
	BOOST_CHECK_EQUAL (spans[4].last (),0);
	BOOST_CHECK_EQUAL (spans[4].size (),1);

	BOOST_CHECK_EQUAL (spans[5].first(),0);
	BOOST_CHECK_EQUAL (spans[5].step (),1);
	BOOST_CHECK_EQUAL (spans[5].last (),2);
	BOOST_CHECK_EQUAL (spans[5].size (),3);

	BOOST_CHECK_EQUAL (spans[6].first(),1);
	BOOST_CHECK_EQUAL (spans[6].step (),1);
	BOOST_CHECK_EQUAL (spans[6].last (),2);
	BOOST_CHECK_EQUAL (spans[6].size (),2);

	BOOST_CHECK_EQUAL (spans[7].first(),0);
	BOOST_CHECK_EQUAL (spans[7].step (),2);
	BOOST_CHECK_EQUAL (spans[7].last (),4);
	BOOST_CHECK_EQUAL (spans[7].size (),3);

	BOOST_CHECK_EQUAL (spans[8].first(),1);
	BOOST_CHECK_EQUAL (spans[8].step (),2);
	BOOST_CHECK_EQUAL (spans[8].last (),3);
	BOOST_CHECK_EQUAL (spans[8].size (),2);

	BOOST_CHECK_EQUAL (spans[9].first(),1);
	BOOST_CHECK_EQUAL (spans[9].step (),3);
	BOOST_CHECK_EQUAL (spans[9].last (),4);
	BOOST_CHECK_EQUAL (spans[9].size (),2);

	BOOST_CHECK_EQUAL (spans[10].first(),1);
	BOOST_CHECK_EQUAL (spans[10].step (),3);
	BOOST_CHECK_EQUAL (spans[10].last (),7);
	BOOST_CHECK_EQUAL (spans[10].size (),3);


	BOOST_CHECK_THROW ( span_type( 1,0,3 ), std::runtime_error  );
	BOOST_CHECK_THROW ( span_type( 1,2,0 ), std::runtime_error  );

}



BOOST_FIXTURE_TEST_CASE( copy_ctor_test, fixture )
{
	using span_type = boost::numeric::ublas::span<>;


	BOOST_CHECK_EQUAL (span_type(spans[0]).first(),0);
	BOOST_CHECK_EQUAL (span_type(spans[0]).step (),1);
	BOOST_CHECK_EQUAL (span_type(spans[0]).last (),boost::numeric::ublas::max);

	BOOST_CHECK_EQUAL (span_type(spans[1]).first(),0);
	BOOST_CHECK_EQUAL (span_type(spans[1]).step (),1);
	BOOST_CHECK_EQUAL (span_type(spans[1]).last (),4);
	BOOST_CHECK_EQUAL (span_type(spans[1]).size (),5);

	BOOST_CHECK_EQUAL (span_type(spans[2]).first(),2);
	BOOST_CHECK_EQUAL (span_type(spans[2]).step (),1);
	BOOST_CHECK_EQUAL (span_type(spans[2]).last (),6);
	BOOST_CHECK_EQUAL (span_type(spans[2]).size (),5);

	BOOST_CHECK_EQUAL (span_type(spans[3]).first(),0);
	BOOST_CHECK_EQUAL (span_type(spans[3]).step (),1);
	BOOST_CHECK_EQUAL (span_type(spans[3]).last (),0);
	BOOST_CHECK_EQUAL (span_type(spans[3]).size (),1);

	BOOST_CHECK_EQUAL (span_type(spans[4]).first(),0);
	BOOST_CHECK_EQUAL (span_type(spans[4]).step (),1);
	BOOST_CHECK_EQUAL (span_type(spans[4]).last (),0);
	BOOST_CHECK_EQUAL (span_type(spans[4]).size (),1);

	BOOST_CHECK_EQUAL (span_type(spans[5]).first(),0);
	BOOST_CHECK_EQUAL (span_type(spans[5]).step (),1);
	BOOST_CHECK_EQUAL (span_type(spans[5]).last (),2);
	BOOST_CHECK_EQUAL (span_type(spans[5]).size (),3);

	BOOST_CHECK_EQUAL (span_type(spans[6]).first(),1);
	BOOST_CHECK_EQUAL (span_type(spans[6]).step (),1);
	BOOST_CHECK_EQUAL (span_type(spans[6]).last (),2);
	BOOST_CHECK_EQUAL (span_type(spans[6]).size (),2);

	BOOST_CHECK_EQUAL (span_type(spans[7]).first(),0);
	BOOST_CHECK_EQUAL (span_type(spans[7]).step (),2);
	BOOST_CHECK_EQUAL (span_type(spans[7]).last (),4);
	BOOST_CHECK_EQUAL (span_type(spans[7]).size (),3);

	BOOST_CHECK_EQUAL (span_type(spans[8]).first(),1);
	BOOST_CHECK_EQUAL (span_type(spans[8]).step (),2);
	BOOST_CHECK_EQUAL (span_type(spans[8]).last (),3);
	BOOST_CHECK_EQUAL (span_type(spans[8]).size (),2);

	BOOST_CHECK_EQUAL (span_type(spans[9]).first(),1);
	BOOST_CHECK_EQUAL (span_type(spans[9]).step (),3);
	BOOST_CHECK_EQUAL (span_type(spans[9]).last (),4);
	BOOST_CHECK_EQUAL (span_type(spans[9]).size (),2);

	BOOST_CHECK_EQUAL (span_type(spans[10]).first(),1);
	BOOST_CHECK_EQUAL (span_type(spans[10]).step (),3);
	BOOST_CHECK_EQUAL (span_type(spans[10]).last (),7);
	BOOST_CHECK_EQUAL (span_type(spans[10]).size (),3);

}


BOOST_FIXTURE_TEST_CASE( assignment_operator_test, fixture )
{
	auto c0 = spans[1];
	BOOST_CHECK_EQUAL ((c0=spans[0]).first(),0);
	BOOST_CHECK_EQUAL ((c0=spans[0]).step (),1);
	BOOST_CHECK_EQUAL ((c0=spans[0]).last (),boost::numeric::ublas::max);

	auto c1 = spans[2];
	BOOST_CHECK_EQUAL ((c1=spans[1]).first(),0);
	BOOST_CHECK_EQUAL ((c1=spans[1]).step (),1);
	BOOST_CHECK_EQUAL ((c1=spans[1]).last (),4);
	BOOST_CHECK_EQUAL ((c1=spans[1]).size (),5);

	auto c2 = spans[3];
	BOOST_CHECK_EQUAL ((c2=spans[2]).first(),2);
	BOOST_CHECK_EQUAL ((c2=spans[2]).step (),1);
	BOOST_CHECK_EQUAL ((c2=spans[2]).last (),6);
	BOOST_CHECK_EQUAL ((c2=spans[2]).size (),5);

	auto c3 = spans[4];
	BOOST_CHECK_EQUAL ((c3=spans[3]).first(),0);
	BOOST_CHECK_EQUAL ((c3=spans[3]).step (),1);
	BOOST_CHECK_EQUAL ((c3=spans[3]).last (),0);
	BOOST_CHECK_EQUAL ((c3=spans[3]).size (),1);

	auto c4 = spans[5];
	BOOST_CHECK_EQUAL ((c4=spans[4]).first(),0);
	BOOST_CHECK_EQUAL ((c4=spans[4]).step (),1);
	BOOST_CHECK_EQUAL ((c4=spans[4]).last (),0);
	BOOST_CHECK_EQUAL ((c4=spans[4]).size (),1);

	auto c5 = spans[6];
	BOOST_CHECK_EQUAL ((c5=spans[5]).first(),0);
	BOOST_CHECK_EQUAL ((c5=spans[5]).step (),1);
	BOOST_CHECK_EQUAL ((c5=spans[5]).last (),2);
	BOOST_CHECK_EQUAL ((c5=spans[5]).size (),3);

	auto c6 = spans[7];
	BOOST_CHECK_EQUAL ((c6=spans[6]).first(),1);
	BOOST_CHECK_EQUAL ((c6=spans[6]).step (),1);
	BOOST_CHECK_EQUAL ((c6=spans[6]).last (),2);
	BOOST_CHECK_EQUAL ((c6=spans[6]).size (),2);

	auto c7 = spans[8];
	BOOST_CHECK_EQUAL ((c7=spans[7]).first(),0);
	BOOST_CHECK_EQUAL ((c7=spans[7]).step (),2);
	BOOST_CHECK_EQUAL ((c7=spans[7]).last (),4);
	BOOST_CHECK_EQUAL ((c7=spans[7]).size (),3);

	auto c8 = spans[9];
	BOOST_CHECK_EQUAL ((c8=spans[8]).first(),1);
	BOOST_CHECK_EQUAL ((c8=spans[8]).step (),2);
	BOOST_CHECK_EQUAL ((c8=spans[8]).last (),3);
	BOOST_CHECK_EQUAL ((c8=spans[8]).size (),2);

	auto c9 = spans[10];
	BOOST_CHECK_EQUAL ((c9=spans[9]).first(),1);
	BOOST_CHECK_EQUAL ((c9=spans[9]).step (),3);
	BOOST_CHECK_EQUAL ((c9=spans[9]).last (),4);
	BOOST_CHECK_EQUAL ((c9=spans[9]).size (),2);

}


BOOST_AUTO_TEST_SUITE_END();
