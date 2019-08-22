
//  Copyright (c) 2018-2019 Cem Bassoy
//
//  Distributed under the Boost Software License, Version 1.0. (See
//  accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#include <boost/numeric/ublas/tensor/slice.hpp>
#include <boost/test/unit_test.hpp>
#include <type_traits>

BOOST_AUTO_TEST_SUITE ( test_slice )

namespace sp = boost::numeric::ublas::span;
using type = ptrdiff_t;

BOOST_AUTO_TEST_CASE(test_slice_basic_slice) {
  auto s1 = sp::basic_slice<type, 2,10,1>{};
  BOOST_CHECK_EQUAL( s1.first(), 2 );
  BOOST_CHECK_EQUAL( s1.last(), 10 );
  BOOST_CHECK_EQUAL( s1.step(), 1 );
  BOOST_CHECK_EQUAL( s1.size(), 9 );

  auto s2 = sp::basic_slice<type, 2,10,3>{};
  BOOST_CHECK_EQUAL( s2.first(), 2 );
  BOOST_CHECK_EQUAL( s2.last(), 8 );
  BOOST_CHECK_EQUAL( s2.step(), 3 );
  BOOST_CHECK_EQUAL( s2.size(), 3 );

  auto s3 = sp::basic_slice<type, 2,10>{};
  BOOST_CHECK_EQUAL( s3.first(), 2 );
  BOOST_CHECK_EQUAL( s3.last(), 10 );
  BOOST_CHECK_EQUAL( s3.step(), 1 );
  BOOST_CHECK_EQUAL( s3.size(), 9 );


  auto s4 = sp::basic_slice<type, 2,2>{};
  BOOST_CHECK_EQUAL( s4.first(), 2 );
  BOOST_CHECK_EQUAL( s4.last(), 2 );
  BOOST_CHECK_EQUAL( s4.step(), 1 );
  BOOST_CHECK_EQUAL( s4.size(), 1 );

  auto s5 = sp::basic_slice<type, 2>{};
  BOOST_CHECK_EQUAL( s5.first(), 2 );
  BOOST_CHECK_EQUAL( s5.last(), 2 );
  BOOST_CHECK_EQUAL( s5.step(), 1 );
  BOOST_CHECK_EQUAL( s5.size(), 1 );

  auto s6 = sp::basic_slice<type>{};
  BOOST_CHECK_EQUAL( s6.first(), 0 );
  BOOST_CHECK_EQUAL( s6.last(), 0 );
  BOOST_CHECK_EQUAL( s6.step(), 1 );
  BOOST_CHECK_EQUAL( s6.size(), 0 );


  auto s7 = sp::basic_slice<type>{2,10,1};
  BOOST_CHECK_EQUAL( s7.first(), 2 );
  BOOST_CHECK_EQUAL( s7.last(), 10 );
  BOOST_CHECK_EQUAL( s7.step(), 1 );
  BOOST_CHECK_EQUAL( s7.size(), 9 );


  auto s8 = sp::basic_slice<type>{2,10};
  BOOST_CHECK_EQUAL( s8.first(), 2 );
  BOOST_CHECK_EQUAL( s8.last(), 10 );
  BOOST_CHECK_EQUAL( s8.step(), 1 );
  BOOST_CHECK_EQUAL( s8.size(), 9 );


  auto s9 = sp::basic_slice<type>{2,2};
  BOOST_CHECK_EQUAL( s9.first(), 2 );
  BOOST_CHECK_EQUAL( s9.last(), 2 );
  BOOST_CHECK_EQUAL( s9.step(), 1 );
  BOOST_CHECK_EQUAL( s9.size(), 1 );

  auto s10 = sp::basic_slice<type>{2};
  BOOST_CHECK_EQUAL( s10.first(), 2 );
  BOOST_CHECK_EQUAL( s10.last(), 2 );
  BOOST_CHECK_EQUAL( s10.step(), 1 );
  BOOST_CHECK_EQUAL( s10.size(), 1 );
}

BOOST_AUTO_TEST_SUITE_END()