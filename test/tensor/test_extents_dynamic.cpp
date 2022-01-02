//
// 	Copyright (c) 2018  Cem Bassoy, cem.bassoy@gmail.com
// 	Copyright (c) 2019, Amit Singh, amitsingh19975@gmail.com
//
//  Distributed under the Boost Software License, Version 1.0. (See
//  accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)
//
//  The authors gratefully acknowledge the support of
//  Google and Fraunhofer IOSB, Ettlingen, Germany
//

#include <boost/test/unit_test.hpp>
#include <boost/numeric/ublas/tensor/extents.hpp>

BOOST_AUTO_TEST_SUITE ( test_extents_dynamic )


struct fixture
{
  using extents = boost::numeric::ublas::extents<>;

//  static inline auto n     = extents{};
  static inline auto n11   = extents{1,1};
  static inline auto n12   = extents{1,2};
  static inline auto n21   = extents{2,1};
  static inline auto n22   = extents{2,2};
  static inline auto n32   = extents{3,2};
  static inline auto n111  = extents{1,1,1};
  static inline auto n211  = extents{2,1,1};
  static inline auto n121  = extents{1,2,1};
  static inline auto n112  = extents{1,1,2};
  static inline auto n123  = extents{1,2,3};
  static inline auto n321  = extents{3,2,1};
  static inline auto n213  = extents{2,1,3};
  static inline auto n432  = extents{4,3,2};
};


BOOST_FIXTURE_TEST_CASE(test_extents_dynamic_empty,
                        fixture,
                        *boost::unit_test::label("dynamic_extents") *boost::unit_test::label("empty"))
{
  namespace ublas = boost::numeric::ublas;

//  BOOST_CHECK( ublas::empty(n   ));
  BOOST_CHECK(!ublas::empty(n11 ));
  BOOST_CHECK(!ublas::empty(n12 ));
  BOOST_CHECK(!ublas::empty(n21 ));
  BOOST_CHECK(!ublas::empty(n22 ));
  BOOST_CHECK(!ublas::empty(n32 ));
  BOOST_CHECK(!ublas::empty(n111));
  BOOST_CHECK(!ublas::empty(n211));
  BOOST_CHECK(!ublas::empty(n121));
  BOOST_CHECK(!ublas::empty(n112));
  BOOST_CHECK(!ublas::empty(n123));
  BOOST_CHECK(!ublas::empty(n321));
  BOOST_CHECK(!ublas::empty(n213));
  BOOST_CHECK(!ublas::empty(n432));

  BOOST_CHECK_THROW( extents({1,1,0}), std::invalid_argument);
  BOOST_CHECK_THROW( extents({1,0})  , std::invalid_argument);
  BOOST_CHECK_THROW( extents({0}  )  , std::length_error    );
  BOOST_CHECK_THROW( extents({0,1})  , std::invalid_argument);
}



BOOST_FIXTURE_TEST_CASE(test_extents_dynamic_size,
                        fixture,
                        *boost::unit_test::label("dynamic_extents") *boost::unit_test::label("size"))
{
  namespace ublas = boost::numeric::ublas;


//  BOOST_CHECK_EQUAL(ublas::size(n   ),0);
  BOOST_CHECK_EQUAL(ublas::size(n11 ),2);
  BOOST_CHECK_EQUAL(ublas::size(n12 ),2);
  BOOST_CHECK_EQUAL(ublas::size(n21 ),2);
  BOOST_CHECK_EQUAL(ublas::size(n22 ),2);
  BOOST_CHECK_EQUAL(ublas::size(n32 ),2);
  BOOST_CHECK_EQUAL(ublas::size(n111),3);
  BOOST_CHECK_EQUAL(ublas::size(n211),3);
  BOOST_CHECK_EQUAL(ublas::size(n121),3);
  BOOST_CHECK_EQUAL(ublas::size(n112),3);
  BOOST_CHECK_EQUAL(ublas::size(n123),3);
  BOOST_CHECK_EQUAL(ublas::size(n321),3);
  BOOST_CHECK_EQUAL(ublas::size(n213),3);
  BOOST_CHECK_EQUAL(ublas::size(n432),3);
}



BOOST_FIXTURE_TEST_CASE(test_extents_dynamic_at_read,
                       fixture,
                       *boost::unit_test::label("dynamic_extents") *boost::unit_test::label("at_read"))
{

  BOOST_CHECK_EQUAL(n11 .at(0),1);
  BOOST_CHECK_EQUAL(n11 .at(1),1);

  BOOST_CHECK_EQUAL(n12 .at(0),1);
  BOOST_CHECK_EQUAL(n12 .at(1),2);

  BOOST_CHECK_EQUAL(n21 .at(0),2);
  BOOST_CHECK_EQUAL(n21 .at(1),1);

  BOOST_CHECK_EQUAL(n22 .at(0),2);
  BOOST_CHECK_EQUAL(n22 .at(1),2);

  BOOST_CHECK_EQUAL(n32 .at(0),3);
  BOOST_CHECK_EQUAL(n32 .at(1),2);

  BOOST_CHECK_EQUAL(n432.at(0),4);
  BOOST_CHECK_EQUAL(n432.at(1),3);
  BOOST_CHECK_EQUAL(n432.at(2),2);


//  BOOST_CHECK_THROW( (void)n  .at(0), std::out_of_range);
  BOOST_CHECK_THROW( (void)n32.at(2), std::out_of_range);
  BOOST_CHECK_THROW( (void)n32.at(5), std::out_of_range);
}


BOOST_FIXTURE_TEST_CASE(test_extents_dynamic_at_write,
                        fixture,
                        *boost::unit_test::label("dynamic_extents") *boost::unit_test::label("at_write"))
{

  auto n34 = extents{1,1};
  n34 = extents{3,4};
  BOOST_CHECK_EQUAL(n34.at(0),3);
  BOOST_CHECK_EQUAL(n34.at(1),4);


  auto n345 = extents{1,1,1};
  n345 = extents{3,4,5};
  BOOST_CHECK_EQUAL(n345.at(0),3);
  BOOST_CHECK_EQUAL(n345.at(1),4);
  BOOST_CHECK_EQUAL(n345.at(2),5);


  auto n5432 = extents{1,1,1,1};
  n5432 = extents{5,4,3,2};
  BOOST_CHECK_EQUAL(n5432.at(0),5);
  BOOST_CHECK_EQUAL(n5432.at(1),4);
  BOOST_CHECK_EQUAL(n5432.at(2),3);
  BOOST_CHECK_EQUAL(n5432.at(3),2);
}


BOOST_FIXTURE_TEST_CASE(test_extents_dynamic_operator_access_read,
                        fixture,
                        *boost::unit_test::label("dynamic_extents") *boost::unit_test::label("operator_access_read"))
{

  BOOST_CHECK_EQUAL(n11 [0],1);
  BOOST_CHECK_EQUAL(n11 [1],1);

  BOOST_CHECK_EQUAL(n12 [0],1);
  BOOST_CHECK_EQUAL(n12 [1],2);

  BOOST_CHECK_EQUAL(n21 [0],2);
  BOOST_CHECK_EQUAL(n21 [1],1);

  BOOST_CHECK_EQUAL(n22 [0],2);
  BOOST_CHECK_EQUAL(n22 [1],2);

  BOOST_CHECK_EQUAL(n32 [0],3);
  BOOST_CHECK_EQUAL(n32 [1],2);

  BOOST_CHECK_EQUAL(n432[0],4);
  BOOST_CHECK_EQUAL(n432[1],3);
  BOOST_CHECK_EQUAL(n432[2],2);
}

BOOST_AUTO_TEST_SUITE_END()
