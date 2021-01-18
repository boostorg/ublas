//
// 	Copyright (c) 2021 Cem Bassoy, cem.bassoy@gmail.com
//
//  Distributed under the Boost Software License, Version 1.0. (See
//  accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)
//

#include <boost/test/unit_test.hpp>
#include <boost/numeric/ublas/tensor/fixed_rank_extents.hpp>

BOOST_AUTO_TEST_SUITE ( test_shape_dynamic_static_rank )


struct fixture
{
  template<std::size_t N>
  using shape_t = boost::numeric::ublas::basic_fixed_rank_extents<unsigned, N>;

  static inline auto n     = shape_t<0>{};
  static inline auto n1    = shape_t<1>{1};
  static inline auto n2    = shape_t<1>{2};
  static inline auto n11   = shape_t<2>{1,1};
  static inline auto n12   = shape_t<2>{1,2};
  static inline auto n21   = shape_t<2>{2,1};
  static inline auto n22   = shape_t<2>{2,2};
  static inline auto n32   = shape_t<2>{3,2};
  static inline auto n111  = shape_t<3>{1,1,1};
  static inline auto n211  = shape_t<3>{2,1,1};
  static inline auto n121  = shape_t<3>{1,2,1};
  static inline auto n112  = shape_t<3>{1,1,2};
  static inline auto n123  = shape_t<3>{1,2,3};
  static inline auto n321  = shape_t<3>{3,2,1};
  static inline auto n213  = shape_t<3>{2,1,3};
  static inline auto n432  = shape_t<3>{4,3,2};
};


BOOST_FIXTURE_TEST_CASE(test_extents_dynamic_rank_static_empty,
                        fixture,
                        *boost::unit_test::label("dynamic_extents_rank_static") *boost::unit_test::label("empty"))
{
  BOOST_CHECK( n   .empty());
  BOOST_CHECK(!n1  .empty());
  BOOST_CHECK(!n2  .empty());
  BOOST_CHECK(!n11 .empty());
  BOOST_CHECK(!n12 .empty());
  BOOST_CHECK(!n21 .empty());
  BOOST_CHECK(!n22 .empty());
  BOOST_CHECK(!n32 .empty());
  BOOST_CHECK(!n111.empty());
  BOOST_CHECK(!n211.empty());
  BOOST_CHECK(!n121.empty());
  BOOST_CHECK(!n112.empty());
  BOOST_CHECK(!n123.empty());
  BOOST_CHECK(!n321.empty());
  BOOST_CHECK(!n213.empty());
  BOOST_CHECK(!n432.empty());

  BOOST_CHECK_THROW( shape_t<3>({1,1,0}), std::invalid_argument);
  BOOST_CHECK_THROW( shape_t<2>({1,0}), std::invalid_argument);
  BOOST_CHECK_THROW( shape_t<1>({0}  ), std::invalid_argument);
  BOOST_CHECK_THROW( shape_t<2>({0,1}), std::invalid_argument);
}



BOOST_FIXTURE_TEST_CASE(test_extents_dynamic_rank_static_size,
                        fixture,
                        *boost::unit_test::label("dynamic_extents_rank_static") *boost::unit_test::label("size"))
{
  BOOST_CHECK_EQUAL(n   .size(),0);
  BOOST_CHECK_EQUAL(n1  .size(),1);
  BOOST_CHECK_EQUAL(n2  .size(),1);
  BOOST_CHECK_EQUAL(n11 .size(),2);
  BOOST_CHECK_EQUAL(n12 .size(),2);
  BOOST_CHECK_EQUAL(n21 .size(),2);
  BOOST_CHECK_EQUAL(n22 .size(),2);
  BOOST_CHECK_EQUAL(n32 .size(),2);
  BOOST_CHECK_EQUAL(n111.size(),3);
  BOOST_CHECK_EQUAL(n211.size(),3);
  BOOST_CHECK_EQUAL(n121.size(),3);
  BOOST_CHECK_EQUAL(n112.size(),3);
  BOOST_CHECK_EQUAL(n123.size(),3);
  BOOST_CHECK_EQUAL(n321.size(),3);
  BOOST_CHECK_EQUAL(n213.size(),3);
  BOOST_CHECK_EQUAL(n432.size(),3);
}



BOOST_FIXTURE_TEST_CASE(test_extents_dynamic_rank_static_at_read,
                        fixture,
                        *boost::unit_test::label("dynamic_extents_rank_static") *boost::unit_test::label("at_read"))
{
  BOOST_CHECK_EQUAL(n1  .at(0),1);
  BOOST_CHECK_EQUAL(n2  .at(0),2);

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


  BOOST_CHECK_THROW( (void)n  .at(0), std::out_of_range);
  BOOST_CHECK_THROW( (void)n32.at(2), std::out_of_range);
  BOOST_CHECK_THROW( (void)n32.at(5), std::out_of_range);
}


BOOST_FIXTURE_TEST_CASE(test_extents_dynamic_rank_static_at_write,
                        fixture,
                        *boost::unit_test::label("dynamic_extents_rank_static") *boost::unit_test::label("at_write"))
{
  auto n3 = shape_t<1>{1};
  n3.at(0)=3;
  BOOST_CHECK_EQUAL(n3.at(0),3);

  auto n34 = shape_t<2>{1,1};
  n34.at(0)=3;
  n34.at(1)=4;
  BOOST_CHECK_EQUAL(n34.at(0),3);
  BOOST_CHECK_EQUAL(n34.at(1),4);


  auto n345 = shape_t<3>{1,1,1};
  n345.at(0)=3;
  n345.at(1)=4;
  n345.at(2)=5;
  BOOST_CHECK_EQUAL(n345.at(0),3);
  BOOST_CHECK_EQUAL(n345.at(1),4);
  BOOST_CHECK_EQUAL(n345.at(2),5);


  auto n5432 = shape_t<4>{1,1,1,1};
  n5432.at(0)=5;
  n5432.at(1)=4;
  n5432.at(2)=3;
  n5432.at(3)=2;
  BOOST_CHECK_EQUAL(n5432.at(0),5);
  BOOST_CHECK_EQUAL(n5432.at(1),4);
  BOOST_CHECK_EQUAL(n5432.at(2),3);
  BOOST_CHECK_EQUAL(n5432.at(3),2);
}


BOOST_FIXTURE_TEST_CASE(test_extents_dynamic_rank_static_operator_access_read,
                        fixture,
                        *boost::unit_test::label("dynamic_extents_rank_static") *boost::unit_test::label("operator_access_read"))
{
  BOOST_CHECK_EQUAL(n1  [0],1);
  BOOST_CHECK_EQUAL(n2  [0],2);

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


BOOST_FIXTURE_TEST_CASE(test_extents_dynamic_rank_static_operator_access_write,
                        fixture,
                        *boost::unit_test::label("dynamic_extents_rank_static") *boost::unit_test::label("operator_access_write"))
{
  auto n3 = shape_t<1>{1};
  n3.at(0)=3;
  BOOST_CHECK_EQUAL(n3[0],3);

  auto n34 = shape_t<2>{1,1};
  n34[0]=3;
  n34[1]=4;
  BOOST_CHECK_EQUAL(n34[0],3);
  BOOST_CHECK_EQUAL(n34[1],4);


  auto n345 = shape_t<3>{1,1,1};
  n345[0]=3;
  n345[1]=4;
  n345[2]=5;
  BOOST_CHECK_EQUAL(n345[0],3);
  BOOST_CHECK_EQUAL(n345[1],4);
  BOOST_CHECK_EQUAL(n345[2],5);


  auto n5432 = shape_t<4>{1,1,1,1};
  n5432[0]=5;
  n5432[1]=4;
  n5432[2]=3;
  n5432[3]=2;
  BOOST_CHECK_EQUAL(n5432[0],5);
  BOOST_CHECK_EQUAL(n5432[1],4);
  BOOST_CHECK_EQUAL(n5432[2],3);
  BOOST_CHECK_EQUAL(n5432[3],2);
}


BOOST_AUTO_TEST_SUITE_END()
