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



#include <boost/numeric/ublas/tensor/operators_comparison.hpp>
#include <boost/numeric/ublas/tensor/operators_arithmetic.hpp>
#include <boost/numeric/ublas/tensor.hpp>
#include <boost/test/unit_test.hpp>
#include <boost/multiprecision/cpp_bin_float.hpp>
#include "utility.hpp"


BOOST_AUTO_TEST_SUITE(test_subtensor_comparison/*, * boost::unit_test::depends_on("test_subtensor")*/)

using double_extended = boost::multiprecision::cpp_bin_float_double_extended;

using test_types = zip<int,float,double_extended>::with_t<boost::numeric::ublas::layout::first_order, boost::numeric::ublas::layout::last_order>;

struct fixture {
  using extents_type = boost::numeric::ublas::extents<>;
  fixture()
    : extents{
        //                extents_type{},    // 0
        extents_type{1,1}, // 1
        extents_type{1,2}, // 2
        extents_type{2,1}, // 3
        extents_type{2,3}, // 4
        extents_type{2,3,1}, // 5
        extents_type{4,1,3}, // 6
        extents_type{1,2,3}, // 7
        extents_type{4,2,3}, // 8
        extents_type{4,2,3,5}} // 9
  {
  }
  std::vector<extents_type> extents;
};


BOOST_FIXTURE_TEST_CASE_TEMPLATE( test_subtensor_comparison, value,  test_types, fixture)
{
  namespace ublas = boost::numeric::ublas;
  using value_type  = typename value::first_type;
  using layout_type = typename value::second_type;
  using tensor_type  = ublas::tensor_dynamic<value_type,layout_type>;
  using subtensor = typename tensor_type::subtensor_type;

  auto check = [](auto const& e)
  {
    auto t  = tensor_type (e);
    auto t2 = tensor_type (e);
    auto v  = value_type  {};
    auto s  = subtensor(t);


    std::iota(t.begin(), t.end(), v);
    std::iota(t2.begin(), t2.end(), v+2);

    BOOST_CHECK( s == s  );
    BOOST_CHECK( s != t2 );

    if(s.empty())
      return;

    BOOST_CHECK(!(s < s));
    BOOST_CHECK(!(s > s));
    BOOST_CHECK( s < t2 );
    BOOST_CHECK( t2 > s );
    BOOST_CHECK( s <= s );
    BOOST_CHECK( s >= s );
    BOOST_CHECK( s <= t2 );
    BOOST_CHECK( t2 >= s );
    BOOST_CHECK( t2 >= t2 );
    BOOST_CHECK( t2 >= s );
  };

  for(auto const& e : extents)
    check(e);

  auto e0 = extents.at(0);
  auto e1 = extents.at(1);
  auto e2 = extents.at(2);

  tensor_type t0(e0);
  tensor_type t1(e1);
  tensor_type t2(e2);

  auto b = false;
  BOOST_CHECK_NO_THROW ( b = (subtensor(t0) == subtensor(t0)));
  BOOST_CHECK_NO_THROW ( b = (subtensor(t1) == subtensor(t2)));
  BOOST_CHECK_NO_THROW ( b = (subtensor(t0) == subtensor(t2)));
  BOOST_CHECK_NO_THROW ( b = (subtensor(t1) != subtensor(t2)));

  BOOST_CHECK_THROW    ( b = (subtensor(t1) >= subtensor(t2)), std::runtime_error  );
  BOOST_CHECK_THROW    ( b = (subtensor(t1) <= subtensor(t2)), std::runtime_error  );
  BOOST_CHECK_THROW    ( b = (subtensor(t1) <  subtensor(t2)), std::runtime_error  );
  BOOST_CHECK_THROW    ( b = (subtensor(t1) >  subtensor(t2)), std::runtime_error  );

}


BOOST_FIXTURE_TEST_CASE_TEMPLATE( test_subtensor_comparison_with_tensor_expressions, value,  test_types, fixture)
{
  namespace ublas = boost::numeric::ublas;
  using value_type  = typename value::first_type;
  using layout_type = typename value::second_type;
  using tensor_type  = ublas::tensor_dynamic<value_type,layout_type>;
  using subtensor = typename tensor_type::subtensor_type;


  auto check = [](auto const& e)
  {
    auto t  = tensor_type (e);
    auto t2 = tensor_type (e);
    auto v  = value_type  {};
    auto s  = subtensor(t);

    std::iota(t.begin(), t.end(), v);
    std::iota(t2.begin(), t2.end(), v+2);

    BOOST_CHECK( s == s  );
    BOOST_CHECK( s != t2 );

    if(s.empty())
      return;

    BOOST_CHECK( !(s < s) );
    BOOST_CHECK( !(s > s) );
    BOOST_CHECK( s < (t2+s) );
    BOOST_CHECK( (t2+s) > s );
    BOOST_CHECK( s <= (s+s) );
    BOOST_CHECK( (s+t2) >= s );
    BOOST_CHECK( (t2+t2+2) >= s);
    BOOST_CHECK( 2*t2 > s );
    BOOST_CHECK( s < 2*t2 );
    BOOST_CHECK( 2*t2 > s);
    BOOST_CHECK( 2*t2 >= t2 );
    BOOST_CHECK( t2 <= 2*t2);
    BOOST_CHECK( 3*t2 >= s );

  };

  for(auto const& e : extents)
    check(e);

  auto e0 = extents.at(0);
  auto e1 = extents.at(1);
  auto e2 = extents.at(2);


  tensor_type t0(e0);
  tensor_type t1(e1);
  tensor_type t2(e2);

  auto b = false;
  BOOST_CHECK_NO_THROW (b = subtensor(t0) == (subtensor(t0) + subtensor(t0))  );
  BOOST_CHECK_NO_THROW (b = subtensor(t1) == (subtensor(t2) + subtensor(t2))  );
  BOOST_CHECK_NO_THROW (b = subtensor(t0) == (subtensor(t2) + 2) );
  BOOST_CHECK_NO_THROW (b = subtensor(t1) != (2 + subtensor(t2)) );

  BOOST_CHECK_NO_THROW (b = (subtensor(t0) + subtensor(t0)) == subtensor(t0) );
  BOOST_CHECK_NO_THROW (b = (subtensor(t2) + subtensor(t2)) == subtensor(t1) );
  BOOST_CHECK_NO_THROW (b = (subtensor(t2) + 2)               == subtensor(t0) );
  BOOST_CHECK_NO_THROW (b = (2 + subtensor(t2))               != subtensor(t1) );

  BOOST_CHECK_THROW    (b = subtensor(t1) >= (subtensor(t2) + subtensor(t2)), std::runtime_error  );
  BOOST_CHECK_THROW    (b = subtensor(t1) <= (subtensor(t2) + subtensor(t2)), std::runtime_error  );
  BOOST_CHECK_THROW    (b = subtensor(t1) <  (subtensor(t2) + subtensor(t2)), std::runtime_error  );
  BOOST_CHECK_THROW    (b = subtensor(t1) >  (subtensor(t2) + subtensor(t2)), std::runtime_error  );

  BOOST_CHECK_THROW    (b = subtensor(t1) >= (subtensor(t2) + 2), std::runtime_error  );
  BOOST_CHECK_THROW    (b = subtensor(t1) <= (2 + subtensor(t2)), std::runtime_error  );
  BOOST_CHECK_THROW    (b = subtensor(t1) <  (subtensor(t2) + 3), std::runtime_error  );
  BOOST_CHECK_THROW    (b = subtensor(t1) >  (4 + subtensor(t2)), std::runtime_error  );

}



BOOST_FIXTURE_TEST_CASE_TEMPLATE( test_subtensor_comparison_with_scalar, value,  test_types, fixture)
{
  namespace ublas = boost::numeric::ublas;
  using value_type  = typename value::first_type;
  using layout_type = typename value::second_type;
  using tensor_type  = ublas::tensor_dynamic<value_type,layout_type>;
  using subtensor = typename tensor_type::subtensor_type;


  auto check = [](auto const& e)
  {

    tensor_type t2 = tensor_type(e,2);
    tensor_type t3 = tensor_type(e,3);

    BOOST_CHECK( subtensor(t2) == subtensor(t2)  );
    BOOST_CHECK( subtensor(t2) != subtensor(t3)  );

    if(ublas::empty(e))
      return;


    BOOST_CHECK( !(subtensor(t2) <  2) );
    BOOST_CHECK( !(subtensor(t2) >  2) );
    BOOST_CHECK(  (subtensor(t2) >= 2) );
    BOOST_CHECK(  (subtensor(t2) <= 2) );
    BOOST_CHECK(  (subtensor(t2) == 2) );
    BOOST_CHECK(  (subtensor(t2) != 3) );

    BOOST_CHECK( !(2 >  subtensor(t2)) );
    BOOST_CHECK( !(2 <  subtensor(t2)) );
    BOOST_CHECK(  (2 <= subtensor(t2)) );
    BOOST_CHECK(  (2 >= subtensor(t2)) );
    BOOST_CHECK(  (2 == subtensor(t2)) );
    BOOST_CHECK(  (3 != subtensor(t2)) );

    BOOST_CHECK( !( subtensor(t2)+3 <  5) );
    BOOST_CHECK( !( subtensor(t2)+3 >  5) );
    BOOST_CHECK(  ( subtensor(t2)+3 >= 5) );
    BOOST_CHECK(  ( subtensor(t2)+3 <= 5) );
    BOOST_CHECK(  ( subtensor(t2)+3 == 5) );
    BOOST_CHECK(  ( subtensor(t2)+3 != 6) );


    BOOST_CHECK( !( 5 >  subtensor(t2)+3) );
    BOOST_CHECK( !( 5 <  subtensor(t2)+3) );
    BOOST_CHECK(  ( 5 >= subtensor(t2)+3) );
    BOOST_CHECK(  ( 5 <= subtensor(t2)+3) );
    BOOST_CHECK(  ( 5 == subtensor(t2)+3) );
    BOOST_CHECK(  ( 6 != subtensor(t2)+3) );


    BOOST_CHECK( !( subtensor(t2)+subtensor(t3) <  5) );
    BOOST_CHECK( !( subtensor(t2)+subtensor(t3) >  5) );
    BOOST_CHECK(  ( subtensor(t2)+subtensor(t3) >= 5) );
    BOOST_CHECK(  ( subtensor(t2)+subtensor(t3) <= 5) );
    BOOST_CHECK(  ( subtensor(t2)+subtensor(t3) == 5) );
    BOOST_CHECK(  ( subtensor(t2)+subtensor(t3) != 6) );


    BOOST_CHECK( !( 5 >  subtensor(t2)+subtensor(t3)) );
    BOOST_CHECK( !( 5 <  subtensor(t2)+subtensor(t3)) );
    BOOST_CHECK(  ( 5 >= subtensor(t2)+subtensor(t3)) );
    BOOST_CHECK(  ( 5 <= subtensor(t2)+subtensor(t3)) );
    BOOST_CHECK(  ( 5 == subtensor(t2)+subtensor(t3)) );
    BOOST_CHECK(  ( 6 != subtensor(t2)+subtensor(t3)) );

  };

  for(auto const& e : extents)
    check(e);

}


BOOST_AUTO_TEST_SUITE_END()
