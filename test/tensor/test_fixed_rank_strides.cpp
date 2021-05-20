//
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

#if 0

#include <boost/test/unit_test.hpp>
#include <boost/numeric/ublas/tensor/strides.hpp>
#include <boost/numeric/ublas/tensor/extents.hpp>

BOOST_AUTO_TEST_SUITE(test_fixed_rank_strides)

using test_types = std::tuple<boost::numeric::ublas::layout::first_order, boost::numeric::ublas::layout::last_order>;

template<std::size_t N, class L>
using strides =boost::numeric::ublas::strides<boost::numeric::ublas::extents<N>,L>;

BOOST_AUTO_TEST_CASE_TEMPLATE( test_fixed_rank_strides_ctor, value, test_types)
{
  namespace ublas = boost::numeric::ublas;
//  using layout_type = value;
//  constexpr auto layout = value{};

  auto s0   = strides<0,value>{};
  auto s1   = strides<1,value>({1}     );
  auto s3   = strides<1,value>({3}     );
  auto s11  = strides<2,value>({1,1}   );
  auto s12  = strides<2,value>({1,2}   );
  auto s21  = strides<2,value>({2,1}   );
  auto s23  = strides<2,value>({2,3}   );
  auto s231 = strides<3,value>({2,3,1} );
  auto s123 = strides<3,value>({1,2,3} );
  auto s423 = strides<3,value>({4,2,3} );

  BOOST_CHECK       (   s0.empty());
  BOOST_CHECK       (!  s1.empty());
  BOOST_CHECK       (!  s3.empty());
  BOOST_CHECK       (! s11.empty());
  BOOST_CHECK       (! s12.empty());
  BOOST_CHECK       (! s21.empty());
  BOOST_CHECK       (! s23.empty());
  BOOST_CHECK       (!s231.empty());
  BOOST_CHECK       (!s123.empty());
  BOOST_CHECK       (!s423.empty());


  BOOST_CHECK_EQUAL (   s0.size(), 0);
  BOOST_CHECK_EQUAL (   s1.size(), 3);
  BOOST_CHECK_EQUAL (   s3.size(), 1);
  BOOST_CHECK_EQUAL (  s11.size(), 2);
  BOOST_CHECK_EQUAL (  s12.size(), 2);
  BOOST_CHECK_EQUAL (  s21.size(), 2);
  BOOST_CHECK_EQUAL (  s23.size(), 2);
  BOOST_CHECK_EQUAL ( s231.size(), 3);
  BOOST_CHECK_EQUAL ( s123.size(), 3);
  BOOST_CHECK_EQUAL ( s423.size(), 3);
}


BOOST_AUTO_TEST_CASE( test_fixed_rank_strides_ctor_access_first_order)
{
  using value = boost::numeric::ublas::layout::first_order;
//  constexpr auto layout = boost::numeric::ublas::layout::first_order{};

  auto s1   = strides<1,value>({1}     );
  auto s3   = strides<1,value>({3}     );
  auto s11  = strides<2,value>({1,1}   );
  auto s12  = strides<2,value>({1,2}   );
  auto s21  = strides<2,value>({2,1}   );
  auto s23  = strides<2,value>({2,3}   );
  auto s231 = strides<3,value>({2,3,1} );
  auto s213 = strides<3,value>({2,3,1} );
  auto s123 = strides<3,value>({1,2,3} );
  auto s423 = strides<3,value>({4,2,3} );


  BOOST_REQUIRE_EQUAL (   s1.size(),1);
  BOOST_REQUIRE_EQUAL (   s3.size(),1);
  BOOST_REQUIRE_EQUAL (  s11.size(),2);
  BOOST_REQUIRE_EQUAL (  s12.size(),2);
  BOOST_REQUIRE_EQUAL (  s21.size(),2);
  BOOST_REQUIRE_EQUAL (  s23.size(),2);
  BOOST_REQUIRE_EQUAL ( s231.size(),3);
  BOOST_REQUIRE_EQUAL ( s213.size(),3);
  BOOST_REQUIRE_EQUAL ( s123.size(),3);
  BOOST_REQUIRE_EQUAL ( s423.size(),3);

  BOOST_CHECK_EQUAL   ( s11[0], 1);
  BOOST_CHECK_EQUAL   ( s11[1], 1);

  BOOST_CHECK_EQUAL   ( s12[0], 1);
  BOOST_CHECK_EQUAL   ( s12[1], 1);

  BOOST_CHECK_EQUAL   ( s21[0], 1);
  BOOST_CHECK_EQUAL   ( s21[1], 1);

  BOOST_CHECK_EQUAL   ( s23[0], 1);
  BOOST_CHECK_EQUAL   ( s23[1], 2);

  BOOST_CHECK_EQUAL   ( s231[0], 1);
  BOOST_CHECK_EQUAL   ( s231[1], 2);
  BOOST_CHECK_EQUAL   ( s231[2], 6);

  BOOST_CHECK_EQUAL   ( s123[0], 1);
  BOOST_CHECK_EQUAL   ( s123[1], 1);
  BOOST_CHECK_EQUAL   ( s123[2], 2);

  BOOST_CHECK_EQUAL   ( s213[0], 1);
  BOOST_CHECK_EQUAL   ( s213[1], 2);
  BOOST_CHECK_EQUAL   ( s213[2], 2);

  BOOST_CHECK_EQUAL   ( s423[0], 1);
  BOOST_CHECK_EQUAL   ( s423[1], 4);
  BOOST_CHECK_EQUAL   ( s423[2], 8);
}

BOOST_AUTO_TEST_CASE( test_fixed_rank_strides_ctor_access_last_order)
{
  using value = boost::numeric::ublas::layout::first_order;
  //  constexpr auto layout = boost::numeric::ublas::layout::first_order{};

  auto s1   = strides<1,value>({1}     );
  auto s3   = strides<1,value>({3}     );
  auto s11  = strides<2,value>({1,1}   );
  auto s12  = strides<2,value>({1,2}   );
  auto s21  = strides<2,value>({2,1}   );
  auto s23  = strides<2,value>({2,3}   );
  auto s231 = strides<3,value>({2,3,1} );
  auto s213 = strides<3,value>({2,3,1} );
  auto s123 = strides<3,value>({1,2,3} );
  auto s423 = strides<3,value>({4,2,3} );

  BOOST_REQUIRE_EQUAL (   s1.size(),1);
  BOOST_REQUIRE_EQUAL (   s3.size(),1);
  BOOST_REQUIRE_EQUAL (  s11.size(),2);
  BOOST_REQUIRE_EQUAL (  s12.size(),2);
  BOOST_REQUIRE_EQUAL (  s21.size(),2);
  BOOST_REQUIRE_EQUAL (  s23.size(),2);
  BOOST_REQUIRE_EQUAL ( s231.size(),3);
  BOOST_REQUIRE_EQUAL ( s213.size(),3);
  BOOST_REQUIRE_EQUAL ( s123.size(),3);
  BOOST_REQUIRE_EQUAL ( s423.size(),3);

  BOOST_CHECK_EQUAL   ( s11[0], 1);
  BOOST_CHECK_EQUAL   ( s11[1], 1);

  BOOST_CHECK_EQUAL   ( s12[0], 1);
  BOOST_CHECK_EQUAL   ( s12[1], 1);

  BOOST_CHECK_EQUAL   ( s21[0], 1);
  BOOST_CHECK_EQUAL   ( s21[1], 1);

  BOOST_CHECK_EQUAL   ( s23[0], 3);
  BOOST_CHECK_EQUAL   ( s23[1], 1);

  BOOST_CHECK_EQUAL   ( s231[0], 3);
  BOOST_CHECK_EQUAL   ( s231[1], 1);
  BOOST_CHECK_EQUAL   ( s231[2], 1);

  BOOST_CHECK_EQUAL   ( s123[0], 6);
  BOOST_CHECK_EQUAL   ( s123[1], 3);
  BOOST_CHECK_EQUAL   ( s123[2], 1);

  BOOST_CHECK_EQUAL   ( s213[0], 3);
  BOOST_CHECK_EQUAL   ( s213[1], 3);
  BOOST_CHECK_EQUAL   ( s213[2], 1);

  BOOST_CHECK_EQUAL   ( s423[0], 6);
  BOOST_CHECK_EQUAL   ( s423[1], 3);
  BOOST_CHECK_EQUAL   ( s423[2], 1);

}

BOOST_AUTO_TEST_SUITE_END()

#endif
