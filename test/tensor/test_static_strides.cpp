//
//  Copyright (c) 2020, Cem Bassoy, cem.bassoy@gmail.com
//  Copyright (c) 2019, Amit Singh, amitsingh19975@gmail.com
//
//  Distributed under the Boost Software License, Version 1.0. (See
//  accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)
//
//  The authors gratefully acknowledge the support of
//  Google and Fraunhofer IOSB, Ettlingen, Germany
//

#include <boost/numeric/ublas/tensor/extents.hpp>
#include <boost/test/unit_test.hpp>

BOOST_AUTO_TEST_SUITE(test_static_strides)

using test_types = std::tuple<boost::numeric::ublas::layout::first_order,
                              boost::numeric::ublas::layout::last_order>;

template<std::size_t ... es>
using extents = boost::numeric::ublas::extents<es...>;

using first_order = boost::numeric::ublas::layout::first_order;
using last_order  = boost::numeric::ublas::layout::last_order;

BOOST_AUTO_TEST_CASE_TEMPLATE(test_static_strides_ctor, value, test_types)
{
  namespace ublas = boost::numeric::ublas;

  constexpr auto s11  = ublas::to_strides(extents  <1,1>{},first_order{});
  constexpr auto s12  = ublas::to_strides(extents  <1,2>{},first_order{});
  constexpr auto s21  = ublas::to_strides(extents  <2,1>{},first_order{});
  constexpr auto s23  = ublas::to_strides(extents  <2,3>{},first_order{});
  constexpr auto s231 = ublas::to_strides(extents<2,3,1>{},first_order{});
  constexpr auto s123 = ublas::to_strides(extents<1,2,3>{},first_order{});
  constexpr auto s423 = ublas::to_strides(extents<4,2,3>{},first_order{});

  BOOST_CHECK_EQUAL(s11.empty(), false);
  BOOST_CHECK_EQUAL(s12.empty(), false);
  BOOST_CHECK_EQUAL(s21.empty(), false);
  BOOST_CHECK_EQUAL(s23.empty(), false);
  BOOST_CHECK_EQUAL(s231.empty(), false);
  BOOST_CHECK_EQUAL(s123.empty(), false);
  BOOST_CHECK_EQUAL(s423.empty(), false);

  BOOST_CHECK_EQUAL(s11.size(), 2);
  BOOST_CHECK_EQUAL(s12.size(), 2);
  BOOST_CHECK_EQUAL(s21.size(), 2);
  BOOST_CHECK_EQUAL(s23.size(), 2);
  BOOST_CHECK_EQUAL(s231.size(), 3);
  BOOST_CHECK_EQUAL(s123.size(), 3);
  BOOST_CHECK_EQUAL(s423.size(), 3);
}

BOOST_AUTO_TEST_CASE(test_static_strides_ctor_access_first_order)
{
  namespace ublas = boost::numeric::ublas;

  constexpr auto s11  = ublas::to_strides(extents  <1,1>{},first_order{});
  constexpr auto s12  = ublas::to_strides(extents  <1,2>{},first_order{});
  constexpr auto s21  = ublas::to_strides(extents  <2,1>{},first_order{});
  constexpr auto s23  = ublas::to_strides(extents  <2,3>{},first_order{});
  constexpr auto s231 = ublas::to_strides(extents<2,3,1>{},first_order{});
  constexpr auto s213 = ublas::to_strides(extents<2,1,3>{},first_order{});
  constexpr auto s123 = ublas::to_strides(extents<1,2,3>{},first_order{});
  constexpr auto s423 = ublas::to_strides(extents<4,2,3>{},first_order{});

  BOOST_REQUIRE_EQUAL(s11.size(), 2);
  BOOST_REQUIRE_EQUAL(s12.size(), 2);
  BOOST_REQUIRE_EQUAL(s21.size(), 2);
  BOOST_REQUIRE_EQUAL(s23.size(), 2);
  BOOST_REQUIRE_EQUAL(s231.size(), 3);
  BOOST_REQUIRE_EQUAL(s213.size(), 3);
  BOOST_REQUIRE_EQUAL(s123.size(), 3);
  BOOST_REQUIRE_EQUAL(s423.size(), 3);


  BOOST_CHECK_EQUAL(s11[0], 1);
  BOOST_CHECK_EQUAL(s11[1], 1);

  BOOST_CHECK_EQUAL(s12[0], 1);
  BOOST_CHECK_EQUAL(s12[1], 1);

  BOOST_CHECK_EQUAL(s21[0], 1);
  BOOST_CHECK_EQUAL(s21[1], 1);

  BOOST_CHECK_EQUAL(s23[0], 1);
  BOOST_CHECK_EQUAL(s23[1], 2);

  BOOST_CHECK_EQUAL(s231[0], 1);
  BOOST_CHECK_EQUAL(s231[1], 2);
  BOOST_CHECK_EQUAL(s231[2], 6);

  BOOST_CHECK_EQUAL(s123[0], 1);
  BOOST_CHECK_EQUAL(s123[1], 1);
  BOOST_CHECK_EQUAL(s123[2], 2);

  BOOST_CHECK_EQUAL(s213[0], 1);
  BOOST_CHECK_EQUAL(s213[1], 2);
  BOOST_CHECK_EQUAL(s213[2], 2);

  BOOST_CHECK_EQUAL(s423[0], 1);
  BOOST_CHECK_EQUAL(s423[1], 4);
  BOOST_CHECK_EQUAL(s423[2], 8);
}

BOOST_AUTO_TEST_CASE(test_static_strides_ctor_access_last_order)
{
  namespace ublas = boost::numeric::ublas;

  constexpr auto s11  = ublas::to_strides(extents  <1,1>{},last_order{});
  constexpr auto s12  = ublas::to_strides(extents  <1,2>{},last_order{});
  constexpr auto s21  = ublas::to_strides(extents  <2,1>{},last_order{});
  constexpr auto s23  = ublas::to_strides(extents  <2,3>{},last_order{});
  constexpr auto s231 = ublas::to_strides(extents<2,3,1>{},last_order{});
  constexpr auto s213 = ublas::to_strides(extents<2,1,3>{},last_order{});
  constexpr auto s123 = ublas::to_strides(extents<1,2,3>{},last_order{});
  constexpr auto s423 = ublas::to_strides(extents<4,2,3>{},last_order{});

  BOOST_REQUIRE_EQUAL(s11.size(), 2);
  BOOST_REQUIRE_EQUAL(s12.size(), 2);
  BOOST_REQUIRE_EQUAL(s21.size(), 2);
  BOOST_REQUIRE_EQUAL(s23.size(), 2);
  BOOST_REQUIRE_EQUAL(s231.size(), 3);
  BOOST_REQUIRE_EQUAL(s213.size(), 3);
  BOOST_REQUIRE_EQUAL(s123.size(), 3);
  BOOST_REQUIRE_EQUAL(s423.size(), 3);


  BOOST_CHECK_EQUAL(s11[0], 1);
  BOOST_CHECK_EQUAL(s11[1], 1);

  BOOST_CHECK_EQUAL(s12[0], 1);
  BOOST_CHECK_EQUAL(s12[1], 1);

  BOOST_CHECK_EQUAL(s21[0], 1);
  BOOST_CHECK_EQUAL(s21[1], 1);

  BOOST_CHECK_EQUAL(s23[0], 3);
  BOOST_CHECK_EQUAL(s23[1], 1);

  BOOST_CHECK_EQUAL(s231[0], 3);
  BOOST_CHECK_EQUAL(s231[1], 1);
  BOOST_CHECK_EQUAL(s231[2], 1);

  BOOST_CHECK_EQUAL(s123[0], 6);
  BOOST_CHECK_EQUAL(s123[1], 3);
  BOOST_CHECK_EQUAL(s123[2], 1);

  BOOST_CHECK_EQUAL(s213[0], 3);
  BOOST_CHECK_EQUAL(s213[1], 3);
  BOOST_CHECK_EQUAL(s213[2], 1);
}

BOOST_AUTO_TEST_SUITE_END()
