//
// 	Copyright (c) 2018-2020, Cem Bassoy, cem.bassoy@gmail.com
// 	Copyright (c) 2019-2020, Amit Singh, amitsingh19975@gmail.com
//
//  Distributed under the Boost Software License, Version 1.0. (See
//  accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)
//
//  The authors gratefully acknowledge the support of
//  Google
//

#include <boost/numeric/ublas/tensor/extents.hpp>
#include <boost/numeric/ublas/tensor/strides.hpp>
#include <boost/test/unit_test.hpp>

BOOST_AUTO_TEST_SUITE(test_static_strides)

using test_types = std::tuple<boost::numeric::ublas::first_order,
                              boost::numeric::ublas::last_order>;

template <size_t... E>
using extents_type = boost::numeric::ublas::basic_static_extents<unsigned, E...>;
template <class E, class L>
using strides_type = boost::numeric::ublas::strides_t<E, L>;

BOOST_AUTO_TEST_CASE_TEMPLATE(test_static_strides_ctor, value, test_types) {
  using namespace boost::numeric;

  strides_type<extents_type<1, 1>, ublas::first_order> s1{};
  BOOST_CHECK_EQUAL(s1.empty(), false);
  BOOST_CHECK_EQUAL(s1.size(), 2);

  strides_type<extents_type<1, 2>, ublas::first_order> s2{};
  BOOST_CHECK_EQUAL(s2.empty(), false);
  BOOST_CHECK_EQUAL(s2.size(), 2);

  strides_type<extents_type<2, 1>, ublas::first_order> s3{};
  BOOST_CHECK_EQUAL(s3.empty(), false);
  BOOST_CHECK_EQUAL(s3.size(), 2);

  strides_type<extents_type<2, 3>, ublas::first_order> s4{};
  BOOST_CHECK_EQUAL(s4.empty(), false);
  BOOST_CHECK_EQUAL(s4.size(), 2);

  strides_type<extents_type<2, 3, 1>, ublas::first_order> s5{};
  BOOST_CHECK_EQUAL(s5.empty(), false);
  BOOST_CHECK_EQUAL(s5.size(), 3);

  strides_type<extents_type<1, 2, 3>, ublas::first_order> s6{};
  BOOST_CHECK_EQUAL(s6.empty(), false);
  BOOST_CHECK_EQUAL(s6.size(), 3);

  strides_type<extents_type<4, 2, 3>, ublas::first_order> s7{};
  BOOST_CHECK_EQUAL(s7.empty(), false);
  BOOST_CHECK_EQUAL(s7.size(), 3);
}

BOOST_AUTO_TEST_CASE(test_static_strides_ctor_access_first_order) {
  using namespace boost::numeric;

  strides_type<extents_type<1, 1>, ublas::first_order> s1{};
  BOOST_REQUIRE_EQUAL(s1.size(), 2);
  BOOST_CHECK_EQUAL(s1[0], 1);
  BOOST_CHECK_EQUAL(s1[1], 1);

  strides_type<extents_type<1, 2>, ublas::first_order> s2{};
  BOOST_REQUIRE_EQUAL(s2.size(), 2);
  BOOST_CHECK_EQUAL(s2[0], 1);
  BOOST_CHECK_EQUAL(s2[1], 1);

  strides_type<extents_type<2, 1>, ublas::first_order> s3{};
  BOOST_REQUIRE_EQUAL(s3.size(), 2);
  BOOST_CHECK_EQUAL(s3[0], 1);
  BOOST_CHECK_EQUAL(s3[1], 1);

  strides_type<extents_type<2, 3>, ublas::first_order> s4{};
  BOOST_REQUIRE_EQUAL(s4.size(), 2);
  BOOST_CHECK_EQUAL(s4[0], 1);
  BOOST_CHECK_EQUAL(s4[1], 2);

  strides_type<extents_type<2, 3, 1>, ublas::first_order> s5{};
  BOOST_REQUIRE_EQUAL(s5.size(), 3);
  BOOST_CHECK_EQUAL(s5[0], 1);
  BOOST_CHECK_EQUAL(s5[1], 2);
  BOOST_CHECK_EQUAL(s5[2], 6);

  strides_type<extents_type<1, 2, 3>, ublas::first_order> s6{};
  BOOST_REQUIRE_EQUAL(s6.size(), 3);
  BOOST_CHECK_EQUAL(s6[0], 1);
  BOOST_CHECK_EQUAL(s6[1], 1);
  BOOST_CHECK_EQUAL(s6[2], 2);

  strides_type<extents_type<2, 1, 3>, ublas::first_order> s7{};
  BOOST_REQUIRE_EQUAL(s7.size(), 3);
  BOOST_CHECK_EQUAL(s7[0], 1);
  BOOST_CHECK_EQUAL(s7[1], 2);
  BOOST_CHECK_EQUAL(s7[2], 2);

  strides_type<extents_type<4, 2, 3>, ublas::first_order> s8{};
  BOOST_REQUIRE_EQUAL(s8.size(), 3);
  BOOST_CHECK_EQUAL(s8[0], 1);
  BOOST_CHECK_EQUAL(s8[1], 4);
  BOOST_CHECK_EQUAL(s8[2], 8);
}

BOOST_AUTO_TEST_CASE(test_static_strides_ctor_access_last_order) {
  using namespace boost::numeric;

  strides_type<extents_type<1, 1>, ublas::last_order> s1{};
  BOOST_REQUIRE_EQUAL(s1.size(), 2);
  BOOST_CHECK_EQUAL(s1[0], 1);
  BOOST_CHECK_EQUAL(s1[1], 1);

  strides_type<extents_type<1, 2>, ublas::last_order> s2{};
  BOOST_REQUIRE_EQUAL(s2.size(), 2);
  BOOST_CHECK_EQUAL(s2[0], 1);
  BOOST_CHECK_EQUAL(s2[1], 1);

  strides_type<extents_type<2, 1>, ublas::last_order> s3{};
  BOOST_REQUIRE_EQUAL(s3.size(), 2);
  BOOST_CHECK_EQUAL(s3[0], 1);
  BOOST_CHECK_EQUAL(s3[1], 1);

  strides_type<extents_type<2, 3>, ublas::last_order> s4{};
  BOOST_REQUIRE_EQUAL(s4.size(), 2);
  BOOST_CHECK_EQUAL(s4[0], 3);
  BOOST_CHECK_EQUAL(s4[1], 1);

  strides_type<extents_type<2, 3, 1>, ublas::last_order> s5{};
  BOOST_REQUIRE_EQUAL(s5.size(), 3);
  BOOST_CHECK_EQUAL(s5[0], 3);
  BOOST_CHECK_EQUAL(s5[1], 1);
  BOOST_CHECK_EQUAL(s5[2], 1);

  strides_type<extents_type<1, 2, 3>, ublas::last_order> s6{};
  BOOST_REQUIRE_EQUAL(s6.size(), 3);
  BOOST_CHECK_EQUAL(s6[0], 6);
  BOOST_CHECK_EQUAL(s6[1], 3);
  BOOST_CHECK_EQUAL(s6[2], 1);

  strides_type<extents_type<2, 1, 3>, ublas::last_order> s7{};
  BOOST_REQUIRE_EQUAL(s7.size(), 3);
  BOOST_CHECK_EQUAL(s7[0], 3);
  BOOST_CHECK_EQUAL(s7[1], 3);
  BOOST_CHECK_EQUAL(s7[2], 1);

  strides_type<extents_type<4, 2, 3>, ublas::last_order> s8{};
  BOOST_REQUIRE_EQUAL(s8.size(), 3);
  BOOST_CHECK_EQUAL(s8[0], 6);
  BOOST_CHECK_EQUAL(s8[1], 3);
  BOOST_CHECK_EQUAL(s8[2], 1);
}

BOOST_AUTO_TEST_SUITE_END()
