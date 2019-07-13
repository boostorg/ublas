//  Copyright (c) 2018-2019 Cem Bassoy
//
//  Distributed under the Boost Software License, Version 1.0. (See
//  accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#include "utility.hpp"
#include <boost/test/unit_test.hpp>
#include <type_traits>
#include <vector>

BOOST_AUTO_TEST_SUITE(test_basic_shape)

template<class T, class U>
constexpr bool is_same_v = std::is_same<T,U>::value;

struct fixture {
  template <ptrdiff_t... E>
  using basic_shape = boost::numeric::ublas::detail::basic_shape<E...>;

  fixture() = default;

  using b_s0 = basic_shape<>;                  // 0
  using b_s1 = basic_shape<1, 1>;              // 1
  using b_s2 = basic_shape<1, 2>;              // 2
  using b_s3 = basic_shape<2, 1>;              // 3
  using b_s4 = basic_shape<2, 3>;              // 4
  using b_s5 = basic_shape<2, 3, 1>;           // 5
  using b_s6 = basic_shape<1, 2, 3>;           // 6
  using b_s7 = basic_shape<1, 1, 2, 3>;        // 7
  using b_s8 = basic_shape<1, 2, 3, 1, 1>;     // 8
  using b_s9 = basic_shape<4, 2, 3>;           // 9
  using b_s10 = basic_shape<4, 2, 1, 3>;       // 10
  using b_s11 = basic_shape<4, 2, 1, 3, 1>;    // 11
  using b_s12 = basic_shape<1, 4, 2, 1, 3, 1>; // 12

  template <ptrdiff_t S, ptrdiff_t E>
  using make_dynamic_basic_shape_impl_t =
      boost::numeric::ublas::detail::make_dynamic_basic_shape_impl_t<S, E>;

  using m_d_b_s_impl0 = make_dynamic_basic_shape_impl_t<0, 0>;   // 0
  using m_d_b_s_impl1 = make_dynamic_basic_shape_impl_t<0, 1>;   // 1
  using m_d_b_s_impl2 = make_dynamic_basic_shape_impl_t<0, 2>;   // 2
  using m_d_b_s_impl3 = make_dynamic_basic_shape_impl_t<0, 3>;   // 3
  using m_d_b_s_impl4 = make_dynamic_basic_shape_impl_t<2, 4>;   // 4
  using m_d_b_s_impl5 = make_dynamic_basic_shape_impl_t<2, 5>;   // 5
  using m_d_b_s_impl6 = make_dynamic_basic_shape_impl_t<2, 6>;   // 6
  using m_d_b_s_impl7 = make_dynamic_basic_shape_impl_t<3, 4>;   // 7
  using m_d_b_s_impl8 = make_dynamic_basic_shape_impl_t<3, 6>;   // 8
  using m_d_b_s_impl9 = make_dynamic_basic_shape_impl_t<4, 7>;   // 9
  using m_d_b_s_impl10 = make_dynamic_basic_shape_impl_t<4, 8>;  // 10
  using m_d_b_s_impl11 = make_dynamic_basic_shape_impl_t<5, 7>;  // 11
  using m_d_b_s_impl12 = make_dynamic_basic_shape_impl_t<6, 10>; // 12
};

BOOST_FIXTURE_TEST_CASE(test_basic_shape, fixture,
                        *boost::unit_test::label("basic_shape") *
                            boost::unit_test::label("type")) {
  using namespace boost::numeric;

  BOOST_CHECK_EQUAL(b_s0::rank, 0);
  BOOST_CHECK(b_s0::empty);

  BOOST_REQUIRE_EQUAL(b_s1::rank, 2);
  BOOST_REQUIRE_EQUAL(b_s2::rank, 2);
  BOOST_REQUIRE_EQUAL(b_s3::rank, 2);
  BOOST_REQUIRE_EQUAL(b_s4::rank, 2);
  BOOST_REQUIRE_EQUAL(b_s5::rank, 3);
  BOOST_REQUIRE_EQUAL(b_s6::rank, 3);
  BOOST_REQUIRE_EQUAL(b_s7::rank, 4);
  BOOST_REQUIRE_EQUAL(b_s8::rank, 5);
  BOOST_REQUIRE_EQUAL(b_s9::rank, 3);
  BOOST_REQUIRE_EQUAL(b_s10::rank, 4);
  BOOST_REQUIRE_EQUAL(b_s11::rank, 5);
  BOOST_REQUIRE_EQUAL(b_s12::rank, 6);

  BOOST_CHECK_EQUAL((get<0, b_s1>()), 1);
  BOOST_CHECK_EQUAL((get<1, b_s1>()), 1);

  BOOST_CHECK_EQUAL((get<0, b_s2>()), 1);
  BOOST_CHECK_EQUAL((get<1, b_s2>()), 2);

  BOOST_CHECK_EQUAL((get<0, b_s3>()), 2);
  BOOST_CHECK_EQUAL((get<1, b_s3>()), 1);

  BOOST_CHECK_EQUAL((get<0, b_s4>()), 2);
  BOOST_CHECK_EQUAL((get<1, b_s4>()), 3);

  BOOST_CHECK_EQUAL((get<0, b_s5>()), 2);
  BOOST_CHECK_EQUAL((get<1, b_s5>()), 3);
  BOOST_CHECK_EQUAL((get<2, b_s5>()), 1);

  BOOST_CHECK_EQUAL((get<0, b_s6>()), 1);
  BOOST_CHECK_EQUAL((get<1, b_s6>()), 2);
  BOOST_CHECK_EQUAL((get<2, b_s6>()), 3);

  BOOST_CHECK_EQUAL((get<0, b_s7>()), 1);
  BOOST_CHECK_EQUAL((get<1, b_s7>()), 1);
  BOOST_CHECK_EQUAL((get<2, b_s7>()), 2);
  BOOST_CHECK_EQUAL((get<3, b_s7>()), 3);

  BOOST_CHECK_EQUAL((get<0, b_s8>()), 1);
  BOOST_CHECK_EQUAL((get<1, b_s8>()), 2);
  BOOST_CHECK_EQUAL((get<2, b_s8>()), 3);
  BOOST_CHECK_EQUAL((get<3, b_s8>()), 1);
  BOOST_CHECK_EQUAL((get<4, b_s8>()), 1);

  BOOST_CHECK_EQUAL((get<0, b_s9>()), 4);
  BOOST_CHECK_EQUAL((get<1, b_s9>()), 2);
  BOOST_CHECK_EQUAL((get<2, b_s9>()), 3);

  BOOST_CHECK_EQUAL((get<0, b_s10>()), 4);
  BOOST_CHECK_EQUAL((get<1, b_s10>()), 2);
  BOOST_CHECK_EQUAL((get<2, b_s10>()), 1);
  BOOST_CHECK_EQUAL((get<3, b_s10>()), 3);

  BOOST_CHECK_EQUAL((get<0, b_s11>()), 4);
  BOOST_CHECK_EQUAL((get<1, b_s11>()), 2);
  BOOST_CHECK_EQUAL((get<2, b_s11>()), 1);
  BOOST_CHECK_EQUAL((get<3, b_s11>()), 3);
  BOOST_CHECK_EQUAL((get<4, b_s11>()), 1);

  BOOST_CHECK_EQUAL((get<0, b_s12>()), 1);
  BOOST_CHECK_EQUAL((get<1, b_s12>()), 4);
  BOOST_CHECK_EQUAL((get<2, b_s12>()), 2);
  BOOST_CHECK_EQUAL((get<3, b_s12>()), 1);
  BOOST_CHECK_EQUAL((get<4, b_s12>()), 3);
  BOOST_CHECK_EQUAL((get<5, b_s12>()), 1);
}

BOOST_FIXTURE_TEST_CASE(
    test_make_dynamic_basic_shape, fixture,
    *boost::unit_test::label("test_make_dynamic_basic_shape_impl_t") *
        boost::unit_test::label("type")) {
  using namespace boost::numeric;

  BOOST_CHECK_EQUAL(m_d_b_s_impl0::rank, 1);
  BOOST_CHECK(!m_d_b_s_impl0::empty);

  BOOST_REQUIRE_EQUAL(m_d_b_s_impl1::rank, 2);
  BOOST_REQUIRE_EQUAL(m_d_b_s_impl2::rank, 3);
  BOOST_REQUIRE_EQUAL(m_d_b_s_impl3::rank, 4);
  BOOST_REQUIRE_EQUAL(m_d_b_s_impl4::rank, 3);
  BOOST_REQUIRE_EQUAL(m_d_b_s_impl5::rank, 4);
  BOOST_REQUIRE_EQUAL(m_d_b_s_impl6::rank, 5);
  BOOST_REQUIRE_EQUAL(m_d_b_s_impl7::rank, 2);
  BOOST_REQUIRE_EQUAL(m_d_b_s_impl8::rank, 4);
  BOOST_REQUIRE_EQUAL(m_d_b_s_impl9::rank, 4);
  BOOST_REQUIRE_EQUAL(m_d_b_s_impl10::rank, 5);
  BOOST_REQUIRE_EQUAL(m_d_b_s_impl11::rank, 3);
  BOOST_REQUIRE_EQUAL(m_d_b_s_impl12::rank, 5);

  BOOST_CHECK_EQUAL(
      (is_same_v<ublas::detail::basic_shape<-1l>, m_d_b_s_impl0>), true);
  BOOST_CHECK_EQUAL(
      (is_same_v<ublas::detail::basic_shape<-1l, -1l>, m_d_b_s_impl1>),
      true);
  BOOST_CHECK_EQUAL((is_same_v<ublas::detail::basic_shape<-1l, -1l, -1l>,
                                    m_d_b_s_impl2>),
                    true);
  BOOST_CHECK_EQUAL(
      (is_same_v<ublas::detail::basic_shape<-1l, -1l, -1l, -1l>,
                      m_d_b_s_impl3>),
      true);
  BOOST_CHECK_EQUAL((is_same_v<ublas::detail::basic_shape<-1l, -1l, -1l>,
                                    m_d_b_s_impl4>),
                    true);
  BOOST_CHECK_EQUAL(
      (is_same_v<ublas::detail::basic_shape<-1l, -1l, -1l, -1l>,
                      m_d_b_s_impl5>),
      true);
  BOOST_CHECK_EQUAL(
      (is_same_v<ublas::detail::basic_shape<-1l, -1l, -1l, -1l, -1l>,
                      m_d_b_s_impl6>),
      true);
  BOOST_CHECK_EQUAL(
      (is_same_v<ublas::detail::basic_shape<-1l, -1l>, m_d_b_s_impl7>),
      true);
  BOOST_CHECK_EQUAL(
      (is_same_v<ublas::detail::basic_shape<-1l, -1l, -1l, -1l>,
                      m_d_b_s_impl8>),
      true);
  BOOST_CHECK_EQUAL(
      (is_same_v<ublas::detail::basic_shape<-1l, -1l, -1l, -1l>,
                      m_d_b_s_impl9>),
      true);
  BOOST_CHECK_EQUAL(
      (is_same_v<ublas::detail::basic_shape<-1l, -1l, -1l, -1l, -1l>,
                      m_d_b_s_impl10>),
      true);
  BOOST_CHECK_EQUAL((is_same_v<ublas::detail::basic_shape<-1l, -1l, -1l>,
                                    m_d_b_s_impl11>),
                    true);
  BOOST_CHECK_EQUAL(
      (is_same_v<ublas::detail::basic_shape<-1l, -1l, -1l, -1l, -1l>,
                      m_d_b_s_impl12>),
      true);
}

BOOST_FIXTURE_TEST_CASE(test_concat_basic_shape, fixture,
                        *boost::unit_test::label("test_concat_basic_shape_t") *
                            boost::unit_test::label("type")) {
  using namespace boost::numeric;

  using b_s0c1 = ublas::detail::concat_basic_shape_t<b_s0, b_s1>;
  using b_s1c2 = ublas::detail::concat_basic_shape_t<b_s1, b_s2>;
  using b_s2c3 = ublas::detail::concat_basic_shape_t<b_s2, b_s3>;
  using b_s3c4 = ublas::detail::concat_basic_shape_t<b_s3, b_s4>;
  using b_s4c5 = ublas::detail::concat_basic_shape_t<b_s4, b_s5>;
  using b_s5c6 = ublas::detail::concat_basic_shape_t<b_s5, b_s6>;
  using b_s6c7 = ublas::detail::concat_basic_shape_t<b_s6, b_s7>;
  using b_s7c8 = ublas::detail::concat_basic_shape_t<b_s7, b_s8>;
  using b_s8c9 = ublas::detail::concat_basic_shape_t<b_s8, b_s9>;
  using b_s9c10 = ublas::detail::concat_basic_shape_t<b_s9, b_s10>;
  using b_s10c11 = ublas::detail::concat_basic_shape_t<b_s10, b_s11>;
  using b_s11c12 = ublas::detail::concat_basic_shape_t<b_s11, b_s12>;

  BOOST_CHECK_EQUAL(b_s0c1::rank, 2);
  BOOST_CHECK(!b_s0c1::empty);

  BOOST_REQUIRE_EQUAL(b_s1c2::rank, 4);
  BOOST_REQUIRE_EQUAL(b_s2c3::rank, 4);
  BOOST_REQUIRE_EQUAL(b_s3c4::rank, 4);
  BOOST_REQUIRE_EQUAL(b_s4c5::rank, 5);
  BOOST_REQUIRE_EQUAL(b_s5c6::rank, 6);
  BOOST_REQUIRE_EQUAL(b_s6c7::rank, 7);
  BOOST_REQUIRE_EQUAL(b_s7c8::rank, 9);
  BOOST_REQUIRE_EQUAL(b_s8c9::rank, 8);
  BOOST_REQUIRE_EQUAL(b_s9c10::rank, 7);
  BOOST_REQUIRE_EQUAL(b_s10c11::rank, 9);
  BOOST_REQUIRE_EQUAL(b_s11c12::rank, 11);

  BOOST_CHECK_EQUAL((get<0, b_s0c1>()), 1);
  BOOST_CHECK_EQUAL((get<1, b_s0c1>()), 1);

  BOOST_CHECK_EQUAL((get<0, b_s1c2>()), 1);
  BOOST_CHECK_EQUAL((get<1, b_s1c2>()), 1);
  BOOST_CHECK_EQUAL((get<2, b_s1c2>()), 1);
  BOOST_CHECK_EQUAL((get<3, b_s1c2>()), 2);

  BOOST_CHECK_EQUAL((get<0, b_s2c3>()), 1);
  BOOST_CHECK_EQUAL((get<1, b_s2c3>()), 2);
  BOOST_CHECK_EQUAL((get<2, b_s2c3>()), 2);
  BOOST_CHECK_EQUAL((get<3, b_s2c3>()), 1);

  BOOST_CHECK_EQUAL((get<0, b_s3c4>()), 2);
  BOOST_CHECK_EQUAL((get<1, b_s3c4>()), 1);
  BOOST_CHECK_EQUAL((get<2, b_s3c4>()), 2);
  BOOST_CHECK_EQUAL((get<3, b_s3c4>()), 3);

  BOOST_CHECK_EQUAL((get<0, b_s4c5>()), 2);
  BOOST_CHECK_EQUAL((get<1, b_s4c5>()), 3);
  BOOST_CHECK_EQUAL((get<2, b_s4c5>()), 2);
  BOOST_CHECK_EQUAL((get<3, b_s4c5>()), 3);
  BOOST_CHECK_EQUAL((get<4, b_s4c5>()), 1);

  BOOST_CHECK_EQUAL((get<0, b_s5c6>()), 2);
  BOOST_CHECK_EQUAL((get<1, b_s5c6>()), 3);
  BOOST_CHECK_EQUAL((get<2, b_s5c6>()), 1);
  BOOST_CHECK_EQUAL((get<3, b_s5c6>()), 1);
  BOOST_CHECK_EQUAL((get<4, b_s5c6>()), 2);
  BOOST_CHECK_EQUAL((get<5, b_s5c6>()), 3);

  BOOST_CHECK_EQUAL((get<0, b_s6c7>()), 1);
  BOOST_CHECK_EQUAL((get<1, b_s6c7>()), 2);
  BOOST_CHECK_EQUAL((get<2, b_s6c7>()), 3);
  BOOST_CHECK_EQUAL((get<3, b_s6c7>()), 1);
  BOOST_CHECK_EQUAL((get<4, b_s6c7>()), 1);
  BOOST_CHECK_EQUAL((get<5, b_s6c7>()), 2);
  BOOST_CHECK_EQUAL((get<6, b_s6c7>()), 3);

  BOOST_CHECK_EQUAL((get<0, b_s7c8>()), 1);
  BOOST_CHECK_EQUAL((get<1, b_s7c8>()), 1);
  BOOST_CHECK_EQUAL((get<2, b_s7c8>()), 2);
  BOOST_CHECK_EQUAL((get<3, b_s7c8>()), 3);
  BOOST_CHECK_EQUAL((get<4, b_s7c8>()), 1);
  BOOST_CHECK_EQUAL((get<5, b_s7c8>()), 2);
  BOOST_CHECK_EQUAL((get<6, b_s7c8>()), 3);
  BOOST_CHECK_EQUAL((get<7, b_s7c8>()), 1);
  BOOST_CHECK_EQUAL((get<8, b_s7c8>()), 1);

  BOOST_CHECK_EQUAL((get<0, b_s8c9>()), 1);
  BOOST_CHECK_EQUAL((get<1, b_s8c9>()), 2);
  BOOST_CHECK_EQUAL((get<2, b_s8c9>()), 3);
  BOOST_CHECK_EQUAL((get<3, b_s8c9>()), 1);
  BOOST_CHECK_EQUAL((get<4, b_s8c9>()), 1);
  BOOST_CHECK_EQUAL((get<5, b_s8c9>()), 4);
  BOOST_CHECK_EQUAL((get<6, b_s8c9>()), 2);
  BOOST_CHECK_EQUAL((get<7, b_s8c9>()), 3);

  BOOST_CHECK_EQUAL((get<0, b_s9c10>()), 4);
  BOOST_CHECK_EQUAL((get<1, b_s9c10>()), 2);
  BOOST_CHECK_EQUAL((get<2, b_s9c10>()), 3);
  BOOST_CHECK_EQUAL((get<3, b_s9c10>()), 4);
  BOOST_CHECK_EQUAL((get<4, b_s9c10>()), 2);
  BOOST_CHECK_EQUAL((get<5, b_s9c10>()), 1);
  BOOST_CHECK_EQUAL((get<6, b_s9c10>()), 3);

  BOOST_CHECK_EQUAL((get<0, b_s10c11>()), 4);
  BOOST_CHECK_EQUAL((get<1, b_s10c11>()), 2);
  BOOST_CHECK_EQUAL((get<2, b_s10c11>()), 1);
  BOOST_CHECK_EQUAL((get<3, b_s10c11>()), 3);
  BOOST_CHECK_EQUAL((get<4, b_s10c11>()), 4);
  BOOST_CHECK_EQUAL((get<5, b_s10c11>()), 2);
  BOOST_CHECK_EQUAL((get<6, b_s10c11>()), 1);
  BOOST_CHECK_EQUAL((get<7, b_s10c11>()), 3);
  BOOST_CHECK_EQUAL((get<8, b_s10c11>()), 1);

  BOOST_CHECK_EQUAL((get<0, b_s11c12>()), 4);
  BOOST_CHECK_EQUAL((get<1, b_s11c12>()), 2);
  BOOST_CHECK_EQUAL((get<2, b_s11c12>()), 1);
  BOOST_CHECK_EQUAL((get<3, b_s11c12>()), 3);
  BOOST_CHECK_EQUAL((get<4, b_s11c12>()), 1);
  BOOST_CHECK_EQUAL((get<5, b_s11c12>()), 1);
  BOOST_CHECK_EQUAL((get<6, b_s11c12>()), 4);
  BOOST_CHECK_EQUAL((get<7, b_s11c12>()), 2);
  BOOST_CHECK_EQUAL((get<8, b_s11c12>()), 1);
  BOOST_CHECK_EQUAL((get<9, b_s11c12>()), 3);
  BOOST_CHECK_EQUAL((get<10, b_s11c12>()), 1);
}

BOOST_AUTO_TEST_CASE(test_make_basic_shape) {
  using namespace boost::numeric;
  using b_s0 = ublas::detail::make_basic_shape_t<1, 1>;
  using b_s1 = ublas::detail::make_basic_shape_t<2, 1, 1>;
  using b_s2 = ublas::detail::make_basic_shape_t<2, 1, 2>;
  using b_s3 = ublas::detail::make_basic_shape_t<2, 2, 1>;
  using b_s4 = ublas::detail::make_basic_shape_t<2, 2, 3>;
  using b_s5 = ublas::detail::make_basic_shape_t<3, 2, 3, 1>;
  using b_s6 = ublas::detail::make_basic_shape_t<3, 1, 2, 3>;
  using b_s7 = ublas::detail::make_basic_shape_t<4, 1, 1, 2, 3>;
  using b_s8 = ublas::detail::make_basic_shape_t<5, 1, 2, 3, 1, 1>;
  using b_s9 = ublas::detail::make_basic_shape_t<3, 4, 2, 3>;
  using b_s10 = ublas::detail::make_basic_shape_t<4, 4, 2, 1, 3>;
  using b_s11 = ublas::detail::make_basic_shape_t<5, 4, 2, 1, 3, 1>;
  using b_s12 = ublas::detail::make_basic_shape_t<6, 1, 4, 2, 1, 3, 1>;

  BOOST_CHECK_EQUAL(b_s0::rank, 1);
  BOOST_CHECK(!b_s0::empty);

  BOOST_REQUIRE_EQUAL(b_s1::rank, 2);
  BOOST_REQUIRE_EQUAL(b_s2::rank, 2);
  BOOST_REQUIRE_EQUAL(b_s3::rank, 2);
  BOOST_REQUIRE_EQUAL(b_s4::rank, 2);
  BOOST_REQUIRE_EQUAL(b_s5::rank, 3);
  BOOST_REQUIRE_EQUAL(b_s6::rank, 3);
  BOOST_REQUIRE_EQUAL(b_s7::rank, 4);
  BOOST_REQUIRE_EQUAL(b_s8::rank, 5);
  BOOST_REQUIRE_EQUAL(b_s9::rank, 3);
  BOOST_REQUIRE_EQUAL(b_s10::rank, 4);
  BOOST_REQUIRE_EQUAL(b_s11::rank, 5);
  BOOST_REQUIRE_EQUAL(b_s12::rank, 6);

  BOOST_CHECK_EQUAL((get<0, b_s1>()), 1);
  BOOST_CHECK_EQUAL((get<1, b_s1>()), 1);

  BOOST_CHECK_EQUAL((get<0, b_s2>()), 1);
  BOOST_CHECK_EQUAL((get<1, b_s2>()), 2);

  BOOST_CHECK_EQUAL((get<0, b_s3>()), 2);
  BOOST_CHECK_EQUAL((get<1, b_s3>()), 1);

  BOOST_CHECK_EQUAL((get<0, b_s4>()), 2);
  BOOST_CHECK_EQUAL((get<1, b_s4>()), 3);

  BOOST_CHECK_EQUAL((get<0, b_s5>()), 2);
  BOOST_CHECK_EQUAL((get<1, b_s5>()), 3);
  BOOST_CHECK_EQUAL((get<2, b_s5>()), 1);

  BOOST_CHECK_EQUAL((get<0, b_s6>()), 1);
  BOOST_CHECK_EQUAL((get<1, b_s6>()), 2);
  BOOST_CHECK_EQUAL((get<2, b_s6>()), 3);

  BOOST_CHECK_EQUAL((get<0, b_s7>()), 1);
  BOOST_CHECK_EQUAL((get<1, b_s7>()), 1);
  BOOST_CHECK_EQUAL((get<2, b_s7>()), 2);
  BOOST_CHECK_EQUAL((get<3, b_s7>()), 3);

  BOOST_CHECK_EQUAL((get<0, b_s8>()), 1);
  BOOST_CHECK_EQUAL((get<1, b_s8>()), 2);
  BOOST_CHECK_EQUAL((get<2, b_s8>()), 3);
  BOOST_CHECK_EQUAL((get<3, b_s8>()), 1);
  BOOST_CHECK_EQUAL((get<4, b_s8>()), 1);

  BOOST_CHECK_EQUAL((get<0, b_s9>()), 4);
  BOOST_CHECK_EQUAL((get<1, b_s9>()), 2);
  BOOST_CHECK_EQUAL((get<2, b_s9>()), 3);

  BOOST_CHECK_EQUAL((get<0, b_s10>()), 4);
  BOOST_CHECK_EQUAL((get<1, b_s10>()), 2);
  BOOST_CHECK_EQUAL((get<2, b_s10>()), 1);
  BOOST_CHECK_EQUAL((get<3, b_s10>()), 3);

  BOOST_CHECK_EQUAL((get<0, b_s11>()), 4);
  BOOST_CHECK_EQUAL((get<1, b_s11>()), 2);
  BOOST_CHECK_EQUAL((get<2, b_s11>()), 1);
  BOOST_CHECK_EQUAL((get<3, b_s11>()), 3);
  BOOST_CHECK_EQUAL((get<4, b_s11>()), 1);

  BOOST_CHECK_EQUAL((get<0, b_s12>()), 1);
  BOOST_CHECK_EQUAL((get<1, b_s12>()), 4);
  BOOST_CHECK_EQUAL((get<2, b_s12>()), 2);
  BOOST_CHECK_EQUAL((get<3, b_s12>()), 1);
  BOOST_CHECK_EQUAL((get<4, b_s12>()), 3);
  BOOST_CHECK_EQUAL((get<5, b_s12>()), 1);
}

BOOST_AUTO_TEST_SUITE_END()
