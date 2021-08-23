//
//  Copyright (c) 2018-2020, Cem Bassoy, cem.bassoy@gmail.com
//  Copyright (c) 2019-2020, Amit Singh, amitsingh19975@gmail.com
//
//  Distributed under the Boost Software License, Version 1.0. (See
//  accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)
//
//  The authors gratefully acknowledge the support of
//  Google and Fraunhofer IOSB, Ettlingen, Germany
//



#include <boost/numeric/ublas/tensor.hpp>

#include <boost/test/unit_test.hpp>
#include <boost/multiprecision/cpp_bin_float.hpp>
#include "utility.hpp"

BOOST_AUTO_TEST_SUITE(test_subtensor_static_rank_arithmetic_operations)

using double_extended = boost::multiprecision::cpp_bin_float_double_extended;

using test_types = zip<int,float,double_extended>::with_t<boost::numeric::ublas::layout::first_order, boost::numeric::ublas::layout::last_order>;

struct fixture
{
  template<size_t N>
  using extents_t = boost::numeric::ublas::extents<N>;

  std::tuple<
    extents_t<2>, // 1
    extents_t<2>, // 2
    extents_t<3>, // 3
    extents_t<3>, // 4
    extents_t<4>  // 5
    > extents = {
      extents_t<2>{1,1},
      extents_t<2>{2,3},
      extents_t<3>{4,1,3},
      extents_t<3>{4,2,3},
      extents_t<4>{4,2,3,5}
  };
};

BOOST_FIXTURE_TEST_CASE_TEMPLATE( test_subtensor_static_rank_binary_arithmetic_operations, value,  test_types, fixture)
{
  namespace ublas = boost::numeric::ublas;
  using value_t  = typename value::first_type;
  using layout_t = typename value::second_type;


  auto check = [](auto const& /*unused*/, auto& e)
  {
    constexpr auto size = std::tuple_size_v<std::decay_t<decltype(e)>>;
    using tensor_t = ublas::tensor_static_rank<value_t, size, layout_t>;
    using subtensor = typename tensor_t::subtensor_type;
    auto t  = tensor_t (e);
    auto t2 = tensor_t (e);
    auto r  = tensor_t (e);
    auto s  = subtensor(subtensor(t));
    auto s2 = subtensor(t2);
    auto v  = value_t  {};

    BOOST_CHECK_EQUAL(t.size(), s.size());

    std::iota(t.begin(), t.end(), v);
    std::iota(t2.begin(), t2.end(), v+2);
    r = s + s + s + s2;

    for(auto i = 0ul; i < s.size(); ++i)
      BOOST_CHECK_EQUAL ( r(i), 3*s(i) + s2(i) );


    r = s2 / (s+3) * (s+1) - s2; // r = ( s2/ ((s+3)*(s+1)) ) - s2

    for(auto i = 0ul; i < s.size(); ++i)
      BOOST_CHECK_EQUAL ( r(i), s2(i) / (s(i)+3)*(s(i)+1) - s2(i) );

    r = 3+s2 / (s+3) * (s+1) * s - s2; // r = 3+( s2/ ((s+3)*(s+1)*s) ) - s2

    for(auto i = 0ul; i < s.size(); ++i)
      BOOST_CHECK_EQUAL ( r(i), 3+s2(i) / (s(i)+3)*(s(i)+1)*s(i) - s2(i) );

    r = s2 - s + s2 - s;

    for(auto i = 0ul; i < r.size(); ++i)
      BOOST_CHECK_EQUAL ( r(i), 4 );


    r = s * s * s * s2;

    for(auto i = 0ul; i < s.size(); ++i)
      BOOST_CHECK_EQUAL ( r(i), s(i)*s(i)*s(i)*s2(i) );

    r = (s2/s2) * (s2/s2);

    for(auto i = 0ul; i < s.size(); ++i)
      BOOST_CHECK_EQUAL ( r(i), 1 );
  };

  for_each_in_tuple(extents,check);
}



BOOST_FIXTURE_TEST_CASE_TEMPLATE( test_subtensor_static_rank_unary_arithmetic_operations, value,  test_types, fixture)
{
  namespace ublas = boost::numeric::ublas;
  using value_t  = typename value::first_type;
  using layout_t = typename value::second_type;


  auto check = [](auto const& /*unused*/, auto& e)
  {
    constexpr auto size = std::tuple_size_v<std::decay_t<decltype(e)>>;
    using tensor_t = ublas::tensor_static_rank<value_t, size, layout_t>;
    using subtensor = typename tensor_t::subtensor_type;

    auto t  = tensor_t (e);
    auto t2 = tensor_t (e);
    auto v  = value_t  {};
    auto s  = subtensor(t);
    auto s2 = subtensor(t2);
    BOOST_CHECK_EQUAL(t.size(), s.size());


    std::iota(t.begin(), t.end(), v);
    std::iota(t2.begin(), t2.end(), v+2);

    tensor_t r1 = s + 2 + s + 2;

    for(auto i = 0ul; i < s.size(); ++i)
      BOOST_CHECK_EQUAL ( r1(i), 2*s(i) + 4 );

    tensor_t r2 = 2 + s + 2 + s;

    for(auto i = 0ul; i < s.size(); ++i)
      BOOST_CHECK_EQUAL ( r2(i), 2*s(i) + 4 );

    tensor_t r3 = (s-2) + (s-2);

    for(auto i = 0ul; i < s.size(); ++i)
      BOOST_CHECK_EQUAL ( r3(i), 2*s(i) - 4 );

    tensor_t r4 = (s*2) * (3*s);

    for(auto i = 0ul; i < s.size(); ++i)
      BOOST_CHECK_EQUAL ( r4(i), 2*3*s(i)*s(i) );

    tensor_t r5 = (s2*2) / (2*s2) * s2;

    for(auto i = 0ul; i < s.size(); ++i)
      BOOST_CHECK_EQUAL ( r5(i), (s2(i)*2) / (2*s2(i)) * s2(i) );

    tensor_t r6 = (s2/2+1) / (2/s2+1) / s2;

    for(auto i = 0ul; i < s.size(); ++i)
      BOOST_CHECK_EQUAL ( r6(i), (s2(i)/2+1) / (2/s2(i)+1) / s2(i) );

  };

  for_each_in_tuple(extents,check);
}





BOOST_FIXTURE_TEST_CASE_TEMPLATE( test_subtensor_static_rank_assign_arithmetic_operations, value,  test_types, fixture)
{
  namespace ublas = boost::numeric::ublas;
  using value_t  = typename value::first_type;
  using layout_t = typename value::second_type;


  auto check = [](auto const& /*unused*/, auto& e)
  {
    constexpr auto size = std::tuple_size_v<std::decay_t<decltype(e)>>;
    using tensor_t = ublas::tensor_static_rank<value_t, size, layout_t>;
    using subtensor = typename tensor_t::subtensor_type;

    auto t  = tensor_t (e);
    auto t2 = tensor_t (e);
    auto r  = tensor_t (e);
    auto v  = value_t  {};
    auto s = subtensor(t);
    auto s2 = subtensor(t2);
    BOOST_CHECK_EQUAL(t.size(), s.size());


    std::iota(t.begin(), t.end(), v);
    std::iota(t2.begin(), t2.end(), v+2);

    r  = s + 2;
    r += s;
    r += 2;

    for(auto i = 0ul; i < s.size(); ++i)
      BOOST_CHECK_EQUAL ( r(i), 2*s(i) + 4 );

    r  = 2 + s;
    r += s;
    r += 2;

    for(auto i = 0ul; i < s.size(); ++i)
      BOOST_CHECK_EQUAL ( r(i), 2*s(i) + 4 );

    for(auto i = 0ul; i < s.size(); ++i)
      BOOST_CHECK_EQUAL ( r(i), 2*s(i) + 4 );

    r = (s-2);
    r += s;
    r -= 2;

    for(auto i = 0ul; i < s.size(); ++i)
      BOOST_CHECK_EQUAL ( r(i), 2*s(i) - 4 );

    r  = (s*2);
    r *= 3;
    r *= s;

    for(auto i = 0ul; i < s.size(); ++i)
      BOOST_CHECK_EQUAL ( r(i), 2*3*s(i)*s(i) );

    r  = (s2*2);
    r /= 2;
    r /= s2;
    r *= s2;

    for(auto i = 0ul; i < s.size(); ++i)
      BOOST_CHECK_EQUAL ( r(i), (s2(i)*2) / (2*s2(i)) * s2(i) );

    r  = (s2/2+1);
    r /= (2/s2+1);
    r /= s2;

    for(auto i = 0ul; i < s.size(); ++i)
      BOOST_CHECK_EQUAL ( r(i), (s2(i)/2+1) / (2/s2(i)+1) / s2(i) );

    tensor_t q = -r;
    for(auto i = 0ul; i < s.size(); ++i)
      BOOST_CHECK_EQUAL ( q(i), -r(i) );

    tensor_t p = +r;
    for(auto i = 0ul; i < s.size(); ++i)
      BOOST_CHECK_EQUAL ( p(i), r(i) );
  };

  for_each_in_tuple(extents,check);
}


BOOST_AUTO_TEST_SUITE_END()
