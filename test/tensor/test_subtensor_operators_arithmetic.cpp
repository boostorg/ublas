//  Copyright (c) 2018-2021 Cem Bassoy
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

BOOST_AUTO_TEST_SUITE(test_subtensor_arithmetic_operations)

using double_extended = boost::multiprecision::cpp_bin_float_double_extended;

using test_types = zip<int,float,double_extended>::with_t<boost::numeric::ublas::layout::first_order, boost::numeric::ublas::layout::last_order>;

struct fixture
{
  using extents_type = boost::numeric::ublas::extents<>;

  std::vector<extents_type> extents =
    {
//      extents_type{},    // 0
      extents_type{1,1}, // 1
      extents_type{1,2}, // 2
      extents_type{2,1}, // 3
      extents_type{2,3}, // 4
      extents_type{2,3,1}, // 5
      extents_type{4,1,3}, // 6
      extents_type{1,2,3}, // 7
      extents_type{4,2,3}, // 8
      extents_type{4,2,3,5} // 9
  };
};

BOOST_FIXTURE_TEST_CASE_TEMPLATE( test_subtensor_binary_arithmetic_operations, value,  test_types, fixture)
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
    auto r  = tensor_type (e);
    auto s  = subtensor(t);
    auto s2 = subtensor(t2);
    auto v  = value_type {};

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

  for(auto const& e : extents)
        check(e);

  auto t0 = tensor_type(extents.at(0));
  auto t1 = tensor_type(extents.at(1));
  auto t2 = tensor_type(extents.at(2));

  BOOST_CHECK_NO_THROW ( tensor_type t = subtensor(t0) + t0  );
  BOOST_CHECK_NO_THROW ( tensor_type t = subtensor(t0) + subtensor(t0)  );
  BOOST_CHECK_THROW    ( tensor_type t = subtensor(t0) + subtensor(t2), std::runtime_error  );
  BOOST_CHECK_THROW    ( tensor_type t = subtensor(t1) + t2, std::runtime_error  );

}



BOOST_FIXTURE_TEST_CASE_TEMPLATE( test_subtensor_unary_arithmetic_operations, value,  test_types, fixture)
{
  namespace ublas = boost::numeric::ublas;
  using value_type  = typename value::first_type;
  using layout_type = typename value::second_type;
  using tensor_type  = ublas::tensor_dynamic<value_type,layout_type>;
  using subtensor = typename tensor_type::subtensor_type;


  auto check = [](auto const&  e)
  {

    auto t  = tensor_type (e);
    auto t2 = tensor_type (e);
    auto v  = value_type  {};
    auto s  = subtensor(t);
    auto s2 = subtensor(t2);
    BOOST_CHECK_EQUAL(t.size(), s.size());
    BOOST_CHECK_EQUAL(t2.size(), s2.size());


    std::iota(t.begin(), t.end(), v);
    std::iota(t2.begin(), t2.end(), v+2);

    tensor_type r1 = s + 2 + s + 2;

    for(auto i = 0ul; i < s.size(); ++i)
      BOOST_CHECK_EQUAL ( r1(i), 2*s(i) + 4 );

    tensor_type r2 = 2 + s + 2 + s;

    for(auto i = 0ul; i < s.size(); ++i)
      BOOST_CHECK_EQUAL ( r2(i), 2*s(i) + 4 );

    tensor_type r3 = (s-2) + (s-2);

    for(auto i = 0ul; i < s.size(); ++i)
      BOOST_CHECK_EQUAL ( r3(i), 2*s(i) - 4 );

    tensor_type r4 = (s*2) * (3*s);

    for(auto i = 0ul; i < s.size(); ++i)
      BOOST_CHECK_EQUAL ( r4(i), 2*3*s(i)*s(i) );

    tensor_type r5 = (s2*2) / (2*s2) * s2;

    for(auto i = 0ul; i < s.size(); ++i)
      BOOST_CHECK_EQUAL ( r5(i), (s2(i)*2) / (2*s2(i)) * s2(i) );

    tensor_type r6 = (s2/2+1) / (2/s2+1) / s2;

    for(auto i = 0ul; i < s.size(); ++i)
      BOOST_CHECK_EQUAL ( r6(i), (s2(i)/2+1) / (2/s2(i)+1) / s2(i) );

  };

  for(auto const& e : extents)
    check(e);


  auto t0 = tensor_type(extents.at(0));
  auto t2 = tensor_type(extents.at(2));


  BOOST_CHECK_NO_THROW ( tensor_type t = subtensor(t0) + 2 + t0  );
  BOOST_CHECK_NO_THROW ( tensor_type t = subtensor(t0) + 2 + subtensor(t0)  );
  BOOST_CHECK_THROW    ( tensor_type t = subtensor(t0) + 2 + t2, std::runtime_error  );
  BOOST_CHECK_THROW    ( tensor_type t = subtensor(t0) + 2 + subtensor(t2), std::runtime_error  );

}





BOOST_FIXTURE_TEST_CASE_TEMPLATE( test_subtensor_assign_arithmetic_operations, value,  test_types, fixture)
{
  namespace ublas = boost::numeric::ublas;
  using value_type  = typename value::first_type;
  using layout_type = typename value::second_type;
  using tensor_type  = ublas::tensor_dynamic<value_type,layout_type>;
  using subtensor = typename tensor_type::subtensor_type;


  auto check = [](auto const&  e)
  {
    auto t  = tensor_type (e);
    auto t2 = tensor_type (e);
    auto r  = tensor_type (e);
    auto v  = value_type  {};
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

    s = -r;
    for(auto i = 0ul; i < s.size(); ++i)
      BOOST_CHECK_EQUAL ( s(i), -r(i) );

    s = +r;
    for(auto i = 0ul; i < s.size(); ++i)
      BOOST_CHECK_EQUAL ( s(i), r(i) );
  };

  for(auto const& e : extents)
    check(e);
}


BOOST_AUTO_TEST_SUITE_END()
