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



#include <boost/numeric/ublas/tensor.hpp>
#include <boost/test/unit_test.hpp>
#include <boost/multiprecision/cpp_bin_float.hpp>
#include "utility.hpp"

BOOST_AUTO_TEST_SUITE(test_tensor_static_rank_comparison)

using double_extended = boost::multiprecision::cpp_bin_float_double_extended;

using test_types = zip<int,float,double_extended>::with_t<boost::numeric::ublas::layout::first_order, boost::numeric::ublas::layout::last_order>;

struct fixture {
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


BOOST_FIXTURE_TEST_CASE_TEMPLATE( test_tensor_comparison, value,  test_types, fixture)
{
  namespace ublas = boost::numeric::ublas;
  using value_t  = typename value::first_type;
  using layout_t = typename value::second_type;

  auto check = [](auto const& /*unused*/, auto& e)
  {
    using extents_t = std::decay_t<decltype (e)>;
    using tensor_t = ublas::tensor_static_rank<value_t, std::tuple_size_v<extents_t>, layout_t>;
    using subtensor = typename tensor_t::subtensor_type;

    auto t  = tensor_t (e);
    auto t2 = tensor_t (e);
    auto v  = value_t  {};
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

  for_each_in_tuple(extents,check);

}


BOOST_FIXTURE_TEST_CASE_TEMPLATE( test_tensor_comparison_with_tensor_expressions, value,  test_types, fixture)
{
  namespace ublas = boost::numeric::ublas;
  using value_t   = typename value::first_type;
  using layout_t  = typename value::second_type;


  for_each_in_tuple(extents,[](auto const& /*unused*/, auto& e) {
    using extents_t = std::decay_t<decltype (e)>;
    using tensor_t  = ublas::tensor_static_rank<value_t, std::tuple_size_v<extents_t>, layout_t>;
    using subtensor = typename tensor_t::subtensor_type;

    auto t  = tensor_t (e);
    auto t2 = tensor_t (e);
    auto v  = value_t  {};
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
  });


}



//BOOST_FIXTURE_TEST_CASE_TEMPLATE( test_tensor_comparison_with_scalar, value,  test_types, fixture)
//{
//  namespace ublas = boost::numeric::ublas;
//  using value_t  = typename value::first_type;
//  using layout_t = typename value::second_type;


//  for_each_in_tuple(extents, [](auto const& /*unused*/, auto& e) {
//    using extents_t = std::decay_t<decltype (e)>;
//    using tensor_t  = ublas::tensor_static_rank<value_t, std::tuple_size_v<extents_t>, layout_t>;

//    BOOST_CHECK( tensor_t(e,value_t{2}) == tensor_t(e,value_t{2})  );
//    BOOST_CHECK( tensor_t(e,value_t{2}) != tensor_t(e,value_t{1})  );

//    if(ublas::empty(e))
//      return;

//    BOOST_CHECK( !(tensor_t(e,2) <  2) );
//    BOOST_CHECK( !(tensor_t(e,2) >  2) );
//    BOOST_CHECK(  (tensor_t(e,2) >= 2) );
//    BOOST_CHECK(  (tensor_t(e,2) <= 2) );
//    BOOST_CHECK(  (tensor_t(e,2) == 2) );
//    BOOST_CHECK(  (tensor_t(e,2) != 3) );

//    BOOST_CHECK( !(2 >  tensor_t(e,2)) );
//    BOOST_CHECK( !(2 <  tensor_t(e,2)) );
//    BOOST_CHECK(  (2 <= tensor_t(e,2)) );
//    BOOST_CHECK(  (2 >= tensor_t(e,2)) );
//    BOOST_CHECK(  (2 == tensor_t(e,2)) );
//    BOOST_CHECK(  (3 != tensor_t(e,2)) );

//    BOOST_CHECK( !( tensor_t(e,2)+3 <  5) );
//    BOOST_CHECK( !( tensor_t(e,2)+3 >  5) );
//    BOOST_CHECK(  ( tensor_t(e,2)+3 >= 5) );
//    BOOST_CHECK(  ( tensor_t(e,2)+3 <= 5) );
//    BOOST_CHECK(  ( tensor_t(e,2)+3 == 5) );
//    BOOST_CHECK(  ( tensor_t(e,2)+3 != 6) );


//    BOOST_CHECK( !( 5 >  tensor_t(e,2)+3) );
//    BOOST_CHECK( !( 5 <  tensor_t(e,2)+3) );
//    BOOST_CHECK(  ( 5 >= tensor_t(e,2)+3) );
//    BOOST_CHECK(  ( 5 <= tensor_t(e,2)+3) );
//    BOOST_CHECK(  ( 5 == tensor_t(e,2)+3) );
//    BOOST_CHECK(  ( 6 != tensor_t(e,2)+3) );


//    BOOST_CHECK( !( tensor_t(e,2)+tensor_t(e,3) <  5) );
//    BOOST_CHECK( !( tensor_t(e,2)+tensor_t(e,3) >  5) );
//    BOOST_CHECK(  ( tensor_t(e,2)+tensor_t(e,3) >= 5) );
//    BOOST_CHECK(  ( tensor_t(e,2)+tensor_t(e,3) <= 5) );
//    BOOST_CHECK(  ( tensor_t(e,2)+tensor_t(e,3) == 5) );
//    BOOST_CHECK(  ( tensor_t(e,2)+tensor_t(e,3) != 6) );


//    BOOST_CHECK( !( 5 >  tensor_t(e,2)+tensor_t(e,3)) );
//    BOOST_CHECK( !( 5 <  tensor_t(e,2)+tensor_t(e,3)) );
//    BOOST_CHECK(  ( 5 >= tensor_t(e,2)+tensor_t(e,3)) );
//    BOOST_CHECK(  ( 5 <= tensor_t(e,2)+tensor_t(e,3)) );
//    BOOST_CHECK(  ( 5 == tensor_t(e,2)+tensor_t(e,3)) );
//    BOOST_CHECK(  ( 6 != tensor_t(e,2)+tensor_t(e,3)) );

//  });

//}


BOOST_AUTO_TEST_SUITE_END()
