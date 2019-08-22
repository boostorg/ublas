//  Copyright (c) 2018-2019 Cem Bassoy
//
//  Distributed under the Boost Software License, Version 1.0. (See
//  accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#include <boost/numeric/ublas/tensor/slice.hpp>
#include <boost/test/unit_test.hpp>
#include <type_traits>

namespace sp = boost::numeric::ublas::span;
using type = ptrdiff_t;

BOOST_AUTO_TEST_CASE(test_slice_helper_normalized_slice) {
  
  using n1 = typename sp::detail::normalized_slice<type,0,0,1>::type;
  BOOST_CHECK( ( std::is_same_v< n1,  sp::detail::slice_helper<type,0,0,1l,1l> > ) );

  using n2 = typename sp::detail::normalized_slice<type,0,9,2>::type;
  BOOST_CHECK( ( std::is_same_v< n2,  sp::detail::slice_helper<type,0,9,2l,5l> > ) );

  using n3 = typename sp::detail::normalized_slice<type,1,9,2>::type;
  BOOST_CHECK( ( std::is_same_v< n3,  sp::detail::slice_helper<type,1,9,2l,5l> > ) );

  using n4 = typename sp::detail::normalized_slice<type,1,1,2>::type;
  BOOST_CHECK( ( std::is_same_v< n4,  sp::detail::slice_helper<type,1,1,2l,1l> > ) );

  using n5 = typename sp::detail::normalized_slice<type,1,-1,2>::type;
  BOOST_CHECK( ( std::is_same_v< n5,  sp::detail::slice_helper<type,1,-1,2l,0l> > ) );

  using n6 = typename sp::detail::normalized_slice<type,-3,9,2>::type;
  BOOST_CHECK( ( std::is_same_v< n6,  sp::detail::slice_helper<type,-3,9,2l,7l> > ) );

  using n7 = typename sp::detail::normalized_slice<type,-1,-1,2>::type;
  BOOST_CHECK( ( std::is_same_v< n7,  sp::detail::slice_helper<type,-1,-1,2l,1l> > ) );
}

BOOST_AUTO_TEST_CASE(test_slice_helper_normalized_slice_helper) {
  
  auto n1 = sp::detail::normalized_slice_helper<type,0,0,1>{}();
  BOOST_CHECK( ( std::is_same_v< decltype( n1 ),  sp::detail::normalized_slice<type,0,0,1l,1l> > ) );

  auto n2 = sp::detail::normalized_slice_helper<type,0,9,2>{}();
  BOOST_CHECK( ( std::is_same_v< decltype( n2 ),  sp::detail::normalized_slice<type,0,8,2l,5l> > ) );

  auto n3 = sp::detail::normalized_slice_helper<type,1,9,2>{}();
  BOOST_CHECK( ( std::is_same_v< decltype( n3 ),  sp::detail::normalized_slice<type,1,9,2l,5l> > ) );

  auto n4 = sp::detail::normalized_slice_helper<type,1,1,2>{}();
  BOOST_CHECK( ( std::is_same_v< decltype( n4 ),  sp::detail::normalized_slice<type,1,1,2l,1l> > ) );

  auto n5 = sp::detail::normalized_slice_helper<type,1,-1,2>{}();
  BOOST_CHECK( ( std::is_same_v< decltype( n5 ),  sp::detail::normalized_slice<type,1,-1,2l,0l> > ) );

  auto n6 = sp::detail::normalized_slice_helper<type,-3,9,2>{}();
  BOOST_CHECK( ( std::is_same_v< decltype( n6 ),  sp::detail::normalized_slice<type,-3,9,2l,0l> > ) );

  auto n7 = sp::detail::normalized_slice_helper<type,-1,-1,2>{}();
  BOOST_CHECK( ( std::is_same_v< decltype( n7 ),  sp::detail::normalized_slice<type,-1,-1,2l,0l> > ) );
}

BOOST_AUTO_TEST_CASE(test_slice_helper_slice_helper) {

  auto n1 = sp::detail::slice_helper<type,0,0,1,1l>{};
  BOOST_CHECK_EQUAL( n1.first(), 0 );
  BOOST_CHECK_EQUAL( n1.last(), 0 );
  BOOST_CHECK_EQUAL( n1.step(), 1 );
  BOOST_CHECK_EQUAL( n1.size(), 1 );

  auto n2 = sp::detail::slice_helper<type,0,9,2,5l>{};
  BOOST_CHECK_EQUAL( n2.first(), 0 );
  BOOST_CHECK_EQUAL( n2.last(), 9 );
  BOOST_CHECK_EQUAL( n2.step(), 2 );
  BOOST_CHECK_EQUAL( n2.size(), 5 );

  auto n3 = sp::detail::slice_helper<type,1,9,2,5l>{};
  BOOST_CHECK_EQUAL( n3.first(), 1 );
  BOOST_CHECK_EQUAL( n3.last(), 9 );
  BOOST_CHECK_EQUAL( n3.step(), 2 );
  BOOST_CHECK_EQUAL( n3.size(), 5 );

  auto n4 = sp::detail::slice_helper<type,1,1,2,1>{};
  BOOST_CHECK_EQUAL( n4.first(), 1 );
  BOOST_CHECK_EQUAL( n4.last(), 1 );
  BOOST_CHECK_EQUAL( n4.step(), 2 );
  BOOST_CHECK_EQUAL( n4.size(), 1 );

  auto n5 = sp::detail::slice_helper<type,1,-1,2,0>{};
  BOOST_CHECK_EQUAL( n5.first(), 1 );
  BOOST_CHECK_EQUAL( n5.last(), -1 );
  BOOST_CHECK_EQUAL( n5.step(), 2 );
  BOOST_CHECK_EQUAL( n5.size(), 0 );

  auto n6 = sp::detail::slice_helper<type,-3,9,2,7>{};
  BOOST_CHECK_EQUAL( n6.first(), -3 );
  BOOST_CHECK_EQUAL( n6.last(), 9 );
  BOOST_CHECK_EQUAL( n6.step(), 2 );
  BOOST_CHECK_EQUAL( n6.size(), 7 );

  auto n7 = sp::detail::slice_helper<type,-1,-1,2,0>{};
  BOOST_CHECK_EQUAL( n7.first(), -1 );
  BOOST_CHECK_EQUAL( n7.last(), -1 );
  BOOST_CHECK_EQUAL( n7.step(), 2 );
  BOOST_CHECK_EQUAL( n7.size(), 0 );

}

BOOST_AUTO_TEST_CASE(test_slice_helper_typelist) {
  using n1 = sp::detail::list<>;
  using n2 = decltype( sp::detail::push_back(n1{}, int{}) );
  BOOST_CHECK( (std::is_same_v< n2, sp::detail::list<int> >) );

  using n3 = decltype(sp::detail::push_front(n2{},float{}));
  BOOST_CHECK( (std::is_same_v< n3, sp::detail::list<float,int> >) );


  using n4 = decltype(sp::detail::pop_front(n3{}));
  BOOST_CHECK( (std::is_same_v< n4, sp::detail::list<int> >) );


  auto n5 = sp::detail::pop_and_get_front(n3{}) ;
  BOOST_CHECK( (std::is_same_v< std::decay_t< decltype(std::get<0>(n5)) >, float >) );
  BOOST_CHECK( (std::is_same_v< std::decay_t< decltype(std::get<1>(n5)) >, sp::detail::list<int> >) );
  
}


BOOST_AUTO_TEST_CASE(test_slice_helper_slice_common_type) {

  BOOST_CHECK( (std::is_same_v< typename sp::detail::slice_common_type< sp::slice<>, int, sp::basic_slice<size_t> >::type, ptrdiff_t >) );
  BOOST_CHECK( (std::is_same_v< typename sp::detail::slice_common_type<  sp::slice<>, sp::basic_slice<int> >::type, ptrdiff_t >) );
  
}

BOOST_AUTO_TEST_CASE(test_slice_helper_noramlize_value) {
  constexpr auto extent = 10l;
  constexpr auto v1 = -3l;
  constexpr auto v2 = -2l;
  constexpr auto v3 = -1l;
  constexpr auto v4 = 0l;
  constexpr auto v5 = 1l;
  constexpr auto v6 = 2l;
  constexpr auto v7 = 3l;

  BOOST_CHECK_EQUAL( sp::detail::noramlize_value(extent,v1), 7 );
  BOOST_CHECK_EQUAL( sp::detail::noramlize_value(extent,v2), 8 );
  BOOST_CHECK_EQUAL( sp::detail::noramlize_value(extent,v3), 9 );
  BOOST_CHECK_EQUAL( sp::detail::noramlize_value(extent,v4), 0 );
  BOOST_CHECK_EQUAL( sp::detail::noramlize_value(extent,v5), 1 );
  BOOST_CHECK_EQUAL( sp::detail::noramlize_value(extent,v6), 2 );
  BOOST_CHECK_EQUAL( sp::detail::noramlize_value(extent,v7), 3 );


  BOOST_CHECK_EQUAL( ( sp::detail::noramlize_value<extent,v1>() ), 7 );
  BOOST_CHECK_EQUAL( ( sp::detail::noramlize_value<extent,v2>() ), 8 );
  BOOST_CHECK_EQUAL( ( sp::detail::noramlize_value<extent,v3>() ), 9 );
  BOOST_CHECK_EQUAL( ( sp::detail::noramlize_value<extent,v4>() ), 0 );
  BOOST_CHECK_EQUAL( ( sp::detail::noramlize_value<extent,v5>() ), 1 );
  BOOST_CHECK_EQUAL( ( sp::detail::noramlize_value<extent,v6>() ), 2 );
  BOOST_CHECK_EQUAL( ( sp::detail::noramlize_value<extent,v7>() ), 3 );
  
}