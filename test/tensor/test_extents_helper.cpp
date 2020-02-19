//  Copyright (c) 2018-2019 Cem Bassoy
//
//  Distributed under the Boost Software License, Version 1.0. (See
//  accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#include <boost/numeric/ublas/tensor/extents_helper.hpp>
#include <boost/test/unit_test.hpp>
#include <map>
#include <numeric>
#include "utility.hpp"
#include <vector>

BOOST_AUTO_TEST_SUITE(test_basic_extents_impl)

template <ptrdiff_t R, ptrdiff_t... E>
using extents = boost::numeric::ublas::detail::basic_extents_impl<
    0, boost::numeric::ublas::detail::make_basic_shape_t<R, E...>>;
constexpr ptrdiff_t dynamic_extent{-1};

BOOST_AUTO_TEST_CASE(test_basic_extents_impl_ctor) {
  using namespace boost::numeric;

  auto e0 = extents<0>{};
  BOOST_CHECK(e0.empty());
  BOOST_CHECK_EQUAL(e0.size(), 0);

  auto e1 = extents<2>{1, 1};
  BOOST_CHECK(!e1.empty());
  BOOST_CHECK_EQUAL(e1.size(), 2);

  auto e2 = extents<2>{1, 2};
  BOOST_CHECK(!e2.empty());
  BOOST_CHECK_EQUAL(e2.size(), 2);

  auto e3 = extents<2>{2, 1};
  BOOST_CHECK(!e3.empty());
  BOOST_CHECK_EQUAL(e3.size(), 2);

  auto e4 = extents<5, 1, dynamic_extent, dynamic_extent, 4, 5>{2, 3};
  BOOST_CHECK(!e4.empty());
  BOOST_CHECK_EQUAL(e4.size(), 5);

  auto e5 =
      extents<4, dynamic_extent, dynamic_extent, dynamic_extent, 4>{2, 3, 1};
  BOOST_CHECK(!e5.empty());
  BOOST_CHECK_EQUAL(e5.size(), 4);

  auto e6 = extents<3>{1, 2, 3}; // 6
  BOOST_CHECK(!e6.empty());
  BOOST_CHECK_EQUAL(e6.size(), 3);

  auto e7 = extents<3>{4, 2, 3}; // 7
  BOOST_CHECK(!e7.empty());
  BOOST_CHECK_EQUAL(e7.size(), 3);
}

struct fixture {
  fixture() = default;
  extents<0> e0{};                              // 0
  extents<4> e1{};                              // 1
  extents<3, 1, 2, 3> e2{};                     // 2
  extents<3, 4, 2, 3> e3{};                     // 3
  extents<4, 4, 2, 1, 3> e4{};                  // 4
  extents<6, 1, 4, 2, 1, 3, 1> e5{};            // 5
  extents<4, 1, dynamic_extent, 2, 3> e6{1};    // 6
  extents<5, 4, 2, 1, 3, dynamic_extent> e7{1}; // 7
  extents<2> e8{1, 1};                          // 8
  extents<2> e9{1, 2};                          // 9
  extents<2> e10{2, 1};                         // 10
  extents<2> e11{2, 3};                         // 11
  extents<3> e12{2, 3, 1};                      // 12
  extents<5, 1, dynamic_extent, dynamic_extent, 1, dynamic_extent> e13{2, 3,
                                                                       1}; // 13
  std::tuple<
    extents<0>
  > rank_0_extents;

  std::tuple<
    extents<1,1>,
    extents<1,2>,
    extents<1,3>,
    extents<1,4>,
    extents<1,5>,
    extents<1,6>
  > rank_1_extents;

  std::tuple<
    extents<2,1,1>,
    extents<2,2,2>,
    extents<2,3,3>,
    extents<2,4,4>,
    extents<2,5,5>,
    extents<2,6,6>
  > rank_2_extents;

  using r1 = extents<1,-1>;
  std::tuple<
    r1,r1,r1,r1,r1,r1
  > rank_1_dynamic_extents{
    r1{1},
    r1{2},
    r1{3},
    r1{4},
    r1{5},
    r1{6},
  };

  using r2 = extents<2,-1,-1>;
  std::tuple<r2,r2,r2,r2,r2,r2> rank_2_dynamic_extents{
    r2{1,1},
    r2{2,2},
    r2{3,3},
    r2{4,4},
    r2{5,5},
    r2{6,6}
  };

  
  std::tuple<
    extents<1,-1>,
    extents<2,-1,-1>,
    extents<3,-1,-1,-1>,
    extents<4,-1,-1,-1,-1>,
    extents<5,-1,-1,-1,-1,-1>,
    extents<6,-1,-1,-1,-1,-1,-1>
  > scalars_dynamic{
    extents<1,-1>{1,1},
    extents<2,-1,-1>{1,1},
    extents<3,-1,-1,-1>{1,1,1},
    extents<4,-1,-1,-1,-1>{1,1,1,1},
    extents<5,-1,-1,-1,-1,-1>{1,1,1,1,1},
    extents<6,-1,-1,-1,-1,-1,-1>{1,1,1,1,1,1}
  };

  std::tuple<
    extents<1,1>,
    extents<2,1,1>,
    extents<3,1,1,1>,
    extents<4,1,1,1,1>,
    extents<5,1,1,1,1,1>,
    extents<6,1,1,1,1,1,1>
  > scalars;

  std::tuple<
    extents<2,1,2>,
    extents<3,1,3,1>,
    extents<4,1,4,1,1>,
    extents<5,5,1,1,1,1>,
    extents<6,6,1,1,1,1,1>
  > vectors;

  std::tuple<
    extents<2,2,2>,
    extents<2,3,2>,
    extents<2,2,3>,
    extents<3,3,3,1>,
    extents<3,2,3,1>,
    extents<3,3,2,1>,
    extents<4,4,4,1,1>,
    extents<4,3,4,1,1>,
    extents<4,4,3,1,1>,
    extents<5,5,5,1,1,1>,
    extents<6,6,6,1,1,1,1>
  > matrices;

  std::tuple<
    extents<3,3,3,3>,
    extents<3,2,3,3>,
    extents<3,3,2,3>,
    extents<3,3,2,2>,
    extents<4,4,4,4,1>,
    extents<4,3,4,4,1>,
    extents<4,3,4,1,4>,
    extents<4,4,3,5,1>,
    extents<4,4,3,3,3>,
    extents<5,5,5,5,1,1>,
    extents<5,5,5,1,5,1>,
    extents<6,6,6,6,1,1,1>,
    extents<6,6,6,1,6,1,1>,
    extents<6,6,6,1,1,6,1>,
    extents<6,6,6,1,1,1,6>
  > tensors;
};

BOOST_FIXTURE_TEST_CASE(test_basic_extents_impl_access, fixture,
                        *boost::unit_test::label("basic_extents_impl") *
                            boost::unit_test::label("access")) {
  using namespace boost::numeric;

  BOOST_CHECK_EQUAL(e0.size(), 0);
  BOOST_CHECK(e0.empty());

  BOOST_REQUIRE_EQUAL(e1.size(), 4);
  BOOST_REQUIRE_EQUAL(e2.size(), 3);
  BOOST_REQUIRE_EQUAL(e3.size(), 3);
  BOOST_REQUIRE_EQUAL(e4.size(), 4);
  BOOST_REQUIRE_EQUAL(e5.size(), 6);
  BOOST_REQUIRE_EQUAL(e6.size(), 4);
  BOOST_REQUIRE_EQUAL(e7.size(), 5);
  BOOST_REQUIRE_EQUAL(e8.size(), 2);
  BOOST_REQUIRE_EQUAL(e9.size(), 2);
  BOOST_REQUIRE_EQUAL(e10.size(), 2);
  BOOST_REQUIRE_EQUAL(e11.size(), 2);
  BOOST_REQUIRE_EQUAL(e12.size(), 3);
  BOOST_REQUIRE_EQUAL(e13.size(), 5);

  BOOST_REQUIRE_EQUAL(decltype(e1)::DynamicRank, 4);
  BOOST_REQUIRE_EQUAL(decltype(e2)::DynamicRank, 0);
  BOOST_REQUIRE_EQUAL(decltype(e3)::DynamicRank, 0);
  BOOST_REQUIRE_EQUAL(decltype(e4)::DynamicRank, 0);
  BOOST_REQUIRE_EQUAL(decltype(e5)::DynamicRank, 0);
  BOOST_REQUIRE_EQUAL(decltype(e6)::DynamicRank, 1);
  BOOST_REQUIRE_EQUAL(decltype(e7)::DynamicRank, 1);
  BOOST_REQUIRE_EQUAL(decltype(e8)::DynamicRank, 2);
  BOOST_REQUIRE_EQUAL(decltype(e9)::DynamicRank, 2);
  BOOST_REQUIRE_EQUAL(decltype(e10)::DynamicRank, 2);
  BOOST_REQUIRE_EQUAL(decltype(e11)::DynamicRank, 2);
  BOOST_REQUIRE_EQUAL(decltype(e12)::DynamicRank, 3);
  BOOST_REQUIRE_EQUAL(decltype(e13)::DynamicRank, 3);

  BOOST_CHECK_EQUAL(e1[0], 0);
  BOOST_CHECK_EQUAL(e1[1], 0);
  BOOST_CHECK_EQUAL(e1[2], 0);
  BOOST_CHECK_EQUAL(e1[3], 0);
  BOOST_CHECK_EQUAL(ublas::detail::in_bounds(e1,0, 0, 0, 0), false);
  BOOST_CHECK_EQUAL(ublas::detail::in_bounds(e1,1, 0, 0, 0), false);
  BOOST_CHECK_EQUAL(ublas::detail::in_bounds(e1,0, 1, 0, 0), false);
  BOOST_CHECK_EQUAL(ublas::detail::in_bounds(e1,0, 0, 1, 0), false);
  BOOST_CHECK_EQUAL(ublas::detail::in_bounds(e1,0, 0, 0, 1), false);
  BOOST_CHECK_EQUAL(ublas::detail::in_bounds(e1,1, 1, 1, 1), false);
  BOOST_CHECK_EQUAL(ublas::detail::in_bounds(e1,1, 1, 1, -1), false);

  BOOST_CHECK_EQUAL(e2[0], 1);
  BOOST_CHECK_EQUAL(e2[1], 2);
  BOOST_CHECK_EQUAL(e2[2], 3);
  BOOST_CHECK_EQUAL(ublas::detail::in_bounds(e2,0, 0, 0), true);
  BOOST_CHECK_EQUAL(ublas::detail::in_bounds(e2,1, 0, 0), false);
  BOOST_CHECK_EQUAL(ublas::detail::in_bounds(e2,0, 1, 0), true);
  BOOST_CHECK_EQUAL(ublas::detail::in_bounds(e2,0, 0, 1), true);
  BOOST_CHECK_EQUAL(ublas::detail::in_bounds(e2,0, 3, 0), false);
  BOOST_CHECK_EQUAL(ublas::detail::in_bounds(e2,1, 1, 1), false);
  BOOST_CHECK_EQUAL(ublas::detail::in_bounds(e2,0, 1, 1), true);

  BOOST_CHECK_EQUAL(e3[0], 4);
  BOOST_CHECK_EQUAL(e3[1], 2);
  BOOST_CHECK_EQUAL(e3[2], 3);
  BOOST_CHECK_EQUAL(ublas::detail::in_bounds(e3,0, 0, 0), true);
  BOOST_CHECK_EQUAL(ublas::detail::in_bounds(e3,3, 0, 0), true);
  BOOST_CHECK_EQUAL(ublas::detail::in_bounds(e3,0, 1, 0), true);
  BOOST_CHECK_EQUAL(ublas::detail::in_bounds(e3,0, 0, 1), true);
  BOOST_CHECK_EQUAL(ublas::detail::in_bounds(e3,0, 3, 0), false);
  BOOST_CHECK_EQUAL(ublas::detail::in_bounds(e3,1, 1, 1), true);
  BOOST_CHECK_EQUAL(ublas::detail::in_bounds(e3,0, 1, 1), true);

  BOOST_CHECK_EQUAL(e4[0], 4);
  BOOST_CHECK_EQUAL(e4[1], 2);
  BOOST_CHECK_EQUAL(e4[2], 1);
  BOOST_CHECK_EQUAL(e4[3], 3);
  BOOST_CHECK_EQUAL(ublas::detail::in_bounds(e4,0, 0, 0, 0), true);
  BOOST_CHECK_EQUAL(ublas::detail::in_bounds(e4,1, 0, 0, 0), true);
  BOOST_CHECK_EQUAL(ublas::detail::in_bounds(e4,0, 1, 0, 0), true);
  BOOST_CHECK_EQUAL(ublas::detail::in_bounds(e4,0, 0, 1, 0), false);
  BOOST_CHECK_EQUAL(ublas::detail::in_bounds(e4,0, 0, 0, 1), true);
  BOOST_CHECK_EQUAL(ublas::detail::in_bounds(e4,1, 1, 1, 1), false);
  BOOST_CHECK_EQUAL(ublas::detail::in_bounds(e4,1, 1, 1, -1), false);

  BOOST_CHECK_EQUAL(e5[0], 1);
  BOOST_CHECK_EQUAL(e5[1], 4);
  BOOST_CHECK_EQUAL(e5[2], 2);
  BOOST_CHECK_EQUAL(e5[3], 1);
  BOOST_CHECK_EQUAL(e5[4], 3);
  BOOST_CHECK_EQUAL(e5[5], 1);
  BOOST_CHECK_EQUAL(ublas::detail::in_bounds(e5,0, 0, 0, 0, 0, 0), true);
  BOOST_CHECK_EQUAL(ublas::detail::in_bounds(e5,1, 0, 0, 0, 0, 0), false);
  BOOST_CHECK_EQUAL(ublas::detail::in_bounds(e5,0, 1, 0, 0, 0, 0), true);
  BOOST_CHECK_EQUAL(ublas::detail::in_bounds(e5,0, 0, 1, 0, 0, 0), true);
  BOOST_CHECK_EQUAL(ublas::detail::in_bounds(e5,0, 0, 0, 1, 0, 0), false);
  BOOST_CHECK_EQUAL(ublas::detail::in_bounds(e5,0, 0, 0, 0, 1, 0), true);
  BOOST_CHECK_EQUAL(ublas::detail::in_bounds(e5,0, 0, 0, 0, 0, 1), false);
  BOOST_CHECK_EQUAL(ublas::detail::in_bounds(e5,1, 1, 1, 1, 1, 1), false);
  BOOST_CHECK_EQUAL(ublas::detail::in_bounds(e5,1, 1, 1, -1, 1), false);

  BOOST_CHECK_EQUAL(e6[0], 1);
  BOOST_CHECK_EQUAL(e6[1], 1);
  BOOST_CHECK_EQUAL(e6[2], 2);
  BOOST_CHECK_EQUAL(e6[3], 3);
  BOOST_CHECK_EQUAL(ublas::detail::in_bounds(e6,0, 0, 0, 0), true);
  BOOST_CHECK_EQUAL(ublas::detail::in_bounds(e6,1, 0, 0, 0), false);
  BOOST_CHECK_EQUAL(ublas::detail::in_bounds(e6,0, 1, 0, 0), false);
  BOOST_CHECK_EQUAL(ublas::detail::in_bounds(e6,0, 0, 1, 0), true);
  BOOST_CHECK_EQUAL(ublas::detail::in_bounds(e6,0, 0, 0, 1), true);
  BOOST_CHECK_EQUAL(ublas::detail::in_bounds(e6,1, 1, 1, 1), false);
  BOOST_CHECK_EQUAL(ublas::detail::in_bounds(e6,1, 1, 1, -1), false);

  BOOST_CHECK_EQUAL(e7[0], 4);
  BOOST_CHECK_EQUAL(e7[1], 2);
  BOOST_CHECK_EQUAL(e7[2], 1);
  BOOST_CHECK_EQUAL(e7[3], 3);
  BOOST_CHECK_EQUAL(e7[4], 1);
  BOOST_CHECK_EQUAL(ublas::detail::in_bounds(e7,0, 0, 0, 0, 0), true);
  BOOST_CHECK_EQUAL(ublas::detail::in_bounds(e7,1, 0, 0, 0, 0), true);
  BOOST_CHECK_EQUAL(ublas::detail::in_bounds(e7,0, 1, 0, 0, 0), true);
  BOOST_CHECK_EQUAL(ublas::detail::in_bounds(e7,0, 0, 1, 0, 0), false);
  BOOST_CHECK_EQUAL(ublas::detail::in_bounds(e7,0, 0, 0, 1, 0), true);
  BOOST_CHECK_EQUAL(ublas::detail::in_bounds(e7,0, 0, 0, 0, 1), false);
  BOOST_CHECK_EQUAL(ublas::detail::in_bounds(e7,1, 1, 1, 1, 0), false);
  BOOST_CHECK_EQUAL(ublas::detail::in_bounds(e7,1, 1, 1, -1), false);

  BOOST_CHECK_EQUAL(e8[0], 1);
  BOOST_CHECK_EQUAL(e8[1], 1);
  BOOST_CHECK_EQUAL(ublas::detail::in_bounds(e8,0, 0), true);
  BOOST_CHECK_EQUAL(ublas::detail::in_bounds(e8,3, 0), false);
  BOOST_CHECK_EQUAL(ublas::detail::in_bounds(e8,0, 1), false);
  BOOST_CHECK_EQUAL(ublas::detail::in_bounds(e8,0, 3), false);
  BOOST_CHECK_EQUAL(ublas::detail::in_bounds(e8,1, 1), false);

  BOOST_CHECK_EQUAL(e9[0], 1);
  BOOST_CHECK_EQUAL(e9[1], 2);
  BOOST_CHECK_EQUAL(ublas::detail::in_bounds(e9,0, 0), true);
  BOOST_CHECK_EQUAL(ublas::detail::in_bounds(e9,3, 0), false);
  BOOST_CHECK_EQUAL(ublas::detail::in_bounds(e9,0, 1), true);
  BOOST_CHECK_EQUAL(ublas::detail::in_bounds(e9,0, 3), false);
  BOOST_CHECK_EQUAL(ublas::detail::in_bounds(e9,1, 1), false);

  BOOST_CHECK_EQUAL(e10[0], 2);
  BOOST_CHECK_EQUAL(e10[1], 1);
  BOOST_CHECK_EQUAL(ublas::detail::in_bounds(e10,0, 0), true);
  BOOST_CHECK_EQUAL(ublas::detail::in_bounds(e10,1, 0), true);
  BOOST_CHECK_EQUAL(ublas::detail::in_bounds(e10,0, 1), false);
  BOOST_CHECK_EQUAL(ublas::detail::in_bounds(e10,0, 3), false);
  BOOST_CHECK_EQUAL(ublas::detail::in_bounds(e10,1, 1), false);

  BOOST_CHECK_EQUAL(e11[0], 2);
  BOOST_CHECK_EQUAL(e11[1], 3);
  BOOST_CHECK_EQUAL(ublas::detail::in_bounds(e11,0, 0), true);
  BOOST_CHECK_EQUAL(ublas::detail::in_bounds(e11,1, 0), true);
  BOOST_CHECK_EQUAL(ublas::detail::in_bounds(e11,0, 1), true);
  BOOST_CHECK_EQUAL(ublas::detail::in_bounds(e11,0, 3), false);
  BOOST_CHECK_EQUAL(ublas::detail::in_bounds(e11,1, 1), true);

  BOOST_CHECK_EQUAL(e12[0], 2);
  BOOST_CHECK_EQUAL(e12[1], 3);
  BOOST_CHECK_EQUAL(e12[2], 1);
  BOOST_CHECK_EQUAL(ublas::detail::in_bounds(e12,0, 0, 0), true);
  BOOST_CHECK_EQUAL(ublas::detail::in_bounds(e12,3, 0, 0), false);
  BOOST_CHECK_EQUAL(ublas::detail::in_bounds(e12,0, 1, 0), true);
  BOOST_CHECK_EQUAL(ublas::detail::in_bounds(e12,0, 0, 1), false);
  BOOST_CHECK_EQUAL(ublas::detail::in_bounds(e12,0, 3, 0), false);
  BOOST_CHECK_EQUAL(ublas::detail::in_bounds(e12,1, 1, 1), false);
  BOOST_CHECK_EQUAL(ublas::detail::in_bounds(e12,0, 1, 1), false);

  BOOST_CHECK_EQUAL(e13[0], 1);
  BOOST_CHECK_EQUAL(e13[1], 2);
  BOOST_CHECK_EQUAL(e13[2], 3);
  BOOST_CHECK_EQUAL(e13[3], 1);
  BOOST_CHECK_EQUAL(e13[4], 1);
  BOOST_CHECK_EQUAL(ublas::detail::in_bounds(e13,0, 0, 0, 0, 0), true);
  BOOST_CHECK_EQUAL(ublas::detail::in_bounds(e13,1, 0, 0, 0, 0), false);
  BOOST_CHECK_EQUAL(ublas::detail::in_bounds(e13,0, 1, 0, 0, 0), true);
  BOOST_CHECK_EQUAL(ublas::detail::in_bounds(e13,0, 0, 1, 0, 0), true);
  BOOST_CHECK_EQUAL(ublas::detail::in_bounds(e13,0, 0, 0, 1, 0), false);
  BOOST_CHECK_EQUAL(ublas::detail::in_bounds(e13,0, 0, 0, 0, 1), false);
  BOOST_CHECK_EQUAL(ublas::detail::in_bounds(e13,1, 1, 1, 0, 0), false);
  BOOST_CHECK_EQUAL(ublas::detail::in_bounds(e13,1, 1, 1, -1), false);

  for_each_tuple(rank_0_extents,[](auto const& I, auto const& e){
    for(auto i = 0; i < e.size(); i++){
      BOOST_CHECK_EQUAL(e[i],1);
    }
  });

  for_each_tuple(rank_1_extents,[](auto const& I, auto const& e){
    for(auto i = 0; i < e.size(); i++){
      BOOST_CHECK_EQUAL(e[i],I + 1);
    }
  });

  for_each_tuple(rank_1_dynamic_extents,[](auto const& I, auto const& e){
    for(auto i = 0; i < e.size(); i++){
      BOOST_CHECK_EQUAL(e[i],I + 1);
    }
  });


  for_each_tuple(rank_2_extents,[](auto const& I, auto const& e){
    for(auto i = 0; i < e.size(); i++){
      BOOST_CHECK_EQUAL(e[i],I + 1);
    }
  });
  for_each_tuple(rank_2_dynamic_extents,[](auto const& I, auto const& e){
    for(auto i = 0; i < e.size(); i++){
      BOOST_CHECK_EQUAL(e[i],I + 1);
    }
  });


  for_each_tuple(scalars,[](auto const& I, auto const& e){
    for(auto i = 0; i < e.size(); i++){
      BOOST_CHECK_EQUAL(e[i],1);
    }
  });

  for_each_tuple(scalars_dynamic,[](auto const& I, auto const& e){
    for(auto i = 0; i < e.size(); i++){
      BOOST_CHECK_EQUAL(e[i],1);
    }
  });

  for_each_tuple(rank_1_extents,[](auto const& I, auto const& e){
    BOOST_CHECK_EQUAL(ublas::detail::in_bounds(e,-100),false);
    BOOST_CHECK_EQUAL(ublas::detail::in_bounds(e,100),false);
    BOOST_CHECK_EQUAL(ublas::detail::in_bounds(e,0),true);
  });

  for_each_tuple(rank_1_dynamic_extents,[](auto const& I, auto const& e){
    BOOST_CHECK_EQUAL(ublas::detail::in_bounds(e,-100),false);
    BOOST_CHECK_EQUAL(ublas::detail::in_bounds(e,100),false);
  });


  for_each_tuple(rank_2_extents,[](auto const& I, auto const& e){
    BOOST_CHECK_EQUAL(ublas::detail::in_bounds(e,-100,0),false);
    BOOST_CHECK_EQUAL(ublas::detail::in_bounds(e,100,0),false);
  });

  for_each_tuple(rank_2_dynamic_extents,[](auto const& I, auto const& e){
    BOOST_CHECK_EQUAL(ublas::detail::in_bounds(e,-100,0),false);
    BOOST_CHECK_EQUAL(ublas::detail::in_bounds(e,100,0),false);
  });


  auto product_lm = [](auto const& e){
    auto p = 1;
    for(auto i = 0; i < e.size();i++){
      p *= e.at(i);
    }
    return p;
  };

  for_each_tuple(rank_1_extents,[&](auto const& I, auto const& e){
    BOOST_CHECK_EQUAL(e.product(),product_lm(e));
  });

  for_each_tuple(rank_1_dynamic_extents,[&](auto const& I, auto const& e){
    BOOST_CHECK_EQUAL(e.product(),product_lm(e));
  });


  for_each_tuple(rank_2_extents,[&](auto const& I, auto const& e){
    BOOST_CHECK_EQUAL(e.product(),product_lm(e));
  });

  for_each_tuple(rank_2_dynamic_extents,[&](auto const& I, auto const& e){
    BOOST_CHECK_EQUAL(e.product(),product_lm(e));
  });

  for_each_tuple(scalars,[&](auto const& I, auto const& e){
    BOOST_CHECK_EQUAL(e.product(),product_lm(e));
  });

  for_each_tuple(scalars_dynamic,[&](auto const& I, auto const& e){
    BOOST_CHECK_EQUAL(e.product(),product_lm(e));
  });


}

BOOST_FIXTURE_TEST_CASE(test_basic_extents_impl_copy_ctor, fixture,
                        *boost::unit_test::label("basic_extents_impl") *
                            boost::unit_test::label("copy_ctor")) {

  auto e_c_0 = e0;   // {}
  auto e_c_1 = e1;   // {0,0,0,0}
  auto e_c_2 = e2;   // {1,2,3}
  auto e_c_3 = e3;   // {4,2,3}
  auto e_c_4 = e4;   // {4,2,1,3}
  auto e_c_5 = e5;   // {1,4,2,1,3,1}
  auto e_c_6 = e6;   // {1,1,2,3}
  auto e_c_7 = e7;   // {4,2,1,3,1}
  auto e_c_8 = e8;   // {1,1}
  auto e_c_9 = e9;   // {1,2}
  auto e_c_10 = e10; // {2,1}
  auto e_c_11 = e11; // {2,3}
  auto e_c_12 = e12; // {2,3,1}
  auto e_c_13 = e13; // {1,2,3,1,1}

  BOOST_CHECK_EQUAL(e_c_0.size(), 0);
  BOOST_CHECK(e_c_0.empty());

  BOOST_REQUIRE_EQUAL(e_c_1.size(), 4);
  BOOST_REQUIRE_EQUAL(e_c_2.size(), 3);
  BOOST_REQUIRE_EQUAL(e_c_3.size(), 3);
  BOOST_REQUIRE_EQUAL(e_c_4.size(), 4);
  BOOST_REQUIRE_EQUAL(e_c_5.size(), 6);
  BOOST_REQUIRE_EQUAL(e_c_6.size(), 4);
  BOOST_REQUIRE_EQUAL(e_c_7.size(), 5);
  BOOST_REQUIRE_EQUAL(e_c_8.size(), 2);
  BOOST_REQUIRE_EQUAL(e_c_9.size(), 2);
  BOOST_REQUIRE_EQUAL(e_c_10.size(), 2);
  BOOST_REQUIRE_EQUAL(e_c_11.size(), 2);
  BOOST_REQUIRE_EQUAL(e_c_12.size(), 3);
  BOOST_REQUIRE_EQUAL(e_c_13.size(), 5);

  BOOST_CHECK_EQUAL(e_c_1[0], 0);
  BOOST_CHECK_EQUAL(e_c_1[1], 0);
  BOOST_CHECK_EQUAL(e_c_1[2], 0);
  BOOST_CHECK_EQUAL(e_c_1[3], 0);

  BOOST_CHECK_EQUAL(e_c_2[0], 1);
  BOOST_CHECK_EQUAL(e_c_2[1], 2);
  BOOST_CHECK_EQUAL(e_c_2[2], 3);

  BOOST_CHECK_EQUAL(e_c_3[0], 4);
  BOOST_CHECK_EQUAL(e_c_3[1], 2);
  BOOST_CHECK_EQUAL(e_c_3[2], 3);

  BOOST_CHECK_EQUAL(e_c_4[0], 4);
  BOOST_CHECK_EQUAL(e_c_4[1], 2);
  BOOST_CHECK_EQUAL(e_c_4[2], 1);
  BOOST_CHECK_EQUAL(e_c_4[3], 3);

  BOOST_CHECK_EQUAL(e_c_5[0], 1);
  BOOST_CHECK_EQUAL(e_c_5[1], 4);
  BOOST_CHECK_EQUAL(e_c_5[2], 2);
  BOOST_CHECK_EQUAL(e_c_5[3], 1);
  BOOST_CHECK_EQUAL(e_c_5[4], 3);
  BOOST_CHECK_EQUAL(e_c_5[5], 1);

  BOOST_CHECK_EQUAL(e_c_6[0], 1);
  BOOST_CHECK_EQUAL(e_c_6[1], 1);
  BOOST_CHECK_EQUAL(e_c_6[2], 2);
  BOOST_CHECK_EQUAL(e_c_6[3], 3);

  BOOST_CHECK_EQUAL(e_c_7[0], 4);
  BOOST_CHECK_EQUAL(e_c_7[1], 2);
  BOOST_CHECK_EQUAL(e_c_7[2], 1);
  BOOST_CHECK_EQUAL(e_c_7[3], 3);
  BOOST_CHECK_EQUAL(e_c_7[4], 1);

  BOOST_CHECK_EQUAL(e_c_8[0], 1);
  BOOST_CHECK_EQUAL(e_c_8[1], 1);

  BOOST_CHECK_EQUAL(e_c_9[0], 1);
  BOOST_CHECK_EQUAL(e_c_9[1], 2);

  BOOST_CHECK_EQUAL(e_c_10[0], 2);
  BOOST_CHECK_EQUAL(e_c_10[1], 1);

  BOOST_CHECK_EQUAL(e_c_11[0], 2);
  BOOST_CHECK_EQUAL(e_c_11[1], 3);

  BOOST_CHECK_EQUAL(e_c_12[0], 2);
  BOOST_CHECK_EQUAL(e_c_12[1], 3);
  BOOST_CHECK_EQUAL(e_c_12[2], 1);

  BOOST_CHECK_EQUAL(e_c_13[0], 1);
  BOOST_CHECK_EQUAL(e_c_13[1], 2);
  BOOST_CHECK_EQUAL(e_c_13[2], 3);
  BOOST_CHECK_EQUAL(e_c_13[3], 1);
  BOOST_CHECK_EQUAL(e_c_13[4], 1);
}

BOOST_FIXTURE_TEST_CASE(test_basic_extents_impl_product, fixture,
                        *boost::unit_test::label("basic_extents_impl") *
                            boost::unit_test::label("product")) {

  auto p0 = e0.product();   // {}
  auto p1 = e1.product();   // {0,0,0,0}
  auto p2 = e2.product();   // {1,2,3}
  auto p3 = e3.product();   // {4,2,3}
  auto p4 = e4.product();   // {4,2,1,3}
  auto p5 = e5.product();   // {1,4,2,1,3,1}
  auto p6 = e6.product();   // {1,1,2,3}
  auto p7 = e7.product();   // {4,2,1,3,1}
  auto p8 = e8.product();   // {1,1}
  auto p9 = e9.product();   // {1,2}
  auto p10 = e10.product(); // {2,1}
  auto p11 = e11.product(); // {2,3}
  auto p12 = e12.product(); // {2,3,1}
  auto p13 = e13.product(); // {1,2,3,1,1}

  BOOST_CHECK_EQUAL(p0, 1);
  BOOST_CHECK_EQUAL(p1, 0);
  BOOST_CHECK_EQUAL(p2, 6);
  BOOST_CHECK_EQUAL(p3, 24);
  BOOST_CHECK_EQUAL(p4, 24);
  BOOST_CHECK_EQUAL(p5, 24);
  BOOST_CHECK_EQUAL(p6, 6);
  BOOST_CHECK_EQUAL(p7, 24);
  BOOST_CHECK_EQUAL(p8, 1);
  BOOST_CHECK_EQUAL(p9, 2);
  BOOST_CHECK_EQUAL(p10, 2);
  BOOST_CHECK_EQUAL(p11, 6);
  BOOST_CHECK_EQUAL(p12, 6);
  BOOST_CHECK_EQUAL(p13, 6);
}

BOOST_AUTO_TEST_CASE(test_basic_extents_impl_initialize_dynamic_extents,
                     *boost::unit_test::label("basic_extents_impl") *
                         boost::unit_test::label("initialize")) {

  using namespace boost::numeric::ublas;
  std::initializer_list<int> li = {1, 2, 3, 4};
  std::vector<int> v = {5, 6, 7, 8};
  std::map<int, int> m = {{1, 1}, {2, 2}, {3, 3}};

  extents<4> e_li(li.begin(), li.end(),
                  detail::iterator_tag_t<decltype(li.begin())>{}); //{1,2,3,4}
  extents<4> e_v(v.begin(), v.end(),
                 detail::iterator_tag_t<decltype(v.begin())>{}); //{5.6.7.8}
  extents<4, 11, 22, dynamic_extent, dynamic_extent> e_p(33,
                                                         44); //{11,22,33,44}

  BOOST_CHECK_THROW(extents<3>(m.begin(), m.end(),
                               detail::iterator_tag_t<decltype(m.begin())>{}),
                    std::runtime_error);

  BOOST_CHECK_EQUAL(e_li[0], 1);
  BOOST_CHECK_EQUAL(e_li[1], 2);
  BOOST_CHECK_EQUAL(e_li[2], 3);
  BOOST_CHECK_EQUAL(e_li[3], 4);

  BOOST_CHECK_EQUAL(e_v[0], 5);
  BOOST_CHECK_EQUAL(e_v[1], 6);
  BOOST_CHECK_EQUAL(e_v[2], 7);
  BOOST_CHECK_EQUAL(e_v[3], 8);

  BOOST_CHECK_EQUAL(e_p[0], 11);
  BOOST_CHECK_EQUAL(e_p[1], 22);
  BOOST_CHECK_EQUAL(e_p[2], 33);
  BOOST_CHECK_EQUAL(e_p[3], 44);
}

// BOOST_AUTO_TEST_CASE(test_basic_extents_impl_access,
//                      *boost::unit_test::label("basic_extents_impl") *
//                          boost::unit_test::label("initialize")) {


// }

BOOST_AUTO_TEST_CASE(test_basic_extents_impl_exception)
{
  using namespace boost::numeric::ublas::detail;
  
  std::vector<size_t> v = {1,0,3};
  BOOST_REQUIRE_THROW( ( basic_extents_impl<0,make_basic_shape_t<3>>(v.begin(),v.end(),iterator_tag{}) ), std::runtime_error );
  
  BOOST_REQUIRE_THROW( (basic_extents_impl<0,make_basic_shape_t<3>>(v.begin(),v.end(),invalid_iterator_tag{})), std::runtime_error);

}

BOOST_AUTO_TEST_SUITE_END()
