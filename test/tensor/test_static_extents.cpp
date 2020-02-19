//  Copyright (c) 2018-2019 Cem Bassoy
//
//  Distributed under the Boost Software License, Version 1.0. (See
//  accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#include <boost/numeric/ublas/tensor/static_extents.hpp>
#include <boost/numeric/ublas/tensor/extents.hpp>
#include <boost/test/unit_test.hpp>
#include <vector>
#include "utility.hpp"

BOOST_AUTO_TEST_SUITE(test_static_extents)

template <ptrdiff_t R, ptrdiff_t... E>
using extents = boost::numeric::ublas::basic_static_extents<unsigned,R, E...>;

constexpr ptrdiff_t dynamic_extent{-1};

BOOST_AUTO_TEST_CASE(test_static_extents_ctor) {
  using namespace boost::numeric;

  auto e0 = extents<0>{};
  BOOST_CHECK(e0.empty());
  BOOST_CHECK_EQUAL(e0.size(), 0);

  auto e1 = extents<2>{1, 1};
  BOOST_CHECK(!e1.empty());
  BOOST_CHECK_EQUAL(e1.size(), 2);

  auto e2 = extents<2, 1, 2>{};
  BOOST_CHECK(!e2.empty());
  BOOST_CHECK_EQUAL(e2.size(), 2);

  auto e3 = extents<2>{2, 1};
  BOOST_CHECK(!e3.empty());
  BOOST_CHECK_EQUAL(e3.size(), 2);

  auto e4 = extents<2, 2, 3>{};
  BOOST_CHECK(!e4.empty());
  BOOST_CHECK_EQUAL(e4.size(), 2);

  auto e5 = extents<3, 2, dynamic_extent, dynamic_extent>{3, 1};
  BOOST_CHECK(!e5.empty());
  BOOST_CHECK_EQUAL(e5.size(), 3);

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

BOOST_FIXTURE_TEST_CASE(test_static_extents_product, fixture,
                        *boost::unit_test::label("static_extents") *
                            boost::unit_test::label("product")) {

  auto p0   = product( e0);   // {}
  auto p1   = product( e1);   // {0,0,0,0}
  auto p2   = product( e2);   // {1,2,3}
  auto p3   = product( e3);   // {4,2,3}
  auto p4   = product( e4);   // {4,2,1,3}
  auto p5   = product( e5);   // {1,4,2,1,3,1}
  auto p6   = product( e6);   // {1, 1, 2, 3}
  auto p7   = product( e7);   // {4,2,1,3,1}
  auto p8   = product( e8);   // {1,1}
  auto p9   = product( e9);   // {1,2}
  auto p10  = product(e10);   // {2,1}
  auto p11  = product(e11);   // {2,3}
  auto p12  = product(e12);   // {2,3,1}
  auto p13  = product(e13);   // {1,2,3,1,1}

  BOOST_CHECK_EQUAL(p0, 0);
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

BOOST_FIXTURE_TEST_CASE(test_static_extents_access, fixture,
                        *boost::unit_test::label("static_extents") *
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

  BOOST_REQUIRE_EQUAL(e1.dynamic_rank(), 4);
  BOOST_REQUIRE_EQUAL(e2.dynamic_rank(), 0);
  BOOST_REQUIRE_EQUAL(e3.dynamic_rank(), 0);
  BOOST_REQUIRE_EQUAL(e4.dynamic_rank(), 0);
  BOOST_REQUIRE_EQUAL(e5.dynamic_rank(), 0);
  BOOST_REQUIRE_EQUAL(e6.dynamic_rank(), 1);
  BOOST_REQUIRE_EQUAL(e7.dynamic_rank(), 1);
  BOOST_REQUIRE_EQUAL(e8.dynamic_rank(), 2);
  BOOST_REQUIRE_EQUAL(e9.dynamic_rank(), 2);
  BOOST_REQUIRE_EQUAL(e10.dynamic_rank(), 2);
  BOOST_REQUIRE_EQUAL(e11.dynamic_rank(), 2);
  BOOST_REQUIRE_EQUAL(e12.dynamic_rank(), 3);
  BOOST_REQUIRE_EQUAL(e13.dynamic_rank(), 3);

  BOOST_CHECK_EQUAL(e1[0], 0);
  BOOST_CHECK_EQUAL(e1[1], 0);
  BOOST_CHECK_EQUAL(e1[2], 0);
  BOOST_CHECK_EQUAL(e1[3], 0);

  BOOST_CHECK_EQUAL(e2[0], 1);
  BOOST_CHECK_EQUAL(e2[1], 2);
  BOOST_CHECK_EQUAL(e2[2], 3);

  BOOST_CHECK_EQUAL(e3[0], 4);
  BOOST_CHECK_EQUAL(e3[1], 2);
  BOOST_CHECK_EQUAL(e3[2], 3);

  BOOST_CHECK_EQUAL(e4[0], 4);
  BOOST_CHECK_EQUAL(e4[1], 2);
  BOOST_CHECK_EQUAL(e4[2], 1);
  BOOST_CHECK_EQUAL(e4[3], 3);

  BOOST_CHECK_EQUAL(e5[0], 1);
  BOOST_CHECK_EQUAL(e5[1], 4);
  BOOST_CHECK_EQUAL(e5[2], 2);
  BOOST_CHECK_EQUAL(e5[3], 1);
  BOOST_CHECK_EQUAL(e5[4], 3);
  BOOST_CHECK_EQUAL(e5[5], 1);

  BOOST_CHECK_EQUAL(e6[0], 1);
  BOOST_CHECK_EQUAL(e6[1], 1);
  BOOST_CHECK_EQUAL(e6[2], 2);
  BOOST_CHECK_EQUAL(e6[3], 3);

  BOOST_CHECK_EQUAL(e7[0], 4);
  BOOST_CHECK_EQUAL(e7[1], 2);
  BOOST_CHECK_EQUAL(e7[2], 1);
  BOOST_CHECK_EQUAL(e7[3], 3);
  BOOST_CHECK_EQUAL(e7[4], 1);

  BOOST_CHECK_EQUAL(e8[0], 1);
  BOOST_CHECK_EQUAL(e8[1], 1);

  BOOST_CHECK_EQUAL(e9[0], 1);
  BOOST_CHECK_EQUAL(e9[1], 2);

  BOOST_CHECK_EQUAL(e10[0], 2);
  BOOST_CHECK_EQUAL(e10[1], 1);

  BOOST_CHECK_EQUAL(e11[0], 2);
  BOOST_CHECK_EQUAL(e11[1], 3);

  BOOST_CHECK_EQUAL(e12[0], 2);
  BOOST_CHECK_EQUAL(e12[1], 3);
  BOOST_CHECK_EQUAL(e12[2], 1);

  BOOST_CHECK_EQUAL(e13[0], 1);
  BOOST_CHECK_EQUAL(e13[1], 2);
  BOOST_CHECK_EQUAL(e13[2], 3);
  BOOST_CHECK_EQUAL(e13[3], 1);
  BOOST_CHECK_EQUAL(e13[4], 1);
}

BOOST_AUTO_TEST_CASE(test_static_extents_initialize_dynamic_extents,
                     *boost::unit_test::label("static_extents") *
                         boost::unit_test::label("initialize")) {

  using namespace boost::numeric::ublas;
  std::initializer_list<int> li = {1, 2, 3, 4};
  std::vector<int> v = {5, 6, 7, 8};
  std::map<int, int> m = {{1, 1}, {2, 2}, {3, 3}};

  extents<4> e_li(li.begin(), li.end()); //{1,2,3,4}
  extents<4> e_v(v.begin(), v.end());    //{5.6.7.8}
  extents<4, 11, 22, dynamic_extent, dynamic_extent> e_p(33,
                                                         44); //{11,22,33,44}

  BOOST_CHECK_THROW(extents<3>(m.begin(), m.end()), std::runtime_error);

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


  for(auto i = 0; i < 10; i++){
    std::vector<size_t> v;
    for(auto j = i * 10 + 1; j < (i + 1) * 10 + 1; j++){
      v.push_back(j);
    }
    extents<10> e(v.begin(),v.end());
    int k = 0;
    for(auto j = i * 10 + 1; j < (i + 1) * 10 + 1; j++){
      BOOST_CHECK_EQUAL(e[k++], j);
    }
  }

}

BOOST_FIXTURE_TEST_CASE(test_static_extents, fixture,
                        *boost::unit_test::label("static_extents") *
                            boost::unit_test::label("query")) {

  // e0  ==> {}
  // e1  ==> {0,0,0,0}
  // e2  ==> {1,2,3}
  // e3  ==> {4,2,3}
  // e4  ==> {4,2,1,3}
  // e5  ==> {1,4,2,1,3,1}
  // e6  ==> {1, 1, 2, 3}
  // e7  ==> {4,2,1,3,1}
  // e8  ==> {1,1}
  // e9  ==> {1,2}
  // e10 ==> {2,1}
  // e11 ==> {2,3}
  // e12 ==> {2,3,1}
  // e13 ==> {1,2,3,1,1}

  BOOST_CHECK(   e0.empty(   ));
  BOOST_CHECK( !is_scalar( e0));
  BOOST_CHECK( !is_vector( e0));
  BOOST_CHECK( !is_matrix( e0));
  BOOST_CHECK( !is_tensor( e0));

  BOOST_CHECK( ! e1.empty(   ) );
  BOOST_CHECK( !is_scalar( e1) );
  BOOST_CHECK( !is_vector( e1) );
  BOOST_CHECK( !is_matrix( e1) );
  BOOST_CHECK( !is_tensor( e1) );

  BOOST_CHECK( ! e2.empty(   ) );
  BOOST_CHECK( !is_scalar( e2) );
  BOOST_CHECK( !is_vector( e2) );
  BOOST_CHECK( !is_matrix( e2) );
  BOOST_CHECK(  is_tensor( e2) );

  BOOST_CHECK( ! e3.empty(   ) );
  BOOST_CHECK( !is_scalar( e3) );
  BOOST_CHECK( !is_vector( e3) );
  BOOST_CHECK( !is_matrix( e3) );
  BOOST_CHECK(  is_tensor( e3) );

  BOOST_CHECK( ! e4.empty(   ) );
  BOOST_CHECK( !is_scalar( e4) );
  BOOST_CHECK( !is_vector( e4) );
  BOOST_CHECK( !is_matrix( e4) );
  BOOST_CHECK(  is_tensor( e4) );

  BOOST_CHECK( ! e5.empty(   ) );
  BOOST_CHECK( !is_scalar( e5) );
  BOOST_CHECK( !is_vector( e5) );
  BOOST_CHECK( !is_matrix( e5) );
  BOOST_CHECK(  is_tensor( e5) );

  BOOST_CHECK( ! e6.empty(   ) );
  BOOST_CHECK( !is_scalar( e6) );
  BOOST_CHECK( !is_vector( e6) );
  BOOST_CHECK( !is_matrix( e6) );
  BOOST_CHECK(  is_tensor( e6) );

  BOOST_CHECK( ! e7.empty(   ) );
  BOOST_CHECK( !is_scalar( e7) );
  BOOST_CHECK( !is_vector( e7) );
  BOOST_CHECK( !is_matrix( e7) );
  BOOST_CHECK(  is_tensor( e7) );

  BOOST_CHECK( ! e8.empty(   ) );
  BOOST_CHECK(  is_scalar( e8) );
  BOOST_CHECK( !is_vector( e8) );
  BOOST_CHECK( !is_matrix( e8) );
  BOOST_CHECK( !is_tensor( e8) );

  BOOST_CHECK( ! e9.empty    () );
  BOOST_CHECK( !is_scalar( e9) );
  BOOST_CHECK(  is_vector( e9) );
  BOOST_CHECK( !is_matrix( e9) );
  BOOST_CHECK( !is_tensor( e9) );

  BOOST_CHECK( ! e10.empty(   ) );
  BOOST_CHECK( ! is_scalar(e10) );
  BOOST_CHECK(   is_vector(e10) );
  BOOST_CHECK( ! is_matrix(e10) );
  BOOST_CHECK( ! is_tensor(e10) );

  BOOST_CHECK( ! e11.empty(   ) );
  BOOST_CHECK( ! is_scalar(e11) );
  BOOST_CHECK( ! is_vector(e11) );
  BOOST_CHECK(   is_matrix(e11) );
  BOOST_CHECK( ! is_tensor(e11) );

  BOOST_CHECK( ! e12.empty(   ) );
  BOOST_CHECK( ! is_scalar(e12) );
  BOOST_CHECK( ! is_vector(e12) );
  BOOST_CHECK(   is_matrix(e12) );
  BOOST_CHECK( ! is_tensor(e12) );

  BOOST_CHECK( ! e13.empty(   ) );
  BOOST_CHECK( ! is_scalar(e13) );
  BOOST_CHECK( ! is_vector(e13) );
  BOOST_CHECK( ! is_matrix(e13) );
  BOOST_CHECK(   is_tensor(e13) );

  boost::numeric::ublas::basic_static_extents<size_t,1,3> e14;
  BOOST_CHECK( ! e14.empty(   ) );
  BOOST_CHECK( ! is_scalar(e14) );
  BOOST_CHECK(   is_vector(e14) );
  BOOST_CHECK( ! is_matrix(e14) );
  BOOST_CHECK( ! is_tensor(e14) );


  for_each_tuple(rank_0_extents,[](auto const& I, auto const& e){
    BOOST_CHECK( !is_scalar(e) );
    BOOST_CHECK( !is_vector(e) );
    BOOST_CHECK( !is_matrix(e) );
    BOOST_CHECK( !is_tensor(e) );
  });


  for_each_tuple(rank_1_extents,[](auto const& I, auto const& e){
    if( I == 0 ){
      BOOST_CHECK(  is_scalar(e) );
      BOOST_CHECK( !is_vector(e) );
      BOOST_CHECK( !is_matrix(e) );
      BOOST_CHECK( !is_tensor(e) );
    }else{
      BOOST_CHECK( !is_scalar(e) );
      BOOST_CHECK(  is_vector(e) );
      BOOST_CHECK( !is_matrix(e) );
      BOOST_CHECK( !is_tensor(e) );
    }
  });

  for_each_tuple(rank_2_extents,[](auto const& I, auto const& e){
    if( I == 0 ){
      BOOST_CHECK(  is_scalar(e) );
      BOOST_CHECK( !is_vector(e) );
      BOOST_CHECK( !is_matrix(e) );
      BOOST_CHECK( !is_tensor(e) );
    }else{
      BOOST_CHECK( !is_scalar(e) );
      BOOST_CHECK( !is_vector(e) );
      BOOST_CHECK(  is_matrix(e) );
      BOOST_CHECK( !is_tensor(e) );
    }
  });

  for_each_tuple(scalars,[](auto const& I, auto const& e){
      BOOST_CHECK(  is_scalar(e) );
      BOOST_CHECK( !is_vector(e) );
      BOOST_CHECK( !is_matrix(e) );
      BOOST_CHECK( !is_tensor(e) );
  });

  for_each_tuple(vectors,[](auto const& I, auto const& e){
      BOOST_CHECK( !is_scalar(e) );
      BOOST_CHECK(  is_vector(e) );
      BOOST_CHECK( !is_matrix(e) );
      BOOST_CHECK( !is_tensor(e) );
  });

  for_each_tuple(matrices,[](auto const& I, auto const& e){
      BOOST_CHECK( !is_scalar(e) );
      BOOST_CHECK( !is_vector(e) );
      BOOST_CHECK(  is_matrix(e) );
      BOOST_CHECK( !is_tensor(e) );
  });

  for_each_tuple(tensors,[](auto const& I, auto const& e){
      BOOST_CHECK( !is_scalar(e) );
      BOOST_CHECK( !is_vector(e) );
      BOOST_CHECK( !is_matrix(e) );
      BOOST_CHECK(  is_tensor(e) );
  });

}

BOOST_FIXTURE_TEST_CASE(test_static_extents_to_functions, fixture, *boost::unit_test::label("static_extents") *boost::unit_test::label("to_functions"))
{
  for_each_tuple(scalars,[](auto const& I, auto const& e){
    auto v = e.to_vector();
    for(auto i = 0; i < v.size(); i++){
      BOOST_CHECK(v[i] == e[i]);
    }
  });

  for_each_tuple(scalars,[](auto const& I, auto const& e){
    auto v = e.to_array();
    for(auto i = 0; i < v.size(); i++){
      BOOST_CHECK(v[i] == e[i]);
    }
  });

  for_each_tuple(scalars,[](auto const& I, auto const& e){
    if (e.size() > 1){
      auto d = e.to_dynamic_extents();
      BOOST_CHECK(d == e);
    }
  });



  for_each_tuple(vectors,[](auto const& I, auto const& e){
    auto v = e.to_vector();
    for(auto i = 0; i < v.size(); i++){
      BOOST_CHECK(v[i] == e[i]);
    }
  });

  for_each_tuple(vectors,[](auto const& I, auto const& e){
    auto v = e.to_array();
    for(auto i = 0; i < v.size(); i++){
      BOOST_CHECK(v[i] == e[i]);
    }
  });

  for_each_tuple(vectors,[](auto const& I, auto const& e){
    auto d = e.to_dynamic_extents();
    BOOST_CHECK(d == e);
  });



  for_each_tuple(matrices,[](auto const& I, auto const& e){
    auto v = e.to_vector();
    for(auto i = 0; i < v.size(); i++){
      BOOST_CHECK(v[i] == e[i]);
    }
  });

  for_each_tuple(matrices,[](auto const& I, auto const& e){
    auto v = e.to_array();
    for(auto i = 0; i < v.size(); i++){
      BOOST_CHECK(v[i] == e[i]);
    }
  });

  for_each_tuple(matrices,[](auto const& I, auto const& e){
    auto d = e.to_dynamic_extents();
    BOOST_CHECK(d == e);
  });



  for_each_tuple(tensors,[](auto const& I, auto const& e){
    auto v = e.to_vector();
    for(auto i = 0; i < v.size(); i++){
      BOOST_CHECK(v[i] == e[i]);
    }
  });

  for_each_tuple(tensors,[](auto const& I, auto const& e){
    auto v = e.to_array();
    for(auto i = 0; i < v.size(); i++){
      BOOST_CHECK(v[i] == e[i]);
    }
  });
  
  for_each_tuple(tensors,[](auto const& I, auto const& e){
    auto d = e.to_dynamic_extents();
    BOOST_CHECK(d == e);
  });

}

BOOST_FIXTURE_TEST_CASE(test_static_extents_valid, fixture, *boost::unit_test::label("static_extents") *boost::unit_test::label("valid"))
{
  for_each_tuple(rank_0_extents,[](auto const& I, auto const& e){
    BOOST_CHECK(!valid(e));
  });

  for_each_tuple(rank_1_extents,[](auto const& I, auto const& e){
    if( I == 0 ){
      BOOST_CHECK(valid(e));
    }else{
      BOOST_CHECK(!valid(e));
    }
  });

  for_each_tuple(rank_2_extents,[](auto const& I, auto const& e){
      BOOST_CHECK(valid(e));
  });
  
  for_each_tuple(scalars,[](auto const& I, auto const& e){
      BOOST_CHECK(valid(e));
  });
  
  for_each_tuple(vectors,[](auto const& I, auto const& e){
      BOOST_CHECK(valid(e));
  });
  
  for_each_tuple(matrices,[](auto const& I, auto const& e){
      BOOST_CHECK(valid(e));
  });
  
  for_each_tuple(tensors,[](auto const& I, auto const& e){
      BOOST_CHECK(valid(e));
  });
}

BOOST_FIXTURE_TEST_CASE(test_static_extents_comparsion_operator, fixture, *boost::unit_test::label("static_extents") *boost::unit_test::label("comparsion_operator"))
{

  auto const compare_extents = [](auto const& e1, auto const& e2){
    if(e1.size() != e2.size()) return false;
    for(auto i = 0; i <e1.size(); i++){
      if(e1[i] != e2[i]){
        return false;
      }
    }
    return true;
  };

  for_each_tuple(rank_0_extents,[&](auto const& I, auto const& e1){
    for_each_tuple(rank_1_extents,[&](auto const& J, auto const& e2){
      BOOST_CHECK(compare_extents(e1,e2) == (e1 == e2));
    });
  });

  for_each_tuple(rank_1_extents,[&](auto const& I, auto const& e1){
    for_each_tuple(rank_1_extents,[&](auto const& J, auto const& e2){
      BOOST_CHECK(compare_extents(e1,e2) == (e1 == e2));
    });
  });

  for_each_tuple(rank_1_extents,[&](auto const& I, auto const& e1){
    for_each_tuple(rank_2_extents,[&](auto const& J, auto const& e2){
      BOOST_CHECK(compare_extents(e1,e2) == (e1 == e2));
    });
  });

  for_each_tuple(scalars,[&](auto const& I, auto const& e1){
    for_each_tuple(scalars,[&](auto const& J, auto const& e2){
      BOOST_CHECK(compare_extents(e1,e2) == (e1 == e2));
    });
  });

  for_each_tuple(scalars,[&](auto const& I, auto const& e1){
    for_each_tuple(vectors,[&](auto const& J, auto const& e2){
      BOOST_CHECK(compare_extents(e1,e2) == (e1 == e2));
    });
  });

  for_each_tuple(scalars,[&](auto const& I, auto const& e1){
    for_each_tuple(matrices,[&](auto const& J, auto const& e2){
      BOOST_CHECK(compare_extents(e1,e2) == (e1 == e2));
    });
  });

  for_each_tuple(scalars,[&](auto const& I, auto const& e1){
    for_each_tuple(tensors,[&](auto const& J, auto const& e2){
      BOOST_CHECK(compare_extents(e1,e2) == (e1 == e2));
    });
  });

}

BOOST_FIXTURE_TEST_CASE(test_static_extents_squeeze, fixture, *boost::unit_test::label("static_extents") *boost::unit_test::label("squeeze"))
{
  using extents_type = boost::numeric::ublas::basic_extents<unsigned>;

  auto e_sq2  = squeeze(e2 )  ;//==> {2,3}
  auto e_sq3  = squeeze(e3 )  ;//==> {4,2,3}
  auto e_sq4  = squeeze(e4 )  ;//==> {4,2,3}
  auto e_sq5  = squeeze(e5 )  ;//==> {4,2,3}
  auto e_sq6  = squeeze(e6 )  ;//==> {2, 3}
  auto e_sq7  = squeeze(e7 )  ;//==> {4,2,3}
  auto e_sq8  = squeeze(e8 )  ;//==> {1,1}
  auto e_sq9  = squeeze(e9 )  ;//==> {1,2}
  auto e_sq10 = squeeze(e10) ; //==> {2,1}
  auto e_sq11 = squeeze(e11) ; //==> {2,3}
  auto e_sq12 = squeeze(e12) ; //==> {2,3}
  auto e_sq13 = squeeze(e13) ; //==> {2,3}

	BOOST_CHECK( (e_sq2  == extents_type{2,3}) );
	BOOST_CHECK( (e_sq3  == extents_type{4,2,3}) );

	BOOST_CHECK( (e_sq4  == extents_type{4,2,3}) );
	BOOST_CHECK( (e_sq5  == extents_type{4,2,3}) );
	BOOST_CHECK( (e_sq6  == extents_type{2,3}) );
	BOOST_CHECK( (e_sq7  == extents_type{4,2,3}) );
	BOOST_CHECK( (e_sq8  == extents_type{1,1}) );

	BOOST_CHECK( (e_sq9  == extents_type{1,2}) );
	BOOST_CHECK( (e_sq10 == extents_type{2,1}) );
	BOOST_CHECK( (e_sq11 == extents_type{2,3}) );
	BOOST_CHECK( (e_sq12 == extents_type{2,3}) );
	BOOST_CHECK( (e_sq13 == extents_type{2,3}) );


}

BOOST_AUTO_TEST_CASE(test_static_extents_exception)
{
  using namespace boost::numeric::ublas;
  
  basic_static_extents<size_t,3,1,2,3> e1;
  for(auto i = e1.size(); i < 100; i++){
    BOOST_REQUIRE_THROW((void)e1.at(i),std::out_of_range);
  }
  BOOST_REQUIRE_THROW((void)e1.at(std::numeric_limits<size_t>::max()),std::out_of_range);
  
  for(auto i = 0; i < 10; i++){
    std::vector<size_t> v;
    for(auto j = 1; j < (i * 2) + 10; j++){
      v.push_back(j);
    }
    auto b = v.begin();
    auto e = v.end();
    BOOST_REQUIRE_THROW((basic_static_extents<size_t,3>(b,e)),std::runtime_error);
  }

}

BOOST_AUTO_TEST_SUITE_END()
