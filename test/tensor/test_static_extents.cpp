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

#include <boost/numeric/ublas/tensor/static_extents.hpp>
#include <boost/numeric/ublas/tensor/extents.hpp>
#include <boost/test/unit_test.hpp>
#include <vector>
#include "utility.hpp"

BOOST_AUTO_TEST_SUITE(test_static_extents)

template <size_t... E>
using extents = boost::numeric::ublas::basic_static_extents<unsigned,E...>;

BOOST_AUTO_TEST_CASE(test_static_extents_ctor) {
  using namespace boost::numeric;

  auto e0 = extents<>{};
  BOOST_CHECK(e0.empty());
  BOOST_CHECK_EQUAL(e0.size(), 0);

  auto e1 = extents<1, 2>{};
  BOOST_CHECK(!e1.empty());
  BOOST_CHECK_EQUAL(e1.size(), 2);

  auto e2 = extents<2, 3>{};
  BOOST_CHECK(!e2.empty());
  BOOST_CHECK_EQUAL(e2.size(), 2);

  auto e3 = extents<4, 2, 3>{}; // 7
  BOOST_CHECK(!e3.empty());
  BOOST_CHECK_EQUAL(e3.size(), 3);
}

struct fixture {
  fixture() = default;
  extents<> e0{};                            // 0
  extents<1, 2, 3, 4> e1{};                  // 1
  extents<1, 2, 3> e2{};                     // 2
  extents<4, 2, 3> e3{};                     // 3
  extents<4, 2, 1, 3> e4{};                  // 4
  extents<1, 4, 2, 1, 3, 1> e5{};            // 5

  std::tuple<
    extents<>
  > rank_0_extents;

  std::tuple<
    extents<1>,
    extents<2>
  > rank_1_extents;

  std::tuple<
    extents<1,1>,
    extents<2,2>
  > rank_2_extents;

  std::tuple<
    extents<1>,
    extents<1,1>,
    extents<1,1,1>,
    extents<1,1,1,1>
  > scalars;

  std::tuple<
    extents<1,2>,
    extents<1,3,1>,
    extents<1,4,1,1>,
    extents<5,1,1,1,1>,
    extents<6,1,1,1,1,1>
  > vectors;

  std::tuple<
    extents<2,3>,
    extents<3,2,1>,
    extents<4,4,1,1>,
    extents<6,6,1,1,1,1>
  > matrices;

  std::tuple<
    extents<3,3,3>,
    extents<4,4,4,1>,
    extents<5,5,5,1,1>,
    extents<6,6,6,1,1,1>,
    extents<6,6,1,1,1,6>
  > tensors;
};

BOOST_FIXTURE_TEST_CASE(test_static_extents_product, fixture,
                        *boost::unit_test::label("static_extents") *
                            boost::unit_test::label("product")) {
  
  using namespace boost::numeric::ublas;

  auto p0   = product( e0);   // {}
  auto p1   = product( e1);   // {1,2,3,4}
  auto p2   = product( e2);   // {1,2,3}
  auto p3   = product( e3);   // {4,2,3}
  auto p4   = product( e4);   // {4,2,1,3}
  auto p5   = product( e5);   // {1,4,2,1,3,1}
  
  auto sp0   = static_product_v< std::decay_t<decltype(e0)> >;   // {}
  auto sp1   = static_product_v< std::decay_t<decltype(e1)> >;   // {1,2,3,4}
  auto sp2   = static_product_v< std::decay_t<decltype(e2)> >;   // {1,2,3}
  auto sp3   = static_product_v< std::decay_t<decltype(e3)> >;   // {4,2,3}
  auto sp4   = static_product_v< std::decay_t<decltype(e4)> >;   // {4,2,1,3}
  auto sp5   = static_product_v< std::decay_t<decltype(e5)> >;   // {1,4,2,1,3,1}

  BOOST_CHECK_EQUAL(p0, 0);
  BOOST_CHECK_EQUAL(p1, 24);
  BOOST_CHECK_EQUAL(p2, 6);
  BOOST_CHECK_EQUAL(p3, 24);
  BOOST_CHECK_EQUAL(p4, 24);
  BOOST_CHECK_EQUAL(p5, 24);
  
  BOOST_CHECK_EQUAL(sp0, 0);
  BOOST_CHECK_EQUAL(sp1, 24);
  BOOST_CHECK_EQUAL(sp2, 6);
  BOOST_CHECK_EQUAL(sp3, 24);
  BOOST_CHECK_EQUAL(sp4, 24);
  BOOST_CHECK_EQUAL(sp5, 24);
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

  BOOST_CHECK_EQUAL(e1[0], 1);
  BOOST_CHECK_EQUAL(e1[1], 2);
  BOOST_CHECK_EQUAL(e1[2], 3);
  BOOST_CHECK_EQUAL(e1[3], 4);

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
}

BOOST_FIXTURE_TEST_CASE(test_static_extents, fixture,
                        *boost::unit_test::label("static_extents") *
                            boost::unit_test::label("query")) {

  using namespace boost::numeric::ublas;
  // e0  ==> {}
  // e1  ==> {0,0,0,0}
  // e2  ==> {1,2,3}
  // e3  ==> {4,2,3}
  // e4  ==> {4,2,1,3}
  // e5  ==> {1,4,2,1,3,1}

  BOOST_CHECK(   e0.empty(   ));
  BOOST_CHECK( !is_scalar( e0));
  BOOST_CHECK( !is_vector( e0));
  BOOST_CHECK( !is_matrix( e0));
  BOOST_CHECK( !is_tensor( e0));
  BOOST_CHECK( !static_traits::is_scalar_v< std::decay_t<decltype(e0)> >);
  BOOST_CHECK( !static_traits::is_vector_v< std::decay_t<decltype(e0)> >);
  BOOST_CHECK( !static_traits::is_matrix_v< std::decay_t<decltype(e0)> >);
  BOOST_CHECK( !static_traits::is_tensor_v< std::decay_t<decltype(e0)> >);

  BOOST_CHECK( ! e1.empty(   ) );
  BOOST_CHECK( !is_scalar( e1) );
  BOOST_CHECK( !is_vector( e1) );
  BOOST_CHECK( !is_matrix( e1) );
  BOOST_CHECK(  is_tensor( e1) );
  BOOST_CHECK( !static_traits::is_scalar_v< std::decay_t<decltype(e1)> >);
  BOOST_CHECK( !static_traits::is_vector_v< std::decay_t<decltype(e1)> >);
  BOOST_CHECK( !static_traits::is_matrix_v< std::decay_t<decltype(e1)> >);
  BOOST_CHECK(  static_traits::is_tensor_v< std::decay_t<decltype(e1)> >);

  BOOST_CHECK( ! e2.empty(   ) );
  BOOST_CHECK( !is_scalar( e2) );
  BOOST_CHECK( !is_vector( e2) );
  BOOST_CHECK( !is_matrix( e2) );
  BOOST_CHECK(  is_tensor( e2) );
  BOOST_CHECK( !static_traits::is_scalar_v< std::decay_t<decltype(e2)> >);
  BOOST_CHECK( !static_traits::is_vector_v< std::decay_t<decltype(e2)> >);
  BOOST_CHECK( !static_traits::is_matrix_v< std::decay_t<decltype(e2)> >);
  BOOST_CHECK(  static_traits::is_tensor_v< std::decay_t<decltype(e2)> >);

  BOOST_CHECK( ! e3.empty(   ) );
  BOOST_CHECK( !is_scalar( e3) );
  BOOST_CHECK( !is_vector( e3) );
  BOOST_CHECK( !is_matrix( e3) );
  BOOST_CHECK(  is_tensor( e3) );
  BOOST_CHECK( !static_traits::is_scalar_v< std::decay_t<decltype(e3)> >);
  BOOST_CHECK( !static_traits::is_vector_v< std::decay_t<decltype(e3)> >);
  BOOST_CHECK( !static_traits::is_matrix_v< std::decay_t<decltype(e3)> >);
  BOOST_CHECK(  static_traits::is_tensor_v< std::decay_t<decltype(e3)> >);

  BOOST_CHECK( ! e4.empty(   ) );
  BOOST_CHECK( !is_scalar( e4) );
  BOOST_CHECK( !is_vector( e4) );
  BOOST_CHECK( !is_matrix( e4) );
  BOOST_CHECK(  is_tensor( e4) );
  BOOST_CHECK( !static_traits::is_scalar_v< std::decay_t<decltype(e4)> >);
  BOOST_CHECK( !static_traits::is_vector_v< std::decay_t<decltype(e4)> >);
  BOOST_CHECK( !static_traits::is_matrix_v< std::decay_t<decltype(e4)> >);
  BOOST_CHECK(  static_traits::is_tensor_v< std::decay_t<decltype(e4)> >);

  BOOST_CHECK( ! e5.empty(   ) );
  BOOST_CHECK( !is_scalar( e5) );
  BOOST_CHECK( !is_vector( e5) );
  BOOST_CHECK( !is_matrix( e5) );
  BOOST_CHECK(  is_tensor( e5) );
  BOOST_CHECK( !static_traits::is_scalar_v< std::decay_t<decltype(e5)> >);
  BOOST_CHECK( !static_traits::is_vector_v< std::decay_t<decltype(e5)> >);
  BOOST_CHECK( !static_traits::is_matrix_v< std::decay_t<decltype(e5)> >);
  BOOST_CHECK(  static_traits::is_tensor_v< std::decay_t<decltype(e5)> >);

  boost::numeric::ublas::basic_static_extents<size_t,1,3> e14;
  BOOST_CHECK( ! e14.empty(   ) );
  BOOST_CHECK( ! is_scalar(e14) );
  BOOST_CHECK(   is_vector(e14) );
  BOOST_CHECK( ! is_matrix(e14) );
  BOOST_CHECK( ! is_tensor(e14) );
  BOOST_CHECK( !static_traits::is_scalar_v< std::decay_t<decltype(e14)> >);
  BOOST_CHECK(  static_traits::is_vector_v< std::decay_t<decltype(e14)> >);
  BOOST_CHECK( !static_traits::is_matrix_v< std::decay_t<decltype(e14)> >);
  BOOST_CHECK( !static_traits::is_tensor_v< std::decay_t<decltype(e14)> >);


  for_each_tuple(rank_0_extents,[](auto const&, auto& e){
    BOOST_CHECK( !is_scalar(e) );
    BOOST_CHECK( !is_vector(e) );
    BOOST_CHECK( !is_matrix(e) );
    BOOST_CHECK( !is_tensor(e) );
    BOOST_CHECK( !static_traits::is_scalar_v< std::decay_t<decltype(e)> >);
    BOOST_CHECK( !static_traits::is_vector_v< std::decay_t<decltype(e)> >);
    BOOST_CHECK( !static_traits::is_matrix_v< std::decay_t<decltype(e)> >);
    BOOST_CHECK( !static_traits::is_tensor_v< std::decay_t<decltype(e)> >);
  });


  for_each_tuple(rank_1_extents,[](auto const& I, auto const& e){
    if( I == 0 ){
      BOOST_CHECK(  is_scalar(e) );
      BOOST_CHECK( !is_vector(e) );
      BOOST_CHECK( !is_matrix(e) );
      BOOST_CHECK( !is_tensor(e) );
      BOOST_CHECK(  static_traits::is_scalar_v< std::decay_t<decltype(e)> >);
      BOOST_CHECK( !static_traits::is_vector_v< std::decay_t<decltype(e)> >);
      BOOST_CHECK( !static_traits::is_matrix_v< std::decay_t<decltype(e)> >);
      BOOST_CHECK( !static_traits::is_tensor_v< std::decay_t<decltype(e)> >);
    }else{
      BOOST_CHECK( !is_scalar(e) );
      BOOST_CHECK(  is_vector(e) );
      BOOST_CHECK( !is_matrix(e) );
      BOOST_CHECK( !is_tensor(e) );
      BOOST_CHECK( !static_traits::is_scalar_v< std::decay_t<decltype(e)> >);
      BOOST_CHECK(  static_traits::is_vector_v< std::decay_t<decltype(e)> >);
      BOOST_CHECK( !static_traits::is_matrix_v< std::decay_t<decltype(e)> >);
      BOOST_CHECK( !static_traits::is_tensor_v< std::decay_t<decltype(e)> >);
    }
  });

  for_each_tuple(rank_2_extents,[](auto const& I, auto const& e){
    if( I == 0 ){
      BOOST_CHECK(  is_scalar(e) );
      BOOST_CHECK( !is_vector(e) );
      BOOST_CHECK( !is_matrix(e) );
      BOOST_CHECK( !is_tensor(e) );
      BOOST_CHECK(  static_traits::is_scalar_v< std::decay_t<decltype(e)> >);
      BOOST_CHECK( !static_traits::is_vector_v< std::decay_t<decltype(e)> >);
      BOOST_CHECK( !static_traits::is_matrix_v< std::decay_t<decltype(e)> >);
      BOOST_CHECK( !static_traits::is_tensor_v< std::decay_t<decltype(e)> >);
    }else{
      BOOST_CHECK( !is_scalar(e) );
      BOOST_CHECK( !is_vector(e) );
      BOOST_CHECK(  is_matrix(e) );
      BOOST_CHECK( !is_tensor(e) );
      BOOST_CHECK( !static_traits::is_scalar_v< std::decay_t<decltype(e)> >);
      BOOST_CHECK( !static_traits::is_vector_v< std::decay_t<decltype(e)> >);
      BOOST_CHECK(  static_traits::is_matrix_v< std::decay_t<decltype(e)> >);
      BOOST_CHECK( !static_traits::is_tensor_v< std::decay_t<decltype(e)> >);
    }
  });

  for_each_tuple(scalars,[](auto const&, auto& e){
      BOOST_CHECK(  is_scalar(e) );
      BOOST_CHECK( !is_vector(e) );
      BOOST_CHECK( !is_matrix(e) );
      BOOST_CHECK( !is_tensor(e) );
      BOOST_CHECK(  static_traits::is_scalar_v< std::decay_t<decltype(e)> >);
      BOOST_CHECK( !static_traits::is_vector_v< std::decay_t<decltype(e)> >);
      BOOST_CHECK( !static_traits::is_matrix_v< std::decay_t<decltype(e)> >);
      BOOST_CHECK( !static_traits::is_tensor_v< std::decay_t<decltype(e)> >);
  });

  for_each_tuple(vectors,[](auto const&, auto& e){
      BOOST_CHECK( !is_scalar(e) );
      BOOST_CHECK(  is_vector(e) );
      BOOST_CHECK( !is_matrix(e) );
      BOOST_CHECK( !is_tensor(e) );
      BOOST_CHECK( !static_traits::is_scalar_v< std::decay_t<decltype(e)> >);
      BOOST_CHECK(  static_traits::is_vector_v< std::decay_t<decltype(e)> >);
      BOOST_CHECK( !static_traits::is_matrix_v< std::decay_t<decltype(e)> >);
      BOOST_CHECK( !static_traits::is_tensor_v< std::decay_t<decltype(e)> >);
  });

  for_each_tuple(matrices,[](auto const&, auto& e){
      BOOST_CHECK( !is_scalar(e) );
      BOOST_CHECK( !is_vector(e) );
      BOOST_CHECK(  is_matrix(e) );
      BOOST_CHECK( !is_tensor(e) );
      BOOST_CHECK( !static_traits::is_scalar_v< std::decay_t<decltype(e)> >);
      BOOST_CHECK( !static_traits::is_vector_v< std::decay_t<decltype(e)> >);
      BOOST_CHECK(  static_traits::is_matrix_v< std::decay_t<decltype(e)> >);
      BOOST_CHECK( !static_traits::is_tensor_v< std::decay_t<decltype(e)> >);
      
  });

  for_each_tuple(tensors,[](auto const&, auto& e){
      BOOST_CHECK( !is_scalar(e) );
      BOOST_CHECK( !is_vector(e) );
      BOOST_CHECK( !is_matrix(e) );
      BOOST_CHECK(  is_tensor(e) );
      BOOST_CHECK( !static_traits::is_scalar_v< std::decay_t<decltype(e)> >);
      BOOST_CHECK( !static_traits::is_vector_v< std::decay_t<decltype(e)> >);
      BOOST_CHECK( !static_traits::is_matrix_v< std::decay_t<decltype(e)> >);
      BOOST_CHECK(  static_traits::is_tensor_v< std::decay_t<decltype(e)> >);
  });

}

BOOST_FIXTURE_TEST_CASE(test_static_extents_to_functions, fixture, *boost::unit_test::label("static_extents") *boost::unit_test::label("to_functions"))
{

  for_each_tuple(scalars,[](auto const&, auto& e){
    if (e.size() > 1){
      auto d = e.to_dynamic_extents();
      BOOST_CHECK(d == e);
    }
  });

  for_each_tuple(vectors,[](auto const&, auto& e){
    auto d = e.to_dynamic_extents();
    BOOST_CHECK(d == e);
  });


  for_each_tuple(matrices,[](auto const&, auto& e){
    auto d = e.to_dynamic_extents();
    BOOST_CHECK(d == e);
  });

  for_each_tuple(tensors,[](auto const&, auto& e){
    auto d = e.to_dynamic_extents();
    BOOST_CHECK(d == e);
  });

}

BOOST_FIXTURE_TEST_CASE(test_static_extents_valid, fixture, *boost::unit_test::label("static_extents") *boost::unit_test::label("valid"))
{
  using namespace boost::numeric::ublas;
  for_each_tuple(rank_0_extents,[](auto const&, auto& e){
    BOOST_CHECK(!valid(e));
    BOOST_CHECK(!static_traits::is_valid_v< std::decay_t<decltype(e)> >);
  });

  for_each_tuple(rank_1_extents,[](auto const& I, auto const& e){
    if( I== 0 ){
      BOOST_CHECK(valid(e));
      BOOST_CHECK(static_traits::is_valid_v< std::decay_t<decltype(e)> >);
    }else{
      BOOST_CHECK(!valid(e));
      BOOST_CHECK(!static_traits::is_valid_v< std::decay_t<decltype(e)> >);
    }
  });

  for_each_tuple(rank_2_extents,[](auto const&, auto& e){
      BOOST_CHECK(valid(e));
      BOOST_CHECK(static_traits::is_valid_v< std::decay_t<decltype(e)> >);
  });
  
  for_each_tuple(scalars,[](auto const&, auto& e){
      BOOST_CHECK(valid(e));
      BOOST_CHECK(static_traits::is_valid_v< std::decay_t<decltype(e)> >);
  });
  
  for_each_tuple(vectors,[](auto const&, auto& e){
      BOOST_CHECK(valid(e));
      BOOST_CHECK(static_traits::is_valid_v< std::decay_t<decltype(e)> >);
  });
  
  for_each_tuple(matrices,[](auto const&, auto& e){
      BOOST_CHECK(valid(e));
      BOOST_CHECK(static_traits::is_valid_v< std::decay_t<decltype(e)> >);
  });
  
  for_each_tuple(tensors,[](auto const&, auto& e){
      BOOST_CHECK(valid(e));
      BOOST_CHECK(static_traits::is_valid_v< std::decay_t<decltype(e)> >);
  });
}

BOOST_FIXTURE_TEST_CASE(test_static_extents_comparsion_operator, fixture, *boost::unit_test::label("static_extents") *boost::unit_test::label("comparsion_operator"))
{

  auto const compare_extents = [](auto const& e1, auto const& e2){
    if(e1.size() != e2.size()) return false;
    for(auto i = 0ul ; i < e1.size(); i++){
      if(e1[i] != e2[i]){
        return false;
      }
    }
    return true;
  };

  for_each_tuple(rank_0_extents,[&](auto const&, auto const& e1){
    for_each_tuple(rank_1_extents,[&](auto const&, auto const& e2){
      BOOST_CHECK(compare_extents(e1,e2) == (e1 == e2));
    });
  });

  for_each_tuple(rank_1_extents,[&](auto const&, auto const& e1){
    for_each_tuple(rank_1_extents,[&](auto const&, auto const& e2){
      BOOST_CHECK(compare_extents(e1,e2) == (e1 == e2));
    });
  });

  for_each_tuple(rank_1_extents,[&](auto const&, auto const& e1){
    for_each_tuple(rank_2_extents,[&](auto const&, auto const& e2){
      BOOST_CHECK(compare_extents(e1,e2) == (e1 == e2));
    });
  });

  for_each_tuple(scalars,[&](auto const&, auto const& e1){
    for_each_tuple(scalars,[&](auto const&, auto const& e2){
      BOOST_CHECK(compare_extents(e1,e2) == (e1 == e2));
    });
  });

  for_each_tuple(scalars,[&](auto const&, auto const& e1){
    for_each_tuple(vectors,[&](auto const&, auto const& e2){
      BOOST_CHECK(compare_extents(e1,e2) == (e1 == e2));
    });
  });

  for_each_tuple(scalars,[&](auto const&, auto const& e1){
    for_each_tuple(matrices,[&](auto const&, auto const& e2){
      BOOST_CHECK(compare_extents(e1,e2) == (e1 == e2));
    });
  });

  for_each_tuple(scalars,[&](auto const&, auto const& e1){
    for_each_tuple(tensors,[&](auto const&, auto const& e2){
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

	BOOST_CHECK( (e_sq2  == extents_type{2,3}) );
	BOOST_CHECK( (e_sq3  == extents_type{4,2,3}) );

	BOOST_CHECK( (e_sq4  == extents_type{4,2,3}) );
	BOOST_CHECK( (e_sq5  == extents_type{4,2,3}) );

}

BOOST_AUTO_TEST_CASE(test_static_extents_exception)
{
  using namespace boost::numeric::ublas;
  
  basic_static_extents<size_t,3,1,2,3> e1;
  for(auto i = e1.size(); i < 3; i++){
    BOOST_REQUIRE_THROW( (void)e1.at(i),std::out_of_range );
  }
  
  BOOST_REQUIRE_THROW((void)e1.at(std::numeric_limits<size_t>::max()),std::out_of_range);

}

BOOST_AUTO_TEST_SUITE_END()
