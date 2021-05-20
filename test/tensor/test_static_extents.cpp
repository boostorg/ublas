//
// 	Copyright (c) 2018, Cem Bassoy, cem.bassoy@gmail.com
// 	Copyright (c) 2019, Amit Singh, amitsingh19975@gmail.com
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
#include <vector>
#include "utility.hpp"

BOOST_AUTO_TEST_SUITE(test_static_extents)


struct fixture
{
  template<std::size_t ... e>
  using extents = boost::numeric::ublas::extents<e...>;

  extents<>                 e0      {};
  extents<1>                e1      {};
  extents<1, 1>             e11     {};
  extents<2, 1>             e21     {};
  extents<1, 2>             e12     {};
  extents<2, 3>             e23     {};
  extents<2, 1, 1>          e211    {};
  extents<2, 3, 1>          e231    {};
  extents<1, 2, 3>          e123    {};
  extents<4, 2, 3>          e423    {};
  extents<1, 2, 3, 4>       e1234   {};
  extents<4, 2, 1, 3>       e4213   {};
  extents<1, 2, 3, 4, 1>    e12341  {};
  extents<4, 2, 1, 3, 1>    e42131  {};
  extents<1, 4, 2, 1, 3, 1> e142131 {};
};

BOOST_FIXTURE_TEST_CASE(test_extents_static_ctor, fixture,
                        *boost::unit_test::label("extents_static") *boost::unit_test::label("ctor"))
{

  namespace ublas = boost::numeric::ublas;

  BOOST_CHECK(  ublas::empty(     e0));
  BOOST_CHECK(! ublas::empty(     e1));
  BOOST_CHECK(! ublas::empty(    e11));
  BOOST_CHECK(! ublas::empty(    e12));
  BOOST_CHECK(! ublas::empty(    e21));
  BOOST_CHECK(! ublas::empty(    e23));
  BOOST_CHECK(! ublas::empty(   e211));
  BOOST_CHECK(! ublas::empty(   e123));
  BOOST_CHECK(! ublas::empty(   e423));
  BOOST_CHECK(! ublas::empty(  e1234));
  BOOST_CHECK(! ublas::empty(  e4213));
  BOOST_CHECK(! ublas::empty(e142131));

  BOOST_CHECK_EQUAL( ublas::size(     e0),0);
  BOOST_CHECK_EQUAL( ublas::size(     e1),1);
  BOOST_CHECK_EQUAL( ublas::size(    e11),2);
  BOOST_CHECK_EQUAL( ublas::size(    e12),2);
  BOOST_CHECK_EQUAL( ublas::size(    e21),2);
  BOOST_CHECK_EQUAL( ublas::size(    e23),2);
  BOOST_CHECK_EQUAL( ublas::size(   e211),3);
  BOOST_CHECK_EQUAL( ublas::size(   e123),3);
  BOOST_CHECK_EQUAL( ublas::size(   e423),3);
  BOOST_CHECK_EQUAL( ublas::size(  e1234),4);
  BOOST_CHECK_EQUAL( ublas::size(  e4213),4);
  BOOST_CHECK_EQUAL( ublas::size(e142131),6);


  BOOST_CHECK_EQUAL( ublas::size_v<decltype(     e0)>,0);
  BOOST_CHECK_EQUAL( ublas::size_v<decltype(     e1)>,1);
  BOOST_CHECK_EQUAL( ublas::size_v<decltype(    e11)>,2);
  BOOST_CHECK_EQUAL( ublas::size_v<decltype(    e12)>,2);
  BOOST_CHECK_EQUAL( ublas::size_v<decltype(    e21)>,2);
  BOOST_CHECK_EQUAL( ublas::size_v<decltype(    e23)>,2);
  BOOST_CHECK_EQUAL( ublas::size_v<decltype(   e211)>,3);
  BOOST_CHECK_EQUAL( ublas::size_v<decltype(   e123)>,3);
  BOOST_CHECK_EQUAL( ublas::size_v<decltype(   e423)>,3);
  BOOST_CHECK_EQUAL( ublas::size_v<decltype(  e1234)>,4);
  BOOST_CHECK_EQUAL( ublas::size_v<decltype(  e4213)>,4);
  BOOST_CHECK_EQUAL( ublas::size_v<decltype(e142131)>,6);

}

BOOST_FIXTURE_TEST_CASE(test_extents_static_product, fixture,
                        *boost::unit_test::label("extents_static") *boost::unit_test::label("product"))
{
  
  namespace ublas = boost::numeric::ublas;

  BOOST_CHECK_EQUAL(ublas::product(     e0),  0);
  //FIXME:  BOOST_CHECK_EQUAL(ublas::product(     e1),  1);
  BOOST_CHECK_EQUAL(ublas::product(    e11),  1);
  BOOST_CHECK_EQUAL(ublas::product(    e12),  2);
  BOOST_CHECK_EQUAL(ublas::product(    e21),  2);
  BOOST_CHECK_EQUAL(ublas::product(    e23),  6);
  BOOST_CHECK_EQUAL(ublas::product(   e211),  2);
  BOOST_CHECK_EQUAL(ublas::product(   e123),  6);
  BOOST_CHECK_EQUAL(ublas::product(   e423), 24);
  BOOST_CHECK_EQUAL(ublas::product(  e1234), 24);
  BOOST_CHECK_EQUAL(ublas::product(  e4213), 24);
  BOOST_CHECK_EQUAL(ublas::product(e142131), 24);


  BOOST_CHECK_EQUAL(ublas::product_v<decltype(     e0)>,  0);
  BOOST_CHECK_EQUAL(ublas::product_v<decltype(     e1)>,  1);
  BOOST_CHECK_EQUAL(ublas::product_v<decltype(    e11)>,  1);
  BOOST_CHECK_EQUAL(ublas::product_v<decltype(    e12)>,  2);
  BOOST_CHECK_EQUAL(ublas::product_v<decltype(    e21)>,  2);
  BOOST_CHECK_EQUAL(ublas::product_v<decltype(    e23)>,  6);
  BOOST_CHECK_EQUAL(ublas::product_v<decltype(   e211)>,  2);
  BOOST_CHECK_EQUAL(ublas::product_v<decltype(   e123)>,  6);
  BOOST_CHECK_EQUAL(ublas::product_v<decltype(   e423)>, 24);
  BOOST_CHECK_EQUAL(ublas::product_v<decltype(  e1234)>, 24);
  BOOST_CHECK_EQUAL(ublas::product_v<decltype(  e4213)>, 24);
  BOOST_CHECK_EQUAL(ublas::product_v<decltype(e142131)>, 24);
}

BOOST_FIXTURE_TEST_CASE(test_static_extents_access, fixture,
                        *boost::unit_test::label("extents_static") *boost::unit_test::label("access"))
{
  namespace ublas = boost::numeric::ublas;

  BOOST_REQUIRE_EQUAL( ublas::size_v<decltype(     e0)>,0);
  BOOST_REQUIRE_EQUAL( ublas::size_v<decltype(     e1)>,1);
  BOOST_REQUIRE_EQUAL( ublas::size_v<decltype(    e11)>,2);
  BOOST_REQUIRE_EQUAL( ublas::size_v<decltype(    e12)>,2);
  BOOST_REQUIRE_EQUAL( ublas::size_v<decltype(    e21)>,2);
  BOOST_REQUIRE_EQUAL( ublas::size_v<decltype(    e23)>,2);
  BOOST_REQUIRE_EQUAL( ublas::size_v<decltype(   e211)>,3);
  BOOST_REQUIRE_EQUAL( ublas::size_v<decltype(   e123)>,3);
  BOOST_REQUIRE_EQUAL( ublas::size_v<decltype(   e423)>,3);
  BOOST_REQUIRE_EQUAL( ublas::size_v<decltype(  e1234)>,4);
  BOOST_REQUIRE_EQUAL( ublas::size_v<decltype(  e4213)>,4);
  BOOST_REQUIRE_EQUAL( ublas::size_v<decltype(e142131)>,6);


  BOOST_CHECK_EQUAL((ublas::get_v<decltype(e1),0>), 1);

  BOOST_CHECK_EQUAL((ublas::get_v<decltype(e11),0>), 1);
  BOOST_CHECK_EQUAL((ublas::get_v<decltype(e11),1>), 1);

  BOOST_CHECK_EQUAL((ublas::get_v<decltype(e12),0>), 1);
  BOOST_CHECK_EQUAL((ublas::get_v<decltype(e12),1>), 2);

  BOOST_CHECK_EQUAL((ublas::get_v<decltype(e21),0>), 2);
  BOOST_CHECK_EQUAL((ublas::get_v<decltype(e21),1>), 1);

  BOOST_CHECK_EQUAL((ublas::get_v<decltype(e23),0>), 2);
  BOOST_CHECK_EQUAL((ublas::get_v<decltype(e23),1>), 3);

  BOOST_CHECK_EQUAL((ublas::get_v<decltype(e211),0>), 2);
  BOOST_CHECK_EQUAL((ublas::get_v<decltype(e211),1>), 1);
  BOOST_CHECK_EQUAL((ublas::get_v<decltype(e211),2>), 1);

  BOOST_CHECK_EQUAL((ublas::get_v<decltype(e123),0>), 1);
  BOOST_CHECK_EQUAL((ublas::get_v<decltype(e123),1>), 2);
  BOOST_CHECK_EQUAL((ublas::get_v<decltype(e123),2>), 3);

  BOOST_CHECK_EQUAL((ublas::get_v<decltype(e423),0>), 4);
  BOOST_CHECK_EQUAL((ublas::get_v<decltype(e423),1>), 2);
  BOOST_CHECK_EQUAL((ublas::get_v<decltype(e423),2>), 3);

  BOOST_CHECK_EQUAL((ublas::get_v<decltype(e1234),0>), 1);
  BOOST_CHECK_EQUAL((ublas::get_v<decltype(e1234),1>), 2);
  BOOST_CHECK_EQUAL((ublas::get_v<decltype(e1234),2>), 3);
  BOOST_CHECK_EQUAL((ublas::get_v<decltype(e1234),3>), 4);

  BOOST_CHECK_EQUAL((ublas::get_v<decltype(e4213),0>), 4);
  BOOST_CHECK_EQUAL((ublas::get_v<decltype(e4213),1>), 2);
  BOOST_CHECK_EQUAL((ublas::get_v<decltype(e4213),2>), 1);
  BOOST_CHECK_EQUAL((ublas::get_v<decltype(e4213),3>), 3);

  //FIXME:  BOOST_CHECK_EQUAL(e1 [0], 1);

  BOOST_CHECK_EQUAL(e11[0], 1);
  BOOST_CHECK_EQUAL(e11[1], 1);

  BOOST_CHECK_EQUAL(e12[0], 1);
  BOOST_CHECK_EQUAL(e12[1], 2);

  BOOST_CHECK_EQUAL(e21[0], 2);
  BOOST_CHECK_EQUAL(e21[1], 1);

  BOOST_CHECK_EQUAL(e23[0], 2);
  BOOST_CHECK_EQUAL(e23[1], 3);

  BOOST_CHECK_EQUAL(e211[0], 2);
  BOOST_CHECK_EQUAL(e211[1], 1);
  BOOST_CHECK_EQUAL(e211[2], 1);

  BOOST_CHECK_EQUAL(e123[0], 1);
  BOOST_CHECK_EQUAL(e123[1], 2);
  BOOST_CHECK_EQUAL(e123[2], 3);

  BOOST_CHECK_EQUAL(e423[0], 4);
  BOOST_CHECK_EQUAL(e423[1], 2);
  BOOST_CHECK_EQUAL(e423[2], 3);

  BOOST_CHECK_EQUAL(e1234[0], 1);
  BOOST_CHECK_EQUAL(e1234[1], 2);
  BOOST_CHECK_EQUAL(e1234[2], 3);
  BOOST_CHECK_EQUAL(e1234[3], 4);

  BOOST_CHECK_EQUAL(e4213[0], 4);
  BOOST_CHECK_EQUAL(e4213[1], 2);
  BOOST_CHECK_EQUAL(e4213[2], 1);
  BOOST_CHECK_EQUAL(e4213[3], 3);
}

struct fixture_second
{
  template<std::size_t ... e>
  using extents = boost::numeric::ublas::extents<e...>;

  std::tuple<
    extents<>
    > empty;

  std::tuple<
    //FIXME:    extents<1>,
    extents<1,1>,
    extents<1,1,1>,
    extents<1,1,1,1>
    > scalars;

  std::tuple<
    extents<1,2>,
    extents<2,1>,
    extents<1,2,1>,
    extents<2,1,1>,
    extents<1,4,1,1>,
    extents<5,1,1,1,1>
    > vectors;

  std::tuple<
    extents<2,3>,
    extents<3,2,1>,
    extents<4,4,1,1>,
    extents<6,6,1,1,1,1>
    > matrices;

  std::tuple<
    extents<1,2,3>,
    extents<1,2,3>,
    extents<1,2,3,1>,
    extents<4,2,3>,
    extents<4,2,3,1>,
    extents<4,2,3,1,1>,
    extents<6,6,6,1,1,1>,
    extents<6,6,1,1,1,6>
    > tensors;
};


BOOST_FIXTURE_TEST_CASE(test_static_extents, fixture_second,
                        *boost::unit_test::label("extents_static") *boost::unit_test::label("is_scalar_vector_matrix_tensor")) {

  namespace ublas = boost::numeric::ublas;

  for_each_in_tuple(scalars,[](auto const& /*unused*/, auto const& e){
    BOOST_CHECK(  ublas::is_scalar(e) );
    BOOST_CHECK(  ublas::is_vector(e) );
    BOOST_CHECK(  ublas::is_matrix(e) );
    BOOST_CHECK( !ublas::is_tensor(e) );

    BOOST_CHECK(  ublas::is_scalar_v<decltype(e)>);
    BOOST_CHECK(  ublas::is_vector_v<decltype(e)>);
    BOOST_CHECK(  ublas::is_matrix_v<decltype(e)>);
    BOOST_CHECK( !ublas::is_tensor_v<decltype(e)>);

  });

  for_each_in_tuple(vectors,[](auto const& /*unused*/, auto& e){
    BOOST_CHECK( !ublas::is_scalar(e) );
    BOOST_CHECK(  ublas::is_vector(e) );
    BOOST_CHECK(  ublas::is_matrix(e) );
    BOOST_CHECK( !ublas::is_tensor(e) );

    BOOST_CHECK( !ublas::is_scalar_v<decltype(e)>);
    BOOST_CHECK(  ublas::is_vector_v<decltype(e)>);
    BOOST_CHECK(  ublas::is_matrix_v<decltype(e)>);
    BOOST_CHECK( !ublas::is_tensor_v<decltype(e)>);
  });

  for_each_in_tuple(matrices,[](auto const& /*unused*/, auto& e){
    BOOST_CHECK( !ublas::is_scalar(e) );
    BOOST_CHECK( !ublas::is_vector(e) );
    BOOST_CHECK(  ublas::is_matrix(e) );
    BOOST_CHECK( !ublas::is_tensor(e) );

    BOOST_CHECK( !ublas::is_scalar_v<decltype(e)>);
    BOOST_CHECK( !ublas::is_vector_v<decltype(e)>);
    BOOST_CHECK(  ublas::is_matrix_v<decltype(e)>);
    BOOST_CHECK( !ublas::is_tensor_v<decltype(e)>);
  });

  for_each_in_tuple(tensors,[](auto const& /*unused*/, auto& e){
    BOOST_CHECK( !ublas::is_scalar(e) );
    BOOST_CHECK( !ublas::is_vector(e) );
    BOOST_CHECK( !ublas::is_matrix(e) );
    BOOST_CHECK(  ublas::is_tensor(e) );

    BOOST_CHECK( !ublas::is_scalar_v<decltype(e)>);
    BOOST_CHECK( !ublas::is_vector_v<decltype(e)>);
    BOOST_CHECK( !ublas::is_matrix_v<decltype(e)>);
    BOOST_CHECK(  ublas::is_tensor_v<decltype(e)>);
  });

}

BOOST_FIXTURE_TEST_CASE(test_static_extents_valid, fixture_second,
                        *boost::unit_test::label("extents_extents") *boost::unit_test::label("valid"))
{
  namespace ublas = boost::numeric::ublas;

//FIXME:  BOOST_CHECK(!ublas::is_valid (extents<0>{}) );
//FIXME:  BOOST_CHECK( ublas::is_valid (extents<2>{}) );
//FIXME:  BOOST_CHECK( ublas::is_valid (extents<3>{}) );

  BOOST_CHECK(!ublas::is_valid_v<extents<0>> );
  BOOST_CHECK( ublas::is_valid_v<extents<2>> );
  BOOST_CHECK( ublas::is_valid_v<extents<3>> );


  for_each_in_tuple(scalars  ,[](auto const& /*unused*/, auto& e){ BOOST_CHECK(  ublas::is_valid (e) ); });
  for_each_in_tuple(vectors  ,[](auto const& /*unused*/, auto& e){ BOOST_CHECK(  ublas::is_valid (e) ); });
  for_each_in_tuple(matrices ,[](auto const& /*unused*/, auto& e){ BOOST_CHECK(  ublas::is_valid (e) ); });
  for_each_in_tuple(tensors  ,[](auto const& /*unused*/, auto& e){ BOOST_CHECK(  ublas::is_valid (e) ); });


  for_each_in_tuple(scalars  ,[](auto const& /*unused*/, auto& e){ BOOST_CHECK(  ublas::is_valid_v<decltype(e)> ); });
  for_each_in_tuple(vectors  ,[](auto const& /*unused*/, auto& e){ BOOST_CHECK(  ublas::is_valid_v<decltype(e)> ); });
  for_each_in_tuple(matrices ,[](auto const& /*unused*/, auto& e){ BOOST_CHECK(  ublas::is_valid_v<decltype(e)> ); });
  for_each_in_tuple(tensors  ,[](auto const& /*unused*/, auto& e){ BOOST_CHECK(  ublas::is_valid_v<decltype(e)> ); });
}


BOOST_FIXTURE_TEST_CASE(test_static_extents_comparsion_operator, fixture,
                        *boost::unit_test::label("extents_static") *boost::unit_test::label("equals"))
{
  namespace ublas = boost::numeric::ublas;

  BOOST_CHECK(      e0 == e0      );
  BOOST_CHECK(      e1 == e1      );
  BOOST_CHECK(     e11 == e11     );
  BOOST_CHECK(     e21 == e21     );
  BOOST_CHECK(     e12 == e12     );
  BOOST_CHECK(     e23 == e23     );
  BOOST_CHECK(    e231 == e231    );
  BOOST_CHECK(    e211 == e211    );
  BOOST_CHECK(    e123 == e123    );
  BOOST_CHECK(    e423 == e423    );
  BOOST_CHECK(   e1234 == e1234   );
  BOOST_CHECK(   e4213 == e4213   );
  BOOST_CHECK( e142131 == e142131 );

}

BOOST_AUTO_TEST_SUITE_END()
