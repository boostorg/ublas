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

#include <boost/test/unit_test.hpp>
#include <boost/numeric/ublas/tensor/extents.hpp>
#include <vector>

BOOST_AUTO_TEST_SUITE ( test_extents_static_size )


//*boost::unit_test::label("extents")
//*boost::unit_test::label("constructor")

BOOST_AUTO_TEST_CASE(test_extents_static_size_ctor)
{
  namespace ub = boost::numeric::ublas;


//  auto e = ub::extents<0>{};
  auto  e11 = ub::extents<2>{1,1};
  auto  e12 = ub::extents<2>{1,2};
  auto  e21 = ub::extents<2>{2,1};
  auto  e23 = ub::extents<2>{2,3};
  auto e231 = ub::extents<3>{2,3,1};
  auto e123 = ub::extents<3>{1,2,3}; // 6
  auto e423 = ub::extents<3>{4,2,3};  // 7


  BOOST_CHECK (!ub::empty(e11));
  BOOST_CHECK (!ub::empty(e12));
  BOOST_CHECK (!ub::empty(e21));
  BOOST_CHECK (!ub::empty(e23));
  BOOST_CHECK (!ub::empty(e231));
  BOOST_CHECK (!ub::empty(e123));
  BOOST_CHECK (!ub::empty(e423));

  BOOST_CHECK ( ub::size (e11) == 2);
  BOOST_CHECK ( ub::size (e12) == 2);
  BOOST_CHECK ( ub::size (e21) == 2);
  BOOST_CHECK ( ub::size (e23) == 2);
  BOOST_CHECK ( ub::size(e231) == 3);
  BOOST_CHECK ( ub::size(e123) == 3);
  BOOST_CHECK ( ub::size(e423) == 3);


  BOOST_CHECK_THROW( ub::extents<2>({1,0}), 	std::invalid_argument);
  BOOST_CHECK_THROW( ub::extents<1>({0}  ), 	std::invalid_argument);
  BOOST_CHECK_THROW( ub::extents<2>({0,1}), 	std::invalid_argument);
  BOOST_CHECK_THROW( ub::extents<2>({1,1,2}), std::length_error);
}


struct fixture {
  template<size_t N>
  using extents = boost::numeric::ublas::extents<N>;

//  extents<0> de       {};

  extents<2> de11     {1,1};
  extents<2> de12     {1,2};
  extents<2> de21     {2,1};

  extents<2> de23     {2,3};
  extents<3> de231    {2,3,1};
  extents<3> de123    {1,2,3};
  extents<4> de1123   {1,1,2,3};
  extents<5> de12311  {1,2,3,1,1};

  extents<3> de423    {4,2,3};
  extents<4> de4213   {4,2,1,3};
  extents<5> de42131  {4,2,1,3,1};
  extents<6> de142131 {1,4,2,1,3,1};

  extents<3> de141    {1,4,1};
  extents<4> de1111   {1,1,1,1};
  extents<5> de14111  {1,4,1,1,1};
  extents<6> de112111 {1,1,2,1,1,1};
  extents<6> de112311 {1,1,2,3,1,1};
};

BOOST_FIXTURE_TEST_CASE(test_extents_static_size_access, fixture, *boost::unit_test::label("basic_fixed_rank_extents") *boost::unit_test::label("access"))
{

  namespace ublas = boost::numeric::ublas;

//  BOOST_REQUIRE_EQUAL(ublas::size(de), 0);
//  BOOST_CHECK        (ublas::empty(de)  );

  BOOST_REQUIRE_EQUAL(ublas::size(de11)    , 2);
  BOOST_REQUIRE_EQUAL(ublas::size(de12)    , 2);
  BOOST_REQUIRE_EQUAL(ublas::size(de21)    , 2);
  BOOST_REQUIRE_EQUAL(ublas::size(de23)    , 2);
  BOOST_REQUIRE_EQUAL(ublas::size(de231)   , 3);
  BOOST_REQUIRE_EQUAL(ublas::size(de123)   , 3);
  BOOST_REQUIRE_EQUAL(ublas::size(de1123)  , 4);
  BOOST_REQUIRE_EQUAL(ublas::size(de12311) , 5);
  BOOST_REQUIRE_EQUAL(ublas::size(de423)   , 3);
  BOOST_REQUIRE_EQUAL(ublas::size(de4213)  , 4);
  BOOST_REQUIRE_EQUAL(ublas::size(de42131) , 5);
  BOOST_REQUIRE_EQUAL(ublas::size(de142131), 6);
  BOOST_REQUIRE_EQUAL(ublas::size(de141)   , 3);
  BOOST_REQUIRE_EQUAL(ublas::size(de1111)  , 4);
  BOOST_REQUIRE_EQUAL(ublas::size(de14111) , 5);
  BOOST_REQUIRE_EQUAL(ublas::size(de112111), 6);
  BOOST_REQUIRE_EQUAL(ublas::size(de112311), 6);


  BOOST_CHECK_EQUAL(de11[0],1);
  BOOST_CHECK_EQUAL(de11[1],1);

  BOOST_CHECK_EQUAL(de12[0],1);
  BOOST_CHECK_EQUAL(de12[1],2);

  BOOST_CHECK_EQUAL(de21[0],2);
  BOOST_CHECK_EQUAL(de21[1],1);

  BOOST_CHECK_EQUAL(de23[0],2);
  BOOST_CHECK_EQUAL(de23[1],3);

  BOOST_CHECK_EQUAL(de231[0],2);
  BOOST_CHECK_EQUAL(de231[1],3);
  BOOST_CHECK_EQUAL(de231[2],1);

  BOOST_CHECK_EQUAL(de123[0],1);
  BOOST_CHECK_EQUAL(de123[1],2);
  BOOST_CHECK_EQUAL(de123[2],3);

  BOOST_CHECK_EQUAL(de1123[0],1);
  BOOST_CHECK_EQUAL(de1123[1],1);
  BOOST_CHECK_EQUAL(de1123[2],2);
  BOOST_CHECK_EQUAL(de1123[3],3);

  BOOST_CHECK_EQUAL(de12311[0],1);
  BOOST_CHECK_EQUAL(de12311[1],2);
  BOOST_CHECK_EQUAL(de12311[2],3);
  BOOST_CHECK_EQUAL(de12311[3],1);
  BOOST_CHECK_EQUAL(de12311[4],1);

  BOOST_CHECK_EQUAL(de423[0],4);
  BOOST_CHECK_EQUAL(de423[1],2);
  BOOST_CHECK_EQUAL(de423[2],3);

  BOOST_CHECK_EQUAL(de4213[0],4);
  BOOST_CHECK_EQUAL(de4213[1],2);
  BOOST_CHECK_EQUAL(de4213[2],1);
  BOOST_CHECK_EQUAL(de4213[3],3);

  BOOST_CHECK_EQUAL(de42131[0],4);
  BOOST_CHECK_EQUAL(de42131[1],2);
  BOOST_CHECK_EQUAL(de42131[2],1);
  BOOST_CHECK_EQUAL(de42131[3],3);
  BOOST_CHECK_EQUAL(de42131[4],1);

  BOOST_CHECK_EQUAL(de142131[0],1);
  BOOST_CHECK_EQUAL(de142131[1],4);
  BOOST_CHECK_EQUAL(de142131[2],2);
  BOOST_CHECK_EQUAL(de142131[3],1);
  BOOST_CHECK_EQUAL(de142131[4],3);
  BOOST_CHECK_EQUAL(de142131[5],1);

  BOOST_CHECK_EQUAL(de141[0],1);
  BOOST_CHECK_EQUAL(de141[1],4);
  BOOST_CHECK_EQUAL(de141[2],1);

  BOOST_CHECK_EQUAL(de1111[0],1);
  BOOST_CHECK_EQUAL(de1111[1],1);
  BOOST_CHECK_EQUAL(de1111[2],1);
  BOOST_CHECK_EQUAL(de1111[3],1);

  BOOST_CHECK_EQUAL(de14111[0],1);
  BOOST_CHECK_EQUAL(de14111[1],4);
  BOOST_CHECK_EQUAL(de14111[2],1);
  BOOST_CHECK_EQUAL(de14111[3],1);
  BOOST_CHECK_EQUAL(de14111[4],1);

  BOOST_CHECK_EQUAL(de112111[0],1);
  BOOST_CHECK_EQUAL(de112111[1],1);
  BOOST_CHECK_EQUAL(de112111[2],2);
  BOOST_CHECK_EQUAL(de112111[3],1);
  BOOST_CHECK_EQUAL(de112111[4],1);
  BOOST_CHECK_EQUAL(de112111[5],1);

  BOOST_CHECK_EQUAL(de112311[0],1);
  BOOST_CHECK_EQUAL(de112311[1],1);
  BOOST_CHECK_EQUAL(de112311[2],2);
  BOOST_CHECK_EQUAL(de112311[3],3);
  BOOST_CHECK_EQUAL(de112311[4],1);
  BOOST_CHECK_EQUAL(de112311[5],1);
}

BOOST_FIXTURE_TEST_CASE(test_extents_static_size_copy_ctor, fixture, *boost::unit_test::label("basic_fixed_rank_extents") *boost::unit_test::label("copy_ctor"))
{
  namespace ublas = boost::numeric::ublas;

//  auto e       = de;
  auto e1      = de11;
  auto e12     = de12;
  auto e21     = de21;
  auto e23     = de23;
  auto e231    = de231;
  auto e123    = de123;
  auto e1123   = de1123;
  auto e12311  = de12311;
  auto e423    = de423;
  auto e4213   = de4213;
  auto e42131  = de42131;
  auto e142131 = de142131;
  auto e141    = de141;
  auto e1111   = de1111;
  auto e14111  = de14111;
  auto e112111 = de112111;
  auto e112311 = de112311;


//  BOOST_CHECK (ublas::empty(e)  );

//  BOOST_REQUIRE_EQUAL(ublas::size(e)      , 0);
  BOOST_REQUIRE_EQUAL(ublas::size(e1)     , 2);
  BOOST_REQUIRE_EQUAL(ublas::size(e12)    , 2);
  BOOST_REQUIRE_EQUAL(ublas::size(e21)    , 2);
  BOOST_REQUIRE_EQUAL(ublas::size(e23)    , 2);
  BOOST_REQUIRE_EQUAL(ublas::size(e231),    3);
  BOOST_REQUIRE_EQUAL(ublas::size(e123),    3);
  BOOST_REQUIRE_EQUAL(ublas::size(e1123),   4);
  BOOST_REQUIRE_EQUAL(ublas::size(e12311),  5);
  BOOST_REQUIRE_EQUAL(ublas::size(e423),    3);
  BOOST_REQUIRE_EQUAL(ublas::size(e4213),   4);
  BOOST_REQUIRE_EQUAL(ublas::size(e42131),  5);
  BOOST_REQUIRE_EQUAL(ublas::size(e142131), 6);
  BOOST_REQUIRE_EQUAL(ublas::size(e141),    3);
  BOOST_REQUIRE_EQUAL(ublas::size(e1111),   4);
  BOOST_REQUIRE_EQUAL(ublas::size(e14111),  5);
  BOOST_REQUIRE_EQUAL(ublas::size(e112111), 6);
  BOOST_REQUIRE_EQUAL(ublas::size(e112311), 6);


  BOOST_CHECK_EQUAL(e1[0],1);
  BOOST_CHECK_EQUAL(e1[1],1);

  BOOST_CHECK_EQUAL(e12[0],1);
  BOOST_CHECK_EQUAL(e12[1],2);

  BOOST_CHECK_EQUAL(e21[0],2);
  BOOST_CHECK_EQUAL(e21[1],1);

  BOOST_CHECK_EQUAL(e23[0],2);
  BOOST_CHECK_EQUAL(e23[1],3);

  BOOST_CHECK_EQUAL(e231[0],2);
  BOOST_CHECK_EQUAL(e231[1],3);
  BOOST_CHECK_EQUAL(e231[2],1);

  BOOST_CHECK_EQUAL(e123[0],1);
  BOOST_CHECK_EQUAL(e123[1],2);
  BOOST_CHECK_EQUAL(e123[2],3);

  BOOST_CHECK_EQUAL(e1123[0],1);
  BOOST_CHECK_EQUAL(e1123[1],1);
  BOOST_CHECK_EQUAL(e1123[2],2);
  BOOST_CHECK_EQUAL(e1123[3],3);

  BOOST_CHECK_EQUAL(e12311[0],1);
  BOOST_CHECK_EQUAL(e12311[1],2);
  BOOST_CHECK_EQUAL(e12311[2],3);
  BOOST_CHECK_EQUAL(e12311[3],1);
  BOOST_CHECK_EQUAL(e12311[4],1);

  BOOST_CHECK_EQUAL(e423[0],4);
  BOOST_CHECK_EQUAL(e423[1],2);
  BOOST_CHECK_EQUAL(e423[2],3);

  BOOST_CHECK_EQUAL(e4213[0],4);
  BOOST_CHECK_EQUAL(e4213[1],2);
  BOOST_CHECK_EQUAL(e4213[2],1);
  BOOST_CHECK_EQUAL(e4213[3],3);

  BOOST_CHECK_EQUAL(e42131[0],4);
  BOOST_CHECK_EQUAL(e42131[1],2);
  BOOST_CHECK_EQUAL(e42131[2],1);
  BOOST_CHECK_EQUAL(e42131[3],3);
  BOOST_CHECK_EQUAL(e42131[4],1);

  BOOST_CHECK_EQUAL(e142131[0],1);
  BOOST_CHECK_EQUAL(e142131[1],4);
  BOOST_CHECK_EQUAL(e142131[2],2);
  BOOST_CHECK_EQUAL(e142131[3],1);
  BOOST_CHECK_EQUAL(e142131[4],3);
  BOOST_CHECK_EQUAL(e142131[5],1);

  BOOST_CHECK_EQUAL(e141[0],1);
  BOOST_CHECK_EQUAL(e141[1],4);
  BOOST_CHECK_EQUAL(e141[2],1);

  BOOST_CHECK_EQUAL(e1111[0],1);
  BOOST_CHECK_EQUAL(e1111[1],1);
  BOOST_CHECK_EQUAL(e1111[2],1);
  BOOST_CHECK_EQUAL(e1111[3],1);

  BOOST_CHECK_EQUAL(e14111[0],1);
  BOOST_CHECK_EQUAL(e14111[1],4);
  BOOST_CHECK_EQUAL(e14111[2],1);
  BOOST_CHECK_EQUAL(e14111[3],1);
  BOOST_CHECK_EQUAL(e14111[4],1);

  BOOST_CHECK_EQUAL(e112111[0],1);
  BOOST_CHECK_EQUAL(e112111[1],1);
  BOOST_CHECK_EQUAL(e112111[2],2);
  BOOST_CHECK_EQUAL(e112111[3],1);
  BOOST_CHECK_EQUAL(e112111[4],1);
  BOOST_CHECK_EQUAL(e112111[5],1);

  BOOST_CHECK_EQUAL(e112311[0],1);
  BOOST_CHECK_EQUAL(e112311[1],1);
  BOOST_CHECK_EQUAL(e112311[2],2);
  BOOST_CHECK_EQUAL(e112311[3],3);
  BOOST_CHECK_EQUAL(e112311[4],1);
  BOOST_CHECK_EQUAL(e112311[5],1);

}

BOOST_FIXTURE_TEST_CASE(test_extents_static_size_is, fixture, *boost::unit_test::label("basic_fixed_rank_extents") *boost::unit_test::label("query"))
{
  namespace ublas = boost::numeric::ublas;


//  auto e       = de;
  auto e11     = de11;
  auto e12     = de12;
  auto e21     = de21;
  auto e23     = de23;
  auto e231    = de231;
  auto e123    = de123;
  auto e1123   = de1123;
  auto e12311  = de12311;
  auto e423    = de423;
  auto e4213   = de4213;
  auto e42131  = de42131;
  auto e142131 = de142131;
  auto e141    = de141;
  auto e1111   = de1111;
  auto e14111  = de14111;
  auto e112111 = de112111;
  auto e112311 = de112311;

//  BOOST_CHECK(   ublas::empty    (e));
//  BOOST_CHECK( ! ublas::is_scalar(e));
//  BOOST_CHECK( ! ublas::is_vector(e));
//  BOOST_CHECK( ! ublas::is_matrix(e));
//  BOOST_CHECK( ! ublas::is_tensor(e));

  BOOST_CHECK( ! ublas::empty    (e11) );
  BOOST_CHECK(   ublas::is_scalar(e11) );
  BOOST_CHECK(   ublas::is_vector(e11) );
  BOOST_CHECK(   ublas::is_matrix(e11) );
  BOOST_CHECK( ! ublas::is_tensor(e11) );

  BOOST_CHECK( ! ublas::empty    (e12) );
  BOOST_CHECK( ! ublas::is_scalar(e12) );
  BOOST_CHECK(   ublas::is_vector(e12) );
  BOOST_CHECK(   ublas::is_matrix(e12) );
  BOOST_CHECK( ! ublas::is_tensor(e12) );

  BOOST_CHECK( ! ublas::empty    (e21) );
  BOOST_CHECK( ! ublas::is_scalar(e21) );
  BOOST_CHECK(   ublas::is_vector(e21) );
  BOOST_CHECK(   ublas::is_matrix(e21) );
  BOOST_CHECK( ! ublas::is_tensor(e21) );

  BOOST_CHECK( ! ublas::empty    (e23) );
  BOOST_CHECK( ! ublas::is_scalar(e23) );
  BOOST_CHECK( ! ublas::is_vector(e23) );
  BOOST_CHECK(   ublas::is_matrix(e23) );
  BOOST_CHECK( ! ublas::is_tensor(e23) );

  BOOST_CHECK( ! ublas::empty    (e231) );
  BOOST_CHECK( ! ublas::is_scalar(e231) );
  BOOST_CHECK( ! ublas::is_vector(e231) );
  BOOST_CHECK(   ublas::is_matrix(e231) );
  BOOST_CHECK( ! ublas::is_tensor(e231) );

  BOOST_CHECK( ! ublas::empty    (e123) );
  BOOST_CHECK( ! ublas::is_scalar(e123) );
  BOOST_CHECK( ! ublas::is_vector(e123) );
  BOOST_CHECK( ! ublas::is_matrix(e123) );
  BOOST_CHECK(   ublas::is_tensor(e123) );

  BOOST_CHECK( ! ublas::empty    (e1123) );
  BOOST_CHECK( ! ublas::is_scalar(e1123) );
  BOOST_CHECK( ! ublas::is_vector(e1123) );
  BOOST_CHECK( ! ublas::is_matrix(e1123) );
  BOOST_CHECK(   ublas::is_tensor(e1123) );

  BOOST_CHECK( ! ublas::empty    (e12311) );
  BOOST_CHECK( ! ublas::is_scalar(e12311) );
  BOOST_CHECK( ! ublas::is_vector(e12311) );
  BOOST_CHECK( ! ublas::is_matrix(e12311) );
  BOOST_CHECK(   ublas::is_tensor(e12311) );

  BOOST_CHECK( ! ublas::empty    (e423) );
  BOOST_CHECK( ! ublas::is_scalar(e423) );
  BOOST_CHECK( ! ublas::is_vector(e423) );
  BOOST_CHECK( ! ublas::is_matrix(e423) );
  BOOST_CHECK(   ublas::is_tensor(e423) );

  BOOST_CHECK( ! ublas::empty    (e4213) );
  BOOST_CHECK( ! ublas::is_scalar(e4213) );
  BOOST_CHECK( ! ublas::is_vector(e4213) );
  BOOST_CHECK( ! ublas::is_matrix(e4213) );
  BOOST_CHECK(   ublas::is_tensor(e4213) );

  BOOST_CHECK( ! ublas::empty    (e42131) );
  BOOST_CHECK( ! ublas::is_scalar(e42131) );
  BOOST_CHECK( ! ublas::is_vector(e42131) );
  BOOST_CHECK( ! ublas::is_matrix(e42131) );
  BOOST_CHECK(   ublas::is_tensor(e42131) );

  BOOST_CHECK( ! ublas::empty    (e142131) );
  BOOST_CHECK( ! ublas::is_scalar(e142131) );
  BOOST_CHECK( ! ublas::is_vector(e142131) );
  BOOST_CHECK( ! ublas::is_matrix(e142131) );
  BOOST_CHECK(   ublas::is_tensor(e142131) );

  BOOST_CHECK( ! ublas::empty    (e141) );
  BOOST_CHECK( ! ublas::is_scalar(e141) );
  BOOST_CHECK(   ublas::is_vector(e141) );
  BOOST_CHECK(   ublas::is_matrix(e141) );
  BOOST_CHECK( ! ublas::is_tensor(e141) );

  BOOST_CHECK( ! ublas::empty    (e1111) );
  BOOST_CHECK(   ublas::is_scalar(e1111) );
  BOOST_CHECK(   ublas::is_vector(e1111) );
  BOOST_CHECK(   ublas::is_matrix(e1111) );
  BOOST_CHECK( ! ublas::is_tensor(e1111) );

  BOOST_CHECK( ! ublas::empty    (e14111) );
  BOOST_CHECK( ! ublas::is_scalar(e14111) );
  BOOST_CHECK(   ublas::is_vector(e14111) );
  BOOST_CHECK(   ublas::is_matrix(e14111) );
  BOOST_CHECK( ! ublas::is_tensor(e14111) );

  BOOST_CHECK( ! ublas::empty    (e112111) );
  BOOST_CHECK( ! ublas::is_scalar(e112111) );
  BOOST_CHECK( ! ublas::is_vector(e112111) );
  BOOST_CHECK( ! ublas::is_matrix(e112111) );
  BOOST_CHECK(   ublas::is_tensor(e112111) );

  BOOST_CHECK( ! ublas::empty    (e112311) );
  BOOST_CHECK( ! ublas::is_scalar(e112311) );
  BOOST_CHECK( ! ublas::is_vector(e112311) );
  BOOST_CHECK( ! ublas::is_matrix(e112311) );
  BOOST_CHECK(   ublas::is_tensor(e112311) );
}

//BOOST_FIXTURE_TEST_CASE(test_extents_static_size_squeeze, fixture, *boost::unit_test::label("basic_fixed_rank_extents") *boost::unit_test::label("squeeze"))
//{
//    auto e1  = squeeze(de1); // {1,1}
//    auto e2  = squeeze(de2); // {1,2}
//    auto 21  = squeeze(d21); // {2,1}

//    auto e4  = squeeze(de4); // {2,3}
//    auto e231  = squeeze(de231); // {2,3}
//    auto e123  = squeeze(de123); // {2,3}
//    auto e1123  = squeeze(de1123); // {2,3}
//    auto e12311  = squeeze(de12311); // {2,3}

//    auto e423  = squeeze(de423); // {4,2,3}
//    auto e4213 = squeeze(de4213); // {4,2,3}
//    auto e11 = squeeze(de11); // {4,2,3}
//    auto e12 = squeeze(e142131); // {4,2,3}

//    auto e141 = squeeze(de141); // {1,4}
//    auto e1111 = squeeze(de1111); // {1,1}
//    auto e14111 = squeeze(de14111); // {1,4}
//    auto e112111 = squeeze(de112111); // {2,1}
//    auto e112311 = squeeze(de112311); // {2,3}

//    BOOST_CHECK( (e1  == extents<2>{1,1}) );
//    BOOST_CHECK( (e2  == extents<2>{1,2}) );
//    BOOST_CHECK( (21  == extents<2>{2,1}) );

//    BOOST_CHECK( (e4  == extents<2>{2,3}) );
//    BOOST_CHECK( (e231  == extents<2>{2,3}) );
//    BOOST_CHECK( (e123  == extents<2>{2,3}) );
//    BOOST_CHECK( (e1123  == extents<2>{2,3}) );
//    BOOST_CHECK( (e12311  == extents<2>{2,3}) );

//    BOOST_CHECK( (e423  == extents<3>{4,2,3}) );
//    BOOST_CHECK( (e4213 == extents<3>{4,2,3}) );
//    BOOST_CHECK( (e11 == extents<3>{4,2,3}) );
//    BOOST_CHECK( (e12 == extents<3>{4,2,3}) );

//    BOOST_CHECK( (e141 == extents<2>{1,4}) );
//    BOOST_CHECK( (e1111 == extents<2>{1,1}) );
//    BOOST_CHECK( (e14111 == extents<2>{1,4}) );
//    BOOST_CHECK( (e112111 == extents<2>{2,1}) );
//    BOOST_CHECK( (e112311 == extents<2>{2,3}) );

//}


BOOST_FIXTURE_TEST_CASE(test_extents_static_size_product, fixture, *boost::unit_test::label("basic_fixed_rank_extents") *boost::unit_test::label("product"))
{
  namespace ublas = boost::numeric::ublas;

//  auto e       = ublas::product( de       );
  auto e11     = ublas::product( de11     );
  auto e12     = ublas::product( de12     );
  auto e21     = ublas::product( de21     );
  auto e23     = ublas::product( de23     );
  auto e231    = ublas::product( de231    );
  auto e123    = ublas::product( de123    );
  auto e1123   = ublas::product( de1123   );
  auto e12311  = ublas::product( de12311  );
  auto e423    = ublas::product( de423    );
  auto e4213   = ublas::product( de4213   );
  auto e42131  = ublas::product( de42131  );
  auto e142131 = ublas::product( de142131 );
  auto e141    = ublas::product( de141    );
  auto e1111   = ublas::product( de1111   );
  auto e14111  = ublas::product( de14111  );
  auto e112111 = ublas::product( de112111 );
  auto e112311 = ublas::product( de112311 );

//  BOOST_CHECK_EQUAL( e      ,  0 );
  BOOST_CHECK_EQUAL( e11    ,  1 );
  BOOST_CHECK_EQUAL( e12    ,  2 );
  BOOST_CHECK_EQUAL( e21    ,  2 );
  BOOST_CHECK_EQUAL( e23    ,  6 );
  BOOST_CHECK_EQUAL( e231   ,  6 );
  BOOST_CHECK_EQUAL( e123   ,  6 );
  BOOST_CHECK_EQUAL( e1123  ,  6 );
  BOOST_CHECK_EQUAL( e12311 ,  6 );
  BOOST_CHECK_EQUAL( e423   , 24 );
  BOOST_CHECK_EQUAL( e4213  , 24 );
  BOOST_CHECK_EQUAL( e42131 , 24 );
  BOOST_CHECK_EQUAL( e142131, 24 );
  BOOST_CHECK_EQUAL( e141   ,  4 );
  BOOST_CHECK_EQUAL( e1111  ,  1 );
  BOOST_CHECK_EQUAL( e14111 ,  4 );
  BOOST_CHECK_EQUAL( e112111,  2 );
  BOOST_CHECK_EQUAL( e112311,  6 );


}

BOOST_AUTO_TEST_SUITE_END()
