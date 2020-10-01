//
// 	Copyright (c) 2018-2020, Cem Bassoy, cem.bassoy@gmail.com
// 	Copyright (c) 2019-2020, Amit Singh, amitsingh19975@gmail.com
//
//  Distributed under the Boost Software License, Version 1.0. (See
//  accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)
//
//  The authors gratefully acknowledge the support of
//  Google and Fraunhofer IOSB, Ettlingen, Germany
//

#include <boost/test/unit_test.hpp>
#include <boost/numeric/ublas/tensor/fixed_rank_extents.hpp>
#include <boost/numeric/ublas/tensor/static_extents.hpp>
#include <boost/numeric/ublas/tensor/dynamic_extents.hpp>
#include <vector>

BOOST_AUTO_TEST_SUITE ( test_extents )


//*boost::unit_test::label("extents")
//*boost::unit_test::label("constructor")

BOOST_AUTO_TEST_CASE(test_extents_ctor)
{
    using namespace boost::numeric;
    using extents = ublas::basic_extents<unsigned>;


    auto e0 = extents{};
    BOOST_CHECK( e0.empty());
    BOOST_CHECK ( e0.size() == 0 );

    auto e1 = extents{1,1};
    BOOST_CHECK(!e1.empty());
    BOOST_CHECK ( e1.size() == 2 );

    auto e2 = extents{1,2};
    BOOST_CHECK(!e2.empty());
    BOOST_CHECK ( e2.size() == 2 );

    auto e3 = extents{2,1};
    BOOST_CHECK (!e3.empty());
    BOOST_CHECK  ( e3.size() == 2 );

    auto e4 = extents{2,3};
    BOOST_CHECK(!e4.empty());
    BOOST_CHECK ( e4.size() == 2 );

    auto e5 = extents{2,3,1};
    BOOST_CHECK (!e5.empty());
    BOOST_CHECK  ( e5.size() == 3 );

    auto e6 = extents{1,2,3}; // 6
    BOOST_CHECK(!e6.empty());
    BOOST_CHECK ( e6.size() == 3 );

    auto e7 = extents{4,2,3};  // 7
    BOOST_CHECK(!e7.empty());
    BOOST_CHECK ( e7.size() == 3 );

    BOOST_CHECK_THROW( extents({1,0}), std::length_error);
    BOOST_CHECK_THROW( extents({0}  ), std::length_error);
    BOOST_CHECK_THROW( extents({3}  ), std::length_error);
    BOOST_CHECK_THROW( extents({0,1}), std::length_error);
}

BOOST_AUTO_TEST_CASE(test_static_rank_extents_ctor)
{
    namespace ub = boost::numeric::ublas;


    auto e0 = ub::extents<0>{};
    BOOST_CHECK( e0.empty());
    BOOST_CHECK ( e0.size() == 0);

    auto e1 = ub::extents<2>{1,1};
    BOOST_CHECK(!e1.empty());
    BOOST_CHECK ( e1.size() == 2);

    auto e2 = ub::extents<2>{1,2};
    BOOST_CHECK(!e2.empty());
    BOOST_CHECK ( e2.size() == 2);

    auto e3 = ub::extents<2>{2,1};
    BOOST_CHECK (!e3.empty());
    BOOST_CHECK ( e3.size() == 2);

    auto e4 = ub::extents<2>{2,3};
    BOOST_CHECK(!e4.empty());
    BOOST_CHECK ( e4.size() == 2);

    auto e5 = ub::extents<3>{2,3,1};
    BOOST_CHECK (!e5.empty());
    BOOST_CHECK  ( e5.size() == 3);

    auto e6 = ub::extents<3>{1,2,3}; // 6
    BOOST_CHECK(!e6.empty());
    BOOST_CHECK ( e6.size() == 3);

    auto e7 = ub::extents<3>{4,2,3};  // 7
    BOOST_CHECK(!e7.empty());
    BOOST_CHECK ( e7.size() == 3);

    BOOST_CHECK_THROW( ub::extents<2>({1,0}), 	std::length_error);
    BOOST_CHECK_THROW( ub::extents<1>({0}  ), 	std::length_error);
    BOOST_CHECK_THROW( ub::extents<1>({3}  ), 	std::length_error);
    BOOST_CHECK_THROW( ub::extents<2>({0,1}), 	std::length_error);
    BOOST_CHECK_THROW( ub::extents<2>({1,1,2}), std::out_of_range);
}


struct fixture {
    using extents_type = boost::numeric::ublas::extents<>;
    template<size_t N>
    using static_rank_extents_type = boost::numeric::ublas::extents<N>;

    fixture() : extents{
                  extents_type{},            // 0

                  extents_type{1,1},         // 1
                  extents_type{1,2},         // 2
                  extents_type{2,1},         // 3

                  extents_type{2,3},         // 4
                  extents_type{2,3,1},       // 5
                  extents_type{1,2,3},       // 6
                  extents_type{1,1,2,3},     // 7
                  extents_type{1,2,3,1,1},   // 8

                  extents_type{4,2,3},       // 9
                  extents_type{4,2,1,3},     // 10
                  extents_type{4,2,1,3,1},   // 11
                  extents_type{1,4,2,1,3,1}, // 12
                  
                  extents_type{1,4,1}, 		 // 13
                  extents_type{1,1,1,1}, 	 // 14
                  extents_type{1,4,1,1,1},   // 15
                  extents_type{1,1,2,1,1,1}, // 16
                  extents_type{1,1,2,3,1,1}, // 17
            }
    {}
    std::vector<extents_type> extents;
};

BOOST_FIXTURE_TEST_CASE(test_extents_access, fixture, *boost::unit_test::label("extents") *boost::unit_test::label("access"))
{
    using namespace boost::numeric;

    BOOST_REQUIRE_EQUAL(extents.size(),18);

    BOOST_CHECK_EQUAL  (extents[ 0].size(), 0);
    BOOST_CHECK (extents[ 0].empty()  );

    BOOST_REQUIRE_EQUAL(extents[ 1].size(), 2);
    BOOST_REQUIRE_EQUAL(extents[ 2].size(), 2);
    BOOST_REQUIRE_EQUAL(extents[ 3].size(), 2);
    BOOST_REQUIRE_EQUAL(extents[ 4].size(), 2);
    BOOST_REQUIRE_EQUAL(extents[ 5].size(), 3);
    BOOST_REQUIRE_EQUAL(extents[ 6].size(), 3);
    BOOST_REQUIRE_EQUAL(extents[ 7].size(), 4);
    BOOST_REQUIRE_EQUAL(extents[ 8].size(), 5);
    BOOST_REQUIRE_EQUAL(extents[ 9].size(), 3);
    BOOST_REQUIRE_EQUAL(extents[10].size(), 4);
    BOOST_REQUIRE_EQUAL(extents[11].size(), 5);
    BOOST_REQUIRE_EQUAL(extents[12].size(), 6);
    BOOST_REQUIRE_EQUAL(extents[13].size(), 3);
    BOOST_REQUIRE_EQUAL(extents[14].size(), 4);
    BOOST_REQUIRE_EQUAL(extents[15].size(), 5);
    BOOST_REQUIRE_EQUAL(extents[16].size(), 6);
    BOOST_REQUIRE_EQUAL(extents[17].size(), 6);


    BOOST_CHECK_EQUAL(extents[1][0],1);
    BOOST_CHECK_EQUAL(extents[1][1],1);

    BOOST_CHECK_EQUAL(extents[2][0],1);
    BOOST_CHECK_EQUAL(extents[2][1],2);

    BOOST_CHECK_EQUAL(extents[3][0],2);
    BOOST_CHECK_EQUAL(extents[3][1],1);

    BOOST_CHECK_EQUAL(extents[4][0],2);
    BOOST_CHECK_EQUAL(extents[4][1],3);

    BOOST_CHECK_EQUAL(extents[5][0],2);
    BOOST_CHECK_EQUAL(extents[5][1],3);
    BOOST_CHECK_EQUAL(extents[5][2],1);

    BOOST_CHECK_EQUAL(extents[6][0],1);
    BOOST_CHECK_EQUAL(extents[6][1],2);
    BOOST_CHECK_EQUAL(extents[6][2],3);

    BOOST_CHECK_EQUAL(extents[7][0],1);
    BOOST_CHECK_EQUAL(extents[7][1],1);
    BOOST_CHECK_EQUAL(extents[7][2],2);
    BOOST_CHECK_EQUAL(extents[7][3],3);

    BOOST_CHECK_EQUAL(extents[8][0],1);
    BOOST_CHECK_EQUAL(extents[8][1],2);
    BOOST_CHECK_EQUAL(extents[8][2],3);
    BOOST_CHECK_EQUAL(extents[8][3],1);
    BOOST_CHECK_EQUAL(extents[8][4],1);

    BOOST_CHECK_EQUAL(extents[9][0],4);
    BOOST_CHECK_EQUAL(extents[9][1],2);
    BOOST_CHECK_EQUAL(extents[9][2],3);

    BOOST_CHECK_EQUAL(extents[10][0],4);
    BOOST_CHECK_EQUAL(extents[10][1],2);
    BOOST_CHECK_EQUAL(extents[10][2],1);
    BOOST_CHECK_EQUAL(extents[10][3],3);

    BOOST_CHECK_EQUAL(extents[11][0],4);
    BOOST_CHECK_EQUAL(extents[11][1],2);
    BOOST_CHECK_EQUAL(extents[11][2],1);
    BOOST_CHECK_EQUAL(extents[11][3],3);
    BOOST_CHECK_EQUAL(extents[11][4],1);

    BOOST_CHECK_EQUAL(extents[12][0],1);
    BOOST_CHECK_EQUAL(extents[12][1],4);
    BOOST_CHECK_EQUAL(extents[12][2],2);
    BOOST_CHECK_EQUAL(extents[12][3],1);
    BOOST_CHECK_EQUAL(extents[12][4],3);
    BOOST_CHECK_EQUAL(extents[12][5],1);

    BOOST_CHECK_EQUAL(extents[13][0],1);
    BOOST_CHECK_EQUAL(extents[13][1],4);
    BOOST_CHECK_EQUAL(extents[13][2],1);

    BOOST_CHECK_EQUAL(extents[14][0],1);
    BOOST_CHECK_EQUAL(extents[14][1],1);
    BOOST_CHECK_EQUAL(extents[14][2],1);
    BOOST_CHECK_EQUAL(extents[14][3],1);

    BOOST_CHECK_EQUAL(extents[15][0],1);
    BOOST_CHECK_EQUAL(extents[15][1],4);
    BOOST_CHECK_EQUAL(extents[15][2],1);
    BOOST_CHECK_EQUAL(extents[15][3],1);
    BOOST_CHECK_EQUAL(extents[15][4],1);

    BOOST_CHECK_EQUAL(extents[16][0],1);
    BOOST_CHECK_EQUAL(extents[16][1],1);
    BOOST_CHECK_EQUAL(extents[16][2],2);
    BOOST_CHECK_EQUAL(extents[16][3],1);
    BOOST_CHECK_EQUAL(extents[16][4],1);
    BOOST_CHECK_EQUAL(extents[16][5],1);

    BOOST_CHECK_EQUAL(extents[17][0],1);
    BOOST_CHECK_EQUAL(extents[17][1],1);
    BOOST_CHECK_EQUAL(extents[17][2],2);
    BOOST_CHECK_EQUAL(extents[17][3],3);
    BOOST_CHECK_EQUAL(extents[17][4],1);
    BOOST_CHECK_EQUAL(extents[17][5],1);
}

BOOST_FIXTURE_TEST_CASE(test_extents_copy_ctor, fixture, *boost::unit_test::label("extents") *boost::unit_test::label("copy_ctor"))
{
    BOOST_REQUIRE_EQUAL(extents.size(),18);

    auto e0  = extents[ 0]; // {}
    auto e1  = extents[ 1]; // {1,1}
    auto e2  = extents[ 2]; // {1,2}
    auto e3  = extents[ 3]; // {2,1}
    auto e4  = extents[ 4]; // {2,3}
    auto e5  = extents[ 5]; // {2,3,1}
    auto e6  = extents[ 6]; // {1,2,3}
    auto e7  = extents[ 7]; // {1,1,2,3}
    auto e8  = extents[ 8]; // {1,2,3,1,1}
    auto e9  = extents[ 9]; // {4,2,3}
    auto e10 = extents[10]; // {4,2,1,3}
    auto e11 = extents[11]; // {4,2,1,3,1}
    auto e12 = extents[12]; // {1,4,2,1,3,1}
    auto e13 = extents[13]; // {1,4,1}
    auto e14 = extents[14]; // {1,1,1,1}
    auto e15 = extents[15]; // {1,4,1,1,1}
    auto e16 = extents[16]; // {1,1,2,1,1,1}
    auto e17 = extents[17]; // {1,1,2,3,1,1}

    BOOST_CHECK_EQUAL  (e0.size(), 0);
    BOOST_CHECK (e0.empty()  );

    BOOST_REQUIRE_EQUAL(e1 .size(), 2);
    BOOST_REQUIRE_EQUAL(e2 .size(), 2);
    BOOST_REQUIRE_EQUAL(e3 .size(), 2);
    BOOST_REQUIRE_EQUAL(e4 .size(), 2);
    BOOST_REQUIRE_EQUAL(e5 .size(), 3);
    BOOST_REQUIRE_EQUAL(e6 .size(), 3);
    BOOST_REQUIRE_EQUAL(e7 .size(), 4);
    BOOST_REQUIRE_EQUAL(e8 .size(), 5);
    BOOST_REQUIRE_EQUAL(e9 .size(), 3);
    BOOST_REQUIRE_EQUAL(e10.size(), 4);
    BOOST_REQUIRE_EQUAL(e11.size(), 5);
    BOOST_REQUIRE_EQUAL(e12.size(), 6);
    BOOST_REQUIRE_EQUAL(e13.size(), 3);
    BOOST_REQUIRE_EQUAL(e14.size(), 4);
    BOOST_REQUIRE_EQUAL(e15.size(), 5);
    BOOST_REQUIRE_EQUAL(e16.size(), 6);
    BOOST_REQUIRE_EQUAL(e17.size(), 6);


    BOOST_CHECK_EQUAL(e1[0],1);
    BOOST_CHECK_EQUAL(e1[1],1);

    BOOST_CHECK_EQUAL(e2[0],1);
    BOOST_CHECK_EQUAL(e2[1],2);

    BOOST_CHECK_EQUAL(e3[0],2);
    BOOST_CHECK_EQUAL(e3[1],1);

    BOOST_CHECK_EQUAL(e4[0],2);
    BOOST_CHECK_EQUAL(e4[1],3);

    BOOST_CHECK_EQUAL(e5[0],2);
    BOOST_CHECK_EQUAL(e5[1],3);
    BOOST_CHECK_EQUAL(e5[2],1);

    BOOST_CHECK_EQUAL(e6[0],1);
    BOOST_CHECK_EQUAL(e6[1],2);
    BOOST_CHECK_EQUAL(e6[2],3);

    BOOST_CHECK_EQUAL(e7[0],1);
    BOOST_CHECK_EQUAL(e7[1],1);
    BOOST_CHECK_EQUAL(e7[2],2);
    BOOST_CHECK_EQUAL(e7[3],3);

    BOOST_CHECK_EQUAL(e8[0],1);
    BOOST_CHECK_EQUAL(e8[1],2);
    BOOST_CHECK_EQUAL(e8[2],3);
    BOOST_CHECK_EQUAL(e8[3],1);
    BOOST_CHECK_EQUAL(e8[4],1);

    BOOST_CHECK_EQUAL(e9[0],4);
    BOOST_CHECK_EQUAL(e9[1],2);
    BOOST_CHECK_EQUAL(e9[2],3);

    BOOST_CHECK_EQUAL(e10[0],4);
    BOOST_CHECK_EQUAL(e10[1],2);
    BOOST_CHECK_EQUAL(e10[2],1);
    BOOST_CHECK_EQUAL(e10[3],3);

    BOOST_CHECK_EQUAL(e11[0],4);
    BOOST_CHECK_EQUAL(e11[1],2);
    BOOST_CHECK_EQUAL(e11[2],1);
    BOOST_CHECK_EQUAL(e11[3],3);
    BOOST_CHECK_EQUAL(e11[4],1);

    BOOST_CHECK_EQUAL(e12[0],1);
    BOOST_CHECK_EQUAL(e12[1],4);
    BOOST_CHECK_EQUAL(e12[2],2);
    BOOST_CHECK_EQUAL(e12[3],1);
    BOOST_CHECK_EQUAL(e12[4],3);
    BOOST_CHECK_EQUAL(e12[5],1);

    BOOST_CHECK_EQUAL(e13[0],1);
    BOOST_CHECK_EQUAL(e13[1],4);
    BOOST_CHECK_EQUAL(e13[2],1);

    BOOST_CHECK_EQUAL(e14[0],1);
    BOOST_CHECK_EQUAL(e14[1],1);
    BOOST_CHECK_EQUAL(e14[2],1);
    BOOST_CHECK_EQUAL(e14[3],1);

    BOOST_CHECK_EQUAL(e15[0],1);
    BOOST_CHECK_EQUAL(e15[1],4);
    BOOST_CHECK_EQUAL(e15[2],1);
    BOOST_CHECK_EQUAL(e15[3],1);
    BOOST_CHECK_EQUAL(e15[4],1);

    BOOST_CHECK_EQUAL(e16[0],1);
    BOOST_CHECK_EQUAL(e16[1],1);
    BOOST_CHECK_EQUAL(e16[2],2);
    BOOST_CHECK_EQUAL(e16[3],1);
    BOOST_CHECK_EQUAL(e16[4],1);
    BOOST_CHECK_EQUAL(e16[5],1);

    BOOST_CHECK_EQUAL(e17[0],1);
    BOOST_CHECK_EQUAL(e17[1],1);
    BOOST_CHECK_EQUAL(e17[2],2);
    BOOST_CHECK_EQUAL(e17[3],3);
    BOOST_CHECK_EQUAL(e17[4],1);
    BOOST_CHECK_EQUAL(e17[5],1);

}

BOOST_FIXTURE_TEST_CASE(test_extents_is, fixture, *boost::unit_test::label("extents") *boost::unit_test::label("query"))
{
    BOOST_REQUIRE_EQUAL(extents.size(),18);

    auto e0  = extents[ 0]; // {}
    auto e1  = extents[ 1]; // {1,1}
    auto e2  = extents[ 2]; // {1,2}
    auto e3  = extents[ 3]; // {2,1}
    auto e4  = extents[ 4]; // {2,3}
    auto e5  = extents[ 5]; // {2,3,1}
    auto e6  = extents[ 6]; // {1,2,3}
    auto e7  = extents[ 7]; // {1,1,2,3}
    auto e8  = extents[ 8]; // {1,2,3,1,1}
    auto e9  = extents[ 9]; // {4,2,3}
    auto e10 = extents[10]; // {4,2,1,3}
    auto e11 = extents[11]; // {4,2,1,3,1}
    auto e12 = extents[12]; // {1,4,2,1,3,1}
    auto e13 = extents[13]; // {1,4,1}
    auto e14 = extents[14]; // {1,1,1,1}
    auto e15 = extents[15]; // {1,4,1,1,1}
    auto e16 = extents[16]; // {1,1,2,1,1,1}
    auto e17 = extents[17]; // {1,1,2,3,1,1}

    BOOST_CHECK(   e0.empty (  ));
    BOOST_CHECK( ! is_scalar(e0));
    BOOST_CHECK( ! is_vector(e0));
    BOOST_CHECK( ! is_matrix(e0));
    BOOST_CHECK( ! is_tensor(e0));

    BOOST_CHECK( ! e1.empty (  ) );
    BOOST_CHECK(   is_scalar(e1) );
    BOOST_CHECK( ! is_vector(e1) );
    BOOST_CHECK( ! is_matrix(e1) );
    BOOST_CHECK( ! is_tensor(e1) );

    BOOST_CHECK( ! e2.empty (  ) );
    BOOST_CHECK( ! is_scalar(e2) );
    BOOST_CHECK(   is_vector(e2) );
    BOOST_CHECK( ! is_matrix(e2) );
    BOOST_CHECK( ! is_tensor(e2) );

    BOOST_CHECK( ! e3.empty (  ) );
    BOOST_CHECK( ! is_scalar(e3) );
    BOOST_CHECK(   is_vector(e3) );
    BOOST_CHECK( ! is_matrix(e3) );
    BOOST_CHECK( ! is_tensor(e3) );

    BOOST_CHECK( ! e4.empty (  ) );
    BOOST_CHECK( ! is_scalar(e4) );
    BOOST_CHECK( ! is_vector(e4) );
    BOOST_CHECK(   is_matrix(e4) );
    BOOST_CHECK( ! is_tensor(e4) );

    BOOST_CHECK( ! e5.empty (  ) );
    BOOST_CHECK( ! is_scalar(e5) );
    BOOST_CHECK( ! is_vector(e5) );
    BOOST_CHECK(   is_matrix(e5) );
    BOOST_CHECK( ! is_tensor(e5) );

    BOOST_CHECK( ! e6.empty (  ) );
    BOOST_CHECK( ! is_scalar(e6) );
    BOOST_CHECK( ! is_vector(e6) );
    BOOST_CHECK( ! is_matrix(e6) );
    BOOST_CHECK(   is_tensor(e6) );

    BOOST_CHECK( ! e7.empty (  ) );
    BOOST_CHECK( ! is_scalar(e7) );
    BOOST_CHECK( ! is_vector(e7) );
    BOOST_CHECK( ! is_matrix(e7) );
    BOOST_CHECK(   is_tensor(e7) );

    BOOST_CHECK( ! e8.empty (  ) );
    BOOST_CHECK( ! is_scalar(e8) );
    BOOST_CHECK( ! is_vector(e8) );
    BOOST_CHECK( ! is_matrix(e8) );
    BOOST_CHECK(   is_tensor(e8) );

    BOOST_CHECK( ! e9.empty (  ) );
    BOOST_CHECK( ! is_scalar(e9) );
    BOOST_CHECK( ! is_vector(e9) );
    BOOST_CHECK( ! is_matrix(e9) );
    BOOST_CHECK(   is_tensor(e9) );

    BOOST_CHECK( ! e10.empty(   ) );
    BOOST_CHECK( ! is_scalar(e10) );
    BOOST_CHECK( ! is_vector(e10) );
    BOOST_CHECK( ! is_matrix(e10) );
    BOOST_CHECK(   is_tensor(e10) );

    BOOST_CHECK( ! e11.empty(   ) );
    BOOST_CHECK( ! is_scalar(e11) );
    BOOST_CHECK( ! is_vector(e11) );
    BOOST_CHECK( ! is_matrix(e11) );
    BOOST_CHECK(   is_tensor(e11) );

    BOOST_CHECK( ! e12.empty(   ) );
    BOOST_CHECK( ! is_scalar(e12) );
    BOOST_CHECK( ! is_vector(e12) );
    BOOST_CHECK( ! is_matrix(e12) );
    BOOST_CHECK(   is_tensor(e12) );

    BOOST_CHECK( ! e13.empty(   ) );
    BOOST_CHECK( ! is_scalar(e13) );
    BOOST_CHECK(   is_vector(e13) );
    BOOST_CHECK( ! is_matrix(e13) );
    BOOST_CHECK( ! is_tensor(e13) );

    BOOST_CHECK( ! e14.empty(   ) );
    BOOST_CHECK(   is_scalar(e14) );
    BOOST_CHECK( ! is_vector(e14) );
    BOOST_CHECK( ! is_matrix(e14) );
    BOOST_CHECK( ! is_tensor(e14) );

    BOOST_CHECK( ! e15.empty(   ) );
    BOOST_CHECK( ! is_scalar(e15) );
    BOOST_CHECK(   is_vector(e15) );
    BOOST_CHECK( ! is_matrix(e15) );
    BOOST_CHECK( ! is_tensor(e15) );

    BOOST_CHECK( ! e16.empty(   ) );
    BOOST_CHECK( ! is_scalar(e16) );
    BOOST_CHECK( ! is_vector(e16) );
    BOOST_CHECK( ! is_matrix(e16) );
    BOOST_CHECK(   is_tensor(e16) );

    BOOST_CHECK( ! e17.empty(   ) );
    BOOST_CHECK( ! is_scalar(e17) );
    BOOST_CHECK( ! is_vector(e17) );
    BOOST_CHECK( ! is_matrix(e17) );
    BOOST_CHECK(   is_tensor(e17) );
}

BOOST_FIXTURE_TEST_CASE(test_extents_squeeze, fixture, *boost::unit_test::label("extents") *boost::unit_test::label("squeeze"))
{
    BOOST_REQUIRE_EQUAL(extents.size(),18);

    auto e0  = squeeze(extents[ 0]); // {}
    auto e1  = squeeze(extents[ 1]); // {1,1}
    auto e2  = squeeze(extents[ 2]); // {1,2}
    auto e3  = squeeze(extents[ 3]); // {2,1}

    auto e4  = squeeze(extents[ 4]); // {2,3}
    auto e5  = squeeze(extents[ 5]); // {2,3}
    auto e6  = squeeze(extents[ 6]); // {2,3}
    auto e7  = squeeze(extents[ 7]); // {2,3}
    auto e8  = squeeze(extents[ 8]); // {2,3}

    auto e9  = squeeze(extents[ 9]); // {4,2,3}
    auto e10 = squeeze(extents[10]); // {4,2,3}
    auto e11 = squeeze(extents[11]); // {4,2,3}
    auto e12 = squeeze(extents[12]); // {4,2,3}

    auto e13 = squeeze(extents[13]); // {1,4}
    auto e14 = squeeze(extents[14]); // {1,1}
    auto e15 = squeeze(extents[15]); // {1,4}
    auto e16 = squeeze(extents[16]); // {2,1}
    auto e17 = squeeze(extents[17]); // {2,3}

    BOOST_CHECK( (e0  == extents_type{}   ) );
    BOOST_CHECK( (e1  == extents_type{1,1}) );
    BOOST_CHECK( (e2  == extents_type{1,2}) );
    BOOST_CHECK( (e3  == extents_type{2,1}) );

    BOOST_CHECK( (e4  == extents_type{2,3}) );
    BOOST_CHECK( (e5  == extents_type{2,3}) );
    BOOST_CHECK( (e6  == extents_type{2,3}) );
    BOOST_CHECK( (e7  == extents_type{2,3}) );
    BOOST_CHECK( (e8  == extents_type{2,3}) );

    BOOST_CHECK( (e9  == extents_type{4,2,3}) );
    BOOST_CHECK( (e10 == extents_type{4,2,3}) );
    BOOST_CHECK( (e11 == extents_type{4,2,3}) );
    BOOST_CHECK( (e12 == extents_type{4,2,3}) );

    BOOST_CHECK( (e13 == extents_type{1,4}) );
    BOOST_CHECK( (e14 == extents_type{1,1}) );
    BOOST_CHECK( (e15 == extents_type{1,4}) );
    BOOST_CHECK( (e16 == extents_type{2,1}) );
    BOOST_CHECK( (e17 == extents_type{2,3}) );

}

BOOST_FIXTURE_TEST_CASE(test_extents_comparison, fixture, *boost::unit_test::label("extents") *boost::unit_test::label("compare"))
{

    using namespace boost::numeric;

    auto s0 = ublas::static_extents<>{};
    auto s1 = ublas::static_extents<1,1>{};
    auto s2 = ublas::static_extents<1,4,2,1,3,1>{};
    auto s3 = ublas::static_extents<1,4,2,1,1,1>{};
    
    auto d0 = ublas::extents<0>{};
    auto d1 = ublas::extents<2>{1,1};
    auto d2 = ublas::extents<6>{1,4,2,1,3,1};
    auto d3 = ublas::extents<6>{1,4,2,1,1,1};

    auto e0  = extents[ 0]; // {}
    auto e1  = extents[ 1]; // {1,1}
    auto e2	 = extents[12]; // {1,4,2,1,3,1}
    
    // static_extents<...> == extents<>
    BOOST_TEST( s0 == e0 );
    BOOST_TEST( s1 == e1 );
    BOOST_TEST( s2 == e2 );
    
    BOOST_TEST( e0 == s0 );
    BOOST_TEST( e1 == s1 );
    BOOST_TEST( e2 == s2 );
    
    BOOST_TEST( s0 != e1 );
    BOOST_TEST( s0 != e2 );
    BOOST_TEST( s1 != e0 );
    BOOST_TEST( s1 != e2 );
    BOOST_TEST( s2 != e0 );
    BOOST_TEST( s2 != e1 );
    BOOST_TEST( s3 != e0 );
    BOOST_TEST( s3 != e1 );
    BOOST_TEST( s3 != e2 );
    
    BOOST_TEST( e1 != s0 );
    BOOST_TEST( e2 != s0 );
    BOOST_TEST( e0 != s1 );
    BOOST_TEST( e2 != s1 );
    BOOST_TEST( e0 != s2 );
    BOOST_TEST( e1 != s2 );
    BOOST_TEST( e0 != s3 );
    BOOST_TEST( e1 != s3 );
    BOOST_TEST( e2 != s3 );
    
    // extents<N> == extents<>
    BOOST_TEST( d0 == e0 );
    BOOST_TEST( d1 == e1 );
    BOOST_TEST( d2 == e2 );
    
    BOOST_TEST( e0 == d0 );
    BOOST_TEST( e1 == d1 );
    BOOST_TEST( e2 == d2 );
    
    BOOST_TEST( d0 != e1 );
    BOOST_TEST( d0 != e2 );
    BOOST_TEST( d1 != e0 );
    BOOST_TEST( d1 != e2 );
    BOOST_TEST( d2 != e0 );
    BOOST_TEST( d2 != e1 );
    BOOST_TEST( d3 != e0 );
    BOOST_TEST( d3 != e1 );
    BOOST_TEST( d3 != e2 );
    
    BOOST_TEST( e1 != d0 );
    BOOST_TEST( e2 != d0 );
    BOOST_TEST( e0 != d1 );
    BOOST_TEST( e2 != d1 );
    BOOST_TEST( e0 != d2 );
    BOOST_TEST( e1 != d2 );
    BOOST_TEST( e0 != d3 );
    BOOST_TEST( e1 != d3 );
    BOOST_TEST( e2 != d3 );
    
    // static_extents<...> == extents<N>
    
    BOOST_TEST( s0 == d0 );
    BOOST_TEST( s1 == d1 );
    BOOST_TEST( s2 == d2 );
    BOOST_TEST( s3 == d3 );
    
    BOOST_TEST( d0 == s0 );
    BOOST_TEST( d1 == s1 );
    BOOST_TEST( d2 == s2 );
    BOOST_TEST( d3 == s3 );
    
    BOOST_TEST( s0 != d1 );
    BOOST_TEST( s0 != d2 );
    BOOST_TEST( s0 != d3 );
    BOOST_TEST( s1 != d0 );
    BOOST_TEST( s1 != d2 );
    BOOST_TEST( s1 != d3 );
    BOOST_TEST( s2 != d0 );
    BOOST_TEST( s2 != d1 );
    BOOST_TEST( s2 != d3 );
    BOOST_TEST( s3 != d0 );
    BOOST_TEST( s3 != d1 );
    BOOST_TEST( s3 != d2 );
    
    BOOST_TEST( d1 != s0 );
    BOOST_TEST( d2 != s0 );
    BOOST_TEST( d3 != s0 );
    BOOST_TEST( d0 != s1 );
    BOOST_TEST( d2 != s1 );
    BOOST_TEST( d3 != s1 );
    BOOST_TEST( d0 != s2 );
    BOOST_TEST( d1 != s2 );
    BOOST_TEST( d3 != s2 );
    BOOST_TEST( d0 != s3 );
    BOOST_TEST( d1 != s3 );
    BOOST_TEST( d2 != s3 );

}


BOOST_FIXTURE_TEST_CASE(test_extents_product, fixture, *boost::unit_test::label("extents") *boost::unit_test::label("product"))
{

    auto e0  = product(extents[ 0]); // {}
    auto e1  = product(extents[ 1]); // {1,1}
    auto e2  = product(extents[ 2]); // {1,2}
    auto e3  = product(extents[ 3]); // {2,1}
    auto e4  = product(extents[ 4]); // {2,3}
    auto e5  = product(extents[ 5]); // {2,3,1}
    auto e6  = product(extents[ 6]); // {1,2,3}
    auto e7  = product(extents[ 7]); // {1,1,2,3}
    auto e8  = product(extents[ 8]); // {1,2,3,1,1}
    auto e9  = product(extents[ 9]); // {4,2,3}
    auto e10 = product(extents[10]); // {4,2,1,3}
    auto e11 = product(extents[11]); // {4,2,1,3,1}
    auto e12 = product(extents[12]); // {1,4,2,1,3,1}
    auto e13 = product(extents[13]); // {1,4,1}
    auto e14 = product(extents[14]); // {1,1,1,1}
    auto e15 = product(extents[15]); // {1,4,1,1,1}
    auto e16 = product(extents[16]); // {1,1,2,1,1,1}
    auto e17 = product(extents[17]); // {1,1,2,3,1,1}

    BOOST_CHECK_EQUAL( e0 ,  0 );
    BOOST_CHECK_EQUAL( e1 ,  1 );
    BOOST_CHECK_EQUAL( e2 ,  2 );
    BOOST_CHECK_EQUAL( e3 ,  2 );
    BOOST_CHECK_EQUAL( e4 ,  6 );
    BOOST_CHECK_EQUAL( e5 ,  6 );
    BOOST_CHECK_EQUAL( e6 ,  6 );
    BOOST_CHECK_EQUAL( e7 ,  6 );
    BOOST_CHECK_EQUAL( e8 ,  6 );
    BOOST_CHECK_EQUAL( e9 , 24 );
    BOOST_CHECK_EQUAL( e10, 24 );
    BOOST_CHECK_EQUAL( e11, 24 );
    BOOST_CHECK_EQUAL( e12, 24 );
    BOOST_CHECK_EQUAL( e13,  4 );
    BOOST_CHECK_EQUAL( e14,  1 );
    BOOST_CHECK_EQUAL( e15,  4 );
    BOOST_CHECK_EQUAL( e16,  2 );
    BOOST_CHECK_EQUAL( e17,  6 );


}

BOOST_AUTO_TEST_SUITE_END()
