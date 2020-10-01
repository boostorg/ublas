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
#include <boost/numeric/ublas/tensor/dynamic_extents.hpp> // Needed for squeeze
#include <boost/numeric/ublas/tensor/fixed_rank_extents.hpp>
#include <vector>

BOOST_AUTO_TEST_SUITE ( test_fixed_rank_extents )


//*boost::unit_test::label("extents")
//*boost::unit_test::label("constructor")

BOOST_AUTO_TEST_CASE(test_fixed_rank_extents_ctor)
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
    template<size_t N>
    using extents = boost::numeric::ublas::extents<N>;

    extents<0> de0{};            // 0

    extents<2> de1{1,1};         // 1
    extents<2> de2{1,2};         // 2
    extents<2> de3{2,1};         // 3

    extents<2> de4{2,3};         // 4
    extents<3> de5{2,3,1};       // 5
    extents<3> de6{1,2,3};       // 6
    extents<4> de7{1,1,2,3};     // 7
    extents<5> de8{1,2,3,1,1};   // 8

    extents<3> de9{4,2,3};       // 9
    extents<4> de10{4,2,1,3};    // 10
    extents<5> de11{4,2,1,3,1};  // 11
    extents<6> de12{1,4,2,1,3,1};// 12
    
    extents<3> de13{1,4,1}; 	   // 13
    extents<4> de14{1,1,1,1};    // 14
    extents<5> de15{1,4,1,1,1};  // 15
    extents<6> de16{1,1,2,1,1,1};// 16
    extents<6> de17{1,1,2,3,1,1};// 17
};

BOOST_FIXTURE_TEST_CASE(test_fixed_rank_extents_access, fixture, *boost::unit_test::label("basic_fixed_rank_extents") *boost::unit_test::label("access"))
{
    using namespace boost::numeric;

    BOOST_CHECK_EQUAL(de0.size(), 0);
    BOOST_CHECK (de0.empty()  );

    BOOST_REQUIRE_EQUAL(de1.size(), 2);
    BOOST_REQUIRE_EQUAL(de2.size(), 2);
    BOOST_REQUIRE_EQUAL(de3.size(), 2);
    BOOST_REQUIRE_EQUAL(de4.size(), 2);
    BOOST_REQUIRE_EQUAL(de5.size(), 3);
    BOOST_REQUIRE_EQUAL(de6.size(), 3);
    BOOST_REQUIRE_EQUAL(de7.size(), 4);
    BOOST_REQUIRE_EQUAL(de8.size(), 5);
    BOOST_REQUIRE_EQUAL(de9.size(), 3);
    BOOST_REQUIRE_EQUAL(de10.size(), 4);
    BOOST_REQUIRE_EQUAL(de11.size(), 5);
    BOOST_REQUIRE_EQUAL(de12.size(), 6);
    BOOST_REQUIRE_EQUAL(de13.size(), 3);
    BOOST_REQUIRE_EQUAL(de14.size(), 4);
    BOOST_REQUIRE_EQUAL(de15.size(), 5);
    BOOST_REQUIRE_EQUAL(de16.size(), 6);
    BOOST_REQUIRE_EQUAL(de17.size(), 6);


    BOOST_CHECK_EQUAL(de1[0],1);
    BOOST_CHECK_EQUAL(de1[1],1);

    BOOST_CHECK_EQUAL(de2[0],1);
    BOOST_CHECK_EQUAL(de2[1],2);

    BOOST_CHECK_EQUAL(de3[0],2);
    BOOST_CHECK_EQUAL(de3[1],1);

    BOOST_CHECK_EQUAL(de4[0],2);
    BOOST_CHECK_EQUAL(de4[1],3);

    BOOST_CHECK_EQUAL(de5[0],2);
    BOOST_CHECK_EQUAL(de5[1],3);
    BOOST_CHECK_EQUAL(de5[2],1);

    BOOST_CHECK_EQUAL(de6[0],1);
    BOOST_CHECK_EQUAL(de6[1],2);
    BOOST_CHECK_EQUAL(de6[2],3);

    BOOST_CHECK_EQUAL(de7[0],1);
    BOOST_CHECK_EQUAL(de7[1],1);
    BOOST_CHECK_EQUAL(de7[2],2);
    BOOST_CHECK_EQUAL(de7[3],3);

    BOOST_CHECK_EQUAL(de8[0],1);
    BOOST_CHECK_EQUAL(de8[1],2);
    BOOST_CHECK_EQUAL(de8[2],3);
    BOOST_CHECK_EQUAL(de8[3],1);
    BOOST_CHECK_EQUAL(de8[4],1);

    BOOST_CHECK_EQUAL(de9[0],4);
    BOOST_CHECK_EQUAL(de9[1],2);
    BOOST_CHECK_EQUAL(de9[2],3);

    BOOST_CHECK_EQUAL(de10[0],4);
    BOOST_CHECK_EQUAL(de10[1],2);
    BOOST_CHECK_EQUAL(de10[2],1);
    BOOST_CHECK_EQUAL(de10[3],3);

    BOOST_CHECK_EQUAL(de11[0],4);
    BOOST_CHECK_EQUAL(de11[1],2);
    BOOST_CHECK_EQUAL(de11[2],1);
    BOOST_CHECK_EQUAL(de11[3],3);
    BOOST_CHECK_EQUAL(de11[4],1);

    BOOST_CHECK_EQUAL(de12[0],1);
    BOOST_CHECK_EQUAL(de12[1],4);
    BOOST_CHECK_EQUAL(de12[2],2);
    BOOST_CHECK_EQUAL(de12[3],1);
    BOOST_CHECK_EQUAL(de12[4],3);
    BOOST_CHECK_EQUAL(de12[5],1);

    BOOST_CHECK_EQUAL(de13[0],1);
    BOOST_CHECK_EQUAL(de13[1],4);
    BOOST_CHECK_EQUAL(de13[2],1);

    BOOST_CHECK_EQUAL(de14[0],1);
    BOOST_CHECK_EQUAL(de14[1],1);
    BOOST_CHECK_EQUAL(de14[2],1);
    BOOST_CHECK_EQUAL(de14[3],1);

    BOOST_CHECK_EQUAL(de15[0],1);
    BOOST_CHECK_EQUAL(de15[1],4);
    BOOST_CHECK_EQUAL(de15[2],1);
    BOOST_CHECK_EQUAL(de15[3],1);
    BOOST_CHECK_EQUAL(de15[4],1);

    BOOST_CHECK_EQUAL(de16[0],1);
    BOOST_CHECK_EQUAL(de16[1],1);
    BOOST_CHECK_EQUAL(de16[2],2);
    BOOST_CHECK_EQUAL(de16[3],1);
    BOOST_CHECK_EQUAL(de16[4],1);
    BOOST_CHECK_EQUAL(de16[5],1);

    BOOST_CHECK_EQUAL(de17[0],1);
    BOOST_CHECK_EQUAL(de17[1],1);
    BOOST_CHECK_EQUAL(de17[2],2);
    BOOST_CHECK_EQUAL(de17[3],3);
    BOOST_CHECK_EQUAL(de17[4],1);
    BOOST_CHECK_EQUAL(de17[5],1);
}

BOOST_FIXTURE_TEST_CASE(test_fixed_rank_extents_copy_ctor, fixture, *boost::unit_test::label("basic_fixed_rank_extents") *boost::unit_test::label("copy_ctor"))
{

    auto e0  = de0; // {}
    auto e1  = de1; // {1,1}
    auto e2  = de2; // {1,2}
    auto e3  = de3; // {2,1}
    auto e4  = de4; // {2,3}
    auto e5  = de5; // {2,3,1}
    auto e6  = de6; // {1,2,3}
    auto e7  = de7; // {1,1,2,3}
    auto e8  = de8; // {1,2,3,1,1}
    auto e9  = de9; // {4,2,3}
    auto e10 = de10; // {4,2,1,3}
    auto e11 = de11; // {4,2,1,3,1}
    auto e12 = de12; // {1,4,2,1,3,1}
    auto e13 = de13; // {1,4,1}
    auto e14 = de14; // {1,1,1,1}
    auto e15 = de15; // {1,4,1,1,1}
    auto e16 = de16; // {1,1,2,1,1,1}
    auto e17 = de17; // {1,1,2,3,1,1}

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

BOOST_FIXTURE_TEST_CASE(test_fixed_rank_extents_is, fixture, *boost::unit_test::label("basic_fixed_rank_extents") *boost::unit_test::label("query"))
{

    auto e0  = de0; // {}
    auto e1  = de1; // {1,1}
    auto e2  = de2; // {1,2}
    auto e3  = de3; // {2,1}
    auto e4  = de4; // {2,3}
    auto e5  = de5; // {2,3,1}
    auto e6  = de6; // {1,2,3}
    auto e7  = de7; // {1,1,2,3}
    auto e8  = de8; // {1,2,3,1,1}
    auto e9  = de9; // {4,2,3}
    auto e10 = de10; // {4,2,1,3}
    auto e11 = de11; // {4,2,1,3,1}
    auto e12 = de12; // {1,4,2,1,3,1}
    auto e13 = de13; // {1,4,1}
    auto e14 = de14; // {1,1,1,1}
    auto e15 = de15; // {1,4,1,1,1}
    auto e16 = de16; // {1,1,2,1,1,1}
    auto e17 = de17; // {1,1,2,3,1,1}

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

BOOST_FIXTURE_TEST_CASE(test_fixed_rank_extents_squeeze, fixture, *boost::unit_test::label("basic_fixed_rank_extents") *boost::unit_test::label("squeeze"))
{
    auto e1  = squeeze(de1); // {1,1}
    auto e2  = squeeze(de2); // {1,2}
    auto e3  = squeeze(de3); // {2,1}

    auto e4  = squeeze(de4); // {2,3}
    auto e5  = squeeze(de5); // {2,3}
    auto e6  = squeeze(de6); // {2,3}
    auto e7  = squeeze(de7); // {2,3}
    auto e8  = squeeze(de8); // {2,3}

    auto e9  = squeeze(de9); // {4,2,3}
    auto e10 = squeeze(de10); // {4,2,3}
    auto e11 = squeeze(de11); // {4,2,3}
    auto e12 = squeeze(de12); // {4,2,3}

    auto e13 = squeeze(de13); // {1,4}
    auto e14 = squeeze(de14); // {1,1}
    auto e15 = squeeze(de15); // {1,4}
    auto e16 = squeeze(de16); // {2,1}
    auto e17 = squeeze(de17); // {2,3}

    BOOST_CHECK( (e1  == extents<2>{1,1}) );
    BOOST_CHECK( (e2  == extents<2>{1,2}) );
    BOOST_CHECK( (e3  == extents<2>{2,1}) );

    BOOST_CHECK( (e4  == extents<2>{2,3}) );
    BOOST_CHECK( (e5  == extents<2>{2,3}) );
    BOOST_CHECK( (e6  == extents<2>{2,3}) );
    BOOST_CHECK( (e7  == extents<2>{2,3}) );
    BOOST_CHECK( (e8  == extents<2>{2,3}) );

    BOOST_CHECK( (e9  == extents<3>{4,2,3}) );
    BOOST_CHECK( (e10 == extents<3>{4,2,3}) );
    BOOST_CHECK( (e11 == extents<3>{4,2,3}) );
    BOOST_CHECK( (e12 == extents<3>{4,2,3}) );

    BOOST_CHECK( (e13 == extents<2>{1,4}) );
    BOOST_CHECK( (e14 == extents<2>{1,1}) );
    BOOST_CHECK( (e15 == extents<2>{1,4}) );
    BOOST_CHECK( (e16 == extents<2>{2,1}) );
    BOOST_CHECK( (e17 == extents<2>{2,3}) );

}


BOOST_FIXTURE_TEST_CASE(test_fixed_rank_extents_product, fixture, *boost::unit_test::label("basic_fixed_rank_extents") *boost::unit_test::label("product"))
{

    auto e0  = product( de0 ); // {}
    auto e1  = product( de1 ); // {1,1}
    auto e2  = product( de2 ); // {1,2}
    auto e3  = product( de3 ); // {2,1}
    auto e4  = product( de4 ); // {2,3}
    auto e5  = product( de5 ); // {2,3,1}
    auto e6  = product( de6 ); // {1,2,3}
    auto e7  = product( de7 ); // {1,1,2,3}
    auto e8  = product( de8 ); // {1,2,3,1,1}
    auto e9  = product( de9 ); // {4,2,3}
    auto e10 = product( de10 ); // {4,2,1,3}
    auto e11 = product( de11 ); // {4,2,1,3,1}
    auto e12 = product( de12 ); // {1,4,2,1,3,1}
    auto e13 = product( de13 ); // {1,4,1}
    auto e14 = product( de14 ); // {1,1,1,1}
    auto e15 = product( de15 ); // {1,4,1,1,1}
    auto e16 = product( de16 ); // {1,1,2,1,1,1}
    auto e17 = product( de17 ); // {1,1,2,3,1,1}

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
