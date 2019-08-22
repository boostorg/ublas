
//  Copyright (c) 2018-2019 Cem Bassoy
//
//  Distributed under the Boost Software License, Version 1.0. (See
//  accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#include <boost/numeric/ublas/tensor/subtensor_helper.hpp>
#include <boost/test/unit_test.hpp>
#include <type_traits>
#include <vector>

namespace ub = boost::numeric::ublas;
namespace sp = ub::span;


struct fixture {
	using dextents_type = ub::basic_extents<size_t>;
  using dstrides_type = ub::strides_t<dextents_type,ub::first_order>;
  template<typename T>
  using sstrides_type = ub::strides_t<T,ub::first_order>;

	fixture() : dstrides{
	              dstrides_type{ dextents_type{1,2} },       
	              dstrides_type{ dextents_type{2,3,1} },     
	              dstrides_type{ dextents_type{4,2,1,3} } 
},dextents{
	              dextents_type{1,2},      
	              dextents_type{2,3,1},     
	              dextents_type{4,2,1,3}
}
	{}
	std::vector<dstrides_type> dstrides;
	std::vector<dextents_type> dextents;
  sp::detail::list<
    sstrides_type<ub::static_extents<1,2>>,
    sstrides_type<ub::static_extents<2,3,1>>,
    sstrides_type<ub::static_extents<4,2,1,3>>
  > sstrides; 

  sp::detail::list<
    ub::static_extents<1,2>,
    ub::static_extents<2,3,1>,
    ub::static_extents<4,2,1,3>
  > sextents; 
};

BOOST_FIXTURE_TEST_CASE(test_slice_generate_span_array, fixture)
{
  {
    auto sps1 = ub::detail::generate_span_array(dextents[0],sp::slice<>{},sp::slice<>{1});
    BOOST_CHECK_EQUAL( sps1[0], ( sp::slice<>{0,0,1} ) );
    BOOST_CHECK_EQUAL( sps1[1], ( sp::slice<>{1,1,1} ) );

    sps1 = ub::detail::generate_span_array(dextents[1],sp::slice<>{-1},sp::slice<>{0,-1},sp::slice<>{});
    BOOST_CHECK_EQUAL( sps1[0], ( sp::slice<>{1,1,1} ) );
    BOOST_CHECK_EQUAL( sps1[1], ( sp::slice<>{0,2,1} ) );
    BOOST_CHECK_EQUAL( sps1[2], ( sp::slice<>{0,0,1} ) );

    sps1 = ub::detail::generate_span_array(dextents[2],sp::slice<>{-3,-1},sp::slice<>{0,-1, 2},sp::slice<>{},2);
    BOOST_CHECK_EQUAL( sps1[0], ( sp::slice<>{1,3,1} ) );
    BOOST_CHECK_EQUAL( sps1[1], ( sp::slice<>{0,0,2} ) );
    BOOST_CHECK_EQUAL( sps1[2], ( sp::slice<>{0,0,1} ) );
    BOOST_CHECK_EQUAL( sps1[3], ( sp::slice<>{2,2,1} ) );

  }

  {
    auto sps1 = ub::detail::generate_span_array( sp::detail::get<0>(sextents),sp::slice<>{},sp::slice<1>{});
    BOOST_CHECK( ( sp::detail::get<0> (sps1) == sp::slice<0,0,1>{} ) );
    BOOST_CHECK( ( sp::detail::get<1>(sps1) == sp::slice<1,1,1>{} ) );

    auto sps2 = ub::detail::generate_span_array( sp::detail::get<1>(sextents) ,sp::slice<-1>{},sp::slice<0,-1>{},sp::slice<>{});
    BOOST_CHECK( ( sp::detail::get<0> (sps2) == sp::slice<1,1,1>{} ) );
    BOOST_CHECK( ( sp::detail::get<1> (sps2) == sp::slice<0,2,1>{} ) );
    BOOST_CHECK( ( sp::detail::get<2> (sps2) == sp::slice<0,0,1>{} ) );

    auto sps3 = ub::detail::generate_span_array(sp::detail::get<2>(sextents) , sp::slice<-3,-1>{},sp::slice<0,-1, 2>{},sp::slice<>{},sp::slice<2>{});
    BOOST_CHECK( ( sp::detail::get<0> (sps3) == sp::slice<1,3,1>{} ) );
    BOOST_CHECK( ( sp::detail::get<1> (sps3) == sp::slice<0,0,2>{} ) );
    BOOST_CHECK( ( sp::detail::get<2> (sps3) == sp::slice<0,0,1>{} ) );
    BOOST_CHECK( ( sp::detail::get<3> (sps3) == sp::slice<2,2,1>{} ) );

  }
}


BOOST_FIXTURE_TEST_CASE(test_slice_span_strides, fixture)
{
  {
    auto sps1 = ub::detail::generate_span_array(dextents[0],sp::slice<>{},sp::slice<>{1});
    auto s1 = ub::detail::span_strides(dstrides[0],sps1);

    BOOST_CHECK_EQUAL( s1[0], dstrides[0][0] );
    BOOST_CHECK_EQUAL( s1[1], dstrides[0][1] );


    sps1 = ub::detail::generate_span_array(dextents[1],sp::slice<>{-1},sp::slice<>{0,-1},sp::slice<>{});
    s1 = ub::detail::span_strides(dstrides[1],sps1);

    BOOST_CHECK_EQUAL( s1[0], dstrides[1][0] );
    BOOST_CHECK_EQUAL( s1[1], dstrides[1][1] );
    BOOST_CHECK_EQUAL( s1[2], dstrides[1][2] );


    sps1 = ub::detail::generate_span_array(dextents[2],sp::slice<>{-3,-1,3},sp::slice<>{0,-1, 2},sp::slice<>{},2);
    s1 = ub::detail::span_strides(dstrides[2],sps1);
    
    BOOST_CHECK_EQUAL( s1[0], dstrides[2][0] * 3 );
    BOOST_CHECK_EQUAL( s1[1], dstrides[2][1] * 2 );
    BOOST_CHECK_EQUAL( s1[2], dstrides[2][2] );
    BOOST_CHECK_EQUAL( s1[3], dstrides[2][3] );

  }

  {
    auto sps1 = ub::detail::generate_span_array( ub::detail::get<0>(sextents),sp::slice<>{},sp::slice<1>{});
    BOOST_REQUIRE_EQUAL( sps1.size(), 2);

    auto s1 = ub::detail::span_strides( ub::detail::get<0>(sstrides),sps1);
    BOOST_CHECK( ( s1[0] == ub::detail::get<0>(sstrides)[0] ) );
    BOOST_CHECK( ( s1[1] == ub::detail::get<0>(sstrides)[1] ) );


    auto sps2 = ub::detail::generate_span_array( ub::detail::get<1>(sextents),sp::slice<-1>{},sp::slice<0,-1>{},sp::slice<>{});
    BOOST_REQUIRE_EQUAL( sps2.size(), 3);
    
    auto s2 = ub::detail::span_strides( ub::detail::get<1>(sstrides),sps2);
    BOOST_CHECK( ( s2[0] == ub::detail::get<1>(sstrides)[0] ) );
    BOOST_CHECK( ( s2[1] == ub::detail::get<1>(sstrides)[1] ) );
    BOOST_CHECK( ( s2[2] == ub::detail::get<1>(sstrides)[2] ) );

    auto sps3 = ub::detail::generate_span_array( ub::detail::get<2>(sextents),sp::slice<-3,-1,3>{},sp::slice<0,-1, 2>{},sp::slice<>{},sp::slice<2>{});
    BOOST_REQUIRE_EQUAL( sps3.size(), 4);
    
    auto s3 = ub::detail::span_strides( ub::detail::get<2>(sstrides),sps3);
    BOOST_CHECK( ( s3[0] == ub::detail::get<2>(sstrides)[0]  * 3 ) );
    BOOST_CHECK( ( s3[1] == ub::detail::get<2>(sstrides)[1]  * 2 ) );
    BOOST_CHECK( ( s3[2] == ub::detail::get<2>(sstrides)[2] ) );
    BOOST_CHECK( ( s3[3] == ub::detail::get<2>(sstrides)[3] ) );

  }
}


BOOST_FIXTURE_TEST_CASE(test_slice_transform_span, fixture)
{
  {
    
    std::vector<sp::slice<>> spans = {
      sp::slice<>{},      //1
      sp::slice<>{2},     //2
      sp::slice<>{-2},    //3
      sp::slice<>{2,6},   //4
      sp::slice<>{-6,-2}, //5
      sp::slice<>{0,-2},  //6
      sp::slice<>{2,6,1}, //7
      sp::slice<>{2,6,2}, //8
      sp::slice<>{2,9,3}, //9
    };

    auto s1 = ub::detail::transform_span(spans[0],10l);
    BOOST_CHECK( (s1 == sp::slice<>{0,9,1}) );

    auto s2 = ub::detail::transform_span(spans[1],10l);
    BOOST_CHECK( (s2 == sp::slice<>{2,2,1}) );

    auto s3 = ub::detail::transform_span(spans[2],10l);
    BOOST_CHECK( (s3 == sp::slice<>{8,8,1}) );

    auto s4 = ub::detail::transform_span(spans[3],10l);
    BOOST_CHECK( (s4 == sp::slice<>{2,6}) );

    auto s5 = ub::detail::transform_span(spans[4],10l);
    BOOST_CHECK( (s5 == sp::slice<>{4,8}) );

    auto s6 = ub::detail::transform_span(spans[5],10l);
    BOOST_CHECK( (s6 == sp::slice<>{0,8}) );

    auto s7 = ub::detail::transform_span(spans[6],10l);
    BOOST_CHECK( (s7 == sp::slice<>{2,6,1}) );

    auto s8 = ub::detail::transform_span(spans[7],10l);
    BOOST_CHECK( (s8 == sp::slice<>{2,6,2}) );

    auto s9 = ub::detail::transform_span(spans[8],10l);
    BOOST_CHECK( (s9 == sp::slice<>{2,9,3}) );

  }

  {
    
    sp::detail::list <
      sp::slice<>,      //1
      sp::slice<2>,     //2
      sp::slice<-2>,    //3
      sp::slice<2,6>,   //4
      sp::slice<-6,-2>, //5
      sp::slice<0,-2>,  //6
      sp::slice<2,6,1>, //7
      sp::slice<2,6,2>, //8
      sp::slice<2,9,3> //9
    > spans;

    auto s1 = ub::detail::transform_span<10>(sp::detail::get<0>( spans ));
    BOOST_CHECK( (s1 == sp::slice<0,9,1>{}) );

    auto s2 = ub::detail::transform_span<10>(sp::detail::get<1>( spans ));
    BOOST_CHECK( (s2 == sp::slice<2,2,1>{}) );

    auto s3 = ub::detail::transform_span<10>(sp::detail::get<2>( spans ));
    BOOST_CHECK( (s3 == sp::slice<8,8,1>{}) );

    auto s4 = ub::detail::transform_span<10>(sp::detail::get<3>( spans ));
    BOOST_CHECK( (s4 == sp::slice<2,6>{}) );

    auto s5 = ub::detail::transform_span<10>(sp::detail::get<4>( spans ));
    BOOST_CHECK( (s5 == sp::slice<4,8>{}) );

    auto s6 = ub::detail::transform_span<10>(sp::detail::get<5>( spans ));
    BOOST_CHECK( (s6 == sp::slice<0,8>{}) );

    auto s7 = ub::detail::transform_span<10>(sp::detail::get<6>( spans ));
    BOOST_CHECK( (s7 == sp::slice<2,6,1>{}) );

    auto s8 = ub::detail::transform_span<10>(sp::detail::get<7>( spans ));
    BOOST_CHECK( (s8 == sp::slice<2,6,2>{}) );

    auto s9 = ub::detail::transform_span<10>(sp::detail::get<8>( spans ));
    BOOST_CHECK( (s9 == sp::slice<2,9,3>{}) );

  }
}


BOOST_FIXTURE_TEST_CASE(test_slice_extents, fixture)
{
  {
    auto sps1 = ub::detail::generate_span_array(dextents[0],sp::slice<>{},sp::slice<>{1});
    auto s1 = ub::detail::extents(sps1);

    BOOST_CHECK_EQUAL( s1[0], 1 );
    BOOST_CHECK_EQUAL( s1[1], 1 );


    sps1 = ub::detail::generate_span_array(dextents[1],sp::slice<>{-1},sp::slice<>{0,-1},sp::slice<>{});
    s1 = ub::detail::extents(sps1);

    BOOST_CHECK_EQUAL( s1[0], 1 );
    BOOST_CHECK_EQUAL( s1[1], 3 );
    BOOST_CHECK_EQUAL( s1[2], 1 );


    sps1 = ub::detail::generate_span_array(dextents[2],sp::slice<>{-3,-1,3},sp::slice<>{0,-1, 2},sp::slice<>{},2);
    s1 = ub::detail::extents(sps1);
    
    BOOST_CHECK_EQUAL( s1[0], 1 );
    BOOST_CHECK_EQUAL( s1[1], 1 );
    BOOST_CHECK_EQUAL( s1[2], 1 );
    BOOST_CHECK_EQUAL( s1[3], 1 );

  }

  {
    auto sps1 = ub::detail::generate_span_array( ub::detail::get<0>(sextents),sp::slice<>{},sp::slice<1>{});
    BOOST_REQUIRE_EQUAL( sps1.size(), 2);

    auto s1 = ub::detail::extents(sps1);
    BOOST_CHECK( ( s1[0] == 1 ) );
    BOOST_CHECK( ( s1[1] == 1 ) );


    auto sps2 = ub::detail::generate_span_array( ub::detail::get<1>(sextents),sp::slice<-1>{},sp::slice<0,-1>{},sp::slice<>{});
    BOOST_REQUIRE_EQUAL( sps2.size(), 3);
    
    auto s2 = ub::detail::extents(sps2);
    BOOST_CHECK( ( s2[0] == 1 ) );
    BOOST_CHECK( ( s2[1] == 3 ) );
    BOOST_CHECK( ( s2[2] == 1 ) );

    auto sps3 = ub::detail::generate_span_array( ub::detail::get<2>(sextents),sp::slice<-3,-1,3>{},sp::slice<0,-1, 2>{},sp::slice<>{},sp::slice<2>{});
    BOOST_REQUIRE_EQUAL( sps3.size(), 4);
    
    auto s3 = ub::detail::extents(sps3);
    BOOST_CHECK( ( s3[0] == 1 ) );
    BOOST_CHECK( ( s3[1] == 1 ) );
    BOOST_CHECK( ( s3[2] == 1 ) );
    BOOST_CHECK( ( s3[3] == 1 ) );

  }
}


BOOST_FIXTURE_TEST_CASE(test_slice_offset, fixture)
{
  {
    auto sps1 = ub::detail::generate_span_array(dextents[0],sp::slice<>{},sp::slice<>{1});
    auto s1 = ub::detail::offset(dstrides[0],sps1);

    BOOST_CHECK_EQUAL( s1, 1 );


    sps1 = ub::detail::generate_span_array(dextents[1],sp::slice<>{-1},sp::slice<>{0,-1},sp::slice<>{});
    s1 = ub::detail::offset(dstrides[1],sps1);

    BOOST_CHECK_EQUAL( s1, 1 );


    sps1 = ub::detail::generate_span_array(dextents[2],sp::slice<>{-3,-1,3},sp::slice<>{0,-1, 2},sp::slice<>{},2);
    s1 = ub::detail::offset(dstrides[2],sps1);
    
    BOOST_CHECK_EQUAL( s1, 17 );

  }

  {
    auto sps1 = ub::detail::generate_span_array( ub::detail::get<0>(sextents),sp::slice<>{},sp::slice<1>{});
    BOOST_REQUIRE_EQUAL( sps1.size(), 2);

    auto s1 = ub::detail::offset( sp::detail::get<0>(sstrides) , sps1);
    BOOST_CHECK_EQUAL(s1,1);

    auto sps2 = ub::detail::generate_span_array( ub::detail::get<1>(sextents),sp::slice<-1>{},sp::slice<0,-1>{},sp::slice<>{});
    BOOST_REQUIRE_EQUAL( sps2.size(), 3);
    
    auto s2 = ub::detail::offset( sp::detail::get<1>(sstrides) , sps2);
    BOOST_CHECK_EQUAL(s2,1);

    auto sps3 = ub::detail::generate_span_array( ub::detail::get<2>(sextents),sp::slice<-3,-1,3>{},sp::slice<0,-1, 2>{},sp::slice<>{},sp::slice<2>{});
    BOOST_REQUIRE_EQUAL( sps3.size(), 4);
    
    auto s3 = ub::detail::offset( sp::detail::get<2>(sstrides) , sps3);
    BOOST_CHECK_EQUAL(s3,17);

  }
}