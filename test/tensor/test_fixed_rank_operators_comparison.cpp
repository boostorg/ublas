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



#include <boost/numeric/ublas/tensor/operators_comparison.hpp>
#include <boost/numeric/ublas/tensor/operators_arithmetic.hpp>
#include <boost/numeric/ublas/tensor/tensor.hpp>
#include <boost/test/unit_test.hpp>
#include <boost/multiprecision/cpp_bin_float.hpp>
#include "utility.hpp"

BOOST_AUTO_TEST_SUITE(test_fixed_rank_tensor_comparison)

using double_extended = boost::multiprecision::cpp_bin_float_double_extended;

using test_types = zip<int,float,double_extended>::with_t<boost::numeric::ublas::first_order, boost::numeric::ublas::last_order>;

struct fixture {
	template<size_t N>
	using extents_type = boost::numeric::ublas::dynamic_extents<N>;

	std::tuple<
		extents_type<2>, // 1
		extents_type<2>, // 2
		extents_type<3>, // 3
		extents_type<3>, // 4
		extents_type<4>  // 5
	> extents = {  
	    extents_type<2>{1,1},  
	    extents_type<2>{2,3}, 
	    extents_type<3>{4,1,3},
	    extents_type<3>{4,2,3},
	    extents_type<4>{4,2,3,5}
	};
};


BOOST_FIXTURE_TEST_CASE_TEMPLATE( test_tensor_comparison, value,  test_types, fixture)
{
	using namespace boost::numeric;
	using value_type  = typename value::first_type;
	using layout_type = typename value::second_type;

	auto check = [](auto const&, auto& e)
	{	
		using extents_type = std::decay_t<decltype(e)>;
		using tensor_type = ublas::tensor<value_type, extents_type, layout_type>;
		auto t  = tensor_type (e);
		auto t2 = tensor_type (e);
		auto v  = value_type  {};

		std::iota(t.begin(), t.end(), v);
		std::iota(t2.begin(), t2.end(), v+2);

		BOOST_CHECK( t == t  );
		BOOST_CHECK( t != t2 );

		if(t.empty())
			return;

		BOOST_CHECK(!(t < t));
		BOOST_CHECK(!(t > t));
		BOOST_CHECK( t < t2 );
		BOOST_CHECK( t2 > t );
		BOOST_CHECK( t <= t );
		BOOST_CHECK( t >= t );
		BOOST_CHECK( t <= t2 );
		BOOST_CHECK( t2 >= t );
		BOOST_CHECK( t2 >= t2 );
		BOOST_CHECK( t2 >= t );
	};

	for_each_tuple(extents,check);

}


BOOST_FIXTURE_TEST_CASE_TEMPLATE( test_tensor_comparison_with_tensor_expressions, value,  test_types, fixture)
{
	using namespace boost::numeric;
	using value_type  = typename value::first_type;
	using layout_type = typename value::second_type;


	auto check = [](auto const&, auto& e)
	{	
		using extents_type = std::decay_t<decltype(e)>;
		using tensor_type = ublas::tensor<value_type, extents_type, layout_type>;

		auto t  = tensor_type (e);
		auto t2 = tensor_type (e);
		auto v  = value_type  {};

		std::iota(t.begin(), t.end(), v);
		std::iota(t2.begin(), t2.end(), v+2);

		BOOST_CHECK( t == t  );
		BOOST_CHECK( t != t2 );

		if(t.empty())
			return;

		BOOST_CHECK( !(t < t) );
		BOOST_CHECK( !(t > t) );
		BOOST_CHECK( t < (t2+t) );
		BOOST_CHECK( (t2+t) > t );
		BOOST_CHECK( t <= (t+t) );
		BOOST_CHECK( (t+t2) >= t );
		BOOST_CHECK( (t2+t2+2) >= t);
		BOOST_CHECK( 2*t2 > t );
		BOOST_CHECK( t < 2*t2 );
		BOOST_CHECK( 2*t2 > t);
		BOOST_CHECK( 2*t2 >= t2 );
		BOOST_CHECK( t2 <= 2*t2);
		BOOST_CHECK( 3*t2 >= t );

	};

	for_each_tuple(extents,check);

}



BOOST_FIXTURE_TEST_CASE_TEMPLATE( test_tensor_comparison_with_scalar, value,  test_types, fixture)
{
	using namespace boost::numeric;
	using value_type  = typename value::first_type;
	using layout_type = typename value::second_type;


	auto check = [](auto const&, auto& e)
	{	
		using extents_type = std::decay_t<decltype(e)>;
		using tensor_type = ublas::tensor<value_type, extents_type, layout_type>;

		BOOST_CHECK( tensor_type(e,value_type{2}) == tensor_type(e,value_type{2})  );
		BOOST_CHECK( tensor_type(e,value_type{2}) != tensor_type(e,value_type{1})  );

		if(e.empty())
			return;

		BOOST_CHECK( !(tensor_type(e,2) <  2) );
		BOOST_CHECK( !(tensor_type(e,2) >  2) );
		BOOST_CHECK(  (tensor_type(e,2) >= 2) );
		BOOST_CHECK(  (tensor_type(e,2) <= 2) );
		BOOST_CHECK(  (tensor_type(e,2) == 2) );
		BOOST_CHECK(  (tensor_type(e,2) != 3) );

		BOOST_CHECK( !(2 >  tensor_type(e,2)) );
		BOOST_CHECK( !(2 <  tensor_type(e,2)) );
		BOOST_CHECK(  (2 <= tensor_type(e,2)) );
		BOOST_CHECK(  (2 >= tensor_type(e,2)) );
		BOOST_CHECK(  (2 == tensor_type(e,2)) );
		BOOST_CHECK(  (3 != tensor_type(e,2)) );

		BOOST_CHECK( !( tensor_type(e,2)+3 <  5) );
		BOOST_CHECK( !( tensor_type(e,2)+3 >  5) );
		BOOST_CHECK(  ( tensor_type(e,2)+3 >= 5) );
		BOOST_CHECK(  ( tensor_type(e,2)+3 <= 5) );
		BOOST_CHECK(  ( tensor_type(e,2)+3 == 5) );
		BOOST_CHECK(  ( tensor_type(e,2)+3 != 6) );


		BOOST_CHECK( !( 5 >  tensor_type(e,2)+3) );
		BOOST_CHECK( !( 5 <  tensor_type(e,2)+3) );
		BOOST_CHECK(  ( 5 >= tensor_type(e,2)+3) );
		BOOST_CHECK(  ( 5 <= tensor_type(e,2)+3) );
		BOOST_CHECK(  ( 5 == tensor_type(e,2)+3) );
		BOOST_CHECK(  ( 6 != tensor_type(e,2)+3) );


		BOOST_CHECK( !( tensor_type(e,2)+tensor_type(e,3) <  5) );
		BOOST_CHECK( !( tensor_type(e,2)+tensor_type(e,3) >  5) );
		BOOST_CHECK(  ( tensor_type(e,2)+tensor_type(e,3) >= 5) );
		BOOST_CHECK(  ( tensor_type(e,2)+tensor_type(e,3) <= 5) );
		BOOST_CHECK(  ( tensor_type(e,2)+tensor_type(e,3) == 5) );
		BOOST_CHECK(  ( tensor_type(e,2)+tensor_type(e,3) != 6) );


		BOOST_CHECK( !( 5 >  tensor_type(e,2)+tensor_type(e,3)) );
		BOOST_CHECK( !( 5 <  tensor_type(e,2)+tensor_type(e,3)) );
		BOOST_CHECK(  ( 5 >= tensor_type(e,2)+tensor_type(e,3)) );
		BOOST_CHECK(  ( 5 <= tensor_type(e,2)+tensor_type(e,3)) );
		BOOST_CHECK(  ( 5 == tensor_type(e,2)+tensor_type(e,3)) );
		BOOST_CHECK(  ( 6 != tensor_type(e,2)+tensor_type(e,3)) );

	};

for_each_tuple(extents,check);

}


BOOST_AUTO_TEST_SUITE_END()
