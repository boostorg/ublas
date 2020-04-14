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



#include <boost/numeric/ublas/tensor/expression_evaluation.hpp>
#include <boost/numeric/ublas/tensor/expression.hpp>
#include <boost/numeric/ublas/tensor/tensor.hpp>
#include "utility.hpp"
#include <boost/test/unit_test.hpp>

#include <functional>

BOOST_AUTO_TEST_SUITE(test_fixed_rank_tensor_expression);

using test_types = zip<int,float,std::complex<float>>::with_t<boost::numeric::ublas::first_order, boost::numeric::ublas::last_order>;


struct fixture
{
	template<size_t N>
	using extents_type = boost::numeric::ublas::dynamic_extents<N>;

	std::tuple<
		extents_type<0>, // 0
		extents_type<2>, // 1
		extents_type<2>, // 2
		extents_type<2>, // 3
		extents_type<2>, // 4
		extents_type<3>, // 5
		extents_type<3>, // 6
		extents_type<3>, // 7
		extents_type<3>, // 8
		extents_type<4>  // 9
	> extents = {
		extents_type<0>{},    
	    extents_type<2>{1,1}, 
	    extents_type<2>{1,2}, 
	    extents_type<2>{2,1}, 
	    extents_type<2>{2,3}, 
	    extents_type<3>{2,3,1},
	    extents_type<3>{4,1,3},
	    extents_type<3>{1,2,3},
	    extents_type<3>{4,2,3},
	    extents_type<4>{4,2,3,5}
	};
};


BOOST_FIXTURE_TEST_CASE_TEMPLATE( test_fixed_rank_tensor_expression_retrieve_extents, value,  test_types, fixture)
{
	using namespace boost::numeric;
	using value_type  = typename value::first_type;
	using layout_type = typename value::second_type;

	auto uplus1 = std::bind(  std::plus<value_type>{}, std::placeholders::_1, value_type(1) );
	auto uplus2 = std::bind(  std::plus<value_type>{}, value_type(2), std::placeholders::_2 );
	auto bplus  = std::plus <value_type>{};
	auto bminus = std::minus<value_type>{};

	for_each_tuple(extents, [&](auto const&, auto & e){
		using extents_type = std::decay_t<decltype(e)>;
		using tensor_type = ublas::tensor<value_type, extents_type, layout_type>;

		auto t = tensor_type(e);
		auto v = value_type{};
		for(auto& tt: t){ tt = v; v+=value_type{1}; }


		BOOST_CHECK( ublas::detail::retrieve_extents( t ) == e );

		// uexpr1 = t+1
		// uexpr2 = 2+t
		auto uexpr1 = ublas::detail::make_unary_tensor_expression<tensor_type>( t, uplus1 );
		auto uexpr2 = ublas::detail::make_unary_tensor_expression<tensor_type>( t, uplus2 );

		BOOST_CHECK( ublas::detail::retrieve_extents( uexpr1 ) == e );
		BOOST_CHECK( ublas::detail::retrieve_extents( uexpr2 ) == e );

		// bexpr_uexpr = (t+1) + (2+t)
		auto bexpr_uexpr = ublas::detail::make_binary_tensor_expression<tensor_type>( uexpr1, uexpr2, bplus );

		BOOST_CHECK( ublas::detail::retrieve_extents( bexpr_uexpr ) == e );


		// bexpr_bexpr_uexpr = ((t+1) + (2+t)) - t
		auto bexpr_bexpr_uexpr = ublas::detail::make_binary_tensor_expression<tensor_type>( bexpr_uexpr, t, bminus );

		BOOST_CHECK( ublas::detail::retrieve_extents( bexpr_bexpr_uexpr ) == e );

	});

	for_each_tuple(extents, [&](auto const& I, auto& e1){

		if ( I >= std::tuple_size_v<decltype(extents)> - 1 ){
			return;
		}
		
		using extents_type1 = std::decay_t<decltype(e1)>;
		using tensor_type1 = ublas::tensor<value_type, extents_type1, layout_type>;

		for_each_tuple(extents, [&](auto const& J, auto& e2){
			if( J != I + 1 ){
				return;
			}

			using extents_type2 = std::decay_t<decltype(e2)>;
			using tensor_type2 = ublas::tensor<value_type, extents_type2, layout_type>;

			auto v = value_type{};

			tensor_type1 t1(e1);
			for(auto& tt: t1){ tt = v; v+=value_type{1}; }

			tensor_type2 t2(e2);
			for(auto& tt: t2){ tt = v; v+=value_type{2}; }

			BOOST_CHECK( ublas::detail::retrieve_extents( t1 ) != ublas::detail::retrieve_extents( t2 ) );

			// uexpr1 = t1+1
			// uexpr2 = 2+t2
			auto uexpr1 = ublas::detail::make_unary_tensor_expression<tensor_type1>( t1, uplus1 );
			auto uexpr2 = ublas::detail::make_unary_tensor_expression<tensor_type2>( t2, uplus2 );

			BOOST_CHECK( ublas::detail::retrieve_extents( t1 )     == ublas::detail::retrieve_extents( uexpr1 ) );
			BOOST_CHECK( ublas::detail::retrieve_extents( t2 )     == ublas::detail::retrieve_extents( uexpr2 ) );
			BOOST_CHECK( ublas::detail::retrieve_extents( uexpr1 ) != ublas::detail::retrieve_extents( uexpr2 ) );

			if constexpr( extents_type1::_size == extents_type2::_size ){
				// bexpr_uexpr = (t1+1) + (2+t2)
				auto bexpr_uexpr = ublas::detail::make_binary_tensor_expression<tensor_type1>( uexpr1, uexpr2, bplus );

				BOOST_CHECK( ublas::detail::retrieve_extents( bexpr_uexpr ) == ublas::detail::retrieve_extents(t1) );


				// bexpr_bexpr_uexpr = ((t1+1) + (2+t2)) - t2
				auto bexpr_bexpr_uexpr1 = ublas::detail::make_binary_tensor_expression<tensor_type1>( bexpr_uexpr, t2, bminus );

				BOOST_CHECK( ublas::detail::retrieve_extents( bexpr_bexpr_uexpr1 ) == ublas::detail::retrieve_extents(t2) );


				// bexpr_bexpr_uexpr = t2 - ((t1+1) + (2+t2))
				auto bexpr_bexpr_uexpr2 = ublas::detail::make_binary_tensor_expression<tensor_type1>( t2, bexpr_uexpr, bminus );

				BOOST_CHECK( ublas::detail::retrieve_extents( bexpr_bexpr_uexpr2 ) == ublas::detail::retrieve_extents(t2) );
			}

		});
	});
}







BOOST_FIXTURE_TEST_CASE_TEMPLATE( test_fixed_rank_tensor_expression_all_extents_equal, value,  test_types, fixture)
{
	using namespace boost::numeric;
	using value_type  = typename value::first_type;
	using layout_type = typename value::second_type;

	auto uplus1 = std::bind(  std::plus<value_type>{}, std::placeholders::_1, value_type(1) );
	auto uplus2 = std::bind(  std::plus<value_type>{}, value_type(2), std::placeholders::_2 );
	auto bplus  = std::plus <value_type>{};
	auto bminus = std::minus<value_type>{};

	for_each_tuple(extents, [&](auto const&, auto& e){
		using extents_type = std::decay_t<decltype(e)>;
		using tensor_type = ublas::tensor<value_type, extents_type, layout_type>;

		auto t = tensor_type(e);
		auto v = value_type{};
		for(auto& tt: t){ tt = v; v+=value_type{1}; }


		BOOST_CHECK( ublas::detail::all_extents_equal( t , e ) );


		// uexpr1 = t+1
		// uexpr2 = 2+t
		auto uexpr1 = ublas::detail::make_unary_tensor_expression<tensor_type>( t, uplus1 );
		auto uexpr2 = ublas::detail::make_unary_tensor_expression<tensor_type>( t, uplus2 );

		BOOST_CHECK( ublas::detail::all_extents_equal( uexpr1, e ) );
		BOOST_CHECK( ublas::detail::all_extents_equal( uexpr2, e ) );

		// bexpr_uexpr = (t+1) + (2+t)
		auto bexpr_uexpr = ublas::detail::make_binary_tensor_expression<tensor_type>( uexpr1, uexpr2, bplus );

		BOOST_CHECK( ublas::detail::all_extents_equal( bexpr_uexpr, e ) );


		// bexpr_bexpr_uexpr = ((t+1) + (2+t)) - t
		auto bexpr_bexpr_uexpr = ublas::detail::make_binary_tensor_expression<tensor_type>( bexpr_uexpr, t, bminus );

		BOOST_CHECK( ublas::detail::all_extents_equal( bexpr_bexpr_uexpr , e ) );

	});


	for_each_tuple(extents, [&](auto const& I, auto& e1){

		if ( I >= std::tuple_size_v<decltype(extents)> - 1){
			return;
		}
		
		using extents_type1 = std::decay_t<decltype(e1)>;
		using tensor_type1 = ublas::tensor<value_type, extents_type1, layout_type>;

		for_each_tuple(extents, [&](auto const& J, auto& e2){
			if( J != I + 1 ){
				return;
			}

			using extents_type2 = std::decay_t<decltype(e2)>;
			using tensor_type2 = ublas::tensor<value_type, extents_type2, layout_type>;

			auto v = value_type{};

			tensor_type1 t1(e1);
			for(auto& tt: t1){ tt = v; v+=value_type{1}; }

			tensor_type2 t2(e2);
			for(auto& tt: t2){ tt = v; v+=value_type{2}; }

			BOOST_CHECK( ublas::detail::all_extents_equal( t1, ublas::detail::retrieve_extents(t1) ) );
			BOOST_CHECK( ublas::detail::all_extents_equal( t2, ublas::detail::retrieve_extents(t2) ) );

			// uexpr1 = t1+1
			// uexpr2 = 2+t2
			auto uexpr1 = ublas::detail::make_unary_tensor_expression<tensor_type1>( t1, uplus1 );
			auto uexpr2 = ublas::detail::make_unary_tensor_expression<tensor_type2>( t2, uplus2 );

			BOOST_CHECK( ublas::detail::all_extents_equal( uexpr1, ublas::detail::retrieve_extents(uexpr1) ) );
			BOOST_CHECK( ublas::detail::all_extents_equal( uexpr2, ublas::detail::retrieve_extents(uexpr2) ) );

			if constexpr( extents_type1::_size == extents_type2::_size ){
				// bexpr_uexpr = (t1+1) + (2+t2)
				auto bexpr_uexpr = ublas::detail::make_binary_tensor_expression<tensor_type1>( uexpr1, uexpr2, bplus );

				BOOST_CHECK( ! ublas::detail::all_extents_equal( bexpr_uexpr, ublas::detail::retrieve_extents( bexpr_uexpr  ) ) );

				// bexpr_bexpr_uexpr = ((t1+1) + (2+t2)) - t2
				auto bexpr_bexpr_uexpr1 = ublas::detail::make_binary_tensor_expression<tensor_type1>( bexpr_uexpr, t2, bminus );

				BOOST_CHECK( ! ublas::detail::all_extents_equal( bexpr_bexpr_uexpr1, ublas::detail::retrieve_extents( bexpr_bexpr_uexpr1  ) ) );

				// bexpr_bexpr_uexpr = t2 - ((t1+1) + (2+t2))
				auto bexpr_bexpr_uexpr2 = ublas::detail::make_binary_tensor_expression<tensor_type1>( t2, bexpr_uexpr, bminus );

				BOOST_CHECK( ! ublas::detail::all_extents_equal( bexpr_bexpr_uexpr2, ublas::detail::retrieve_extents( bexpr_bexpr_uexpr2  ) ) );


				// bexpr_uexpr2 = (t1+1) + t2
				auto bexpr_uexpr2 = ublas::detail::make_binary_tensor_expression<tensor_type1>( uexpr1, t2, bplus );
				BOOST_CHECK( ! ublas::detail::all_extents_equal( bexpr_uexpr2, ublas::detail::retrieve_extents( bexpr_uexpr2  ) ) );


				// bexpr_uexpr2 = ((t1+1) + t2) + t1
				auto bexpr_bexpr_uexpr3 = ublas::detail::make_binary_tensor_expression<tensor_type1>( bexpr_uexpr2, t1, bplus );
				BOOST_CHECK( ! ublas::detail::all_extents_equal( bexpr_bexpr_uexpr3, ublas::detail::retrieve_extents( bexpr_bexpr_uexpr3  ) ) );

				// bexpr_uexpr2 = t1 + (((t1+1) + t2) + t1)
				auto bexpr_bexpr_uexpr4 = ublas::detail::make_binary_tensor_expression<tensor_type1>( t1, bexpr_bexpr_uexpr3, bplus );
				BOOST_CHECK( ! ublas::detail::all_extents_equal( bexpr_bexpr_uexpr4, ublas::detail::retrieve_extents( bexpr_bexpr_uexpr4  ) ) );
			}

		});
	});

}

BOOST_AUTO_TEST_SUITE_END()
