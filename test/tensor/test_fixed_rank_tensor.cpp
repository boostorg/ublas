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



#include <random>
#include <boost/numeric/ublas/tensor/tensor.hpp> 

#ifndef BOOST_TEST_DYN_LINK
#define BOOST_TEST_DYN_LINK 
#endif

#define BOOST_TEST_MODULE TestFixedRankedTensor

#include <boost/test/unit_test.hpp>
#include "utility.hpp"

BOOST_AUTO_TEST_SUITE ( test_fixed_rank_tensor )

using test_types = zip<int,float,std::complex<float>>::with_t<boost::numeric::ublas::first_order, boost::numeric::ublas::last_order>;


BOOST_AUTO_TEST_CASE_TEMPLATE( test_tensor_ctor, value,  test_types)
{
	using namespace boost::numeric;
	using value_type  = typename value::first_type;
	using layout_type = typename value::second_type;

	auto a1 = ublas::tensor<value_type, ublas::dynamic_extents<0>,layout_type>{};
	BOOST_CHECK_EQUAL( a1.size() , 0ul );
	BOOST_CHECK( a1.empty() );
	BOOST_CHECK_EQUAL( a1.data() , nullptr);

	auto a2 = ublas::tensor<value_type, ublas::dynamic_extents<2>,layout_type>{1,1};
	BOOST_CHECK_EQUAL(  a2.size() , 1 );
	BOOST_CHECK( !a2.empty() );
	BOOST_CHECK_NE(  a2.data() , nullptr);

	auto a3 = ublas::tensor<value_type, ublas::dynamic_extents<2>,layout_type>{2,1};
	BOOST_CHECK_EQUAL(  a3.size() , 2 );
	BOOST_CHECK( !a3.empty() );
	BOOST_CHECK_NE(  a3.data() , nullptr);

	auto a4 = ublas::tensor<value_type, ublas::dynamic_extents<2>,layout_type>{1,2};
	BOOST_CHECK_EQUAL(  a4.size() , 2 );
	BOOST_CHECK( !a4.empty() );
	BOOST_CHECK_NE(  a4.data() , nullptr);

	auto a5 = ublas::tensor<value_type, ublas::dynamic_extents<2>,layout_type>{2,1};
	BOOST_CHECK_EQUAL(  a5.size() , 2 );
	BOOST_CHECK( !a5.empty() );
	BOOST_CHECK_NE(  a5.data() , nullptr);

	auto a6 = ublas::tensor<value_type, ublas::dynamic_extents<3>,layout_type>{4,3,2};
	BOOST_CHECK_EQUAL(  a6.size() , 4*3*2 );
	BOOST_CHECK( !a6.empty() );
	BOOST_CHECK_NE(  a6.data() , nullptr);

	auto a7 = ublas::tensor<value_type, ublas::dynamic_extents<3>,layout_type>{4,1,2};
	BOOST_CHECK_EQUAL(  a7.size() , 4*1*2 );
	BOOST_CHECK( !a7.empty() );
	BOOST_CHECK_NE(  a7.data() , nullptr);

}


struct fixture
{
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


BOOST_FIXTURE_TEST_CASE_TEMPLATE( test_tensor_ctor_extents, value,  test_types, fixture )
{
	using namespace boost::numeric;
	using value_type  = typename value::first_type;
	using layout_type = typename value::second_type;

	for_each_tuple(extents, [](auto const&, auto& e){
		using extents_type = std::decay_t<decltype(e)>;
		auto t = ublas::tensor<value_type, extents_type, layout_type>{e};

		BOOST_CHECK_EQUAL (  t.size() , product(e) );
		BOOST_CHECK_EQUAL (  t.rank() , e.size() );
		if(e.empty()) {
			BOOST_CHECK       ( t.empty()    );
			BOOST_CHECK_EQUAL ( t.data() , nullptr);
		}
		else{
			BOOST_CHECK       ( !t.empty()    );
			BOOST_CHECK_NE    (  t.data() , nullptr);
		}
	});

}


BOOST_FIXTURE_TEST_CASE_TEMPLATE( test_tensor_copy_ctor, value,  test_types, fixture )
{
	using namespace boost::numeric;
	using value_type  = typename value::first_type;
	using layout_type = typename value::second_type;


	for_each_tuple(extents, [](auto const&, auto& e){
		using extents_type = std::decay_t<decltype(e)>;
		auto r = ublas::tensor<value_type, extents_type, layout_type>{e};

		auto t = r;
		BOOST_CHECK_EQUAL (  t.size() , r.size() );
		BOOST_CHECK_EQUAL (  t.rank() , r.rank() );
		BOOST_CHECK ( t.strides() == r.strides() );
		BOOST_CHECK ( t.extents() == r.extents() );

		if(e.empty()) {
			BOOST_CHECK       ( t.empty()    );
			BOOST_CHECK_EQUAL ( t.data() , nullptr);
		}
		else{
			BOOST_CHECK       ( !t.empty()    );
			BOOST_CHECK_NE    (  t.data() , nullptr);
		}

		for(auto i = 0ul; i < t.size(); ++i)
			BOOST_CHECK_EQUAL( t[i], r[i]  );

	});
}


BOOST_FIXTURE_TEST_CASE_TEMPLATE( test_tensor_copy_ctor_layout, value,  test_types, fixture )
{
	using namespace boost::numeric;
	using value_type  = typename value::first_type;
	using layout_type = typename value::second_type;
	using other_layout_type = std::conditional_t<std::is_same<ublas::first_order,layout_type>::value, ublas::last_order, ublas::first_order>;


	for_each_tuple(extents, [](auto const&, auto& e){
		using extents_type = std::decay_t<decltype(e)>;
		using tensor_type = ublas::tensor<value_type, extents_type, layout_type>;
		auto r = tensor_type{e};
		ublas::tensor<value_type, extents_type, other_layout_type> t = r;
		tensor_type q = t;

		BOOST_CHECK_EQUAL (  t.size() , r.size() );
		BOOST_CHECK_EQUAL (  t.rank() , r.rank() );
		BOOST_CHECK ( t.extents() == r.extents() );

		BOOST_CHECK_EQUAL (  q.size() , r.size() );
		BOOST_CHECK_EQUAL (  q.rank() , r.rank() );
		BOOST_CHECK ( q.strides() == r.strides() );
		BOOST_CHECK ( q.extents() == r.extents() );

		for(auto i = 0ul; i < t.size(); ++i)
			BOOST_CHECK_EQUAL( q[i], r[i]  );

	});
}


BOOST_FIXTURE_TEST_CASE_TEMPLATE( test_tensor_copy_move_ctor, value,  test_types, fixture )
{
	using namespace boost::numeric;
	using value_type  = typename value::first_type;
	using layout_type = typename value::second_type;

	auto check = [](auto const&, auto& e)
	{
		using extents_type = std::decay_t<decltype(e)>;
		using tensor_type = ublas::tensor<value_type, extents_type, layout_type>;
		auto r = tensor_type{e};
		auto t = std::move(r);
		BOOST_CHECK_EQUAL (  t.size() , product(e) );
		BOOST_CHECK_EQUAL (  t.rank() , e.size() );

		if(e.empty()) {
			BOOST_CHECK       ( t.empty()    );
			BOOST_CHECK_EQUAL ( t.data() , nullptr);
		}
		else{
			BOOST_CHECK       ( !t.empty()    );
			BOOST_CHECK_NE    (  t.data() , nullptr);
		}

	};

	for_each_tuple(extents,check);
}


BOOST_FIXTURE_TEST_CASE_TEMPLATE( test_tensor_ctor_extents_init, value,  test_types, fixture )
{
	using namespace boost::numeric;
	using value_type  = typename value::first_type;
	using layout_type = typename value::second_type;

	std::random_device device{};
	std::minstd_rand0 generator(device());

	using distribution_type = std::conditional_t<std::is_integral_v<value_type>, std::uniform_int_distribution<>, std::uniform_real_distribution<> >;
	auto distribution = distribution_type(1,6);

	for_each_tuple(extents, [&](auto const&, auto const& e){
		using extents_type = std::decay_t<decltype(e)>;
		using tensor_type = ublas::tensor<value_type, extents_type, layout_type>;
		
		auto r = value_type( static_cast< inner_type_t<value_type> >(distribution(generator)) );
		auto t = tensor_type{e,r};
		for(auto i = 0ul; i < t.size(); ++i)
			BOOST_CHECK_EQUAL( t[i], r );

	});
}



BOOST_FIXTURE_TEST_CASE_TEMPLATE( test_tensor_ctor_extents_array, value,  test_types, fixture)
{
	using namespace boost::numeric;
	using value_type  = typename value::first_type;
	using layout_type = typename value::second_type;

	for_each_tuple(extents, [](auto const&, auto& e){
		using extents_type = std::decay_t<decltype(e)>;
		using tensor_type = ublas::tensor<value_type, extents_type, layout_type>;
		using array_type  = typename tensor_type::array_type;
		
		auto a = array_type(product(e));
		auto v = value_type {};

		for(auto& aa : a){
			aa = v;
			v += value_type{1};
		}
		auto t = tensor_type{e, a};
		v = value_type{};

		for(auto i = 0ul; i < t.size(); ++i, v+=value_type{1})
			BOOST_CHECK_EQUAL( t[i], v);

	});
}



BOOST_FIXTURE_TEST_CASE_TEMPLATE( test_tensor_read_write_single_index_access, value,  test_types, fixture)
{
	using namespace boost::numeric;
	using value_type  = typename value::first_type;
	using layout_type = typename value::second_type;

	for_each_tuple(extents, [](auto const&, auto& e){
		using extents_type = std::decay_t<decltype(e)>;
		using tensor_type = ublas::tensor<value_type, extents_type, layout_type>;
		
		auto t = tensor_type{e};
		auto v = value_type {};
		for(auto i = 0ul; i < t.size(); ++i, v+=value_type{1}){
			t[i] = v;
			BOOST_CHECK_EQUAL( t[i], v );

			t(i) = v;
			BOOST_CHECK_EQUAL( t(i), v );
		}

	});
}



BOOST_FIXTURE_TEST_CASE_TEMPLATE( test_tensor_read_write_multi_index_access_at, value,  test_types, fixture)
{
	using namespace boost::numeric;
	using value_type  = typename value::first_type;
	using layout_type = typename value::second_type;

	auto check1 = [](const auto& t)
	{
		auto v = value_type{};
		for(auto k = 0ul; k < t.size(); ++k){
			BOOST_CHECK_EQUAL(t[k], v);
			v+=value_type{1};
		}
	};

	auto check2 = [](const auto& t)
	{
		std::array<unsigned,2> k;
		auto r = std::is_same<layout_type,ublas::first_order>::value ? 1 : 0;
		auto q = std::is_same<layout_type,ublas::last_order >::value ? 1 : 0;
		auto v = value_type{};
		for(k[r] = 0ul; k[r] < t.size(r); ++k[r]){
			for(k[q] = 0ul; k[q] < t.size(q); ++k[q]){
				BOOST_CHECK_EQUAL(t.at(k[0],k[1]), v);
				v+=value_type{1};
			}
		}
	};

	auto check3 = [](const auto& t)
	{
		std::array<unsigned,3> k;
		using op_type = std::conditional_t<std::is_same_v<layout_type,ublas::first_order>, std::minus<>, std::plus<>>;
		auto r = std::is_same_v<layout_type,ublas::first_order> ? 2 : 0;
		auto o = op_type{};
		auto v = value_type{};
		for(k[r] = 0ul; k[r] < t.size(r); ++k[r]){
			for(k[o(r,1)] = 0ul; k[o(r,1)] < t.size(o(r,1)); ++k[o(r,1)]){
				for(k[o(r,2)] = 0ul; k[o(r,2)] < t.size(o(r,2)); ++k[o(r,2)]){
					BOOST_CHECK_EQUAL(t.at(k[0],k[1],k[2]), v);
					v+=value_type{1};
				}
			}
		}
	};

	auto check4 = [](const auto& t)
	{
		std::array<unsigned,4> k;
		using op_type = std::conditional_t<std::is_same_v<layout_type,ublas::first_order>, std::minus<>, std::plus<>>;
		auto r = std::is_same_v<layout_type,ublas::first_order> ? 3 : 0;
		auto o = op_type{};
		auto v = value_type{};
		for(k[r] = 0ul; k[r] < t.size(r); ++k[r]){
			for(k[o(r,1)] = 0ul; k[o(r,1)] < t.size(o(r,1)); ++k[o(r,1)]){
				for(k[o(r,2)] = 0ul; k[o(r,2)] < t.size(o(r,2)); ++k[o(r,2)]){
					for(k[o(r,3)] = 0ul; k[o(r,3)] < t.size(o(r,3)); ++k[o(r,3)]){
						BOOST_CHECK_EQUAL(t.at(k[0],k[1],k[2],k[3]), v);
						v+=value_type{1};
					}
				}
			}
		}
	};

	auto check = [check1,check2,check3,check4](auto const&, auto const& e) {
		using extents_type = std::decay_t<decltype(e)>;
		using tensor_type = ublas::tensor<value_type, extents_type, layout_type>;
		auto t = tensor_type{e};
		auto v = value_type {};
		for(auto i = 0ul; i < t.size(); ++i){
			t[i] = v;
			v+=value_type{1};
		}

		if(t.rank() == 1) check1(t);
		else if(t.rank() == 2) check2(t);
		else if(t.rank() == 3) check3(t);
		else if(t.rank() == 4) check4(t);

	};

	for_each_tuple(extents,check);
}




BOOST_FIXTURE_TEST_CASE_TEMPLATE( test_tensor_reshape, value,  test_types, fixture)
{
	using namespace boost::numeric;
	using value_type  = typename value::first_type;
	using layout_type = typename value::second_type;

	for_each_tuple(extents,[&](auto const&, auto& efrom){
		
		using extents_type = std::decay_t<decltype(efrom)>;
		using tensor_type = ublas::tensor<value_type, extents_type, layout_type>;
		
		for_each_tuple(extents,[&](auto const&, auto& eto){
			
			if ( efrom.size() == eto.size() ){
				
				auto v = value_type {};
				v+=value_type{1};
				auto t = tensor_type{efrom, v};
				for(auto i = 0ul; i < t.size(); ++i)
					BOOST_CHECK_EQUAL( t[i], v );

				t.reshape(eto);
				for(auto i = 0ul; i < std::min(product(efrom),product(eto)); ++i)
					BOOST_CHECK_EQUAL( t[i], v );

				BOOST_CHECK_EQUAL (  t.size() , product(eto) );
				BOOST_CHECK_EQUAL (  t.rank() , eto.size() );
				BOOST_CHECK ( t.extents() == eto );

				if(efrom != eto){
					for(auto i = product(efrom); i < t.size(); ++i)
						BOOST_CHECK_EQUAL( t[i], value_type{} );
				}
			}
		});
	});
}




BOOST_FIXTURE_TEST_CASE_TEMPLATE( test_tensor_swap, value,  test_types, fixture)
{
	using namespace boost::numeric;
	using value_type  = typename value::first_type;
	using layout_type = typename value::second_type;
	
	for_each_tuple(extents,[&](auto const&, auto& e_t){
		
		using extents_type = std::decay_t<decltype(e_t)>;
		using tensor_type = ublas::tensor<value_type, extents_type, layout_type>;
		
		for_each_tuple(extents,[&](auto const&, auto& e_r){
				
			if ( e_t.size() == e_r.size() ){
				
				auto v = value_type {} + value_type{1};
				auto w = value_type {} + value_type{2};
				auto t = tensor_type{e_t, v};
				auto r = tensor_type{e_r, w};

				std::swap( r, t );

				for(auto i = 0ul; i < t.size(); ++i)
					BOOST_CHECK_EQUAL( t[i], w );

				BOOST_CHECK_EQUAL (  t.size() , product(e_r) );
				BOOST_CHECK_EQUAL (  t.rank() , e_r.size() );
				BOOST_CHECK ( t.extents() == e_r );

				for(auto i = 0ul; i < r.size(); ++i)
					BOOST_CHECK_EQUAL( r[i], v );

				BOOST_CHECK_EQUAL (  r.size() , product(e_t) );
				BOOST_CHECK_EQUAL (  r.rank() , e_t.size() );
				BOOST_CHECK ( r.extents() == e_t );

			}
		});
	});
}



BOOST_FIXTURE_TEST_CASE_TEMPLATE( test_tensor_standard_iterator, value,  test_types, fixture)
{
	using namespace boost::numeric;
	using value_type  = typename value::first_type;
	using layout_type = typename value::second_type;

	for_each_tuple(extents,[](auto const&, auto& e){
		using extents_type = std::decay_t<decltype(e)>;
		using tensor_type = ublas::tensor<value_type, extents_type, layout_type>;
		
		auto v = value_type {} + value_type{1};
		auto t = tensor_type{e, v};

		BOOST_CHECK_EQUAL( std::distance(t.begin(),  t.end ()), t.size()  );
		BOOST_CHECK_EQUAL( std::distance(t.rbegin(), t.rend()), t.size()  );

		BOOST_CHECK_EQUAL( std::distance(t.cbegin(),  t.cend ()), t.size() );
		BOOST_CHECK_EQUAL( std::distance(t.crbegin(), t.crend()), t.size() );

		if(t.size() > 0) {
			BOOST_CHECK(  t.data() ==  std::addressof( *t.begin () )  ) ;
			BOOST_CHECK(  t.data() ==  std::addressof( *t.cbegin() )  ) ;
		}
	});

}

BOOST_FIXTURE_TEST_CASE_TEMPLATE( test_tensor_throw, value, test_types, fixture)
{
  using namespace boost::numeric;
  using value_type  = typename value::first_type;
  using layout_type = typename value::second_type;
  using tensor_type = ublas::tensor<value_type, ublas::dynamic_extents<2>, layout_type>;

  std::vector<value_type> vec(30);
  BOOST_CHECK_THROW(tensor_type({5,5},vec), std::runtime_error);

  auto t = tensor_type{{5,5}};
  auto i = ublas::index::index_type<4>{};
  BOOST_CHECK_THROW(t.operator()(i,i,i), std::runtime_error);

}

BOOST_AUTO_TEST_SUITE_END()
