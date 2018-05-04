//  Copyright (c) 2018 Cem Bassoy
//
//  Distributed under the Boost Software License, Version 1.0. (See
//  accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#include <random>
#include <boost/numeric/ublas/tensor.hpp>

#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE TestTensor
#include <boost/test/unit_test.hpp>

BOOST_AUTO_TEST_SUITE ( test_tensor, * boost::unit_test::depends_on("test_extents") ) ;





template<class ... types>
struct zip_helper;

template<class type1, class ... types3>
struct zip_helper<std::tuple<types3...>, type1>
{
	template<class ... types2>
	struct with
	{
		using type = std::tuple<types3...,std::pair<type1,types2>...>;
	};
	template<class ... types2>
	using with_t = typename with<types2...>::type;
};


template<class type1, class ... types3, class ... types1>
struct zip_helper<std::tuple<types3...>, type1, types1...>
{
	template<class ... types2>
	struct with
	{
		using next_tuple = std::tuple<types3...,std::pair<type1,types2>...>;
		using type       = typename zip_helper<next_tuple, types1...>::template with<types2...>::type;
	};

	template<class ... types2>
	using with_t = typename with<types2...>::type;
};

template<class ... types>
using zip = zip_helper<std::tuple<>,types...>;

using test_types = zip<int,long,float,double>::with_t<boost::numeric::ublas::first_order, boost::numeric::ublas::last_order>;

// creates e.g.
//using test_types =
//std::tuple<
//std::pair<float, boost::numeric::ublas::first_order>,
//std::pair<float, boost::numeric::ublas::last_order >,
//std::pair<double,boost::numeric::ublas::first_order>,
//std::pair<double,boost::numeric::ublas::last_order >
//>;
//static_assert(std::is_same< std::tuple_element_t<0,std::tuple_element_t<0,test_types2>>, float>::value,"should be float ");
//static_assert(std::is_same< std::tuple_element_t<1,std::tuple_element_t<0,test_types2>>, boost::numeric::ublas::first_order>::value,"should be boost::numeric::ublas::first_order ");


BOOST_AUTO_TEST_CASE_TEMPLATE( test_tensor_ctor, value,  test_types)
{
	using namespace boost::numeric;
	using value_type  = typename value::first_type;
	using layout_type = typename value::second_type;
	using tensor_type = ublas::tensor<value_type, layout_type>;

	auto a1 = tensor_type{};
	BOOST_CHECK_EQUAL( a1.size() , 0ul );
	BOOST_CHECK( a1.empty() );
	BOOST_CHECK_EQUAL( a1.data() , nullptr);

	auto a2 = tensor_type{1,1};
	BOOST_CHECK_EQUAL(  a2.size() , 1 );
	BOOST_CHECK( !a2.empty() );
	BOOST_CHECK_NE(  a2.data() , nullptr);

	auto a3 = tensor_type{2,1};
	BOOST_CHECK_EQUAL(  a3.size() , 2 );
	BOOST_CHECK( !a3.empty() );
	BOOST_CHECK_NE(  a3.data() , nullptr);

	auto a4 = tensor_type{1,2};
	BOOST_CHECK_EQUAL(  a4.size() , 2 );
	BOOST_CHECK( !a4.empty() );
	BOOST_CHECK_NE(  a4.data() , nullptr);

	auto a5 = tensor_type{2,1};
	BOOST_CHECK_EQUAL(  a5.size() , 2 );
	BOOST_CHECK( !a5.empty() );
	BOOST_CHECK_NE(  a5.data() , nullptr);

	auto a6 = tensor_type{4,3,2};
	BOOST_CHECK_EQUAL(  a6.size() , 4*3*2 );
	BOOST_CHECK( !a6.empty() );
	BOOST_CHECK_NE(  a6.data() , nullptr);

	auto a7 = tensor_type{4,1,2};
	BOOST_CHECK_EQUAL(  a7.size() , 4*1*2 );
	BOOST_CHECK( !a7.empty() );
	BOOST_CHECK_NE(  a7.data() , nullptr);
}


struct fixture {
	using extents_type = boost::numeric::ublas::extents;
	fixture() : extents{
				extents_type{},    // 0
				extents_type{1,1}, // 1
				extents_type{1,2}, // 2
				extents_type{2,1}, // 3
				extents_type{2,3}, // 4
				extents_type{2,3,1}, // 5
				extents_type{4,1,3}, // 6
				extents_type{1,2,3}, // 7
				extents_type{4,2,3}, // 8
				extents_type{4,2,3,5} // 9
				}
	{}
	std::vector<extents_type> extents;
};


BOOST_FIXTURE_TEST_CASE_TEMPLATE( test_tensor_ctor_extents, value,  test_types, fixture )
{
	using namespace boost::numeric;
	using value_type  = typename value::first_type;
	using layout_type = typename value::second_type;
	using tensor_type = ublas::tensor<value_type, layout_type>;

	auto check = [](ublas::extents const& e) {
		auto t = tensor_type{e};
		BOOST_CHECK_EQUAL (  t.size() , e.product() );
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

	for(auto const& e : extents)
		check(e);
}


BOOST_FIXTURE_TEST_CASE_TEMPLATE( test_tensor_copy_ctor, value,  test_types, fixture )
{
	using namespace boost::numeric;
	using value_type  = typename value::first_type;
	using layout_type = typename value::second_type;
	using tensor_type = ublas::tensor<value_type, layout_type>;

	auto check = [](ublas::extents const& e)
	{
		auto r = tensor_type{e};
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
	};

	for(auto const& e : extents)
		check(e);
}


BOOST_FIXTURE_TEST_CASE_TEMPLATE( test_tensor_copy_move_ctor, value,  test_types, fixture )
{
	using namespace boost::numeric;
	using value_type  = typename value::first_type;
	using layout_type = typename value::second_type;
	using tensor_type = ublas::tensor<value_type, layout_type>;

	auto check = [](ublas::extents const& e)
	{
		auto r = tensor_type{e};
		auto t = std::move(r);
		BOOST_CHECK_EQUAL (  t.size() , e.product() );
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

	for(auto const& e : extents)
		check(e);
}


BOOST_FIXTURE_TEST_CASE_TEMPLATE( test_tensor_ctor_extents_init, value,  test_types, fixture )
{
	using namespace boost::numeric;
	using value_type  = typename value::first_type;
	using layout_type = typename value::second_type;
	using tensor_type = ublas::tensor<value_type, layout_type>;

	std::random_device device{};
	std::minstd_rand0 generator(device());

	using distribution_type = std::conditional_t<std::is_integral_v<value_type>, std::uniform_int_distribution<>, std::uniform_real_distribution<> >;
	auto distribution = distribution_type(1,6);

	auto check = [&distribution,&generator](ublas::extents const& e) {
		auto r = static_cast<value_type>(distribution(generator));
		auto t = tensor_type{e,r};
		for(auto i = 0ul; i < t.size(); ++i)
			BOOST_CHECK_EQUAL( t[i], r );
	};

	for(auto const& e : extents)
		check(e);
}



BOOST_FIXTURE_TEST_CASE_TEMPLATE( test_tensor_ctor_extents_array, value,  test_types, fixture)
{
	using namespace boost::numeric;
	using value_type  = typename value::first_type;
	using layout_type = typename value::second_type;
	using tensor_type = ublas::tensor<value_type, layout_type>;
	using array_type  = typename tensor_type::array_type;

	auto check = [](ublas::extents const& e) {
		auto a = array_type(e.product());
		std::iota(a.begin(), a.end(), value_type{});
		auto t = tensor_type{e, a};
		auto v = value_type {};
		for(auto i = 0ul; i < t.size(); ++i, ++v)
			BOOST_CHECK_EQUAL( t[i], v);
	};

	for(auto const& e : extents)
		check(e);
}



BOOST_FIXTURE_TEST_CASE_TEMPLATE( test_tensor_read_write_single_index_access, value,  test_types, fixture)
{
	using namespace boost::numeric;
	using value_type  = typename value::first_type;
	using layout_type = typename value::second_type;
	using tensor_type = ublas::tensor<value_type, layout_type>;

	auto check = [](ublas::extents const& e) {
		auto t = tensor_type{e};
		auto v = value_type {};
		for(auto i = 0ul; i < t.size(); ++i, ++v){
			t[i] = v;
			BOOST_CHECK_EQUAL( t[i], v );
		}
	};

	for(auto const& e : extents)
		check(e);
}



BOOST_FIXTURE_TEST_CASE_TEMPLATE( test_tensor_read_write_multi_index_access_at, value,  test_types, fixture)
{
	using namespace boost::numeric;
	using value_type  = typename value::first_type;
	using layout_type = typename value::second_type;
	using tensor_type = ublas::tensor<value_type, layout_type>;

	auto check1 = [](const tensor_type& t)
	{
		auto v = value_type{};
		for(auto k = 0ul; k < t.size(); ++k)
			BOOST_CHECK_EQUAL(t[k], v++);
	};

	auto check2 = [](const tensor_type& t)
	{
		std::array<unsigned,2> k;
		auto r = std::is_same_v<layout_type,ublas::first_order> ? 1 : 0;
		auto q = std::is_same_v<layout_type,ublas::last_order > ? 1 : 0;
		auto v = value_type{};
		for(k[r] = 0ul; k[r] < t.size(r); ++k[r])
			for(k[q] = 0ul; k[q] < t.size(q); ++k[q])
				BOOST_CHECK_EQUAL(t.at(k[0],k[1]), v++);
	};

	auto check3 = [](const tensor_type& t)
	{
		std::array<unsigned,3> k;
		using op_type = std::conditional_t<std::is_same_v<layout_type,ublas::first_order>, std::minus<>, std::plus<>>;
		auto r = std::is_same_v<layout_type,ublas::first_order> ? 2 : 0;
		auto o = op_type{};
		auto v = value_type{};
		for(k[r] = 0ul; k[r] < t.size(r); ++k[r])
			for(k[o(r,1)] = 0ul; k[o(r,1)] < t.size(o(r,1)); ++k[o(r,1)])
				for(k[o(r,2)] = 0ul; k[o(r,2)] < t.size(o(r,2)); ++k[o(r,2)])
					BOOST_CHECK_EQUAL(t.at(k[0],k[1],k[2]), v++);
	};

	auto check4 = [](const tensor_type& t)
	{
		std::array<unsigned,4> k;
		using op_type = std::conditional_t<std::is_same_v<layout_type,ublas::first_order>, std::minus<>, std::plus<>>;
		auto r = std::is_same_v<layout_type,ublas::first_order> ? 3 : 0;
		auto o = op_type{};
		auto v = value_type{};
		for(k[r] = 0ul; k[r] < t.size(r); ++k[r])
			for(k[o(r,1)] = 0ul; k[o(r,1)] < t.size(o(r,1)); ++k[o(r,1)])
				for(k[o(r,2)] = 0ul; k[o(r,2)] < t.size(o(r,2)); ++k[o(r,2)])
					for(k[o(r,3)] = 0ul; k[o(r,3)] < t.size(o(r,3)); ++k[o(r,3)])
						BOOST_CHECK_EQUAL(t.at(k[0],k[1],k[2],k[3]), v++);
	};

	auto check = [check1,check2,check3,check4](ublas::extents const& e) {
		auto t = tensor_type{e};
		auto v = value_type {};
		for(auto i = 0ul; i < t.size(); ++i)
			t[i] = v++;

			 if(t.rank() == 1) check1(t);
		else if(t.rank() == 2) check2(t);
		else if(t.rank() == 3) check3(t);
		else if(t.rank() == 4) check4(t);

	};

	for(auto const& e : extents)
		check(e);
}




BOOST_FIXTURE_TEST_CASE_TEMPLATE( test_tensor_reshape, value,  test_types, fixture)
{
	using namespace boost::numeric;
	using value_type  = typename value::first_type;
	using layout_type = typename value::second_type;
	using tensor_type = ublas::tensor<value_type, layout_type>;

	auto check = [](ublas::extents const& efrom, ublas::extents const& eto)
	{
		auto v = value_type {};
		++v;
		auto t = tensor_type{efrom, v};
		for(auto i = 0ul; i < t.size(); ++i)
			BOOST_CHECK_EQUAL( t[i], v );

		t.reshape(eto);
		for(auto i = 0ul; i < std::min(efrom.product(),eto.product()); ++i)
			BOOST_CHECK_EQUAL( t[i], v );

		BOOST_CHECK_EQUAL (  t.size() , eto.product() );
		BOOST_CHECK_EQUAL (  t.rank() , eto.size() );
		BOOST_CHECK ( t.extents() == eto );

		if(efrom != eto){
			for(auto i = efrom.product(); i < t.size(); ++i)
				BOOST_CHECK_EQUAL( t[i], value_type{} );
		}
	};

	for(auto const& efrom : extents)
		for(auto const& eto : extents)
			check(efrom,eto);
}


BOOST_FIXTURE_TEST_CASE_TEMPLATE( test_tensor_swap, value,  test_types, fixture)
{
	using namespace boost::numeric;
	using value_type  = typename value::first_type;
	using layout_type = typename value::second_type;
	using tensor_type = ublas::tensor<value_type, layout_type>;

	auto check = [](ublas::extents const& e_t, ublas::extents const& e_r)
	{
		auto v = value_type {} + 1;
		auto w = value_type {} + 2;
		auto t = tensor_type{e_t, v};
		auto r = tensor_type{e_r, w};

		std::swap( r, t );

		for(auto i = 0ul; i < t.size(); ++i)
			BOOST_CHECK_EQUAL( t[i], w );

		BOOST_CHECK_EQUAL (  t.size() , e_r.product() );
		BOOST_CHECK_EQUAL (  t.rank() , e_r.size() );
		BOOST_CHECK ( t.extents() == e_r );

		for(auto i = 0ul; i < r.size(); ++i)
			BOOST_CHECK_EQUAL( r[i], v );

		BOOST_CHECK_EQUAL (  r.size() , e_t.product() );
		BOOST_CHECK_EQUAL (  r.rank() , e_t.size() );
		BOOST_CHECK ( r.extents() == e_t );


	};

	for(auto const& efrom : extents)
		for(auto const& eto : extents)
			check(efrom,eto);
}



BOOST_FIXTURE_TEST_CASE_TEMPLATE( test_tensor_standard_iterator, value,  test_types, fixture)
{
	using namespace boost::numeric;
	using value_type  = typename value::first_type;
	using layout_type = typename value::second_type;
	using tensor_type = ublas::tensor<value_type, layout_type>;
	using iterator_type = typename tensor_type::iterator;
	using const_iterator_type = typename tensor_type::const_iterator;

	auto check = [](ublas::extents const& e)
	{
		auto v = value_type {} + 1;
		auto t = tensor_type{e, v};

		BOOST_CHECK_EQUAL( std::distance(t.begin(),  t.end ()), t.size()  );
		BOOST_CHECK_EQUAL( std::distance(t.rbegin(), t.rend()), t.size()  );

		BOOST_CHECK_EQUAL( std::distance(t.cbegin(),  t.cend ()), t.size() );
		BOOST_CHECK_EQUAL( std::distance(t.crbegin(), t.crend()), t.size() );

		BOOST_CHECK(  iterator_type       ( t.data() ) ==  t.begin ()  ) ;
		BOOST_CHECK(  const_iterator_type ( t.data() ) ==  t.cbegin()  ) ;

		BOOST_CHECK(  iterator_type       ( t.data()+t.size() ) ==  t.end ()  ) ;
		BOOST_CHECK(  const_iterator_type ( t.data()+t.size() ) ==  t.cend()  ) ;

	};

	for(auto const& e : extents)
		check(e);
}

BOOST_AUTO_TEST_SUITE_END();
