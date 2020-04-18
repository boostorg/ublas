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


#include <iostream>
#include <random>
#include <boost/numeric/ublas/tensor.hpp>
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/test/unit_test.hpp>

#include "utility.hpp"



BOOST_AUTO_TEST_SUITE ( test_static_tensor_matrix_interoperability ) ;

using test_types = zip<int,float>::with_t<boost::numeric::ublas::first_order, boost::numeric::ublas::last_order>;


BOOST_AUTO_TEST_CASE_TEMPLATE( test_tensor_matrix_copy_ctor, value,  test_types)
{
	using namespace boost::numeric;
	using value_type  = typename value::first_type;
	using layout_type = typename value::second_type;
	using tensor_type  = ublas::tensor<value_type, ublas::dynamic_extents<>,layout_type>;
	using matrix_type = typename tensor_type::matrix_type;

	ublas::tensor<value_type, ublas::static_extents<1,1>,layout_type> a2 = matrix_type(1,1);
	BOOST_CHECK_EQUAL(  a2.size() , 1 );
	BOOST_CHECK( !a2.empty() );

	ublas::tensor<value_type, ublas::static_extents<2,1>,layout_type> a3 = matrix_type(2,1);
	BOOST_CHECK_EQUAL(  a3.size() , 2 );
	BOOST_CHECK( !a3.empty() );

	ublas::tensor<value_type, ublas::static_extents<1,2>,layout_type> a4 = matrix_type(1,2);
	BOOST_CHECK_EQUAL(  a4.size() , 2 );
	BOOST_CHECK( !a4.empty() );

	ublas::tensor<value_type, ublas::static_extents<2,3>,layout_type> a5 = matrix_type(2,3);
	BOOST_CHECK_EQUAL(  a5.size() , 6 );
	BOOST_CHECK( !a5.empty() );
}


BOOST_AUTO_TEST_CASE_TEMPLATE( test_tensor_vector_copy_ctor, value,  test_types)
{
	using namespace boost::numeric;
	using value_type  = typename value::first_type;
	using layout_type = typename value::second_type;
	using tensor_type  = ublas::tensor<value_type, ublas::dynamic_extents<>,layout_type>;
	using vector_type = typename tensor_type::vector_type;

	ublas::tensor<value_type, ublas::static_extents<1,1>,layout_type> a2 = vector_type(1);
	BOOST_CHECK_EQUAL(  a2.size() , 1 );
	BOOST_CHECK( !a2.empty() );

	ublas::tensor<value_type, ublas::static_extents<2,1>,layout_type> a3 = vector_type(2);
	BOOST_CHECK_EQUAL(  a3.size() , 2 );
	BOOST_CHECK( !a3.empty() );

	ublas::tensor<value_type, ublas::static_extents<2,1>,layout_type> a4 = vector_type(2);
	BOOST_CHECK_EQUAL(  a4.size() , 2 );
	BOOST_CHECK( !a4.empty() );

	ublas::tensor<value_type, ublas::static_extents<3,1>,layout_type> a5 = vector_type(3);
	BOOST_CHECK_EQUAL(  a5.size() , 3 );
	BOOST_CHECK( !a5.empty() );
}


struct fixture
{
	template<size_t... N>
	using extents_type = boost::numeric::ublas::static_extents<N...>;

	fixture() {}

	std::tuple<
		extents_type<1,1>, // 0
		extents_type<2,3>, // 1
		extents_type<9,7>, // 2
		extents_type<15,17> // 3
	> extents;;
};




BOOST_FIXTURE_TEST_CASE_TEMPLATE( test_tensor_matrix_copy_ctor_extents, value,  test_types, fixture )
{
	using namespace boost::numeric;
	using value_type  = typename value::first_type;
	using layout_type = typename value::second_type;
	using tensor_type  = ublas::tensor<value_type, ublas::dynamic_extents<>,layout_type>;
	using matrix_type = typename tensor_type::matrix_type;

	auto check = [](auto const&, auto& e) {
		using extents_type = std::decay_t<decltype(e)>;
		using etensor_type = ublas::tensor<value_type, extents_type,layout_type>;

		assert(e.size()==2);
		etensor_type t = matrix_type{e[0],e[1]};
		BOOST_CHECK_EQUAL (  t.size() , product(e) );
		BOOST_CHECK_EQUAL (  t.rank() , e.size() );
		BOOST_CHECK       ( !t.empty()    );
	};

	for_each_tuple(extents,check);
}


BOOST_FIXTURE_TEST_CASE_TEMPLATE( test_tensor_vector_copy_ctor_extents, value,  test_types, fixture )
{
	using namespace boost::numeric;
	using value_type  = typename value::first_type;
	using layout_type = typename value::second_type;
	using tensor_type  = ublas::tensor<value_type, ublas::dynamic_extents<>,layout_type>;
	using vector_type = typename tensor_type::vector_type;

	auto check = [](auto const&, auto& e) {
		using extents_type = std::decay_t<decltype(e)>;
		using etensor_type = ublas::tensor<value_type, extents_type,layout_type>;

		if constexpr( extents_type::at(1) == 1 ){
			assert(e.size()==2);
			if(e.empty())
				return;

			etensor_type t = vector_type(product(e));
			BOOST_CHECK_EQUAL (  t.size() , product(e) );
			BOOST_CHECK_EQUAL (  t.rank() , e.size() );
			BOOST_CHECK       ( !t.empty()    );
		}

	};

	for_each_tuple(extents,check);
}



BOOST_FIXTURE_TEST_CASE_TEMPLATE( test_tensor_matrix_copy_assignment, value,  test_types, fixture )
{
	using namespace boost::numeric;
	using value_type  = typename value::first_type;
	using layout_type = typename value::second_type;
	using tensor_type  = ublas::tensor<value_type, ublas::dynamic_extents<>,layout_type>;
	using matrix_type = typename tensor_type::matrix_type;

	auto check = [](auto const&, auto& e) {
		using extents_type = std::decay_t<decltype(e)>;
		using etensor_type = ublas::tensor<value_type, extents_type,layout_type>;

		assert(e.size() == 2);
		auto t = etensor_type{};
		auto r = matrix_type(e[0],e[1]);
		std::iota(r.data().begin(),r.data().end(), 1);
		t = r;

		BOOST_CHECK_EQUAL (  t.extents().at(0) , e.at(0) );
		BOOST_CHECK_EQUAL (  t.extents().at(1) , e.at(1) );
		BOOST_CHECK_EQUAL (  t.size() , product(e) );
		BOOST_CHECK_EQUAL (  t.rank() , e.size() );
		BOOST_CHECK       ( !t.empty()    );

		for(auto j = 0ul; j < t.size(1); ++j){
			for(auto i = 0ul; i < t.size(0); ++i){
				BOOST_CHECK_EQUAL( t.at(i,j), r(i,j)  );
			}
		}
	};

	for_each_tuple(extents,check);
}


BOOST_FIXTURE_TEST_CASE_TEMPLATE( test_tensor_vector_copy_assignment, value,  test_types, fixture )
{
	using namespace boost::numeric;
	using value_type  = typename value::first_type;
	using layout_type = typename value::second_type;
	using tensor_type  = ublas::tensor<value_type, ublas::dynamic_extents<>,layout_type>;
	using vector_type = typename tensor_type::vector_type;

	auto check = [](auto const&, auto& e) {
		using extents_type = std::decay_t<decltype(e)>;
		using etensor_type = ublas::tensor<value_type, extents_type,layout_type>;

		assert(e.size() == 2);

		if constexpr( extents_type::at(1) == 1 ){
			auto t = etensor_type{};
			auto r = vector_type(e[0]*e[1]);
			std::iota(r.data().begin(),r.data().end(), 1);
			t = r;

			BOOST_CHECK_EQUAL (  t.extents().at(0) , e.at(0)*e.at(1) );
			BOOST_CHECK_EQUAL (  t.extents().at(1) , 1);
			BOOST_CHECK_EQUAL (  t.size() , product(e) );
			BOOST_CHECK_EQUAL (  t.rank() , e.size() );
			BOOST_CHECK       ( !t.empty()    );

			for(auto i = 0ul; i < t.size(); ++i){
				BOOST_CHECK_EQUAL( t[i], r(i)  );
			}
		}
	};

	for_each_tuple(extents,check);
}

BOOST_FIXTURE_TEST_CASE_TEMPLATE( test_tensor_matrix_move_assignment, value,  test_types, fixture )
{
	using namespace boost::numeric;
	using value_type  = typename value::first_type;
	using layout_type = typename value::second_type;
	using tensor_type  = ublas::tensor<value_type, ublas::dynamic_extents<>,layout_type>;
	using matrix_type = typename tensor_type::matrix_type;

	auto check = [](auto const&, auto& e) {
		using extents_type = std::decay_t<decltype(e)>;
		using etensor_type = ublas::tensor<value_type, extents_type,layout_type>;

		assert(e.size() == 2);
		auto t = etensor_type{};
		auto r = matrix_type(e[0],e[1]);
		std::iota(r.data().begin(),r.data().end(), 1);
		auto q = r;
		t = std::move(r);

		BOOST_CHECK_EQUAL (  t.extents().at(0) , e.at(0) );
		BOOST_CHECK_EQUAL (  t.extents().at(1) , e.at(1) );
		BOOST_CHECK_EQUAL (  t.size() , product(e) );
		BOOST_CHECK_EQUAL (  t.rank() , e.size() );
		BOOST_CHECK       ( !t.empty()    );

		for(auto j = 0ul; j < t.size(1); ++j){
			for(auto i = 0ul; i < t.size(0); ++i){
				BOOST_CHECK_EQUAL( t.at(i,j), q(i,j)  );
			}
		}
	};

	for_each_tuple(extents,check);
}




BOOST_FIXTURE_TEST_CASE_TEMPLATE( test_tensor_vector_move_assignment, value,  test_types, fixture )
{
	using namespace boost::numeric;
	using value_type  = typename value::first_type;
	using layout_type = typename value::second_type;
	using tensor_type  = ublas::tensor<value_type, ublas::dynamic_extents<>,layout_type>;
	using vector_type = typename tensor_type::vector_type;

	auto check = [](auto const&, auto& e) {
		using extents_type = std::decay_t<decltype(e)>;
		using etensor_type = ublas::tensor<value_type, extents_type,layout_type>;

		assert(e.size() == 2);
		if constexpr( extents_type::at(1) == 1 ){
			auto t = etensor_type{};
			auto r = vector_type(e[0]*e[1]);
			std::iota(r.data().begin(),r.data().end(), 1);
			auto q = r;
			t = std::move(r);

			BOOST_CHECK_EQUAL (  t.extents().at(0) , e.at(0) * e.at(1));
			BOOST_CHECK_EQUAL (  t.extents().at(1) , 1);
			BOOST_CHECK_EQUAL (  t.size() , product(e) );
			BOOST_CHECK_EQUAL (  t.rank() , e.size() );
			BOOST_CHECK       ( !t.empty()    );

			for(auto i = 0ul; i < t.size(); ++i){
				BOOST_CHECK_EQUAL( t[i], q(i)  );
			}
		}
	};

	for_each_tuple(extents,check);
}





BOOST_FIXTURE_TEST_CASE_TEMPLATE( test_tensor_matrix_expressions, value,  test_types, fixture )
{
	using namespace boost::numeric;
	using value_type  = typename value::first_type;
	using layout_type = typename value::second_type;
	using tensor_type  = ublas::tensor<value_type, ublas::dynamic_extents<>,layout_type>;
	using matrix_type = typename tensor_type::matrix_type;

	auto check = [](auto const&, auto& e) {
		using extents_type = std::decay_t<decltype(e)>;
		using etensor_type = ublas::tensor<value_type, extents_type,layout_type>;

		assert(e.size() == 2);
		auto t = etensor_type{};
		auto r = matrix_type(e[0],e[1]);
		std::iota(r.data().begin(),r.data().end(), 1);
		t = r + 3*r;
		etensor_type s = r + 3*r;
		etensor_type q = s + r + 3*r + s; // + 3*r


		BOOST_CHECK_EQUAL (  t.extents().at(0) , e.at(0) );
		BOOST_CHECK_EQUAL (  t.extents().at(1) , e.at(1) );
		BOOST_CHECK_EQUAL (  t.size() , product(e) );
		BOOST_CHECK_EQUAL (  t.rank() , e.size() );
		BOOST_CHECK       ( !t.empty()    );

		BOOST_CHECK_EQUAL (  s.extents().at(0) , e.at(0) );
		BOOST_CHECK_EQUAL (  s.extents().at(1) , e.at(1) );
		BOOST_CHECK_EQUAL (  s.size() , product(e) );
		BOOST_CHECK_EQUAL (  s.rank() , e.size() );
		BOOST_CHECK       ( !s.empty()    );

		BOOST_CHECK_EQUAL (  q.extents().at(0) , e.at(0) );
		BOOST_CHECK_EQUAL (  q.extents().at(1) , e.at(1) );
		BOOST_CHECK_EQUAL (  q.size() , product(e) );
		BOOST_CHECK_EQUAL (  q.rank() , e.size() );
		BOOST_CHECK       ( !q.empty()    );


		for(auto j = 0ul; j < t.size(1); ++j){
			for(auto i = 0ul; i < t.size(0); ++i){
				BOOST_CHECK_EQUAL( t.at(i,j), 4*r(i,j)  );
				BOOST_CHECK_EQUAL( s.at(i,j), t.at(i,j)  );
				BOOST_CHECK_EQUAL( q.at(i,j), 3*s.at(i,j)  );
			}
		}
	};

	for_each_tuple(extents,check);
}






BOOST_FIXTURE_TEST_CASE_TEMPLATE( test_tensor_vector_expressions, value,  test_types, fixture )
{
	using namespace boost::numeric;
	using value_type  = typename value::first_type;
	using layout_type = typename value::second_type;
	using tensor_type  = ublas::tensor<value_type, ublas::dynamic_extents<>,layout_type>;
	using vector_type = typename tensor_type::vector_type;

	auto check = [](auto const&, auto& e) {
		using extents_type = std::decay_t<decltype(e)>;
		using etensor_type = ublas::tensor<value_type, extents_type,layout_type>;

		assert(e.size() == 2);
		if constexpr( extents_type::at(1) == 1 ){
			auto t = etensor_type{};
			auto r = vector_type(e[0]*e[1]);
			std::iota(r.data().begin(),r.data().end(), 1);
			t = r + 3*r;
			etensor_type s = r + 3*r;
			etensor_type q = s + r + 3*r + s; // + 3*r


			BOOST_CHECK_EQUAL (  t.extents().at(0) , e.at(0)*e.at(1) );
			BOOST_CHECK_EQUAL (  t.extents().at(1) , 1);
			BOOST_CHECK_EQUAL (  t.size() , product(e) );
			BOOST_CHECK_EQUAL (  t.rank() , e.size() );
			BOOST_CHECK       ( !t.empty()    );

			BOOST_CHECK_EQUAL (  s.extents().at(0) , e.at(0)*e.at(1) );
			BOOST_CHECK_EQUAL (  s.extents().at(1) , 1);
			BOOST_CHECK_EQUAL (  s.size() , product(e) );
			BOOST_CHECK_EQUAL (  s.rank() , e.size() );
			BOOST_CHECK       ( !s.empty()    );

			BOOST_CHECK_EQUAL (  q.extents().at(0) , e.at(0)*e.at(1) );
			BOOST_CHECK_EQUAL (  q.extents().at(1) , 1);
			BOOST_CHECK_EQUAL (  q.size() , product(e) );
			BOOST_CHECK_EQUAL (  q.rank() , e.size() );
			BOOST_CHECK       ( !q.empty()    );



			for(auto i = 0ul; i < t.size(); ++i){
				BOOST_CHECK_EQUAL( t.at(i), 4*r(i)  );
				BOOST_CHECK_EQUAL( s.at(i), t.at(i)  );
				BOOST_CHECK_EQUAL( q.at(i), 3*s.at(i)  );
			}
		}
	};

	for_each_tuple(extents,check);
}



BOOST_FIXTURE_TEST_CASE_TEMPLATE( test_tensor_matrix_vector_expressions, value,  test_types, fixture )
{
	using namespace boost::numeric;
	using value_type  = typename value::first_type;
	using layout_type = typename value::second_type;
	using tensor_type = ublas::tensor<value_type, ublas::dynamic_extents<>,layout_type>;
	using matrix_type = typename tensor_type::matrix_type;
	using vector_type = typename tensor_type::vector_type;

	auto check = [](auto const&, auto& e) {
		using extents_type = std::decay_t<decltype(e)>;

		if(product(e) <= 2)
			return;

		assert(e.size() == 2);
		auto Q = ublas::tensor<value_type, ublas::static_extents<extents_type::at(0),1>,layout_type>{} ;
		auto A = matrix_type(e[0],e[1]);
		auto b = vector_type(e[1]);
		auto c = vector_type(e[0]);
		std::iota(b.data().begin(),b.data().end(), 1);
		std::fill(A.data().begin(),A.data().end(), 1);
		std::fill(c.data().begin(),c.data().end(), 2);
		std::fill(Q.begin(),Q.end(), 2);

		decltype(Q) T = Q + (ublas::prod(A , b) + 2*c) + 3*Q;

		BOOST_CHECK_EQUAL (  T.extents().at(0) , Q.extents().at(0) );
		BOOST_CHECK_EQUAL (  T.extents().at(1) , Q.extents().at(1));
		BOOST_CHECK_EQUAL (  T.size() , Q.size() );
		BOOST_CHECK_EQUAL (  T.size() , c.size() );
		BOOST_CHECK_EQUAL (  T.rank() , Q.rank() );
		BOOST_CHECK       ( !T.empty()    );

		for(auto i = 0ul; i < T.size(); ++i){
			auto n = e[1];
			auto ab = n * (n+1) / 2;
			BOOST_CHECK_EQUAL( T(i), ab+4*Q(0)+2*c(0)  );
		}

	};



	for_each_tuple(extents,check);
}


BOOST_AUTO_TEST_SUITE_END()

