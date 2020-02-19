//  Copyright (c) 2018-2019 Cem Bassoy
//
//  Distributed under the Boost Software License, Version 1.0. (See
//  accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)
//
//  The authors gratefully acknowledge the support of
//  Fraunhofer and Google in producing this work
//  which started as a Google Summer of Code project.
//
//  And we acknowledge the support from all contributors.


#include <iostream>
#include <algorithm>
#include <boost/numeric/ublas/tensor.hpp>
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/vector.hpp>

#include <boost/test/unit_test.hpp>

#include "utility.hpp"

BOOST_AUTO_TEST_SUITE ( test_tensor_functions, * boost::unit_test::depends_on("test_tensor_contraction") )
// BOOST_AUTO_TEST_SUITE ( test_tensor_functions)


using test_types = zip<int,long,float,double,std::complex<float>>::with_t<boost::numeric::ublas::first_order, boost::numeric::ublas::last_order>;

//using test_types = zip<int>::with_t<boost::numeric::ublas::first_order>;


struct fixture
{
	using extents_type = boost::numeric::ublas::dynamic_extents<>;
	template<ptrdiff_t R, ptrdiff_t... E>
	using static_extents_type = boost::numeric::ublas::shape<R,E...>;
	fixture()
	  : extents {
	      extents_type{1,1}, // 1
	      extents_type{1,2}, // 2
	      extents_type{2,1}, // 3
	      extents_type{2,3}, // 4
	      extents_type{2,3,1}, // 5
	      extents_type{4,1,3}, // 6
	      extents_type{1,2,3}, // 7
	      extents_type{4,2,3}, // 8
	      extents_type{4,2,3,5}} // 9
	{
	}
	
	std::tuple<
		static_extents_type<2,1,1>, // 1
		static_extents_type<2,1,2>, // 2
		static_extents_type<2,2,1>, // 3
		static_extents_type<2,2,3>, // 4
		static_extents_type<3,2,3,1>, // 5
		static_extents_type<3,4,1,3>, // 6
		static_extents_type<3,1,2,3>, // 7
		static_extents_type<3,4,2,3>, // 8
		static_extents_type<4,4,2,3,5> // 9
	> static_extents{};

	std::vector<extents_type> extents;
};




BOOST_FIXTURE_TEST_CASE_TEMPLATE( test_tensor_prod_vector, value,  test_types, fixture )
{
	using namespace boost::numeric;
	using value_type   = typename value::first_type;
	using layout_type  = typename value::second_type;
	using tensor_type  = ublas::tensor<value_type, ublas::dynamic_extents<>,layout_type>;
	using vector_type  = typename tensor_type::vector_type;


	for(auto const& n : extents){

		auto a = tensor_type(n, value_type{2});

		for(auto m = 0u; m < n.size(); ++m){

			auto b = vector_type  (n[m], value_type{1} );

			auto c = ublas::prod(a, b, m+1);

			for(auto i = 0u; i < c.size(); ++i)
				BOOST_CHECK_EQUAL( c[i] , value_type(n[m]) * a[i] );

		}
	}
  auto n = extents[8];
  auto a = tensor_type(n, value_type{2});
  auto b = vector_type(n[0], value_type{1});

  auto zero_rank_empty_tensor = tensor_type{};
  auto empty = vector_type{};

  BOOST_CHECK_THROW(prod(a, b, 0), std::length_error);
  BOOST_CHECK_THROW(prod(a, b, 9), std::length_error);
  BOOST_CHECK_THROW(prod(zero_rank_empty_tensor, b, 1), std::length_error);
  BOOST_CHECK_THROW(prod(a, empty, 2), std::length_error);

}



BOOST_FIXTURE_TEST_CASE_TEMPLATE( test_static_tensor_prod_vector, value,  test_types, fixture )
{
	using namespace boost::numeric;
	using value_type   = typename value::first_type;
	using layout_type  = typename value::second_type;

	for_each_tuple(static_extents,[](auto const& I, auto const& n){                                   
		using extents_type = typename std::decay<decltype(n)>::type;              
		using tensor_type = ublas::tensor<value_type, extents_type, layout_type>; 
		using vector_type = typename tensor_type::vector_type;                    
		auto a = tensor_type(extents_type{}, value_type{2});
		
		for (auto m = 0u; m < n.size(); ++m) {                                    
																				
		auto b = vector_type(n[m], value_type{1});                               
																				
		auto c = ublas::prod(a, b, m + 1);                                       
																				
		for (auto i = 0u; i < c.size(); ++i)                                     
			BOOST_CHECK_EQUAL(c[i], value_type(n[m]) * a[i]);                     
		}                                   
	});

}



BOOST_AUTO_TEST_CASE( test_tensor_prod_vector_exception )
{
	using namespace boost::numeric;
	using value_type   = float;
	using layout_type  = ublas::first_order;
	using d_extents_type = ublas::dynamic_extents<>;
	using s_extents_type = ublas::shape<3,1,2,3>;
	using d_tensor_type  = ublas::tensor<value_type,d_extents_type,layout_type>;
	using s_tensor_type  = ublas::tensor<value_type,s_extents_type,layout_type>;
	using vector_type  = typename d_tensor_type::vector_type;

	auto t1 = d_tensor_type{d_extents_type{},1.f};
	auto v1 = vector_type{3,value_type{1}};

	BOOST_REQUIRE_THROW(prod(t1,v1,0),std::length_error);
	BOOST_REQUIRE_THROW(prod(t1,v1,1),std::length_error);
	BOOST_REQUIRE_THROW(prod(t1,v1,3),std::length_error);

	auto t2 = s_tensor_type{};
	BOOST_REQUIRE_THROW(prod(t2,v1,2),std::length_error);

	auto t3 = s_tensor_type{s_extents_type{},value_type{1}};
	auto v2 = vector_type{0,value_type{1}};
	BOOST_REQUIRE_THROW(prod(t3,v2,2),std::length_error);
}




BOOST_FIXTURE_TEST_CASE_TEMPLATE( test_tensor_prod_matrix, value,  test_types, fixture )
{
	using namespace boost::numeric;
	using value_type   = typename value::first_type;
	using layout_type  = typename value::second_type;
	using tensor_type  = ublas::tensor<value_type, ublas::dynamic_extents<>,layout_type>;
	using matrix_type  = typename tensor_type::matrix_type;


	for(auto const& n : extents) {

		auto a = tensor_type(n, value_type{2});

		for(auto m = 0u; m < n.size(); ++m){

			auto b  = matrix_type  ( n[m], n[m], value_type{1} );

			auto c = ublas::prod(a, b, m+1);

			for(auto i = 0u; i < c.size(); ++i)
				BOOST_CHECK_EQUAL( c[i] , value_type(n[m]) * a[i] );

		}
	}

  auto n = extents[8];
  auto a = tensor_type(n, value_type{2});
  auto b = matrix_type(n[0], n[0], value_type{1});

  auto zero_rank_empty_tensor = tensor_type{};
  auto empty = matrix_type{};

  BOOST_CHECK_THROW(prod(a, b, 0), std::length_error);
  BOOST_CHECK_THROW(prod(a, b, 9), std::length_error);
  BOOST_CHECK_THROW(prod(zero_rank_empty_tensor, b, 1), std::length_error);
  BOOST_CHECK_THROW(prod(a, empty, 2), std::length_error);
}


BOOST_FIXTURE_TEST_CASE_TEMPLATE( test_static_tensor_prod_matrix, value,  test_types, fixture )
{
	using namespace boost::numeric;
	using value_type   = typename value::first_type;
	using layout_type  = typename value::second_type;

	for_each_tuple(static_extents,[](auto const& I, auto const& n){                                                                   
		using extents_type = typename std::decay<decltype(n)>::type;             
		using tensor_type = ublas::tensor<value_type, extents_type, layout_type>;
		using matrix_type = typename tensor_type::matrix_type;                   
		auto a = tensor_type(extents_type{}, value_type{2});                     
		for (auto m = 0u; m < n.size(); ++m) {                                   
																				
			auto b = matrix_type  ( n[m], n[m], value_type{1} );                     
																					
			auto c = ublas::prod(a, b, m + 1);                                       
																					
			for (auto i = 0u; i < c.size(); ++i)                                     
				BOOST_CHECK_EQUAL(c[i], value_type(n[m]) * a[i]);                    
		}                                   
	});

}


BOOST_AUTO_TEST_CASE( test_tensor_prod_matrix_exception )
{
	using namespace boost::numeric;
	using value_type   = float;
	using layout_type  = ublas::first_order;
	using d_extents_type = ublas::dynamic_extents<>;
	using s_extents_type = ublas::shape<3,1,2,3>;
	using d_tensor_type  = ublas::tensor<value_type,d_extents_type,layout_type>;
	using s_tensor_type  = ublas::tensor<value_type,s_extents_type,layout_type>;
	using matrix_type  = typename d_tensor_type::matrix_type;

	auto t1 = d_tensor_type{d_extents_type{},1.f};
	auto m1 = matrix_type{3,3,value_type{1}};


	BOOST_REQUIRE_THROW(prod(t1,m1,0),std::length_error);
	BOOST_REQUIRE_THROW(prod(t1,m1,1),std::length_error);
	BOOST_REQUIRE_THROW(prod(t1,m1,3),std::length_error);

	auto t2 = s_tensor_type{};

	BOOST_REQUIRE_THROW(prod(t2,m1,2),std::length_error);

	auto t3 = s_tensor_type{s_extents_type{},value_type{1}};
	auto m2 = matrix_type{0,0,value_type{1}};
	BOOST_REQUIRE_THROW(prod(t3,m2,2),std::length_error);
}

BOOST_FIXTURE_TEST_CASE_TEMPLATE( test_tensor_prod_tensor_1, value,  test_types, fixture )
{
	using namespace boost::numeric;
	using value_type   = typename value::first_type;
	using layout_type  = typename value::second_type;
	using tensor_type  = ublas::tensor<value_type, ublas::dynamic_extents<>,layout_type>;

	// left-hand and right-hand side have the
	// the same number of elements

	for(auto const& na : extents) {

		auto a  = tensor_type( na, value_type{2} );
		auto b  = tensor_type( na, value_type{3} );

		auto const pa = a.rank();

		// the number of contractions is changed.
		for( auto q = 0ul; q <= pa; ++q) { // pa

			auto phi = std::vector<std::size_t> ( q );

			std::iota(phi.begin(), phi.end(), 1ul);

			auto c = ublas::prod(a, b, phi);

			auto acc = value_type(1);
			for(auto i = 0ul; i < q; ++i)
				acc *= a.extents().at(phi.at(i)-1);

			for(auto i = 0ul; i < c.size(); ++i)
				BOOST_CHECK_EQUAL( c[i] , acc * a[0] * b[0] );

		}
	}
}


BOOST_FIXTURE_TEST_CASE_TEMPLATE( test_static_tensor_prod_tensor_1, value,  test_types, fixture )
{
	using namespace boost::numeric;
	using value_type   = typename value::first_type;
	using layout_type  = typename value::second_type;

	auto const body = [](auto const& a, auto const& b){																	
		auto const pa = a.rank();                                                 
																				
		for (auto q = 0ul; q <= pa; ++q) {                                        
																				
		auto phi = std::vector<std::size_t>(q);                                  
																				
		std::iota(phi.begin(), phi.end(), 1ul);                                  
																				
		auto c = ublas::prod(a, b, phi);                                         
																				
		auto acc = value_type(1);                                                
		for (auto i = 0ul; i < q; ++i)                                           
			acc *= a.extents().at(phi.at(i) - 1);                                 
																				
		for (auto i = 0ul; i < c.size(); ++i)                                    
			BOOST_CHECK_EQUAL(c[i], acc *a[0] * b[0]);                            
		}                
	};


	//static extents and static_extents
	for_each_tuple(static_extents,[&](auto const& I, auto const& n){                                                                   
		auto n1 = n;                                                              
		auto n2 = n;                                                              
		using extents_type_1 = typename std::decay<decltype(n1)>::type;           
		using extents_type_2 = typename std::decay<decltype(n2)>::type;           
		using tensor_type_1 =                                                     
			ublas::tensor<value_type, extents_type_1, layout_type>;               
		using tensor_type_2 =                                                     
			ublas::tensor<value_type, extents_type_2, layout_type>;               
		auto a = tensor_type_1(n1, value_type{2});                                
		auto b = tensor_type_2(n2, value_type{3});                                
		body(a,b);      
	});

	for_each_tuple(static_extents,[&](auto const& I, auto const& n){                                                                   
		auto n1 = n;                                                              
		auto n2 = extents[I];                                                              
		using extents_type_1 = typename std::decay<decltype(n1)>::type;           
		using extents_type_2 = typename std::decay<decltype(n2)>::type;           
		using tensor_type_1 =                                                     
			ublas::tensor<value_type, extents_type_1, layout_type>;               
		using tensor_type_2 =                                                     
			ublas::tensor<value_type, extents_type_2, layout_type>;               
		auto a = tensor_type_1(n1, value_type{2});                                
		auto b = tensor_type_2(n2, value_type{3});                                
		body(a,b);      
	});

	for_each_tuple(static_extents,[&](auto const& I, auto const& n){                                                                   
		auto n1 = extents[I];                                                              
		auto n2 = n;                                                              
		using extents_type_1 = typename std::decay<decltype(n1)>::type;           
		using extents_type_2 = typename std::decay<decltype(n2)>::type;           
		using tensor_type_1 =                                                     
			ublas::tensor<value_type, extents_type_1, layout_type>;               
		using tensor_type_2 =                                                     
			ublas::tensor<value_type, extents_type_2, layout_type>;               
		auto a = tensor_type_1(n1, value_type{2});                                
		auto b = tensor_type_2(n2, value_type{3});                                
		body(a,b);      
	});

}



BOOST_AUTO_TEST_CASE( test_tensor_prod_tensor_1_exception )
{
	using namespace boost::numeric;
	using value_type   = float;
	using layout_type  = ublas::first_order;
	using d_extents_type = ublas::dynamic_extents<>;
	using s_extents_type = ublas::shape<3,1,2,3>;
	using d_tensor_type  = ublas::tensor<value_type,d_extents_type,layout_type>;
	using s_tensor_type  = ublas::tensor<value_type,s_extents_type,layout_type>;

	auto t1 = d_tensor_type{};
	auto t2 = s_tensor_type{s_extents_type{},1.f};
	std::vector<std::size_t> phia = {1,2,3};
	std::vector<std::size_t> phib = {1,2,3,4,5};

	BOOST_REQUIRE_THROW(prod(t1,t2,phia,phib),std::runtime_error);
	BOOST_REQUIRE_THROW(prod(t2,t1,phia,phib),std::runtime_error);


	auto t3 = d_tensor_type{d_extents_type{1,2},1.f};
	auto t4 = d_tensor_type{d_extents_type{1,2},1.f};
	BOOST_REQUIRE_THROW(prod(t3,t4,phia,phib),std::runtime_error);


	auto t5 = d_tensor_type{d_extents_type{1,2,3,4},1.f};
	auto t6 = d_tensor_type{d_extents_type{1,2},1.f};
	BOOST_REQUIRE_THROW(prod(t5,t6,phia,phib),std::runtime_error);


	auto t7 = d_tensor_type{d_extents_type{1,2,3,4,5},1.f};
	auto t8 = d_tensor_type{d_extents_type{1,2,3,4,5},1.f};
	BOOST_REQUIRE_THROW(prod(t7,t8,phia,phib),std::runtime_error);

	std::vector<std::size_t> phia_2 = {1,2,3,5,4};
	std::vector<std::size_t> phib_2 = {1,2,3,4,5};
	auto t9 = d_tensor_type{d_extents_type{1,2,3,4,5,6},1.f};
	auto t10 = d_tensor_type{d_extents_type{1,2,3,4,5,6},1.f};
	BOOST_REQUIRE_THROW(prod(t9,t10,phia_2,phib_2),std::runtime_error);
}

BOOST_FIXTURE_TEST_CASE_TEMPLATE( test_tensor_prod_tensor_2, value,  test_types, fixture )
{
	using namespace boost::numeric;
	using value_type   = typename value::first_type;
	using layout_type  = typename value::second_type;
	using tensor_type  = ublas::tensor<value_type, ublas::dynamic_extents<>,layout_type>;


	auto compute_factorial = [](auto const& p){
		auto f = 1ul;
		for(auto i = 1u; i <= p; ++i)
			f *= i;
		return f;
	};

	auto permute_extents = [](auto const& pi, auto const& na){
		auto nb = na;
		assert(pi.size() == na.size());
		for(auto j = 0u; j < pi.size(); ++j)
			nb[pi[j]-1] = na[j];
		return nb;
	};


	// left-hand and right-hand side have the
	// the same number of elements

	for(auto const& na : extents) {

		auto a  = tensor_type( na, value_type{2} );
		auto const pa = a.rank();


		auto pi   = std::vector<std::size_t>(pa);
		auto fac = compute_factorial(pa);
		std::iota( pi.begin(), pi.end(), 1 );

		for(auto f = 0ul; f < fac; ++f)
		{
			auto nb = permute_extents( pi, na  );
			auto b  = tensor_type( nb, value_type{3} );

			// the number of contractions is changed.
			for( auto q = 0ul; q <= pa; ++q) { // pa

				auto phia = std::vector<std::size_t> ( q );  // concatenation for a
				auto phib = std::vector<std::size_t> ( q );  // concatenation for b

				std::iota(phia.begin(), phia.end(), 1ul);
				std::transform(  phia.begin(), phia.end(), phib.begin(),
				                 [&pi] ( std::size_t i ) { return pi.at(i-1); } );

				auto c = ublas::prod(a, b, phia, phib);

				auto acc = value_type(1);
				for(auto i = 0ul; i < q; ++i)
					acc *= a.extents().at(phia.at(i)-1);

				for(auto i = 0ul; i < c.size(); ++i)
					BOOST_CHECK_EQUAL( c[i] , acc * a[0] * b[0] );

			}

			std::next_permutation(pi.begin(), pi.end());
		}
	}

	auto phia = std::vector<std::size_t >(3);
	auto sphia = std::vector<std::size_t>(2);

	BOOST_CHECK_THROW(ublas::prod(tensor_type{}, tensor_type({2,1,2}), phia, phia), std::runtime_error);
        BOOST_CHECK_THROW(ublas::prod(tensor_type({1,2,3}), tensor_type(), phia, phia), std::runtime_error);
        BOOST_CHECK_THROW(ublas::prod(tensor_type({1,2,4}), tensor_type({2,1}), phia, phia), std::runtime_error);
        BOOST_CHECK_THROW(ublas::prod(tensor_type({1,2}), tensor_type({2,1,2}), phia, phia), std::runtime_error);
        BOOST_CHECK_THROW(ublas::prod(tensor_type({1,2}), tensor_type({2,1,3}), sphia, phia), std::runtime_error);
        BOOST_CHECK_THROW(ublas::prod(tensor_type({1,2}), tensor_type({2,2}), phia, sphia), std::runtime_error);
        BOOST_CHECK_THROW(ublas::prod(tensor_type({1,2}), tensor_type({4,4}), sphia, phia), std::runtime_error);
}



BOOST_FIXTURE_TEST_CASE_TEMPLATE( test_static_tensor_prod_tensor_2, value,  test_types, fixture )
{
	using namespace boost::numeric;
	using value_type   = typename value::first_type;
	using layout_type  = typename value::second_type;



	auto compute_factorial = [](auto const& p){
		auto f = 1ul;
		for(auto i = 1u; i <= p; ++i)
			f *= i;
		return f;
	};

	auto permute_extents_d = [](auto const& pi, auto const& na){
		auto nb = na;
		assert(pi.size() == na.size());
		for(auto j = 0u; j < pi.size(); ++j)
			nb[pi[j]-1] = na[j];
		return nb;
	};

	auto permute_extents_s_1 = [](auto const& pi, auto const& na){
		auto nb = na.to_dynamic_extents();
		assert(pi.size() == na.size());
		for(auto j = 0u; j < pi.size(); ++j)
			nb[pi[j]-1] = na[j];
		return nb;
	};
	auto permute_extents_s_2 = [](auto const& pi, auto const& na){
		auto tempn = na.to_vector();
		assert(pi.size() == na.size());
		for(auto j = 0u; j < pi.size(); ++j)
			tempn[pi[j]-1] = na[j];
		return ublas::shape<std::decay<decltype(na)>::type::rank()>(tempn.begin(),tempn.end());
	};

// 	static and dynamic
	for_each_tuple(static_extents,[&](auto const& I, auto const& n){
		auto na = n;                                                              
		using extents_type_1 = typename std::decay<decltype(na)>::type;             
		using tensor_type_1 = ublas::tensor<value_type, extents_type_1, layout_type>; 
		auto a = tensor_type_1(na, value_type{2});                                  
		auto const pa = a.rank();                                                 
																				
		auto pi = std::vector<std::size_t>(pa);                                   
		auto fac = compute_factorial(pa);                                         
		std::iota(pi.begin(), pi.end(), 1);                                       
																				
		for (auto f = 0ul; f < fac; ++f) {                                        
			auto nb = permute_extents_s_1(pi, na); 
			using extents_type_2 = typename std::decay<decltype(nb)>::type;  
			using tensor_type_2 = ublas::tensor<value_type, extents_type_2, layout_type>;                                                     
			auto b = tensor_type_2(nb, value_type{3});                                 
																					
			for (auto q = 0ul; q <= pa; ++q) {                                       
																					
				auto phia = std::vector<std::size_t>(q);                              
				auto phib = std::vector<std::size_t>(q);                              
																					
				std::iota(phia.begin(), phia.end(), 1ul);                             
				std::transform(phia.begin(), phia.end(), phib.begin(),                
							[&pi](std::size_t i) { return pi.at(i - 1); });         
																					
				auto c = ublas::prod(a, b, phia, phib);                               
																					
				auto acc = value_type(1);                                             
				for (auto i = 0ul; i < q; ++i)                                        
				acc *= a.extents().at(phia.at(i) - 1);                               
																					
				for (auto i = 0ul; i < c.size(); ++i)                                 
				BOOST_CHECK_EQUAL(c[i], acc *a[0] * b[0]);                           
			}                                                                        
																					
			std::next_permutation(pi.begin(), pi.end());                             
		}  
	});

	//dynamic and static
	for_each_tuple(static_extents,[&](auto const& I, auto const& n){
		auto na = extents[I];                                                              
		using extents_type_1 = typename std::decay<decltype(na)>::type;             
		using tensor_type_1 = ublas::tensor<value_type, extents_type_1, layout_type>; 
		auto a = tensor_type_1(na, value_type{2});                                  
		auto const pa = a.rank();                                                 
																				
		auto pi = std::vector<std::size_t>(pa);                                   
		auto fac = compute_factorial(pa);                                         
		std::iota(pi.begin(), pi.end(), 1);                                       
																				
		for (auto f = 0ul; f < fac; ++f) {                                        
			auto nb = permute_extents_d(pi, na); 
			using extents_type_2 = typename std::decay<decltype(nb)>::type;  
			using tensor_type_2 = ublas::tensor<value_type, extents_type_2, layout_type>;                                                     
			auto b = tensor_type_2(nb, value_type{3});                                 
																					
			for (auto q = 0ul; q <= pa; ++q) {                                       
																					
				auto phia = std::vector<std::size_t>(q);                              
				auto phib = std::vector<std::size_t>(q);                              
																					
				std::iota(phia.begin(), phia.end(), 1ul);                             
				std::transform(phia.begin(), phia.end(), phib.begin(),                
							[&pi](std::size_t i) { return pi.at(i - 1); });         
																					
				auto c = ublas::prod(a, b, phia, phib);                               
																					
				auto acc = value_type(1);                                             
				for (auto i = 0ul; i < q; ++i)                                        
				acc *= a.extents().at(phia.at(i) - 1);                               
																					
				for (auto i = 0ul; i < c.size(); ++i)                                 
				BOOST_CHECK_EQUAL(c[i], acc *a[0] * b[0]);                           
			}                                                                        
																					
			std::next_permutation(pi.begin(), pi.end());                             
		}  
	});
	//static and static
	for_each_tuple(static_extents,[&](auto const& I, auto const& n){
		auto na = n;                                                              
		using extents_type_1 = typename std::decay<decltype(na)>::type;             
		using tensor_type_1 = ublas::tensor<value_type, extents_type_1, layout_type>; 
		auto a = tensor_type_1(na, value_type{2});                                  
		auto const pa = a.rank();                                                 
																				
		auto pi = std::vector<std::size_t>(pa);                                   
		auto fac = compute_factorial(pa);                                         
		std::iota(pi.begin(), pi.end(), 1);                                       
																				
		for (auto f = 0ul; f < fac; ++f) {                                        
			auto nb = permute_extents_s_2(pi, na); 
			
			using extents_type_2 = typename std::decay<decltype(nb)>::type;  
			using tensor_type_2 = ublas::tensor<value_type, extents_type_2, layout_type>;                                                     
			auto b = tensor_type_2(nb, value_type{3});                                 
																					
			for (auto q = 0ul; q <= pa; ++q) {                                       
																					
				auto phia = std::vector<std::size_t>(q);                              
				auto phib = std::vector<std::size_t>(q);                              
																					
				std::iota(phia.begin(), phia.end(), 1ul);                             
				std::transform(phia.begin(), phia.end(), phib.begin(),                
							[&pi](std::size_t i) { return pi.at(i - 1); });         
																					
				auto c = ublas::prod(a, b, phia, phib);                               
																					
				auto acc = value_type(1);                                             
				for (auto i = 0ul; i < q; ++i)                                        
				acc *= a.extents().at(phia.at(i) - 1);                               
																					
				for (auto i = 0ul; i < c.size(); ++i)                                 
				BOOST_CHECK_EQUAL(c[i], acc *a[0] * b[0]);                           
			}                                                                        
																					
			std::next_permutation(pi.begin(), pi.end());                             
		}  
	});

}


BOOST_FIXTURE_TEST_CASE_TEMPLATE( test_tensor_inner_prod, value,  test_types, fixture )
{
	using namespace boost::numeric;
	using value_type   = typename value::first_type;
	using layout_type  = typename value::second_type;
	using tensor_type  = ublas::tensor<value_type, ublas::dynamic_extents<>,layout_type>;


	for(auto const& n : extents) {

		auto a  = tensor_type(n, value_type(2));
		auto b  = tensor_type(n, value_type(1));

		auto c = ublas::inner_prod(a, b);
		auto r = std::inner_product(a.begin(),a.end(), b.begin(),value_type(0));

		BOOST_CHECK_EQUAL( c , r );

	}
  BOOST_CHECK_THROW(ublas::inner_prod(tensor_type({1,2,3}), tensor_type({1,2,3,4})), std::length_error); // rank different
  BOOST_CHECK_THROW(ublas::inner_prod(tensor_type(), tensor_type()), std::length_error); //empty tensor
  BOOST_CHECK_THROW(ublas::inner_prod(tensor_type({1,2,3}), tensor_type({3,2,1})), std::length_error); // different extent
}


BOOST_AUTO_TEST_CASE( test_tensor_inner_prod_exception )
{
	using namespace boost::numeric;
	using value_type   = float;
	using layout_type  = ublas::first_order;
	using d_extents_type = ublas::dynamic_extents<>;
	using s_extents_type = ublas::shape<3,1,2,3>;
	using d_tensor_type  = ublas::tensor<value_type,d_extents_type,layout_type>;
	using s_tensor_type  = ublas::tensor<value_type,s_extents_type,layout_type>;

	auto t1 = d_tensor_type{d_extents_type{1,2},1.f};
	auto t2 = d_tensor_type{d_extents_type{1,2,3},1.f};
	BOOST_REQUIRE_THROW( ublas::inner_prod(t1, t2), std::length_error);

	auto t3 = s_tensor_type{s_extents_type{},1.f};
	auto t4 = d_tensor_type{d_extents_type{1,2,4,5},1.f};
	BOOST_REQUIRE_THROW( ublas::inner_prod(t3, t4), std::length_error);
}



BOOST_FIXTURE_TEST_CASE_TEMPLATE( test_static_tensor_inner_prod, value,  test_types, fixture )
{
	using namespace boost::numeric;
	using value_type   = typename value::first_type;
	using layout_type  = typename value::second_type;

	auto const body = [&](auto const& a, auto const& b){
		auto c = ublas::inner_prod(a, b);
		auto r = std::inner_product(a.begin(),a.end(), b.begin(),value_type(0));

		BOOST_CHECK_EQUAL( c , r );
	};

	for_each_tuple(static_extents,[&](auto const& I, auto const& n){
		using extents_type_1 = typename std::decay<decltype(n)>::type;             
		using extents_type_2 = typename std::decay<decltype(n)>::type;             
		using tensor_type_1 = ublas::tensor<value_type, extents_type_1, layout_type>;
		using tensor_type_2 = ublas::tensor<value_type, extents_type_2, layout_type>;
		auto a  = tensor_type_1(n, value_type(2));
		auto b  = tensor_type_2(n, value_type(1));
		body(a,b);

	});

	for_each_tuple(static_extents,[&](auto const& I, auto const& n){
		using extents_type_1 = typename std::decay<decltype(n)>::type;             
		using extents_type_2 = typename std::decay<decltype(extents[I])>::type;             
		using tensor_type_1 = ublas::tensor<value_type, extents_type_1, layout_type>;
		using tensor_type_2 = ublas::tensor<value_type, extents_type_2, layout_type>;
		auto a  = tensor_type_1(n, value_type(2));
		auto b  = tensor_type_2(extents[I], value_type(1));
		body(a,b);

	});

	for_each_tuple(static_extents,[&](auto const& I, auto const& n){
		using extents_type_1 = typename std::decay<decltype(extents[I])>::type;             
		using extents_type_2 = typename std::decay<decltype(n)>::type;             
		using tensor_type_1 = ublas::tensor<value_type, extents_type_1, layout_type>;
		using tensor_type_2 = ublas::tensor<value_type, extents_type_2, layout_type>;
		auto a  = tensor_type_1(extents[I], value_type(2));
		auto b  = tensor_type_2(n, value_type(1));
		body(a,b);

	});

}


BOOST_FIXTURE_TEST_CASE_TEMPLATE( test_tensor_norm, value,  test_types, fixture )
{
	using namespace boost::numeric;
	using value_type   = typename value::first_type;
	using layout_type  = typename value::second_type;
	using tensor_type  = ublas::tensor<value_type, ublas::dynamic_extents<>,layout_type>;


	for(auto const& n : extents) {

		auto a  = tensor_type(n);

		auto one = value_type(1);
		auto v = one;
		for(auto& aa: a)
			aa = v, v += one;


		auto c = ublas::inner_prod(a, a);
		auto r = std::inner_product(a.begin(),a.end(), a.begin(),value_type(0));

		tensor_type var = (a+a)/2.0f; // std::complex<float>/int not allowed as expression is captured
		auto r2 = ublas::norm( var );

		BOOST_CHECK_THROW(ublas::norm(tensor_type{}), std::runtime_error);

		BOOST_CHECK_EQUAL( c , r );
		BOOST_CHECK_EQUAL( std::sqrt( c ) , r2 );

	}
}

BOOST_FIXTURE_TEST_CASE( test_tensor_real_imag_conj, fixture )
{
	using namespace boost::numeric;
	using value_type   = float;
	using complex_type = std::complex<value_type>;
	using layout_type  = ublas::first_order;

	using tensor_complex_type  = ublas::tensor<complex_type, ublas::dynamic_extents<>,layout_type>;
	using tensor_type  = ublas::tensor<value_type, ublas::dynamic_extents<>,layout_type>;

	for(auto const& n : extents) {

		auto a   = tensor_type(n);
		auto r0  = tensor_type(n);
		auto r00 = tensor_complex_type(n);


		auto one = value_type(1);
		auto v = one;
		for(auto& aa: a)
			aa = v, v += one;

		tensor_type b = (a+a) / value_type( 2 );
		tensor_type r1 = ublas::real( (a+a) / value_type( 2 )  );
		std::transform(  b.begin(), b.end(), r0.begin(), [](auto const& l){ return std::real( l );  }   );
		BOOST_CHECK( (bool) (r0 == r1) );

		tensor_type r2 = ublas::imag( (a+a) / value_type( 2 )  );
		std::transform(  b.begin(), b.end(), r0.begin(), [](auto const& l){ return std::imag( l );  }   );
		BOOST_CHECK( (bool) (r0 == r2) );

		tensor_complex_type r3 = ublas::conj( (a+a) / value_type( 2 )  );
		std::transform(  b.begin(), b.end(), r00.begin(), [](auto const& l){ return std::conj( l );  }   );
		BOOST_CHECK( (bool) (r00 == r3) );

	}

	for(auto const& n : extents) {




		auto a   = tensor_complex_type(n);

		auto r00 = tensor_complex_type(n);
		auto r0  = tensor_type(n);


		auto one = complex_type(1,1);
		auto v = one;
		for(auto& aa: a)
			aa = v, v = v + one;

		tensor_complex_type b = (a+a) / complex_type( 2,2 );


		tensor_type r1 = ublas::real( (a+a) / complex_type( 2,2 )  );
		std::transform(  b.begin(), b.end(), r0.begin(), [](auto const& l){ return std::real( l );  }   );
		BOOST_CHECK( (bool) (r0 == r1) );

		tensor_type r2 = ublas::imag( (a+a) / complex_type( 2,2 )  );
		std::transform(  b.begin(), b.end(), r0.begin(), [](auto const& l){ return std::imag( l );  }   );
		BOOST_CHECK( (bool) (r0 == r2) );

		tensor_complex_type r3 = ublas::conj( (a+a) / complex_type( 2,2 )  );
		std::transform(  b.begin(), b.end(), r00.begin(), [](auto const& l){ return std::conj( l );  }   );
		BOOST_CHECK( (bool) (r00 == r3) );



	}



}





BOOST_FIXTURE_TEST_CASE_TEMPLATE( test_tensor_outer_prod, value,  test_types, fixture )
{
	using namespace boost::numeric;
	using value_type   = typename value::first_type;
	using layout_type  = typename value::second_type;
	using tensor_type  = ublas::tensor<value_type, ublas::dynamic_extents<>,layout_type>;

	for(auto const& n1 : extents) {
		auto a  = tensor_type(n1, value_type(2));
		for(auto const& n2 : extents) {

			auto b  = tensor_type(n2, value_type(1));
			auto c  = ublas::outer_prod(a, b);

			for(auto const& cc : c)
				BOOST_CHECK_EQUAL( cc , a[0]*b[0] );
		}
	}
}


BOOST_FIXTURE_TEST_CASE_TEMPLATE( test_static_tensor_outer_prod, value,  test_types, fixture )
{
	using namespace boost::numeric;
	using value_type   = typename value::first_type;
	using layout_type  = typename value::second_type;

	for_each_tuple(static_extents,[&](auto const& I, auto const& n1){
		using extents_type_1 = typename std::decay<decltype(n1)>::type;             
		using tensor_type_1 = ublas::tensor<value_type, extents_type_1, layout_type>;
		auto a  = tensor_type_1(n1, value_type(2));
		for_each_tuple(static_extents,[&](auto const& J, auto const& n2){
			using extents_type_2 = typename std::decay<decltype(n2)>::type;             
			using tensor_type_2 = ublas::tensor<value_type, extents_type_2, layout_type>;
			auto b  = tensor_type_2(n2, value_type(1));
			auto c  = ublas::outer_prod(a, b);

			for(auto const& cc : c)
				BOOST_CHECK_EQUAL( cc , a[0]*b[0] );
			
		});

	});

	for_each_tuple(static_extents,[&](auto const& I, auto const& n1){
		using extents_type_1 = typename std::decay<decltype(extents[I])>::type;             
		using tensor_type_1 = ublas::tensor<value_type, extents_type_1, layout_type>;
		auto a  = tensor_type_1(extents[I], value_type(2));
		for_each_tuple(static_extents,[&](auto const& J, auto const& n2){
			using extents_type_2 = typename std::decay<decltype(n2)>::type;             
			using tensor_type_2 = ublas::tensor<value_type, extents_type_2, layout_type>;
			auto b  = tensor_type_2(n2, value_type(1));
			auto c  = ublas::outer_prod(a, b);

			for(auto const& cc : c)
				BOOST_CHECK_EQUAL( cc , a[0]*b[0] );
		});

	});

	for_each_tuple(static_extents,[&](auto const& I, auto const& n1){
		using extents_type_1 = typename std::decay<decltype(n1)>::type;             
		using tensor_type_1 = ublas::tensor<value_type, extents_type_1, layout_type>;
		auto a  = tensor_type_1(n1, value_type(2));
		for(auto n2 : extents){
			using extents_type_2 = typename std::decay<decltype(n2)>::type;             
			using tensor_type_2 = ublas::tensor<value_type, extents_type_2, layout_type>;
			auto b  = tensor_type_2(n2, value_type(1));
			auto c  = ublas::outer_prod(a, b);

			for(auto const& cc : c)
				BOOST_CHECK_EQUAL( cc , a[0]*b[0] );
		}

	});

}

template<class V>
void init(std::vector<V>& a)
{
	auto v = V(1);
	for(auto i = 0u; i < a.size(); ++i, ++v){
		a[i] = v;
	}
}

template<class V>
void init(std::vector<std::complex<V>>& a)
{
	auto v = std::complex<V>(1,1);
	for(auto i = 0u; i < a.size(); ++i){
		a[i] = v;
		v.real(v.real()+1);
		v.imag(v.imag()+1);
	}
}

BOOST_FIXTURE_TEST_CASE_TEMPLATE( test_tensor_trans, value,  test_types, fixture )
{
	using namespace boost::numeric;
	using value_type   = typename value::first_type;
	using layout_type  = typename value::second_type;
	using tensor_type  = ublas::tensor<value_type, ublas::dynamic_extents<>,layout_type>;

	auto fak = [](auto const& p){
		auto f = 1ul;
		for(auto i = 1u; i <= p; ++i)
			f *= i;
		return f;
	};

	auto inverse = [](auto const& pi){
		auto pi_inv = pi;
		for(auto j = 0u; j < pi.size(); ++j)
			pi_inv[pi[j]-1] = j+1;
		return pi_inv;
	};

	for(auto const& n : extents)
	{
		auto const p = n.size();
		auto const s = product(n);
		auto aref = tensor_type(n);
		auto v    = value_type{};
		for(auto i = 0u; i < s; ++i, v+=1)
			aref[i] = v;
		auto a    = aref;


		auto pi = std::vector<std::size_t>(p);
		std::iota(pi.begin(), pi.end(), 1);
		a = ublas::trans( a, pi );
		bool res1 = a == aref;
		BOOST_CHECK( res1 );


		auto const pfak = fak(p);
		auto i = 0u;
		for(; i < pfak-1; ++i) {
			std::next_permutation(pi.begin(), pi.end());
			a = ublas::trans( a, pi );
		}
		std::next_permutation(pi.begin(), pi.end());
		for(; i > 0; --i) {
			std::prev_permutation(pi.begin(), pi.end());
			auto pi_inv = inverse(pi);
			a = ublas::trans( a, pi_inv );
		}
                bool res2 = a == aref; // it was an expression. so evaluate into bool
		BOOST_CHECK( res2 );

	}
}

BOOST_FIXTURE_TEST_CASE_TEMPLATE( test_static_tensor_trans, value,  test_types, fixture )
{
	using namespace boost::numeric;
	using value_type   = typename value::first_type;
	using layout_type  = typename value::second_type;

	auto fac = [](auto const& p){
		auto f = 1ul;
		for(auto i = 1u; i <= p; ++i)
			f *= i;
		return f;
	};

	auto inverse = [](auto const& pi){
		auto pi_inv = pi;
		for(auto j = 0u; j < pi.size(); ++j)
			pi_inv[pi[j]-1] = j+1;
		return pi_inv;
	};

	for_each_tuple(static_extents,[&](auto const& I, auto const& n){
		using extents_type = typename std::decay<decltype(n)>::type;
		using tensor_type  = ublas::tensor<value_type, extents_type,layout_type>;
		auto const p = n.size();
		auto const s = product(n);
		auto aref = tensor_type(n);
		auto v    = value_type{};
		for(auto i = 0u; i < s; ++i, v+=1)
			aref[i] = v;

		auto pi = std::vector<std::size_t>(p);
		std::iota(pi.begin(), pi.end(), 1);
		
		auto a = ublas::trans( aref, pi );
		
		for(auto i = 0; i < a.size(); i++){
			BOOST_CHECK( a[i] == aref[i]  );
		}


		auto const pfac = fac(p);
		auto i = 0u;
		for(; i < pfac-1; ++i) {
			std::next_permutation(pi.begin(), pi.end());
			a = ublas::trans( a, pi );
		}
		std::next_permutation(pi.begin(), pi.end());
		for(; i > 0; --i) {
			std::prev_permutation(pi.begin(), pi.end());
			auto pi_inv = inverse(pi);
			a = ublas::trans( a, pi_inv );
		}

		for(auto i = 0; i < a.size(); i++){
			BOOST_CHECK( a[i] == aref[i]  );
		}
	});

}

BOOST_AUTO_TEST_SUITE_END()

